import numpy as np
import math
from sparse import sparse
from numba import jit, int32, float32

def PowerAxis(minval, maxval, n_bins, power):
    l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
    bin_edges = np.power(l, power)
    return bin_edges

@jit((float32, float32, float32, float32, float32, float32, float32, float32, float32), nopython=True, nogil=True, fastmath=True)
def numba_bin_indices(t, x, y, z, radius, t_scaling_factor, r_scaling_factor, theta_scaling_factor, phi_scaling_factor):
    t_index = int(t * t_scaling_factor)
    r_index = int(math.sqrt(radius * r_scaling_factor))
    if radius == 0.:
        cos_theta = 1.
    else:
        cos_theta = z / radius
    theta_index = int((-cos_theta + 1.) * theta_scaling_factor)
    phi = math.atan2(y, x)
    phi_index = int(phi * phi_scaling_factor)
    return (t_index, r_index, theta_index, phi_index)

@jit()
def numba_create_photon_matrix(t, x, y, z, theta_v, phi_v, trck_length, cscd_photons, time_increment, scaled_time_increment, time_increment_scaling, min_time_increment, n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_min, t_max, r_min, r_max):
    #create dictionary
    z_dict = {}
    #declare constants
    speed_of_light = 2.99e8
    photons_per_meter = 2451.4544553
    time_increment = time_increment * 1e-9
    scaled_time_increment = scaled_time_increment
    segment_length = time_increment * speed_of_light
    cumulative_track_length = 0.
    #calculate frequently used values
    radius = math.sqrt(x * x + y * y + z * z)
    sin_theta_v = math.sin(theta_v)
    cos_theta_v = math.cos(theta_v)
    sin_phi_v = math.sin(phi_v)
    cos_phi_v = math.cos(phi_v)
    speed_x = speed_of_light * sin_theta_v * cos_phi_v
    speed_y = speed_of_light * sin_theta_v * sin_phi_v
    speed_z = speed_of_light * cos_theta_v
    #calculate scaling factors
    t_scaling_factor = (t_max - t_min) / n_t_bins
    r_scaling_factor = n_r_bins * n_r_bins / r_max
    theta_scaling_factor = n_theta_bins / 2.
    phi_scaling_factor = n_phi_bins / 2.
    #add cascade photons
    if radius < r_max:
        t_index = int(t * t_scaling_factor)
        r_index = int(math.sqrt(radius * r_scaling_factor))
        if radius == 0.:
            cos_theta = 1.
        else:
            cos_theta = z / radius
        theta_index = int((-cos_theta + 1.) * theta_scaling_factor)
        phi = math.atan2(y, x)
        phi_index = int(phi * phi_scaling_factor)
        key = (t_index, r_index, theta_index, phi_index)
        if key in dict.keys(z_dict):
            z_dict[key] += cscd_photons
        else:
            z_dict[key] = cscd_photons
    # traverse track and add track photons if within radius of the dom
    while cumulative_track_length < trck_length and t < t_max:
        radius = math.sqrt(x * x + y * y + z * z)
        if radius < r_max:
            t_index = int(t * t_scaling_factor)
            r_index = int(math.sqrt(radius * r_scaling_factor))
            if radius == 0.:
                cos_theta = 1.
            else:
                cos_theta = z / radius
            theta_index = int((-cos_theta + 1.) * theta_scaling_factor)
            phi = math.atan2(y, x)
            phi_index = int(phi * phi_scaling_factor)
            key = (t_index, r_index, theta_index, phi_index)
            if key in dict.keys(z_dict):
                z_dict[key] += segment_length * photons_per_meter
            else:
                z_dict[key] = segment_length * photons_per_meter
        if scaled_time_increment == True:
            if radius * time_increment_scaling > min_time_increment:
                time_increment = radius * time_increment_scaling
                segment_length = speed_of_light * time_increment
            else:
                time_increment = min_time_increment
                segment_length = speed_of_light * time_increment
        cumulative_track_length += segment_length
        t += time_increment
        x += speed_x * time_increment
        y += speed_y * time_increment
        z += speed_z * time_increment
        return z_dict   


class segment_hypo(object):
    '''
    create hypo using individual segments and retrieve matrix that contains expected photons in each cell in spherical coordinate system with dom at origin.
    binnnings and location of the dom must be set
    '''
    def __init__(self, t_v, x_v, y_v, z_v, theta_v, phi_v, trck_energy, cscd_energy, time_increment=1.):
        '''
        provide vertex and track information
        t_v : time (ns)
        x_v, y_v, z_v : vertex position (m)
        theta : zenith (rad)
        phi : azimuth (rad)
        trck_energy : track energy (GeV)
        cscd_energy : cascade energy (GeV)
        time_increment : if using constant time increments, it is the length of time between photon dumps in ns
        '''     
        #assign vertex
        self.t = t_v * 1e-9
        self.x = x_v
        self.y = y_v
        self.z = z_v
        self.theta_v = theta_v
        self.phi_v = phi_v
        #declaring constants 
        self.time_increment = time_increment * 1e-9
        self.scaled_time_increment = False
        self.speed_of_light = 2.99e8
        self.segment_length = self.time_increment * self.speed_of_light
        #calculate frequently used values
        self.radius = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.sin_theta_v = math.sin(self.theta_v)
        self.cos_theta_v = math.cos(self.theta_v)
        self.sin_phi_v = math.sin(self.phi_v)
        self.cos_phi_v = math.cos(self.phi_v)
        self.speed_x = self.speed_of_light * self.sin_theta_v * self.cos_phi_v
        self.speed_y = self.speed_of_light * self.sin_theta_v * self.sin_phi_v
        self.speed_z = self.speed_of_light * self.cos_theta_v
        #convert track energy to length
        self.trck_length = 15. / 3.3 * trck_energy
        #precalculated (nphotons.py) to avoid icetray
        self.photons_per_meter = 2451.4544553
        self.cscd_photons = 12805.3383311 * cscd_energy
        self.trck_photons = self.trck_length * self.photons_per_meter
        self.tot_photons = self.cscd_photons + self.trck_photons

    
    def set_binning(self, n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_max, r_max, t_min=0, r_min=0):
        '''
        define binnings of spherical coordinates
        t_min : min time (ns)
        t_max : max time (ns)
        r_min : min radius (m)
        r_max : max radius (m)
        '''
        self.n_t_bins = n_t_bins
        self.n_r_bins = n_r_bins
        self.n_theta_bins = n_theta_bins
        self.n_phi_bins = n_phi_bins
        self.t_min = t_min * 1e-9
        self.t_max = t_max * 1e-9
        self.r_min = r_min
        self.r_max = r_max
        self.t_scaling_factor = self.n_t_bins / (self.t_max - self.t_min)
        self.r_scaling_factor = self.n_r_bins * self.n_r_bins / self.r_max
        self.theta_scaling_factor = self.n_theta_bins / 2.
        self.phi_scaling_factor = self.n_phi_bins / np.pi / 2.

    def set_bin_index(self):
        '''
        takes t, x, y, z position and creates indices in t, r, theta, and phi
        '''
        #self.t_index = int(self.t * self.t_scaling_factor)
        #self.r_index = int(math.sqrt(self.radius * self.r_scaling_factor))
        #if self.radius == 0.:
        #    self.cos_theta = 1.
        #else:
        #    self.cos_theta = self.z / self.radius
        #self.theta_index = int((-self.cos_theta + 1.) * self.theta_scaling_factor)
        #self.phi = math.atan2(self.y, self.x)
        #self.phi_index = int(self.phi * self.phi_scaling_factor)
        self.t_index, self.r_index, self.theta_index, self.phi_index = numba_bin_indices(self.t, self.x, self.y, self.z, self.radius, self.t_scaling_factor, self.r_scaling_factor, self.theta_scaling_factor, self.phi_scaling_factor)
    
    def create_photon_matrix(self):
        '''
        uses track information from __init__ to create a sparce matrix containing the number of photons in each bin from spherical coordinates which must be pre-defined
        '''
        self.z_kevin = sparse((self.n_t_bins, self.n_r_bins, self.n_theta_bins, self.n_phi_bins))
        self.cumulative_track_length = 0.
        self.number_of_segments = 0

        # add cscd photons
        if self.radius < self.r_max:
            self.set_bin_index()
            self.z_kevin[self.t_index, self.r_index, self.theta_index, self.phi_index] += self.cscd_photons

        # traverse track and add track photons if within radius of the dom
        while self.cumulative_track_length < self.trck_length and self.t < self.t_max:
            self.radius = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
            if self.radius < self.r_max:
                self.set_bin_index()
                self.z_kevin[self.t_index, self.r_index, self.theta_index, self.phi_index] += self.segment_length * self.photons_per_meter
            if self.scaled_time_increment == True:
                if self.radius * self.time_increment_scaling > self.min_time_increment:
                    self.time_increment = self.radius * self.time_increment_scaling
                    self.segment_length = self.speed_of_light * self.time_increment
                else:
                    self.time_increment = self.min_time_increment
                    self.segment_length = self.speed_of_light * self.time_increment
            self.cumulative_track_length += self.segment_length
            self.number_of_segments += 1
            self.t += self.time_increment
            self.x += self.speed_x * self.time_increment
            self.y += self.speed_y * self.time_increment
            self.z += self.speed_z * self.time_increment

    def set_dom_location(self, t_dom=0., x_dom=0., y_dom=0., z_dom=0.):
        '''
        changes the track vertex to be relative to a dom at a given position with
        t_dom : time (ns)
        x_dom, y_dom, z_dom : position of the dom (m)
        '''
        self.t = self.t - t_dom * 1e-9
        self.x = self.x - x_dom
        self.y = self.y - y_dom
        self.z = self.z - z_dom
        self.radius = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def use_scaled_time_increments(self, scaling=0.01, min_time_increment=1.):
        '''
        causes create_photon_matrix to use time increments that scale based off of the radius
        scaling : time increment per radius (ns/m)
        min_time_increment : the lowest possible time increment (ns)
        '''
        self.scaled_time_increment = True
        self.time_increment_scaling = scaling * 1e-9
        self.min_time_increment = min_time_increment * 1e-9

    def use_numba_create_photon_matrix(self):
        '''
        uses numba to create the photon matrix
        '''
        vertex_info = {'t':self.t, 'x':self.x, 'y':self.y, 'z':self.z, 'theta_v':self.theta_v, 'phi_v':self.phi_v, 'trck_length':self.trck_length, 'cscd_photons':self.cscd_photons}
        time_increment_info = {'time_increment':self.time_increment, 'scaled_time_increment':self.scaled_time_increment, 'time_increment_scaling':self.time_increment_scaling, 'min_time_increment':self.min_time_increment}
        bin_info = {'n_t_bins':self.n_t_bins, 'n_r_bins':self.n_r_bins, 'n_theta_bins':self.n_theta_bins, 'n_phi_bins':self.n_phi_bins, 't_min':self.t_min, 't_max':self.t_max, 'r_min':self.r_min, 'r_max':self.r_max}
        kwargs = {}
        kwargs.update(vertex_info)
        kwargs.update(time_increment_info)
        kwargs.update(bin_info)
        self.z_dict = numba_create_photon_matrix(**kwargs)

    def vector_photon_matrix(self):
        '''
        uses a single time array to simultaneously calculate all of the positions along the track, using information from __init__
        '''
        #create initial time array
        self.t_array_init = np.arange(self.t, min(self.t_max, self.trck_length / self.speed_of_light + self.t), self.time_increment, dtype=np.float32)
        #set the number of time increments in the track
        self.number_of_increments = int(len(self.t_array_init))
        #create array with variables
        self.variables_array = np.empty((8, self.number_of_increments), dtype=np.float32)
        self.t_array = self.variables_array[0, :]
        self.t_array[:] = self.t_array_init 
        self.x_array = self.variables_array[1, :]
        self.x_array[:] = self.x + self.speed_x * self.t_array
        self.y_array = self.variables_array[2, :]
        self.y_array[:] = self.y + self.speed_y * self.t_array
        self.z_array = self.variables_array[3, :]
        self.z_array[:] = self.z + self.speed_z * self.t_array
        self.r_array = self.variables_array[4, :]
        self.r_array[:] = np.sqrt(np.square(self.x_array) + np.square(self.y_array) + np.square(self.z_array))
        self.cos_theta_array = self.variables_array[5, :]
        self.cos_theta_array[:] = self.z_array / self.r_array
        self.phi_array = self.variables_array[6, :]
        self.phi_array[:] = np.arctan2(self.y_array, self.x_array) % (2 * np.pi)
        self.photons_array = self.variables_array[7, :]
        self.photons_array[:] = self.segment_length * self.photons_per_meter
        #add cascade photons
        self.photons_array[0] += self.cscd_photons 
        
        #create array with indices
        self.indices_array = np.empty((4, self.number_of_increments), dtype=np.uint16)
        self.t_index_array = self.indices_array[0, :]
        self.t_index_array[:] = self.t_array * self.t_scaling_factor
        self.r_index_array = self.indices_array[1, :]
        self.r_index_array[:] = np.sqrt(self.r_array * self.r_scaling_factor)
        self.theta_index_array = self.indices_array[2, :]
        self.theta_index_array[:] = (-self.cos_theta_array + 1.) * self.theta_scaling_factor
        self.phi_index_array = self.indices_array[3, :]
        self.phi_index_array[:] = self.phi_array * self.phi_scaling_factor
