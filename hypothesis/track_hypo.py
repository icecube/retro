import numpy as np
import math
from sparse import sparse

def PowerAxis(minval, maxval, n_bins, power):
    l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
    bin_edges = np.power(l, power)
    return bin_edges

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
        self.radius = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        self.sin_theta_v = np.sin(self.theta_v)
        self.cos_theta_v = np.cos(self.theta_v)
        self.sin_phi_v = np.sin(self.phi_v)
        self.cos_phi_v = np.cos(self.phi_v)
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
        self.t_scaling_factor = (self.t_max - self.t_min) / self.n_t_bins
        self.r_scaling_factor = self.n_r_bins * self.n_r_bins / self.r_max
        self.theta_scaling_factor = self.n_theta_bins / 2.
        self.phi_scaling_factor = self.n_phi_bins / 2.

    def set_bin_index(self):
        '''
        takes t, x, y, z position and creates indices in t, r, theta, and phi
        '''
        self.t_index = int(self.t * self.t_scaling_factor)
        self.r_index = int(np.sqrt(self.radius * self.r_scaling_factor))
        if self.radius == 0.:
            self.cos_theta = 1.
        else:
            self.cos_theta = self.z / self.radius
        self.theta_index = int((-self.cos_theta + 1.) * self.theta_scaling_factor)
        self.phi = np.arctan2(self.y, self.x)
        self.phi_index = int(self.phi * self.phi_scaling_factor)
    
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
            self.radius = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
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
        self.radius = np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def use_scaled_time_increments(self, scaling=0.01, min_time_increment=1.):
        self.scaled_time_increment = True
        self.time_increment_scaling = scaling * 1e-9
        self.min_time_increment = min_time_increment * 1e-9
