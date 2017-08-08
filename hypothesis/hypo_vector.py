import numpy as np
import math

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
        self.speed_of_light = 2.99e8
        self.segment_length = self.time_increment * self.speed_of_light
        #calculate frequently used values
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

    def vector_photon_matrix(self):
        '''
        uses a single time array to simultaneously calculate all of the positions along the track, using information from __init__
        '''
        #create initial time array, using the midpoints of each time increment
        self.t_array_init = np.arange(self.t, min(self.t_max, self.trck_length / self.speed_of_light + self.t), self.time_increment, dtype=np.float32) - self.time_increment / 2
        self.t_array_init[0] = self.t
        #set the number of time increments in the track
        self.number_of_increments = int(len(self.t_array_init))
        #create array with variables
        self.variables_array = np.empty((8, self.number_of_increments), dtype=np.float32)
        self.t_array = self.variables_array[0, :]
        self.t_array[:] = self.t_array_init 
        self.x_array = self.variables_array[1, :]
        self.x_array[:] = self.x + self.speed_x * (self.t_array - self.t)
        self.y_array = self.variables_array[2, :]
        self.y_array[:] = self.y + self.speed_y * (self.t_array - self.t)
        self.z_array = self.variables_array[3, :]
        self.z_array[:] = self.z + self.speed_z * (self.t_array - self.t)
        self.r_array = self.variables_array[4, :]
        self.r_array[:] = np.sqrt(np.square(self.x_array) + np.square(self.y_array) + np.square(self.z_array))
        self.cos_theta_array = self.variables_array[5, :]
        self.cos_theta_array[:] = self.z_array / self.r_array
        self.phi_array = self.variables_array[6, :]
        self.phi_array[:] = np.arctan2(self.y_array, self.x_array) % (2 * np.pi)
        self.photons_array = self.variables_array[7, :]
        self.photons_array[:] = self.segment_length * self.photons_per_meter
        #add cascade photons
        self.photons_array[0] = self.cscd_photons 
        
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
