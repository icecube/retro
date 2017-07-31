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
    def __init__(self, t_v, x_v, y_v, z_v, theta, phi, trck_energy, cscd_energy, time_increment=1e-9):
        '''
        provide vertex and track information
        t_v : time (ns)
        x_v, y_v, z_v : vertex position (m)
        theta : zenith (rad)
        phi : azimuth (rad)
        trck_energy : track energy (GeV)
        cscd_energy : cascade energy (GeV)
        '''     
        #assign vertex
        self.t = t_v
        self.x = x_v
        self.y = y_v
        self.z = z_v
        self.theta = theta
        self.phi = phi
        self.radius = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.radius_squared = self.radius ** 2
        #convert track energy to length
        self.trck_length = 15. / 3.3 * trck_energy
        #precalculated (nphotons.py) to avoid icetray
        self.photons_per_meter = 2451.4544553
        self.cscd_photons = 12805.3383311 * cscd_energy
        self.trck_photons = self.trck_length * self.photons_per_meter
        self.tot_photons = self.cscd_photons + self.trck_photons
        #declaring constants 
        self.time_increment = time_increment
        self.speed_of_light = 2.99e8
        self.segment_length = self.time_increment * self.speed_of_light

    
    def set_binning(self, n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_max, r_max):
        '''
        define binnings of spherical coordinates
        '''
        self.n_t_bins = n_t_bins
        self.n_r_bins = n_r_bins
        self.n_theta_bins = n_theta_bins
        self.n_phi_bins = n_phi_bins
        self.t_max = t_max
        self.r_max = r_max
        self.r_scaling_factor = self.n_r_bins ** 2 / self.r_max

    def set_bin_index(self):
        '''
        takes t, x, y, z position and creates indices in t, r, theta, and phi
        '''
        self.t_index = int(math.floor(self.t * 1e8))
        self.radius = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        self.r_index = int(math.floor(np.sqrt(self.radius * self.r_scaling_factor)))
        if self.radius == 0.:
            self.cos_theta = 1.
        else:
            self.cos_theta = z / radius
        self.theta_index = int(math.floor((self.cos_theta / -1. + 1.) * self.n_theta_bins / 2.))
        self.phi = np.arctan2(self.y, self.x)
        self.phi_index = int(math.floor(self.phi * 18 / np.pi))

    def create_photon_matrix(self):
        '''
        uses track information from __init__ to create a sparce matrix containing the number of photons in each bin from spherical coordinates which must be pre-defined
        '''
        self.z_kevin = sparse((self.n_t_bins, self.n_r_bins, self.n_theta_bins, self.n_phi_bins))
        self.cumulative_track_length = 0.
        
        # add cscd photons
        if self.radius < r_max:
            self.set_bin_index()
            self.z_kevin[self.t_index, self.r_index, self.theta_index, self.phi_index] += cscd_energy * photons_per_gev_cscd

        # traverse track and add track photons if within radius of the dom
        while self.cumulative_track_length < self.trck_length and t < self.t_max:
            if self.radius_squared < self.r_max ** 2:
                self.set_bin_index()
                self.z_kevin[self.t_index, self.r_index, self.theta_index, self.phi_index] += self.segment_length * self.photons_per_meter
            self.cumulative_track_length += self.segment_length
            self.x += self.speed_of_light * np.sin(self.theta) * np.cos(self.phi) * self.time_increment
            self.y += speed_of_light * np.sin(self.theta) * np.sin(self.phi) * self.time_increment
            self.z += speed_of_light * np.cos(self.theta) * self.time_increment
            self.t += self.time_increment
            self.radius_squared = self.x ** 2 + self.y ** 2 + self.z ** 2
