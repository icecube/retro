import numpy as np
import math
from sparse import sparse

def PowerAxis(minval, maxval, n_bins, power):
    l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
    bin_edges = np.power(l, power)
    return bin_edges

def get_bin_index(t, x, y, z):
    t_index = int(math.floor(t * 1e8))
    radius = (x ** 2 + y ** 2 + z ** 2)**0.5
    r_index = int(math.floor((radius * 2)**0.5))
    if radius == 0:
        theta = 0
    else:
        theta = np.arccos(z / radius)
    theta_index = int(math.floor((np.cos(theta) / -1 + 1) * 25))
    phi = np.arctan2(y, x)
    phi_index = int(math.floor(phi * 18 / np.pi))
    bin_index = (t_index, r_index, theta_index, phi_index)
    return bin_index

def get_track_lengths(t, x, y, z, theta, phi, total_track_length):
    n_t_bins = 50
    n_r_bins = 20
    n_theta_bins = 50
    n_phi_bins = 36
    time_increment = 1e-9
    speed_of_light = 2.99e8
    track_segment = time_increment * speed_of_light
    cumulative_track_length = 0
    radius = (x ** 2 + y ** 2 + z ** 2)**0.5
    
    #t_bin_edges = np.linspace(0, 500e-9, n_t_bins+1)
    #r_bin_edges = PowerAxis(0, 200, n_r_bins, 2)
    #theta_bin_edges = np.linspace(-1, 1, n_theta_bins+1)
    #phi_bin_edges = np.linspace(0, 2*np.pi, n_phi_bins+1)
    #z_kevin = np.zeros((n_t_bins, n_r_bins, n_theta_bins, n_phi_bins))
    z_kevin = sparse((n_t_bins, n_r_bins, n_theta_bins, n_phi_bins))
    
    while cumulative_track_length < total_track_length and radius < 200 and t < 500e-9:
        z_kevin[get_bin_index(t, x, y, z)] += track_segment
        cumulative_track_length += track_segment
        x = speed_of_light * np.sin(theta) * np.cos(phi) * time_increment + x
        y = speed_of_light * np.sin(theta) * np.sin(phi) * time_increment + y
        z = speed_of_light * np.cos(theta) * time_increment + z
        t = time_increment + t
        radius = (x ** 2 + y ** 2 + z ** 2)**0.5

        
    return z_kevin
