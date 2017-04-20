import numpy as np

# define binning
theta_bin_edges = np.linspace(0, np.pi, 21)
delta_phi_bin_edges = np.linspace(0, np.pi, 21)

# bin_centers
theta_centers = 0.5 * (theta_bin_edges[1:] + theta_bin_edges[:-1])
delta_phi_centers = 0.5 * (delta_phi_bin_edges[1:] + delta_phi_bin_edges[:-1])

# the histogram with photon counts in it
n_photons = np.zeros((len(theta_centers), len(delta_phi_centers)))
# fill in some values
n_photons[0,0] = 1
n_photons[1,-1] = 3
n_photons[-1,0] = 1

# proejct phi values
n_photons_theta = np.sum(n_photons, axis=1)
average_direction = np.average(theta_centers, weights=n_photons_theta)
print 'average direction (theta)', average_direction

# delta angles to average for all bins
delta_thetas = theta_centers - average_direction
# project onto average direction
projected = np.outer(np.cos(delta_thetas), np.cos(delta_phi_centers))
length = np.average(projected, weights=n_photons)
print 'correlation (length) ',length
