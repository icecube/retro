import numpy as np

# define binning
theta_bin_edges = np.linspace(0, np.pi, 21)
delta_phi_bin_edges = np.linspace(0, np.pi, 21)

# bin_centers
theta_centers = 0.5 * (theta_bin_edges[1:] + theta_bin_edges[:-1])
delta_phi_centers = 0.5 * (delta_phi_bin_edges[1:] + delta_phi_bin_edges[:-1])
print delta_phi_centers

# the histogram with photon counts in it
n_photons = np.zeros((len(theta_centers), len(delta_phi_centers)))
# fill in some values
#n_photons[0,-1] = 1
#n_photons[0,7] = 3
n_photons[1,7] = 1
n_photons[-1,-1] = 1

n_photons_theta = np.sum(n_photons, axis=1)
average_theta = np.average(theta_centers, weights=n_photons_theta)

projected_n_photons = n_photons * np.sin(theta_centers)[:, np.newaxis]

n_photons_phi = np.sum(projected_n_photons, axis=0)
average_phi = np.average(delta_phi_centers, weights=n_photons_phi)
print 'average direction (theta, delta_phi)', average_theta, average_phi

# cos(angle) between average vector and all angles
coscos = np.cos(theta_centers)*np.cos(average_theta)
sinsin = np.sin(theta_centers)*np.sin(average_theta)
cosphi = np.cos(delta_phi_centers - average_phi)
cospsi = coscos[:, np.newaxis] + np.outer(sinsin, cosphi)

# delta angles to average for all bins
#delta_thetas = theta_centers - average_theta
#delta_phis = delta_phi_centers - average_phi
# project onto average direction
#projected = np.outer(np.cos(delta_thetas), np.cos(delta_phis))
length = np.average(cospsi, weights=n_photons)
print 'correlation (length) ',length
