import pyfits
import numpy as np
import sys, os

fname = sys.argv[1]
if not os.path.isfile(fname):
    print 'table %s does not exist'%fname
    sys.exit()

path, ext = fname.split('.')
new_fname = path + '_r_cz_t_angles.' + ext
if os.path.isfile(new_fname):
    print 'summed table %s exists already! Skipping!'%new_fname
    sys.exit()

table = pyfits.open(fname)
#table.info()
# cut off under and overflow bins
data = table[0].data[1:-1,1:-1,1:-1,1:-1,1:-1]
nphotons = table[0].header['_i3_n_photons']
nphase =  table[0].header['_i3_n_phase']

# norm N_photons * (speed of light / phase refractive index)
norm = nphotons * (2.99792458 / nphase)
# correct for DOM angular acceptance
norm /= 0.338019664877

r_bin_edges = table[1].data
theta_bin_edges = np.arccos(table[2].data)
t_bin_edges = table[3].data
# photon arrival directions
p_theta_bin_edges = table[4].data
p_delta_phi_bin_edges = table[5].data
p_theta_centers = 0.5 * (p_theta_bin_edges[1:] + p_theta_bin_edges[:-1])
p_delta_phi_centers = 0.5 * (p_delta_phi_bin_edges[1:] + p_delta_phi_bin_edges[:-1])

n_photons = data.sum(axis=(3,4))
n_photons /= norm

average_thetas = np.zeros_like(n_photons)
average_phis = np.zeros_like(n_photons)
lengths = np.zeros_like(n_photons)

for i in xrange(n_photons.shape[0]):
    for j in xrange(n_photons.shape[1]):
        for k in xrange(n_photons.shape[2]):
            weights = data[i][j][k]
            if weights.sum() == 0:
                # if no photons, just set the average direction to the theate of the bin center
                average_theta = 0.5 * (theta_bin_edges[j] + theta_bin_edges[j + 1])
                # and lengths to 0
                length = 0.
                average_phi = 0.
            else:
                # average theta
                weights_theta = np.sum(weights, axis=1)
                average_theta = np.average(p_theta_centers, weights=weights_theta)

                # average delta phi
                projected_n_photons = weights * np.sin(p_theta_centers)[:, np.newaxis]
                weights_phi = np.sum(projected_n_photons, axis=0)
                average_phi = np.average(p_delta_phi_centers, weights=weights_phi)

                # length of vector (using projections from all vectors onto average vector
                # cos(angle) between average vector and all angles
                coscos = np.cos(p_theta_centers)*np.cos(average_theta)
                sinsin = np.sin(p_theta_centers)*np.sin(average_theta)
                cosphi = np.cos(p_delta_phi_centers - average_phi)
                cospsi = coscos[:, np.newaxis] + np.outer(sinsin, cosphi)
                length = np.average(cospsi, weights=weights)

            average_thetas[i][j][k] = average_theta
            average_phis[i][j][k] = average_phi
            lengths[i][j][k] = length

# invert tables (r, cz, t) -> (-t, r, cz)
# and also flip coszen binning!
n_photons = np.flipud(np.rollaxis(np.fliplr(n_photons), 2, 0))
average_thetas = np.flipud(np.rollaxis(np.fliplr(average_thetas), 2, 0))
average_phis = np.flipud(np.rollaxis(np.fliplr(average_phis), 2, 0))
lengths = np.flipud(np.rollaxis(np.fliplr(lengths), 2, 0))

a = pyfits.PrimaryHDU(n_photons)
b = pyfits.ImageHDU(average_thetas)
c = pyfits.ImageHDU(average_phis)
d = pyfits.ImageHDU(lengths)
e = pyfits.ImageHDU(t_bin_edges)
f = pyfits.ImageHDU(r_bin_edges)
g = pyfits.ImageHDU(theta_bin_edges[::-1])

hdulist = pyfits.HDUList([a, b, c, d, e, f, g])
hdulist.writeto(new_fname)
