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
theta_bin_edges = table[2].data
t_bin_edges = table[3].data
# photon arrival directions
p_theta_bin_edges = table[4].data
p_delta_phi_bin_edges = table[5].data
p_theta_centers = 0.5 * (p_theta_bin_edges[1:] + p_theta_bin_edges[:-1])
p_delta_phi_centers = 0.5 * (p_delta_phi_bin_edges[1:] + p_delta_phi_bin_edges[:-1])

n_photons = data.sum(axis=(3,4))
n_photons /= norm

average_thetas = np.zeros_like(n_photons)
lengths = np.zeros_like(n_photons)

for i in xrange(n_photons.shape[0]):
    for j in xrange(n_photons.shape[1]):
        for k in xrange(n_photons.shape[2]):
            weights = data[i][j][k]
            if weights.sum() == 0:
                # if no photons, just set the average direction to the theate of the bin center
                average_thetas[i][j][k] = 0.5 * (theta_bin_edges[j] + theta_bin_edges[j + 1])
                # and lengths to 0
                lengths[i][j][k] = 0.
            else:
                # proejct phi values
                weights_theta = np.sum(weights, axis=1)
                average_thetas[i][j][k] = np.average(p_theta_centers, weights=weights_theta)

                # delta angles to average for all bins
                delta_thetas = p_theta_centers - average_thetas[i][j][k]
                # project onto average direction
                projected = np.outer(np.cos(delta_thetas), np.cos(p_delta_phi_centers))
                lengths[i][j][k] = np.average(projected, weights=weights)

# invert tables (r, cz, t) -> (-t, r, cz)
# and also flip coszen binning!
n_photons = np.flipud(np.rollaxis(np.fliplr(n_photons), 2, 0))
average_thetas = np.flipud(np.rollaxis(np.fliplr(average_thetas), 2, 0))
lengths = np.flipud(np.rollaxis(np.fliplr(lengths), 2, 0))

a = pyfits.PrimaryHDU(n_photons)
b = pyfits.ImageHDU(average_thetas)
c = pyfits.ImageHDU(lengths)
d = pyfits.ImageHDU(t_bin_edges)
e = pyfits.ImageHDU(r_bin_edges)
f = pyfits.ImageHDU(theta_bin_edges[::-1])

hdulist = pyfits.HDUList([a, b, c, d, e, f])
hdulist.writeto(new_fname)
#tbhdu = pyfits.BinTableHDU.from_columns([
#    pyfits.Column(name='n_photons', format='E', array=n_photons),
#    pyfits.Column(name='average_thetas', format='E', array=average_thetas),
#    pyfits.Column(name='lengths', format='E', array=lengths)])
#
#tbhdu.writeto(new_fname)
