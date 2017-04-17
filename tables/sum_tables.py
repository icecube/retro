import pyfits
import numpy as np
import sys, os

fname = sys.argv[1]
if not os.path.isfile(fname):
    print 'table %s does not exist'%fname
    sys.exit()

path, ext = fname.split('.')
new_fname = path + '_r_cz_t.' + ext
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
data /= norm
data = data.sum(axis=(3,4))
hdu = pyfits.PrimaryHDU(data)
hdu.writeto(new_fname)
