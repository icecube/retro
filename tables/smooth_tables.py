import pyfits
import numpy as np
import sys
from scipy.ndimage.filters import gaussian_filter

table = pyfits.open(sys.argv[1])
path, ext = sys.argv[1].split('.')
new_fname = path + '_smooth_2.' + ext
data = table[0].data
#smooth with gaussian filter
data = gaussian_filter(data, sigma = 2.)
hdu = pyfits.PrimaryHDU(data)
hdu.writeto(new_fname)
