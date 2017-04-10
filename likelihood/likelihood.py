import pyfits
import numpy as np
import sys

IC = {}
DC = {}

for dom in range(60):
    table = pyfits.open('tables/summed/retro_nevts1000_IC_DOM%i_r_cz_t.fits'%dom)
    IC[dom] = table[0].data
    table = pyfits.open('tables/summed/retro_nevts1000_DC_DOM%i_r_cz_t.fits'%dom)
    DC[dom] = table[0].data
for key, val in IC.items():
    print 'IC DOM %i, sum = %.2f'%(key, val.sum())
for key, val in DC.items():
    print 'DC DOM %i, sum = %.2f'%(key, val.sum())
