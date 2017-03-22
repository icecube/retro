import pyfits
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys



table = pyfits.open(sys.argv[1])
#table.info()
data = table[0].data
r_bin_edges = table[1].data
az_bin_edges = table[2].data/180*np.pi
cos_bin_edges = table[3].data
# time residulas
dt_bin_edges = table[4].data
if len(table) == 6:
    impact_angle_bin_edges = table[5].data

r_az_data = data.sum(axis=(2,3,4))[1:-1,1:-1]
r_cos_data = data.sum(axis=(1,3,4))[1:-1,1:-1]
r_dt_data = data.sum(axis=(1,2,4))[1:-1,1:-1]
cos_dt_data = data.sum(axis=(0,1,4))[1:-1,1:-1]
r_ia_data = data.sum(axis=(1,2,3))[1:-1,1:-1]
cos_ia_data = data.sum(axis=(0,1,3))[1:-1,1:-1]

