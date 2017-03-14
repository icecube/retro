import pyfits
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors



table = pyfits.open('retro/testtable.fits')
#table.info()
data = table[0].data
r_bin_edges = table[1].data
az_bin_edges = table[2].data/180*np.pi
cosphi_bin_edges = table[3].data
# time residulas
dt_bin_edges = table[4].data
if len(table) == 6:
    impact_angle_bins = table[5].data

r_az_data = data.sum(axis=(2,3,4))[1:-1,1:-1]

rr, azaz = np.meshgrid(r_bin_edges, az_bin_edges)

# plot sth.
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111, projection='polar')
z = r_az_data.T
z[z != 0] = np.log(z[z != 0])
im = ax1.pcolormesh(azaz, rr, z, cmap='Purples')
cb = fig.colorbar(im)
cb.set_label(r'$log(N_\gamma)$')
#ax1.set_xlabel('azimuth')
#ax1.set_ylabel('radius')
plt.show()
ax1.grid(True)
plt.savefig('testtable.png',dpi=150)
