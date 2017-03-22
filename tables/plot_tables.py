import pyfits
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, bitrate=20000)

log = True

table = pyfits.open(sys.argv[1])
#table.info()
data = table[0].data
r_bin_edges = table[1].data
az_bin_edges = table[2].data
cos_bin_edges = table[3].data
# time residulas
dt_bin_edges = table[4].data
ia_bin_edges = table[5].data
# cut off under and overflow bins
data = data[1:-1,1:-1,1:-1,1:-1,1:-1]

norm = table[0].header['_i3_n_photons']
data /= norm

r_az_data = data.sum(axis=(2,3,4))
r_cos_data = data.sum(axis=(1,3,4))
r_dt_data = data.sum(axis=(1,2,4))
cos_dt_data = data.sum(axis=(0,1,4))
r_ia_data = data.sum(axis=(1,2,3))
cos_ia_data = data.sum(axis=(0,1,3))

r_cos_t_data = data.sum(axis=4).mean(axis=1)
r_cos_t_data[r_cos_t_data != 0] = 1./r_cos_t_data[r_cos_t_data != 0]


# plot sth.
fig = plt.figure(figsize=(14,15))
ax1 = fig.add_subplot(321, projection='polar')
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

cmap = 'CMRmap_r'

# plot 1
rr, azaz = np.meshgrid(r_bin_edges, az_bin_edges)
z = r_az_data.T
z[z != 0] = np.log(z[z != 0])
im = ax1.pcolormesh(azaz, rr, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax1)
cb.set_label(r'$log(N_\gamma)$')
ax1.set_ylim([r_bin_edges[0],r_bin_edges[-1]])

# plot 2
rr, coscos = np.meshgrid(r_bin_edges, cos_bin_edges)
z = r_cos_data.T
z[z != 0] = np.log(z[z != 0])
im = ax2.pcolormesh(coscos, rr, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax2)
cb.set_label(r'$log(N_\gamma)$')

ax2.set_xlabel(r'$\cos{\theta}$')
ax2.set_ylabel('radius')
ax2.set_xlim([cos_bin_edges[0],cos_bin_edges[-1]])
ax2.set_ylim([r_bin_edges[0],r_bin_edges[-1]])

# plot 3
rr, dtdt = np.meshgrid(r_bin_edges, dt_bin_edges)
z = r_dt_data.T
z[z != 0] = np.log(z[z != 0])
im = ax3.pcolormesh(dtdt, rr, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax3)
cb.set_label(r'$log(N_\gamma)$')
ax3.set_xlabel(r'time')
ax3.set_ylabel('radius')
ax3.set_xlim([dt_bin_edges[0],dt_bin_edges[-1]])
ax3.set_ylim([r_bin_edges[0],r_bin_edges[-1]])

# plot 4
coscos, dtdt = np.meshgrid(cos_bin_edges, dt_bin_edges)
z = cos_dt_data.T
z[z != 0] = np.log(z[z != 0])
im = ax4.pcolormesh(coscos, dtdt, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax4)
cb.set_label(r'$log(N_\gamma)$')
ax4.set_xlabel(r'$\cos{\theta}$')
ax4.set_ylabel(r'time')
ax4.set_xlim([cos_bin_edges[0],cos_bin_edges[-1]])
ax4.set_ylim([dt_bin_edges[0],dt_bin_edges[-1]])

# plot 5
rr, iaia = np.meshgrid(r_bin_edges, ia_bin_edges)
z = r_ia_data.T
z[z != 0] = np.log(z[z != 0])
im = ax5.pcolormesh(iaia, rr, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax5)
cb.set_label(r'$log(N_\gamma)$')
ax5.set_xlabel('impact angle')
ax5.set_ylabel('radius')
ax5.set_xlim([ia_bin_edges[0],ia_bin_edges[-1]])
ax5.set_ylim([r_bin_edges[0],r_bin_edges[-1]])

# plot 
coscos, iaia = np.meshgrid(cos_bin_edges, ia_bin_edges)
z = cos_ia_data.T
z[z != 0] = np.log(z[z != 0])
im = ax6.pcolormesh(iaia, coscos, z, cmap=cmap)
cb = plt.colorbar(im, ax=ax6)
cb.set_label(r'$log(N_\gamma)$')
ax6.set_xlabel('impact angle')
ax6.set_ylabel(r'$\cos{\theta}$')
ax6.set_xlim([ia_bin_edges[0],ia_bin_edges[-1]])
ax6.set_ylim([cos_bin_edges[0],cos_bin_edges[-1]])

plt.show()
ax1.grid(True)
plt.savefig(sys.argv[1].split('.')[0]+'.png',dpi=150)


plt.clf()
# plot 1d
r_data = data[:,:,:,:,:].sum(axis=(1,2,3,4))
az_data = data[:,:,:,:,:].sum(axis=(0,2,3,4))
cos_data = data[:,:,:,:,:].sum(axis=(0,1,3,4))
dt_data = data.sum(axis=(0,1,2,4))
ia_data = data[:,:,:,:,:].sum(axis=(0,1,2,3))


# plot sth.
fig = plt.figure(figsize=(14,15))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

ax1.set_yscale('log')
ax4.set_yscale('log')


# r
ax1.hist(0.5*(r_bin_edges[:-1] + r_bin_edges[1:]), weights=r_data, bins=r_bin_edges, histtype='step', lw=1.5)
ax1.set_xlabel(r'$r$')
# az
ax2.hist(0.5*(az_bin_edges[:-1] + az_bin_edges[1:]), weights=az_data, bins=az_bin_edges, histtype='step', lw=1.5)
ax2.set_xlabel(r'$\phi$')
# cos
ax3.hist(0.5*(cos_bin_edges[:-1] + cos_bin_edges[1:]), weights=cos_data, bins=cos_bin_edges, histtype='step', lw=1.5)
ax3.set_xlabel(r'$\cos{\theta}$')
# dt
ax4.hist(0.5*(dt_bin_edges[:-1] + dt_bin_edges[1:]), weights=dt_data, bins=dt_bin_edges, histtype='step', lw=1.5)
ax4.set_xlabel(r'$t$')
# ia
ax5.hist(0.5*(ia_bin_edges[:-1] + ia_bin_edges[1:]), weights=ia_data, bins=ia_bin_edges, histtype='step', lw=1.5)
ax5.set_xlabel('impact angle')
plt.savefig(sys.argv[1].split('.')[0]+'_1d.png',dpi=150)

plt.clf()
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.set_xlabel(r'$\cos{\theta}$')
ax1.set_ylabel('radius')
ax1.set_xlim([cos_bin_edges[0],cos_bin_edges[-1]])
ax1.set_ylim([r_bin_edges[0],r_bin_edges[-1]])
ims = []
#cmap = 'CMRmap'

if log:
    vmin = np.log(r_cos_t_data[r_cos_t_data != 0].min())
    vmax = np.log(r_cos_t_data.max())
else:
    vmin = 0
    vmax = r_cos_t_data.max()
for tidx in range(r_cos_t_data.shape[2]):
    rr, coscos = np.meshgrid(r_bin_edges, cos_bin_edges)
    z = r_cos_t_data[:,:,tidx].T
    if log:
        z[z == 0] = vmin
        z[z != vmin] = np.log(z[z != vmin])
    im = ax1.pcolormesh(coscos, rr, z, cmap=cmap, vmin=vmin, vmax=vmax)
    #cb = plt.colorbar(im, ax=ax1)
    #cb.set_label(r'$log(N_\gamma)$')
    ims.append([im])

im_ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=3000, blit=True)
im_ani.save(sys.argv[1].split('.')[0]+'.mp4', writer=writer)
