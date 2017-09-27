#!/usr/bin/env python

"""
Make plots from the 3-d retro tables. The output will be 2-d maps plus the time
dimension as the video time.
"""


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
# cut off under and overflow bins
n_photons = table[0].data

n_photons = table[0].data
average_thetas = table[1].data
average_phis = table[2].data
lengths = table[3].data
t_bin_edges = table[4].data
r_bin_edges = table[5].data
theta_bin_edges = table[6].data

r = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
theta = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])
#print r
#print theta

fig = plt.figure(figsize=(14,15))
ax = fig.add_subplot(111, polar=True)



rr, thetatheta = np.meshgrid(r, theta)
# first time bin
a_thetas = average_thetas[-100].T- thetatheta + np.pi/4.
a_lengths = lengths[-100].T
#print a_lengths
#print a_thetas[:,-1]

#ax.quiver(thetatheta, rr, a_thetas, a_lengths, angles='xy')

#plt.savefig(sys.argv[1].split('.')[0]+'.png',dpi=150)

#sys.exit()
# plot movie
#cmap='Accent'
#cmap='viridis'
#cmap='nipy_spectral'


datas = [n_photons, average_thetas, average_phis, lengths]
names = ['survival', 'theta', 'delta_phi', 'corr_length']
cmaps = ['BuPu', 'nipy_spectral', 'nipy_spectral', 'afmhot_r']

for data, name, cmap in zip(datas, names, cmaps):
    plt.clf()
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    ims = []
    #data = lengths
    #data = average_thetas
    #r_cos_t_data = data.sum(axis=(3,4))
    #vmin = np.log(data[data != 0].min())
    #vmax = np.log(data.max())
    vmin = data.min()
    vmax = data.max()
    for tidx in range(data.shape[0])[::-1]:
        #im = plot_2d(data[tidx], [r_bin_edges, theta_bin_edges], ['r','theta'], ax1, 1, 0, log=True, vmax=vmax, vmin=vmin, cb=False)
        xx, yy = np.meshgrid(r_bin_edges, theta_bin_edges)
        im = ax1.pcolormesh(xx, yy, data[tidx].T, cmap=cmap, vmax=vmax, vmin=vmin)
        ax1.set_xlim([r_bin_edges[0],r_bin_edges[-1]])
        ax1.set_ylim([theta_bin_edges[0],theta_bin_edges[-1]])
        ax1.set_xlabel('r')
        ax1.set_ylabel('theta')
        ims.append([im])
        
    im_ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=3000, blit=True)
    im_ani.save(sys.argv[1].split('.')[0]+'_'+name+'.mp4', writer=writer)
