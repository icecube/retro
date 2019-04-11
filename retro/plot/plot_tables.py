# -*- coding: utf-8 -*-

from __future__ import absolute_import, division

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from astropy.io import fits
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

table = fits.open(sys.argv[1])
#table.info()
# cut off under and overflow bins
data = table[0].data[1:-1, 1:-1, 1:-1, 1:-1, 1:-1]
nphotons = table[0].header['_i3_n_photons']
nphase = table[0].header['_i3_n_phase']
# norm N_photons * (speed of light / phase refractive index)
norm = nphotons * (2.99792458 / nphase)
# correct for DOM angular acceptance
norm /= 0.338019664877
data /= norm

lables = [
    r'$r\ (m)$', r'$\cos{\vartheta}$', r'$t\ (ns)$',
    r'$\cos{\vartheta_\gamma}$', r'$\phi_\gamma$'
]
bin_edges = [
    table[1].data, table[2].data, table[3].data, table[4].data, table[5].data
]


def plot_2d(
    data,
    bin_edges,
    lables,
    ax,
    x_idx,
    y_idx,
    log=False,
    cmap='CMRmap_r',
    cb=True,
    vmin=None,
    vmax=None
):
    idx = range(data.ndim)
    idx.remove(x_idx)
    idx.remove(y_idx)
    z = data.sum(axis=tuple(idx))
    if log:
        if vmin is None:
            vmin = np.log(z[z != 0].min())
        if vmax is None:
            vmax = np.log(z.max())
        z[z == 0] = vmin
        z[z != vmin] = np.log(z[z != vmin])
    else:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = z.max()
    xx, yy = np.meshgrid(bin_edges[x_idx], bin_edges[y_idx])
    if y_idx > x_idx:
        z = z.T
    im = ax.pcolormesh(xx, yy, z, cmap=cmap, vmax=vmax, vmin=vmin)
    if cb:
        cb = plt.colorbar(im, ax=ax)
        if log:
            cb.set_label(r'$log(N_\gamma)$')
        else:
            cb.set_label(r'$N_\gamma$')
    ax.set_xlim([bin_edges[x_idx][0], bin_edges[x_idx][-1]])
    ax.set_ylim([bin_edges[y_idx][0], bin_edges[y_idx][-1]])
    ax.set_xlabel(lables[x_idx])
    ax.set_ylabel(lables[y_idx])
    return im


def plot_1d(data, bin_edges, lables, ax, x_idx, log=False):
    idx = range(data.ndim)
    idx.remove(x_idx)
    y = data.sum(axis=tuple(idx))

    ax.hist(
        0.5 * (bin_edges[x_idx][:-1] + bin_edges[x_idx][1:]),
        weights=y,
        bins=bin_edges[x_idx],
        histtype='step',
        lw=1.5
    )
    ax.set_xlim([bin_edges[x_idx][0], bin_edges[x_idx][-1]])
    if log:
        ax.set_yscale('log')
    ax.set_ylabel(r'$N_\gamma$')
    ax.set_xlabel(lables[x_idx])


fig = plt.figure(figsize=(14, 15))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

plot_2d(data, bin_edges, lables, ax1, 1, 0, log=True)
plot_2d(data, bin_edges, lables, ax2, 2, 0, log=True)
plot_2d(data, bin_edges, lables, ax3, 1, 2, log=True)
plot_2d(data, bin_edges, lables, ax4, 3, 1, log=True)
plot_2d(data, bin_edges, lables, ax5, 3, 4, log=True)
plot_2d(data, bin_edges, lables, ax6, 3, 0, log=True)

plt.savefig(sys.argv[1].split('.')[0] + '.png', dpi=150)

# plot 1d
plt.clf()
fig = plt.figure(figsize=(14, 15))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

plot_1d(data, bin_edges, lables, ax1, 0)
plot_1d(data, bin_edges, lables, ax2, 1)
plot_1d(data, bin_edges, lables, ax3, 2)
plot_1d(data, bin_edges, lables, ax4, 2, log=True)
plot_1d(data, bin_edges, lables, ax5, 3)
plot_1d(data, bin_edges, lables, ax6, 4)

plt.savefig(sys.argv[1].split('.')[0] + '_1d.png', dpi=150)

# plot movie
plt.clf()
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(111)
ims = []

r_cos_t_data = data.sum(axis=(3, 4))
vmin = np.log(r_cos_t_data[r_cos_t_data != 0].min())
vmax = np.log(r_cos_t_data.max())
for tidx in range(r_cos_t_data.shape[2]):
    im = plot_2d(
        r_cos_t_data[:, :, tidx],
        bin_edges[:2],
        lables[:2],
        ax1,
        1,
        0,
        log=True,
        vmax=vmax,
        vmin=vmin,
        cb=False
    )
    ims.append([im])

im_ani = animation.ArtistAnimation(
    fig, ims, interval=200, repeat_delay=3000, blit=True
)
im_ani.save(sys.argv[1].split('.')[0] + '.mp4', writer=writer)
