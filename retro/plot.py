#!/usr/bin/env python

# pylint: disable=print-statement, wrong-import-position


from __future__ import absolute_import, division

import math
import os
from os.path import abspath, dirname
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import BinningCoords, FTYPE, HypoParams8D
from power_axis import PowerAxis
from hypo_vector import SegmentedHypo
from hypo_fast import Hypo

#@profile
def main():
    # Same as CLsim
    n_t_bins = 50
    n_r_bins = 20
    n_theta_bins = 50
    n_phi_bins = 36

    t_min, t_max = 0, 500
    r_min, r_max = 0, 200

    bin_edges = BinningCoords(
        t=np.linspace(t_min, t_max, n_t_bins + 1),
        r=PowerAxis(r_min, r_max, n_r_bins, 2),
        theta=np.arccos(np.linspace(-1, 1, n_theta_bins + 1))[::-1],
        phi=np.linspace(0, 2*np.pi, n_phi_bins + 1)
    )
    binning_shape = tuple(len(edges) - 1 for edges in bin_edges)

    hypo_params = HypoParams8D(t=65, x=1, y=10, z=-50, track_zenith=1.08,
                               track_azimuth=0.96, track_energy=20,
                               cascade_energy=25)

    t0 = time.time()
    my_hypo = Hypo(hypo_params)
    my_hypo.set_binning(bin_edges)
    hits, n_t, n_p, n_l = my_hypo.get_matrices(50, 0, 10, 0)
    print ('took %5.2f ms to calculate philipp z matrix'
           % ((time.time() - t0)*1000))

    z = np.zeros(binning_shape)
    for hit in hits:
        #print hit
        idx, count = hit
        z[idx] = count
    print ('total number of photons in philipp z matrix = %i (%0.2f %%)'
           % (z.sum(), z.sum()/my_hypo.tot_photons*100))
    print ''

    # kevin array
    t0 = time.time()
    kevin_hypo = SegmentedHypo(*hypo_params, time_increment=1)
    # NOTE: jitclass w/ SegmentedHypo doesn't allow for kwargs to methods. (?)
    kevin_hypo.set_binning(n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_max, r_max, r_min, t_min)
    kevin_hypo.set_dom_location(50, 0, 10, 0)
    kevin_hypo.vector_photon_matrix()
    print ('took %5.2f ms to calculate kevin z matrix'
           % ((time.time() - t0)*1000))
    z_indices = kevin_hypo.indices_array
    z_values = kevin_hypo.values_array
    z_matrix = np.zeros(binning_shape, dtype=FTYPE)

    for col in xrange(kevin_hypo.number_of_increments):
        idx = (int(z_indices[0, col]),
               int(z_indices[1, col]),
               int(z_indices[2, col]),
               int(z_indices[3, col]))
        if z_indices[1, col] < kevin_hypo.r_max:
            z_matrix[idx] += z_values[0, col]

    print ('total number of photons in kevin z_matrix ='
           ' %i (%.2f %%)'
           %(z_matrix.sum(), z_matrix.sum()/my_hypo.tot_photons*100))
    print ''

    print 'total_residual = ', (z - z_matrix).sum() / z.sum()
    print ''
    return

    # Create differential matrix
    z_diff = z_matrix - z

    # Create percent differnt matrix
    z_per = np.zeros_like(z_matrix)
    mask = z != 0
    z_per[mask] = z_matrix[mask] / z[mask] -1

    # Plot setup
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    plt_lim = 50

    ax.set_xlim((-plt_lim, plt_lim))
    ax.set_ylim((-plt_lim, plt_lim))
    ax.set_zlim((-plt_lim, plt_lim))
    ax.grid(True)

    #cmap = 'gnuplot_r'
    cmap = mpl.cm.get_cmap('bwr')
    cmap.set_under('w')
    cmap.set_bad('w')

    # plot the track as a line
    x_0, y_0, z_0 = my_hypo.track.point(my_hypo.track.t0)
    #print 'track vertex', x_0, y_0, z_0
    x_e, y_e, z_e  = my_hypo.track.point(my_hypo.track.t0 + my_hypo.track.dt)
    ax.plot([x_0, x_e], [y_0, y_e], zs=[z_0, z_e])
    ax.plot([-plt_lim, -plt_lim], [y_0, y_e], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [plt_lim, plt_lim], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [y_0, y_e], zs=[-plt_lim, -plt_lim], alpha=0.3, c='k')

    # Make first plot
    print 'Making first plot'
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z_diff.sum(axis=(2, 3))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z_diff.sum(axis=(1, 3))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z_diff.sum(axis=(1, 2))
    z_vmax = np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    plt.show()
    plt.savefig('hypo_diff11.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    # Make second plot
    print 'Making second plot'
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z_per.sum(axis=(2, 3))
    z_vmax = 5#np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z_per.sum(axis=(1, 3))
    z_vmax = 5#np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z_per.sum(axis=(1, 2))
    z_vmax = 5#np.maximum(np.abs(np.min(zz)), np.max(zz))
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    plt.show()
    plt.savefig('hypo_per11.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    # Change colorbar for non diverging data
    #cmap = 'gnuplot_r'
    cmap = mpl.cm.get_cmap('Blues')
    cmap.set_under('w')
    cmap.set_bad('w')

    # Make third plot
    print 'Making third plot'
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z_matrix.sum(axis=(2, 3))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax2.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z_matrix.sum(axis=(1, 3))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax3.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z_matrix.sum(axis=(1, 2))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax4.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    plt.show()
    plt.savefig('hypo_kevin11.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    # Make fourth plot
    print 'Making fourth plot'
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z.sum(axis=(2, 3))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax2.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z.sum(axis=(1, 3))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax3.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z.sum(axis=(1, 2))
    z_vmax = np.partition(zz.flatten(), -2)[-2]
    mg = ax4.pcolormesh(tt, yy, zz.T, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    plt.show()
    plt.savefig('hypo_philipp11.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()


if __name__ == '__main__':
    main()
