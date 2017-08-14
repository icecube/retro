#!/usr/bin/env python

# pylint: disable=wrong-import-position


from __future__ import absolute_import, division, print_function

import os
from os.path import abspath, dirname
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, binspec_to_edges, FTYPE, HypoParams8D,
                   TimeSpaceCoord)
from hypo_vector import IDX_R_IX, SegmentedHypo
from hypo_fast import Hypo


def main():
    # Binning defined as same as that used for CLsim
    bin_start = BinningCoords(t=0, r=0, theta=0, phi=0)
    bin_stop = BinningCoords(t=500, r=200, theta=np.pi, phi=2*np.pi)
    num_bins = BinningCoords(t=50, r=20, theta=50, phi=36)
    bin_edges = binspec_to_edges(start=bin_start, stop=bin_stop,
                                 num_bins=num_bins)

    # An arbitrary hypothesis for testing
    hypo_params = HypoParams8D(t=65, x=1, y=10, z=-50, track_zenith=1.08,
                               track_azimuth=0.96, track_energy=20,
                               cascade_energy=25)

    # An arbitrary hit coordinate for testing
    hit_dom_coord = TimeSpaceCoord(t=50, x=0, y=10, z=0)

    t0 = time.time()
    hypo_ana_fast = Hypo(hypo_params)
    hypo_ana_fast.set_binning(bin_edges)
    hypo_ana_fast.compute_matrices(hit_dom_coord=hit_dom_coord)
    print('took %5.2f ms to calculate philipp z matrix'
          % ((time.time() - t0)*1000))

    z = np.zeros(num_bins)
    for hit in hypo_ana_fast.photon_counts:
        idx, count = hit
        z[idx] = count
    print('total number of photons in philipp z matrix = %i (%0.2f %%)'
          % (z.sum(), z.sum() / hypo_ana_fast.tot_photons * 100))

    print('')

    # kevin array
    t0 = time.time()
    hypo_approx = SegmentedHypo(params=hypo_params, time_increment=1)
    hypo_approx.set_binning(start=bin_start, stop=bin_stop, num_bins=num_bins)
    hypo_approx.compute_matrices(hit_dom_coord)

    z_indices = hypo_approx.indices_array
    z_values = hypo_approx.values_array
    z_matrix = np.zeros(num_bins, dtype=FTYPE)

    for incr_idx in xrange(hypo_approx.number_of_increments):
        zmat_idx = tuple(z_indices[:, incr_idx])
        if z_indices[IDX_R_IX, incr_idx] < hypo_approx.bin_max.r:
            z_matrix[zmat_idx] += z_values[0, incr_idx]

    print('took %5.2f ms to calculate kevin z matrix'
          % ((time.time() - t0)*1000))

    print('total number of photons in kevin z_matrix ='
          ' %i (%.2f %%)'
          % (z_matrix.sum(), z_matrix.sum() / hypo_ana_fast.tot_photons * 100))

    print('')

    print('total_residual = ', (z - z_matrix).sum() / z.sum())

    print('')

    # Create difference matrix
    z_diff = z_matrix - z

    # Create fractional difference matrix
    z_fractdiff = np.zeros_like(z_matrix)
    mask = z != 0
    z_fractdiff[mask] = z_matrix[mask] / z[mask] - 1

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

    # Plot the track as a line
    x_0, y_0, z_0 = hypo_ana_fast.track.point(hypo_ana_fast.track.t0)
    x_e, y_e, z_e = hypo_ana_fast.track.point(hypo_ana_fast.track.t0
                                              + hypo_ana_fast.track.dt)
    ax.plot([x_0, x_e], [y_0, y_e], zs=[z_0, z_e])
    ax.plot([-plt_lim, -plt_lim], [y_0, y_e], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [plt_lim, plt_lim], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [y_0, y_e], zs=[-plt_lim, -plt_lim], alpha=0.3, c='k')

    print('Plotting differences in photon counts')
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z_diff.sum(axis=(2, 3))
    z_vmax = np.abs(zz).max()
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z_diff.sum(axis=(1, 3))
    z_vmax = np.abs(zz).max()
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z_diff.sum(axis=(1, 2))
    z_vmax = np.abs(zz).max()
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    fig.suptitle('Differences in photon counts: approx - analytic')

    plt.show()
    plt.savefig('hypo_diff.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    print('Plotting fractional differences in photon counts')
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    zz = z_fractdiff.sum(axis=(2, 3))
    z_vmax = np.abs(zz).max()
    mg = ax2.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    zz = z_fractdiff.sum(axis=(1, 3))
    z_vmax = np.abs(zz).max()
    mg = ax3.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    zz = z_fractdiff.sum(axis=(1, 2))
    z_vmax = np.abs(zz).max()
    mg = ax4.pcolormesh(tt, yy, zz.T, vmin=-z_vmax, vmax=z_vmax, cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    fig.suptitle('Fractional differences in photon counts:'
                 ' (approx - analytic) / analytic')

    plt.show()
    plt.savefig('hypo_fractdiff.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    # Change colorbar for non diverging data
    #cmap = 'gnuplot_r'
    cmap = mpl.cm.get_cmap('Blues')
    cmap.set_under('w')
    cmap.set_bad('w')

    zz_approx_23 = z_matrix.sum(axis=(2, 3))
    zz_approx_13 = z_matrix.sum(axis=(1, 3))
    zz_approx_12 = z_matrix.sum(axis=(1, 2))

    zz_ana_23 = z.sum(axis=(2, 3))
    zz_ana_13 = z.sum(axis=(1, 3))
    zz_ana_12 = z.sum(axis=(1, 2))

    vmin_23 = min(zz_approx_23[zz_approx_23 > 0].min(),
                  zz_ana_23[zz_ana_23 > 0].min())
    vmin_13 = min(zz_approx_13[zz_approx_13 > 0].min(),
                  zz_ana_13[zz_ana_13 > 0].min())
    vmin_12 = min(zz_approx_12[zz_approx_12 > 0].min(),
                  zz_ana_12[zz_ana_12 > 0].min())

    vmax_23 = max(zz_approx_23.max(), zz_ana_23.max())
    vmax_13 = max(zz_approx_13.max(), zz_ana_13.max())
    vmax_12 = max(zz_approx_12.max(), zz_ana_12.max())

    print("Plotting approximate method's photon counts")

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    mg = ax2.pcolormesh(tt, yy, zz_approx_23.T,
                        norm=colors.LogNorm(vmin=vmin_23, vmax=vmax_23),
                        cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    mg = ax3.pcolormesh(tt, yy, zz_approx_13.T,
                        norm=colors.LogNorm(vmin=vmin_13, vmax=vmax_13),
                        cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    mg = ax4.pcolormesh(tt, yy, zz_approx_12.T,
                        norm=colors.LogNorm(vmin=vmin_12, vmax=vmax_12),
                        cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    fig.suptitle('Photon counts, approximate method')

    plt.show()
    plt.savefig('hypo_photon_counts_approx.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()

    print("Plotting analytical method's photon counts")
    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    mg = ax2.pcolormesh(tt, yy, zz_ana_23.T,
                        norm=colors.LogNorm(vmin=vmin_23, vmax=vmax_23),
                        cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    mg = ax3.pcolormesh(tt, yy, zz_ana_13.T,
                        norm=colors.LogNorm(vmin=vmin_13, vmax=vmax_13),
                        cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    mg = ax4.pcolormesh(tt, yy, zz_ana_12.T,
                        norm=colors.LogNorm(vmin=vmin_12, vmax=vmax_12),
                        cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    fig.suptitle('Photon counts, analytic method')

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    plt.show()
    plt.savefig('hypo_photon_counts_ana.png', dpi=300)

    # Clear colorbars
    cb2.remove()
    cb3.remove()
    cb4.remove()


if __name__ == '__main__':
    main()
