#!/usr/bin/env python
# pylint: disable=wrong-import-position, too-many-locals, too-many-statements, redefined-outer-name

"""
Make plots for a single hypothesis, comparing segmented to analytic hypotheses.
"""


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
from retro import HypoParams8D, TimeSphCoord, TimeCart3DCoord
from segmented_hypo import SegmentedHypo
from analytic_hypo import AnalyticHypo


def plot_hypo_comparisons():
    """Main"""
    # Binning defined to be same as that used for clsim
    bin_min = TimeSphCoord(t=-3000, r=0, theta=0, phi=0)
    bin_max = TimeSphCoord(t=0, r=200, theta=np.pi, phi=2*np.pi)
    num_bins = TimeSphCoord(t=50, r=20, theta=50, phi=36)

    # An arbitrary hypothesis for testing
    hypo_params = HypoParams8D(
        t=-1000, x=1, y=10, z=-50, track_zenith=1.08, track_azimuth=0.96,
        track_energy=20, cascade_energy=25
    )

    hypo_params_inv = HypoParams8D(
        t=-1000, x=1, y=10, z=-50, track_zenith=np.pi - 1.08,
        track_azimuth=-0.96,
        track_energy=20, cascade_energy=25
    )

    # An arbitrary hit coordinate for testing
    hit_dom_coord = TimeCart3DCoord(t=0, x=0, y=10, z=0)

    t0 = time.time()
    analytic_hypo = AnalyticHypo(hypo_params, cascade_e_scale=1,
                                 track_e_scale=1)
    analytic_hypo.set_binning(start=bin_min, stop=bin_max, num_bins=num_bins)
    analytic_hypo.compute_matrices(hit_dom_coord=hit_dom_coord)
    print('took %5.2f ms to calculate philipp z matrix'
          % ((time.time() - t0)*1000))

    z = np.zeros(num_bins)
    for bin_idx, count in analytic_hypo.photon_counts:
        z[bin_idx] = count
    print('total number of photons in philipp z matrix = %i (%0.2f %%)'
          % (z.sum(), z.sum() / analytic_hypo.tot_photons * 100))

    print('')

    # kevin array
    t0 = time.time()
    segmented_hypo = SegmentedHypo(params=hypo_params_inv, cascade_e_scale=1,
                                   track_e_scale=1, time_increment=1)
    segmented_hypo.set_binning(start=bin_min, stop=bin_max, num_bins=num_bins)
    segmented_hypo.compute_matrices(hit_dom_coord)

    print('took %5.2f ms to calculate kevin z matrix'
          % ((time.time() - t0)*1000))

    z_matrix = np.zeros(num_bins)
    for bin_idx, info in segmented_hypo.photon_info.iteritems():
        z_matrix[bin_idx] = info.count

    print('total number of photons in kevin z_matrix ='
          ' %i (%.2f %%)'
          % (z_matrix.sum(), z_matrix.sum() / analytic_hypo.tot_photons * 100))

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
    x_0, y_0, z_0 = analytic_hypo.track.point(analytic_hypo.track.t0)
    x_e, y_e, z_e = analytic_hypo.track.point(analytic_hypo.track.t0
                                              + analytic_hypo.track.dt)
    ax.plot([x_0, x_e], [y_0, y_e], zs=[z_0, z_e])
    ax.plot([-plt_lim, -plt_lim], [y_0, y_e], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [plt_lim, plt_lim], zs=[z_0, z_e], alpha=0.3, c='k')
    ax.plot([x_0, x_e], [y_0, y_e], zs=[-plt_lim, -plt_lim], alpha=0.3, c='k')

    print('Plotting differences in photon counts')
    bin_edges = analytic_hypo.bin_edges
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

    fig.suptitle('Differences in photon counts: segmented - analytic')

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
                 ' (segmented - analytic) / analytic')

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

    zz_seg_23 = z_matrix.sum(axis=(2, 3))
    zz_seg_13 = z_matrix.sum(axis=(1, 3))
    zz_seg_12 = z_matrix.sum(axis=(1, 2))

    zz_ana_23 = z.sum(axis=(2, 3))
    zz_ana_13 = z.sum(axis=(1, 3))
    zz_ana_12 = z.sum(axis=(1, 2))

    vmin_23 = min(zz_seg_23[zz_seg_23 > 0].min(),
                  zz_ana_23[zz_ana_23 > 0].min())
    vmin_13 = min(zz_seg_13[zz_seg_13 > 0].min(),
                  zz_ana_13[zz_ana_13 > 0].min())
    vmin_12 = min(zz_seg_12[zz_seg_12 > 0].min(),
                  zz_ana_12[zz_ana_12 > 0].min())

    vmax_23 = max(zz_seg_23.max(), zz_ana_23.max())
    vmax_13 = max(zz_seg_13.max(), zz_ana_13.max())
    vmax_12 = max(zz_seg_12.max(), zz_ana_12.max())

    print("Plotting segmented hypo photon counts")

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.r)
    mg = ax2.pcolormesh(tt, yy, zz_seg_23.T,
                        norm=colors.LogNorm(vmin=vmin_23, vmax=vmax_23),
                        cmap=cmap)
    ax2.set_xlabel('t')
    ax2.set_ylabel('r')
    cb2 = plt.colorbar(mg, ax=ax2)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.theta)
    mg = ax3.pcolormesh(tt, yy, zz_seg_13.T,
                        norm=colors.LogNorm(vmin=vmin_13, vmax=vmax_13),
                        cmap=cmap)
    ax3.set_xlabel('t')
    ax3.set_ylabel(r'$\theta$')
    ax3.set_ylim((0, np.pi))
    cb3 = plt.colorbar(mg, ax=ax3)

    tt, yy = np.meshgrid(bin_edges.t, bin_edges.phi)
    mg = ax4.pcolormesh(tt, yy, zz_seg_12.T,
                        norm=colors.LogNorm(vmin=vmin_12, vmax=vmax_12),
                        cmap=cmap)
    ax4.set_xlabel('t')
    ax4.set_ylabel(r'$\phi$')
    ax4.set_ylim((0, 2*np.pi))
    cb4 = plt.colorbar(mg, ax=ax4)

    ax2.grid(True, 'both', color='g')
    ax3.grid(True, 'both', color='g')
    ax4.grid(True, 'both', color='g')

    fig.suptitle('Photon counts, segmented hypo')

    plt.show()
    plt.savefig('hypo_photon_counts_segmented.png', dpi=300)

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

    return analytic_hypo, segmented_hypo


if __name__ == '__main__':
    analytic_hypo, segmented_hypo = plot_hypo_comparisons() # pylint: disable=invalid-name
