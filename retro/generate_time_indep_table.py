#!/usr/bin/env python
# pylint: disable=wrong-import-position, range-builtin-not-iterating, too-many-locals

"""
Create time-independent "whole-detector" table.

First define a cartesian grid covering all of the IceCube fiducial volume and
sum for each voxel info from the tables placed at the locations of each DOM.

The new table is in (x, y, z), independent of time, for purposes of matching
with a hypothesis (which will be defined in a small subset of the voxels) to
determine the total charge expected for that hypothesis.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import os
from os.path import abspath, dirname, join

import numpy as np
import pyfits

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DETECTOR_GEOM_FILE
from retro import (expand, extract_photon_info, powerspace, spherical_volume,
                   sph2cart)


NUM_PHI_BINS = 40
NUM_SUBDIV = 2
RETRO_TABLE_T_IDX = 0

IC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_IC_DOM{dom:d}_r_cz_t_angles.fits'
)

DC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_DC_DOM{dom:d}_r_cz_t_angles.fits'
)


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--nx', type=int,
        help='''Number of x bins'''
    )
    parser.add_argument(
        '--ny', type=int,
        help='''Number of y bins'''
    )
    parser.add_argument(
        '--nz', type=int,
        help='''Number of z bins'''
    )
    parser.add_argument(
        '--xlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-700, 700),
        help='''limits on binning in x dimension'''
    )
    parser.add_argument(
        '--ylims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on binning in y dimension'''
    )
    parser.add_argument(
        '--zlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on binning in z dimension'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/data/icecube/retro_tables/full1000',
        help='''Directory containing retro tables''',
    )
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''NPY file containing DOM locations as (string, dom, x, y, z)
        entries'''
    )
    parser.add_argument(
        '--test', action='store_true',
        help='''Run a simple test using a single DOM'''
    )
    args = parser.parse_args()
    return args


def main(tables_dir, geom_file, xlims, ylims, zlims, nx, ny, nz, test=False):
    """Perform scans and minimization for events.

    Parameters
    ----------
    tables_dir : string
        Path to directory containing the retro tables

    geom_file : string
        File containing detector geometry

    xlims, ylims, zlims : sequence of two floats (each)
        Lower and upper limits for volume that will be divided.

    nx, ny, nz : int
        Number of voxels in each dimension. Note that there will be e.g. ``nx +
        1`` bin edges in the x-dimension.

    """
    tables_dir = expand(tables_dir)
    shape = (nx, ny, nz)
    #x_min, x_max = min(xlims), max(xlims)
    #x_extent = x_max - x_min
    #x_bin_num_factor = x_extent / nx

    #y_min, y_max = min(ylims), max(ylims)
    #y_extent = y_max - y_min
    #y_bin_num_factor = y_extent / ny

    #z_min, z_max = min(zlims), max(zlims)
    #z_extent = z_max - z_min
    #z_bin_num_factor = z_extent / nz

    # Create double precision arrays for maintaining precision during
    # accumulation (will convert to single to store to disk)

    # Total photons in the bin
    p_counts_per_xyz_bin = np.zeros(shape=shape, dtype=np.float64)

    # Average photon direction & length in the bin, encoded as a single vector
    # with x-, y-, and z-components
    avg_photon_x = np.zeros(shape=shape, dtype=np.float64)
    avg_photon_y = np.zeros(shape=shape, dtype=np.float64)
    avg_photon_z = np.zeros(shape=shape, dtype=np.float64)

    # Load detector geometry
    detector_geometry = np.load(geom_file)

    for table_kind in ['ic', 'dc']:
        if table_kind == 'ic':
            table_fpath_proto = IC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            strings_slice = slice(None, 79)
        elif table_kind == 'dc':
            table_fpath_proto = DC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            strings_slice = slice(79, 86)

        for dom_depth_idx in dom_depth_indices:
            # Get the (x, y, z) coords for all DOMs of this type and at this
            # depth index
            subdetector_dom_xyzs = detector_geometry[strings_slice, ...]

            # Load the 4D table for the DOM type / depth index
            fpath = table_fpath_proto.format(
                tables_dir=tables_dir, dom=dom_depth_idx
            )
            photon_info, bin_edges = extract_photon_info(
                fpath=expand(fpath), dom_depth_index=dom_depth_idx
            )
            #print('photon_info bin_edges:', bin_edges)
            p_count = photon_info.count[dom_depth_idx]
            print('p_count.shape:', p_count.shape)
            p_theta = photon_info.theta[dom_depth_idx]
            p_phi = photon_info.phi[dom_depth_idx]
            p_length = photon_info.length[dom_depth_idx]

            #print(p_count.shape)
            #print(p_theta.shape)
            #print(p_phi.shape)
            #print(p_length.shape)

            # Volumes of spherical-binning elements are independent of ``phi``
            # but will depend on ``theta`` and ``r``, so compute these once.
            costheta_edges = np.cos(bin_edges.theta)
            dr, dcostheta = np.meshgrid(
                np.diff(bin_edges.r),
                np.diff(costheta_edges),
                indexing='ij'
            )
            # Note: ``dphi`` is set to 1 so scaling for different phi binnings
            # just requires multiplying by the binning's ``dphi``.
            vols = spherical_volume(dr=dr, dcostheta=dcostheta, dphi=1)
            print('vols.shape:', vols.shape)

            # Marginalize out time for binned photon counts
            t_indep_p_count = p_count.sum(axis=RETRO_TABLE_T_IDX)
            print('t_indep_p_count.shape:', t_indep_p_count.shape)
            counts_per_vols = t_indep_p_count / vols
            print('counts_per_vols.shape:', counts_per_vols.shape)

            # Convert avg photon direction and length to x, y, z components
            # (note that the _binning_ for these is still spherical, just in
            # each bin is a vector quantity described in Cartesian coordinates)
            avg_gamma_x, avg_gamma_y, avg_gamma_z = sph2cart(
                r=p_length, theta=p_theta, phi=p_phi
            )

            #print(p_count.sum())
            #print(avg_gamma_x.shape)

            # Find weighted average over time binning dimension, marginalizing
            # out the time dimension (again, binning is spherical but contains
            # a vector quantity that is described in Cartesian coordinates)
            avg_gamma_x = np.nan_to_num(
                (avg_gamma_x * p_count).sum(axis=RETRO_TABLE_T_IDX)
                / p_count.sum(axis=RETRO_TABLE_T_IDX)
            )
            avg_gamma_y = np.nan_to_num(
                (avg_gamma_y * p_count).sum(axis=RETRO_TABLE_T_IDX)
                / p_count.sum(axis=RETRO_TABLE_T_IDX)
            )
            avg_gamma_z = np.nan_to_num(
                (avg_gamma_z * p_count).sum(axis=RETRO_TABLE_T_IDX)
                / p_count.sum(axis=RETRO_TABLE_T_IDX)
            )

            #avg_gamma_x = np.average(
            #    avg_gamma_x, axis=RETRO_TABLE_T_IDX, weights=p_count
            #)
            #avg_gamma_y = np.average(
            #    avg_gamma_y, axis=RETRO_TABLE_T_IDX, weights=p_count
            #)
            #avg_gamma_z = np.average(
            #    avg_gamma_z, axis=RETRO_TABLE_T_IDX, weights=p_count
            #)

            # Subdivide macro cells into micro cells for achieving higher
            # accuracy in trransferring their counts and avg photon vectors to
            # a cartesian grid.
            r_upsamp_edges = np.concatenate(
                [powerspace(r0, r1, NUM_SUBDIV + 1, 2)[:-1]
                 for r0, r1 in zip(bin_edges.r[:-1], bin_edges.r[1:])]
                + [[bin_edges.r[-1]]] # include final bin edge
            )
            costheta_upsamp_edges = np.concatenate(
                [np.linspace(ct0, ct1, NUM_SUBDIV + 1)[:-1]
                 for ct0, ct1 in zip(costheta_edges[:-1], costheta_edges[1:])]
                + [[costheta_edges[-1]]] # include final bin edge
            )
            phi_upsamp_edges = np.linspace(0, 2*np.pi, NUM_PHI_BINS + 1)

            # TODO: is the actual midpoint the right choice for ``r`` here?
            # E.g., center of mass will have dependence on range on theta and
            # phi of the volume element. This would be more correct in some
            # sense but more difficult to treat and for a "ring" the result is
            # erroneous for transferring to Cartesian coordinates. Maybe
            # instead just enforce that there be a minimum number of bins in
            # each dimension, so we avoid such corner cases, and can get away
            # with doing the "simple" thing as it's a reasonable approximation.
            r_upsamp_centers = 0.5 * (r_upsamp_edges[:-1] + r_upsamp_edges[1:])
            costheta_upsamp_centers = 0.5 * (costheta_upsamp_edges[:-1]
                                             + costheta_upsamp_edges[1:])
            theta_upsamp_centers = np.arccos(costheta_upsamp_centers)
            phi_upsamp_centers = 0.5 * (phi_upsamp_edges[:-1]
                                        + phi_upsamp_edges[1:])

            #print('n_r_upsamp_bins    :', len(r_upsamp_centers))
            #print('n_theta_upsamp_bins:', len(costheta_upsamp_centers))
            #print('n_phi_upsamp_bins  :', len(phi_upsamp_centers))

            # Convert upsampled bin centers to Cartesian coordinates
            #print('r_upsamp_centers:\n', r_upsamp_centers)
            print('theta_upsamp_centers:\n', theta_upsamp_centers)
            print('phi_upsamp_centers:\n', phi_upsamp_centers)
            phi_uc_mg, r_uc_mg, theta_uc_mg = np.meshgrid(
                phi_upsamp_centers, r_upsamp_centers, theta_upsamp_centers,
                indexing='ij'
            )
            print('phi_uc_mg.shape:', phi_uc_mg.shape)
            x_upsamp_centers, y_upsamp_centers, z_upsamp_centers = sph2cart(
                r=r_uc_mg, theta=theta_uc_mg, phi=phi_uc_mg
            )
            print('x_upsamp_centers:\n', x_upsamp_centers)

            dcostheta_upsamp, dr_upsamp = np.meshgrid(
                np.diff(r_upsamp_edges),
                np.diff(costheta_upsamp_edges),
                indexing='ij'
            )
            vols_upsamp = spherical_volume(
                dr=dr_upsamp,
                dcostheta=dcostheta_upsamp,
                dphi=1
            )

            print('vols_upsamp.shape:', vols_upsamp.shape)

            counts_upsamp = np.empty_like(vols_upsamp)
            avg_gamma_x_upsamp = np.empty_like(vols_upsamp)
            avg_gamma_y_upsamp = np.empty_like(vols_upsamp)
            avg_gamma_z_upsamp = np.empty_like(vols_upsamp)

            # Slice up the upsampled grid into chunks the same size as the
            # original and work on one such chunk at a time since the sizes
            # match.
            for row in range(NUM_SUBDIV):
                row_slice = slice(row, None, NUM_SUBDIV)
                for col in range(NUM_SUBDIV):
                    col_slice = slice(col, None, NUM_SUBDIV)
                    idx = (row_slice, col_slice)
                    print('vols_upsamp[idx].shape:', vols_upsamp[idx].shape)
                    counts_upsamp[idx] = vols_upsamp[idx] * counts_per_vols
                    avg_gamma_x_upsamp[idx] = avg_gamma_x * counts_upsamp[idx]
                    avg_gamma_y_upsamp[idx] = avg_gamma_y * counts_upsamp[idx]
                    avg_gamma_z_upsamp[idx] = avg_gamma_z * counts_upsamp[idx]

            counts_upsamp = np.broadcast_to(counts_upsamp,
                                            x_upsamp_centers.shape)
            avg_gamma_x_upsamp = np.broadcast_to(avg_gamma_x_upsamp,
                                                 x_upsamp_centers.shape)
            avg_gamma_y_upsamp = np.broadcast_to(avg_gamma_y_upsamp,
                                                 x_upsamp_centers.shape)
            avg_gamma_z_upsamp = np.broadcast_to(avg_gamma_z_upsamp,
                                                 x_upsamp_centers.shape)

            # Loop through all of the DOMs in this depth index, shifting the
            # coordinates and summing the expected counts in the Cartesian grid
            #for depth_idx in dom_depth_indices:
            for string_dom_xyzs in subdetector_dom_xyzs:
                print('string_dom_xyzs[0, :]:', string_dom_xyzs[0, :])
                string_x, string_y = string_dom_xyzs[0, :2]
                x_rel_centers = (x_upsamp_centers + string_x).flatten()
                print('x_rel_centers range:', x_rel_centers.min(),
                      x_rel_centers.max())
                y_rel_centers = (y_upsamp_centers + string_y).flatten()
                print('y_rel_centers range:', y_rel_centers.min(),
                      y_rel_centers.max())
                #xbin_indices = (
                #    (x_upsamp_centers + (string_x - x_min)) * x_bin_num_factor
                #).astype(np.int)
                #ybin_indices = (
                #    (y_upsamp_centers + (string_y - y_min)) * y_bin_num_factor
                #).astype(np.int)
                #valid_xy = (0 <= xbin_indices < nx) & (0 <= ybin_indices < ny)
                for dom_xyz in string_dom_xyzs:
                    dom_z = dom_xyz[2]
                    z_rel_centers = (z_upsamp_centers + dom_z).flatten()
                    print('z_rel_centers range:', z_rel_centers.min(),
                          z_rel_centers.max())
                    #zbin_indices = (
                    #    (z_upsamp_centers + (dom_z - z_min))
                    #    * z_bin_num_factor
                    #).astype(np.int)
                    #valid = valid_xy & (0 <= zbin_indices < nz)
                    if test:
                        print(x_rel_centers.shape)
                        print(counts_upsamp.shape)
                    tmp_bincounts, _ = np.histogramdd(
                        (x_rel_centers, y_rel_centers, z_rel_centers),
                        bins=(nx, ny, nz),
                        range=(xlims, ylims, zlims),
                        normed=False,
                        weights=counts_upsamp.flatten()
                    )
                    print('x_rel_centers:\n', x_rel_centers)
                    tmp_ag_x, _ = np.histogramdd(
                        (x_rel_centers, y_rel_centers, z_rel_centers),
                        bins=(nx, ny, nz),
                        range=(xlims, ylims, zlims),
                        normed=False,
                        weights=avg_gamma_x_upsamp.flatten()
                    )
                    tmp_ag_y, _ = np.histogramdd(
                        (x_rel_centers, y_rel_centers, z_rel_centers),
                        bins=(nx, ny, nz),
                        range=(xlims, ylims, zlims),
                        normed=False,
                        weights=avg_gamma_y_upsamp.flatten()
                    )
                    tmp_ag_z, _ = np.histogramdd(
                        (x_rel_centers, y_rel_centers, z_rel_centers),
                        bins=(nx, ny, nz),
                        range=(xlims, ylims, zlims),
                        normed=False,
                        weights=avg_gamma_z_upsamp.flatten()
                    )

                    p_counts_per_xyz_bin += tmp_bincounts
                    avg_photon_x += tmp_ag_x * tmp_bincounts
                    avg_photon_y += tmp_ag_y * tmp_bincounts
                    avg_photon_z += tmp_ag_z * tmp_bincounts

                    if test:
                        break
                if test:
                    break

            if test:
                break

            #num_bins_upsamp = BinningCoords(
            #    t=num_bins.t,
            #    r=(num_bins.r if num_bins.r > 100 else
            #       int(num_bins.r * np.ceil(100 / num_bins.r))),
            #    theta=(num_bins.theta if num_bins.theta > 100 else
            #           int(num_bins.theta * np.ceil(100 / num_bins.theta))),
            #    phi=100
            #)

            #for r_bin_num in range(num_bins.r):
            #    for theta_bin_num in range(num_bins.theta):
            #        pass

        if test:
            break

    avg_photon_x /= p_counts_per_xyz_bin
    avg_photon_y /= p_counts_per_xyz_bin
    avg_photon_z /= p_counts_per_xyz_bin

    avg_photon_x = np.nan_to_num(avg_photon_x)
    avg_photon_y = np.nan_to_num(avg_photon_y)
    avg_photon_z = np.nan_to_num(avg_photon_z)
    #print(avg_photon_x)
    #print(avg_photon_y)
    print('number of non-zero counts:', np.sum(p_counts_per_xyz_bin > 0.001))
    print('number of non-zero avg_photon_x:', np.sum(avg_photon_x > 0.001))
    print('number of non-zero avg_photon_y:', np.sum(avg_photon_y > 0.001))
    print('number of non-zero avg_photon_z:', np.sum(avg_photon_z > 0.001))

    arrays_names = [
        (p_counts_per_xyz_bin, 'counts'),
        (avg_photon_x, 'avg_photon_x'),
        (avg_photon_y, 'avg_photon_y'),
        (avg_photon_z, 'avg_photon_z')
    ]
    test_str = '_test' if test else ''
    for array, name in arrays_names:
        fname = ('qdeficit_cart_table_%dx%dx%d_%s%s.fits'
                 % (nx, ny, nz, name, test_str))
        fpath = join(tables_dir, fname)
        hdu = pyfits.PrimaryHDU(array.astype(np.float32))
        hdu.writeto(fpath, clobber=True)

    if test:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-variable

        fig = plt.figure(1, figsize=(12, 12), dpi=72)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')
        x_half_bw = (xlims[1] - xlims[0]) / 2
        y_half_bw = (ylims[1] - ylims[0]) / 2
        z_half_bw = (zlims[1] - zlims[0]) / 2
        x, y, z = np.meshgrid(
            np.linspace(xlims[0] + x_half_bw, xlims[1] - x_half_bw, nx),
            np.linspace(ylims[0] + y_half_bw, ylims[1] - y_half_bw, ny),
            np.linspace(zlims[0] + z_half_bw, zlims[1] - z_half_bw, nz),
            indexing='ij'
        )
        print('x.shape:', x.shape)
        ax.quiver(
            x, y, z, avg_photon_x, avg_photon_y, avg_photon_z,
        )
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        fig.tight_layout()
        plt.draw()
        plt.show()
        fig.savefig('vecfield.png')


if __name__ == '__main__':
    main(**vars(parse_args()))
