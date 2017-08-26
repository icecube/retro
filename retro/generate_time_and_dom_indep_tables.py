#!/usr/bin/env python
# pylint: disable=wrong-import-position, range-builtin-not-iterating, too-many-locals, too-many-statements

"""
Create time- and DOM-independent "whole-detector" retro table.

Define a Cartesian grid that covers all of the IceCube fiducial volume, then
tabulate for each voxel the survival probability for photons coming from any
DOM at any time to reach that voxel. Also, tabulate the "average surviving
photon," defined by its x, y, and z components (which differs from the original
time- and DOM-dependent retro tables, wherein length, theta, and deltaphi are
used to characterize the average surviving photon).

The new table is in (x, y, z)--independent of time and DOM--and can be used to
scale the photons expected to reach any DOM at any time due to a hypothesis
that generates some number of photons (with an average direction / length) in
any of the voxel(s) of this table.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import os
from os.path import abspath, dirname, join

import numpy as np
import pyfits

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (DETECTOR_GEOM_FILE, DC_TABLE_FPATH_PROTO,
                   IC_TABLE_FPATH_PROTO)
from retro import (expand, extract_photon_info, powerspace, spherical_volume,
                   sph2cart)


__all__ = ['RETRO_T_IDX', 'CALC_FTYPE', 'TABLE_FTYPE', 'parse_args',
           'generate_time_and_dom_indep_tables']


RETRO_T_IDX = 0
"""Dimension number of time in the (t, r, theta) retro tables"""

RETRO_R_IDX = 1
"""Dimension number of radius in the (t, r, theta) retro tables"""

RETRO_THETA_IDX = 2
"""Dimension number of zenith angle in the (t, r, theta) retro tables"""

CALC_FTYPE = np.float64
"""Float type to use for intermediate calculations"""

TABLE_FTYPE = np.float32
"""Float type to use for storing tables to disk"""


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--xlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-700, 700),
        help='''limits on time-independent Cartesian binning in x dimension'''
    )
    parser.add_argument(
        '--ylims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on time-independent Cartesian binning in y dimension'''
    )
    parser.add_argument(
        '--zlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-650, 650),
        help='''limits on time-independent Cartesian binning in z dimension'''
    )
    parser.add_argument(
        '--nx', type=int, required=True,
        help='''Number of time-independent Cartesian x bins'''
    )
    parser.add_argument(
        '--ny', type=int, required=True,
        help='''Number of time-independent Cartesian y bins'''
    )
    parser.add_argument(
        '--nz', type=int, required=True,
        help='''Number of time-independent Cartesian z bins'''
    )
    parser.add_argument(
        '--oversample-r', type=int, required=True,
        help='''Subdivide original retro tables' r- and theta-binning this many
        times before placing in time-independent Cartesian binning.'''
    )
    parser.add_argument(
        '--oversample-theta', type=int, required=True,
        help='''Subdivide original retro tables' r- and theta-binning this many
        times before placing in time-independent Cartesian binning.'''
    )
    parser.add_argument(
        '--nphi', type=int, required=True,
        help='''Number of phi bins to use (retro tables are currently
        independent of phi, so --oversample has no effect on the phi
        dimension).'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/data/icecube/retro_tables/full1000',
        help='''Directory containing source retro tables; time-independent
        Cartesian table will be stored in this same directory, too.''',
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


def generate_time_and_dom_indep_tables(xlims, ylims, zlims, nx, ny, nz,
                                       oversample_r, oversample_theta, nphi,
                                       tables_dir, geom_file, test=False):
    """Generate time- and DOM-independent tables. Note that these tables are in
    Cartesian coordinates, defining nx x ny x nz voxels. One table contains the
    photon survival probability for each voxel (which can be > 1 since photons

    Parameters
    ----------
    xlims, ylims, zlims : sequence of two floats (each)
        Lower and upper limits for volume that will be divided.

    nx, ny, nz : int
        Number of voxels in each dimension. Note that there will be e.g. ``nx +
        1`` bin edges in the x-dimension.

    oversample_r, oversample_theta : int >= 1
        Number of times to subdivide original retro tables' r- and
        theta-binning prior to populating the time-independent Cartesian table.
        Presumably the larger oversample_{r,theta}, the more accurate the
        transfer of information will be (and the longer it will take).

    nphi : int >= 4
        Number of phi bins. It doesn't make sense for this to be a small
        number, probably at least O(10) if not O(100) would make sense. Note
        that e.g. "center of spherical volume element" calculations will yield
        erroneous results if this number is "small".

    tables_dir : string
        Path to directory containing the retro tables

    geom_file : string
        File containing detector geometry

    test : bool
        Only compute for single DOM, produce extra debugging messages, and
        produce quiver plot for generated table.

    """
    if test:
        print('xlims:', xlims)
        print('ylims:', ylims)
        print('zlims:', zlims)
    assert oversample_r >= 1
    assert oversample_theta >= 1
    assert int(oversample_r) == oversample_r
    assert int(oversample_theta) == oversample_theta
    oversample_r = int(oversample_r)
    oversample_theta = int(oversample_theta)

    tables_dir = expand(tables_dir)
    shape = (nx, ny, nz)

    # Create double precision arrays for maintaining precision during
    # accumulation (will convert to single to store to disk)

    # Total survival probability in each voxel (can be > 1 since same event can
    # cause hits at multiple times and/or multiple DOMs, both of which are
    # dimensions we're summing over for the Cartesian time- and DOM-independent
    # retro table)
    one_minus_survival_prob = np.ones(shape=shape, dtype=CALC_FTYPE)
    binned_p_survival_prob_by_vol = np.zeros(shape=shape, dtype=CALC_FTYPE)
    """overall normalizion that is applied after looping through all DOMs for
    avg. photon (i.e., p_x, p_y, p_z components)"""

    # Average photon direction & length in the bin, encoded as a single vector
    # with x-, y-, and z-components
    avg_photon_x = np.zeros(shape=shape, dtype=CALC_FTYPE)
    avg_photon_y = np.zeros(shape=shape, dtype=CALC_FTYPE)
    avg_photon_z = np.zeros(shape=shape, dtype=CALC_FTYPE)

    # Load detector geometry
    detector_geometry = np.load(geom_file)

    # Phi bins are not defined in the tables, so these are constants that can
    # be defined outside the loops
    phi_upsamp_edges = np.linspace(0, 2*np.pi, nphi + 1)
    phi_upsamp_centers = 0.5 * (phi_upsamp_edges[:-1] + phi_upsamp_edges[1:])

    doms_used = []

    for table_kind in ['ic', 'dc']:
        if table_kind == 'ic':
            table_fpath_proto = IC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            strings_slice = slice(None, 79)
        elif table_kind == 'dc':
            table_fpath_proto = DC_TABLE_FPATH_PROTO
            dom_depth_indices = range(60)
            strings_slice = slice(79, 86)

        # There is one unique reco table per DOM depth index, so do the
        # following just once per depth index and generalize to all DOMs at the
        # same (approx.) depth

        for dom_depth_idx in dom_depth_indices:
            if test and dom_depth_idx not in [28, 29, 30]:
                continue

            # Get the (x, y, z) coords for all DOMs of this type and at this
            # depth index
            subdet_depth_dom_coords = (
                detector_geometry[strings_slice, dom_depth_idx, ...]
            )

            # Load the 4D table for the DOM type / depth index
            fpath = table_fpath_proto.format(
                tables_dir=tables_dir, dom=dom_depth_idx
            )
            photon_info, bin_edges = extract_photon_info(
                fpath=expand(fpath), dom_depth_index=dom_depth_idx
            )

            p_survival_prob = photon_info.survival_prob[dom_depth_idx]
            p_theta = photon_info.theta[dom_depth_idx]
            p_deltaphi = photon_info.deltaphi[dom_depth_idx]
            p_length = photon_info.length[dom_depth_idx]
            if test:
                print('p_survival_prob range:', p_survival_prob.min(),
                      p_survival_prob.max())
                print('p_theta range:', p_theta.min(), p_theta.max())
                print('p_deltaphi range:', p_deltaphi.min(), p_deltaphi.max())
                print('p_length range:', p_length.min(), p_length.max())

            # TODO: How do we handle deltaphi in [0, pi]? How does this make
            # sense? Only thing I can think of is that nonzero deltaphi will
            # cause p_length to be reduced since photons could be coming from
            # either of two directions, ergo how directional the average photon
            # is must be reduced
            #assert np.all(p_deltaphi <= np.pi/2)
            p_length *= np.cos(p_deltaphi)
            p_deltaphi = np.full_like(a=p_deltaphi, fill_value=0)

            orig_p_info_shape = p_survival_prob.shape
            p_info_w_phi_shape = (nphi,) + orig_p_info_shape

            # Marginalize out time for photon survival probabilities; this
            # implementes, effectively, the logic
            #   ``Prob(t0 or t1 or ... or tN)``
            t_indep_p_survival_prob = (
                1 - (1 - p_survival_prob).prod(axis=RETRO_T_IDX)
            )
            if test:
                print('t_indep_p_survival_prob.max():',
                      t_indep_p_survival_prob.max())

            # Broadcast p_theta and p_length--which currently have 3
            # dimensions, representing (t, r, theta)--to 4D, (phi, t, r, theta)
            p_theta = np.broadcast_to(p_theta, p_info_w_phi_shape)
            p_length = np.broadcast_to(p_length, p_info_w_phi_shape)

            # Do the same for p_survival_prob, which will be used as the
            # weights for averaging the average photon properties over time
            weights = np.broadcast_to(p_survival_prob, p_info_w_phi_shape)
            t_indep_weights = weights.sum(axis=1 + RETRO_T_IDX)
            mask = t_indep_weights != 0
            scale = 1 / t_indep_weights[mask]

            t_indep_p_info_shape = tuple(
                [nphi]
                + [orig_p_info_shape[d] for d in range(3) if d != RETRO_T_IDX]
            )

            t_indep_p_info_upsamp_shape = [nphi]
            for d in range(3):
                if d == RETRO_T_IDX:
                    continue
                elif d == RETRO_R_IDX:
                    oversample = oversample_r
                elif d == RETRO_THETA_IDX:
                    oversample = oversample_theta
                t_indep_p_info_upsamp_shape.append(
                    oversample * orig_p_info_shape[d]
                )
            t_indep_p_info_upsamp_shape = tuple(t_indep_p_info_upsamp_shape)

            # Create p_phi from nphi phi-bin centers and p_deltaphi
            p_phi = np.array([phi + p_deltaphi for phi in phi_upsamp_centers])

            # Convert avg photon info to Cartesian coordinates
            p_x, p_y, p_z = sph2cart(r=p_length, theta=p_theta, phi=p_phi)

            # Weighted-average out the time dimension (since we prepended a phi
            # axis, time axis is now one more than in original retro tables)
            t_indep_p_x = np.zeros(t_indep_p_info_shape)
            t_indep_p_y = np.zeros(t_indep_p_info_shape)
            t_indep_p_z = np.zeros(t_indep_p_info_shape)

            t_indep_p_x[mask] = (
                (p_x * weights).sum(axis=1 + RETRO_T_IDX)[mask] * scale
            )
            t_indep_p_y[mask] = (
                (p_y * weights).sum(axis=1 + RETRO_T_IDX)[mask] * scale
            )
            t_indep_p_z[mask] = (
                (p_z * weights).sum(axis=1 + RETRO_T_IDX)[mask] * scale
            )

            # Subdivide macro cells into micro cells for achieving higher
            # accuracy in trransferring their survival probabilities and avg
            # photon vectors to the Cartesian grid
            r_upsamp_edges = np.concatenate(
                [powerspace(start=a, stop=b, num=oversample_r + 1, power=2)[:-1]
                 for a, b in zip(bin_edges.r[:-1], bin_edges.r[1:])]
                + [[bin_edges.r[-1]]] # include final bin edge
            )

            costheta_edges = np.cos(bin_edges.theta)
            costheta_upsamp_edges = np.concatenate(
                [np.linspace(start=a, stop=b, num=oversample_theta + 1)[:-1]
                 for a, b in zip(costheta_edges[:-1], costheta_edges[1:])]
                + [[costheta_edges[-1]]] # include final bin edge
            )

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

            # Convert upsampled bin centers to Cartesian coordinates
            (phi_upsamp_grid,
             r_upsamp_centers_grid,
             theta_upsamp_centers_grid) = np.meshgrid(phi_upsamp_centers,
                                                      r_upsamp_centers,
                                                      theta_upsamp_centers,
                                                      indexing='ij')
            x_upsamp_centers, y_upsamp_centers, z_upsamp_centers = sph2cart(
                r=r_upsamp_centers_grid,
                theta=theta_upsamp_centers_grid,
                phi=phi_upsamp_grid
            )

            dcostheta_upsamp, dr_upsamp = np.meshgrid(
                np.diff(r_upsamp_edges),
                np.diff(costheta_upsamp_edges),
                indexing='ij'
            )
            vol_upsamp = np.broadcast_to(
                spherical_volume(
                    dr=dr_upsamp,
                    dcostheta=dcostheta_upsamp,
                    dphi=2*np.pi / nphi
                ),
                t_indep_p_info_upsamp_shape
            )

            t_indep_p_survival_prob_by_vol_upsamp = (
                np.empty(t_indep_p_info_upsamp_shape)
            )
            p_x_by_sp_by_vol_upsamp = np.empty(t_indep_p_info_upsamp_shape)
            p_y_by_sp_by_vol_upsamp = np.empty(t_indep_p_info_upsamp_shape)
            p_z_by_sp_by_vol_upsamp = np.empty(t_indep_p_info_upsamp_shape)

            # Slice up the upsampled grid into chunks the same size as the
            # original and work on one such chunk at a time (the sizes will
            # match, making numpy operations possible). This performs an
            # interleaved tiling.
            phi_slice = slice(None)
            for r_subbin in range(oversample_r):
                r_slice = slice(r_subbin, None, oversample_r)
                for theta_subbin in range(oversample_theta):
                    theta_slice = slice(theta_subbin, None, oversample_theta)
                    idx = (phi_slice, r_slice, theta_slice)

                    t_indep_p_survival_prob_by_vol = (
                        t_indep_p_survival_prob * vol_upsamp[idx]
                    )

                    t_indep_p_survival_prob_by_vol_upsamp[idx] = (
                        t_indep_p_survival_prob_by_vol
                    )

                    p_x_by_sp_by_vol_upsamp[idx] = (
                        t_indep_p_x * t_indep_p_survival_prob_by_vol
                    )
                    p_y_by_sp_by_vol_upsamp[idx] = (
                        t_indep_p_y * t_indep_p_survival_prob_by_vol
                    )
                    p_z_by_sp_by_vol_upsamp[idx] = (
                        t_indep_p_z * t_indep_p_survival_prob_by_vol
                    )

            p_x_by_sp_by_vol_upsamp = p_x_by_sp_by_vol_upsamp.flatten()
            p_y_by_sp_by_vol_upsamp = p_y_by_sp_by_vol_upsamp.flatten()
            p_z_by_sp_by_vol_upsamp = p_z_by_sp_by_vol_upsamp.flatten()

            if test:
                print('t_indep_p_survival_prob_by_vol_upsamp.max():',
                      t_indep_p_survival_prob_by_vol_upsamp.max())

            # Loop through all of the DOMs in this subdetector index, shifting
            # the coordinates and aggregating the expected survival
            # probabilities  and average photon info in the Cartesian grid
            for str_idx, string_dom_xyz in enumerate(subdet_depth_dom_coords):
                if test and str_idx not in [25, 26, 34, 35, 36, 44, 45]:
                    continue

                string_x, string_y, dom_z = string_dom_xyz
                x_rel_centers = (string_x - x_upsamp_centers).flatten()
                y_rel_centers = (string_y - y_upsamp_centers).flatten()
                z_rel_centers = (dom_z - z_upsamp_centers).flatten()

                doms_used.append([string_x, string_y, dom_z])

                if test:
                    print('z_rel_centers range:', z_rel_centers.min(),
                          z_rel_centers.max())
                    print('x_rel_centers.shape:', x_rel_centers.shape)
                    print('t_indep_p_survival_prob_by_vol_upsamp.shape:',
                          t_indep_p_survival_prob_by_vol_upsamp.shape)
                    print('p_x_by_sp_by_vol_upsamp.shape:',
                          p_x_by_sp_by_vol_upsamp.shape)
                    print('vol_upsamp.shape:', vol_upsamp.shape)
                    print('x_rel_centers:\n', x_rel_centers)

                tmp_binned_vol, _ = np.histogramdd(
                    (x_rel_centers, y_rel_centers, z_rel_centers),
                    bins=(nx, ny, nz),
                    range=(xlims, ylims, zlims),
                    normed=False,
                    weights=vol_upsamp.flatten()
                )

                # For a single DOM, the survival probability in each voxel
                # is the volume-weighted average of the component survival
                # probabilities. Compute this by histogramming the
                # vol-weighted survival probabilities and then normalize by
                # dividing by the total volume accumulated in each
                # histogram bin.
                tmp_binned_p_survival_prob_by_vol, _ = np.histogramdd(
                    (x_rel_centers, y_rel_centers, z_rel_centers),
                    bins=(nx, ny, nz),
                    range=(xlims, ylims, zlims),
                    normed=False,
                    weights=t_indep_p_survival_prob_by_vol_upsamp.flatten()
                )

                binned_p_survival_prob_by_vol += (
                    tmp_binned_p_survival_prob_by_vol
                )

                mask = tmp_binned_vol != 0
                normed_surv_prob = (tmp_binned_p_survival_prob_by_vol[mask]
                                    / tmp_binned_vol[mask])

                if test:
                    print('')
                    print('mask.sum():', mask.sum())
                    print('t_indep_p_survival_prob_by_vol_upsamp.max():',
                          t_indep_p_survival_prob_by_vol_upsamp.max())
                    print('tmp_binned_p_survival_prob_by_vol.max():',
                          tmp_binned_p_survival_prob_by_vol.max())
                    print('tmp_binned_vol range:', tmp_binned_vol.min(),
                          tmp_binned_vol.max())
                    print('normed_surv_prob range:', normed_surv_prob.min(),
                          normed_surv_prob.max())
                    print('')

                one_minus_survival_prob[mask] *= (
                    1 - normed_surv_prob
                )

                # The average photon that survives for a voxel is the
                # ((survival probability) x (volume))-weighted average of
                # the average photons from each spherical volume element
                # that falls within the voxel.
                tmp_ag_x, _ = np.histogramdd(
                    (x_rel_centers, y_rel_centers, z_rel_centers),
                    bins=(nx, ny, nz),
                    range=(xlims, ylims, zlims),
                    normed=False,
                    weights=p_x_by_sp_by_vol_upsamp
                )
                tmp_ag_y, _ = np.histogramdd(
                    (x_rel_centers, y_rel_centers, z_rel_centers),
                    bins=(nx, ny, nz),
                    range=(xlims, ylims, zlims),
                    normed=False,
                    weights=p_y_by_sp_by_vol_upsamp
                )
                tmp_ag_z, _ = np.histogramdd(
                    (x_rel_centers, y_rel_centers, z_rel_centers),
                    bins=(nx, ny, nz),
                    range=(xlims, ylims, zlims),
                    normed=False,
                    weights=p_z_by_sp_by_vol_upsamp
                )
                if test:
                    print('tmp_ag_x.max():', tmp_ag_x.max())

                avg_photon_x += tmp_ag_x
                avg_photon_y += tmp_ag_y
                avg_photon_z += tmp_ag_z

    doms_used = np.array(doms_used)

    mask = binned_p_survival_prob_by_vol != 0
    scale = 1 / binned_p_survival_prob_by_vol[mask]
    avg_photon_x[mask] *= scale
    avg_photon_y[mask] *= scale
    avg_photon_z[mask] *= scale

    survival_prob = 1 - one_minus_survival_prob
    arrays_names = [
        (survival_prob, 'survival_prob'),
        (avg_photon_x, 'avg_photon_x'),
        (avg_photon_y, 'avg_photon_y'),
        (avg_photon_z, 'avg_photon_z')
    ]
    test_str = '_test' if test else ''
    fbasename = (
        'qdeficit_cart_table_%dx%dx%d_os_r%d_zen%d%s'
        % (nx, ny, nz, oversample_r, oversample_theta, test_str)
        )
    for array, name in arrays_names:
        fname = '%s_%s.fits' % (fbasename, name)
        fpath = join(tables_dir, fname)
        hdu = pyfits.PrimaryHDU(array.astype(np.float32))
        hdu.writeto(fpath, clobber=True)

    if test:
        #import matplotlib as mpl
        #mpl.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-variable

        x_half_bw = (xlims[1] - xlims[0]) / nx / 2
        y_half_bw = (ylims[1] - ylims[0]) / ny / 2
        z_half_bw = (zlims[1] - zlims[0]) / nz / 2
        x, y, z = np.meshgrid(
            np.linspace(xlims[0] + x_half_bw, xlims[1] - x_half_bw, nx),
            np.linspace(ylims[0] + y_half_bw, ylims[1] - y_half_bw, ny),
            np.linspace(zlims[0] + z_half_bw, zlims[1] - z_half_bw, nz),
            indexing='ij'
        )

        print('survival_prob.max:', survival_prob.max())

        print('x.shape:', x.shape)
        mask = survival_prob > 0.0001
        print('number of non-zero counts:', np.sum(mask))
        print('at x:', (x[mask].min(), x[mask].max()))
        print('at y:', (y[mask].min(), y[mask].max()))
        print('at z:', (z[mask].min(), z[mask].max()))
        length = np.sqrt(avg_photon_x**2 + avg_photon_y**2 + avg_photon_z**2)
        mask = length > 0.001
        print('number of avg_photon len > 0.001:', np.sum(mask))
        print('maxlen:', length.max())

        fig1 = plt.figure(1, figsize=(12, 12), dpi=72)
        fig1.clf()
        ax1 = fig1.add_subplot(111, projection='3d')
        ax1.quiver(
            x[mask], y[mask], z[mask],
            avg_photon_x[mask]*100,
            avg_photon_y[mask]*100,
            avg_photon_z[mask]*100,
            alpha=0.7
        )
        ax1.plot(
            doms_used[:, 0], doms_used[:, 1], doms_used[:, 2],
            marker='o', linestyle='none', markersize=2, color='k'
        )

        fig2 = plt.figure(2, figsize=(12, 12), dpi=72)
        fig2.clf()
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot(
            doms_used[:, 0], doms_used[:, 1], doms_used[:, 2],
            marker='o', markersize=2, linestyle='none', color='k'
        )
        mask = survival_prob > 0.001
        ax2.scatter(
            x[mask], y[mask], z[mask], c=survival_prob[mask],
            cmap='YlOrBr',
            alpha=0.3
        )

        for ax, title in [(ax1, 'Average surviving photon'),
                          (ax2, 'Survival probability')]:
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_zlim(zlims)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
            ax.set_title(title)

        for fig, name in [(fig1, 'avg_photon'), (fig2, 'survival_prob')]:
            fig.tight_layout()
            fname = join(tables_dir, fbasename + '_' + name)
            fig.savefig(fname + '.png', dpi=300)
            fig.savefig(fname + '.pdf')

        plt.draw()
        plt.show()


if __name__ == '__main__':
    generate_time_and_dom_indep_tables(**vars(parse_args()))