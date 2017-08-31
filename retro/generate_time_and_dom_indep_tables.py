#!/usr/bin/env python
# pylint: disable=wrong-import-position, range-builtin-not-iterating, too-many-locals, too-many-statements, line-too-long

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
import time

import numba
import numpy as np
import pyfits
#from scipy.ndimage.filters import gaussian_filter

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (DETECTOR_GEOM_FILE, DC_TABLE_FPATH_PROTO,
                   IC_TABLE_FPATH_PROTO)
from retro import (expand, extract_photon_info, pol2cart, spherical_volume,
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
"""Float type to use for intermediate calculations (particularly aggregation)"""

TABLE_FTYPE = np.float32
"""Float type to use for storing tables to disk"""


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--xlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-900, 900),
        help='''limits on time-independent Cartesian binning in x dimension'''
    )
    parser.add_argument(
        '--ylims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-900, 900),
        help='''limits on time-independent Cartesian binning in y dimension'''
    )
    parser.add_argument(
        '--zlims', metavar='LOWER, UPPER', type=float, nargs=2,
        default=(-750, 650),
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
        '--nphi', type=int, required=True,
        help='''Number of phi bins to use (retro tables input to this script
        are currently independent of phi).'''
    )
    parser.add_argument(
        '--t-start', type=int, default=None,
        help='''Time slice start index (from 0 to however many time slices are
        in the binning).'''
    )
    parser.add_argument(
        '--t-stop', type=int, default=None,
        help='''Time slice start index (from 0 to however many time slices are
        in the binning).'''
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


def generate_time_and_dom_indep_tables(xlims, ylims, zlims, nx, ny, nz, nphi,
                                       tables_dir, geom_file, test=False,
                                       t_start=None, t_stop=None):
    """Generate time- and DOM-independent tables. Note that these tables are in
    Cartesian coordinates, defining nx x ny x nz voxels. One table contains the
    photon survival probability for each voxel (which can be > 1 since photons

    Parameters
    ----------
    xlims, ylims, zlims : sequence of two floats (each)
        Lower and upper limits for volume that will be divided.

    nx, ny, nz : int
        Number of voxels in each dimension. Note that there will be e.g.
        ``nx + 1`` bin edges in the x-dimension.

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

    plot : bool
        Whether to produce plots of the table (this is not recommended for
        ``nx*ny*nz`` much larger than 10,000)

    interactive : bool
        If plotting, displays the plot to the user. This flag has no effect if
        ``plot=False``

    """
    start_time = time.time()
    if test:
        print('xlims:', xlims)
        print('ylims:', ylims)
        print('zlims:', zlims)

    tables_dir = expand(tables_dir)

    xyz_shape = (nx, ny, nz)
    xb0, xb1 = CALC_FTYPE(np.min(xlims)), CALC_FTYPE(np.max(xlims))
    yb0, yb1 = CALC_FTYPE(np.min(ylims)), CALC_FTYPE(np.max(ylims))
    zb0, zb1 = CALC_FTYPE(np.min(zlims)), CALC_FTYPE(np.max(zlims))
    xbscale = CALC_FTYPE(1 / ((xb1 - xb0) / nx))
    ybscale = CALC_FTYPE(1 / ((yb1 - yb0) / ny))
    zbscale = CALC_FTYPE(1 / ((zb1 - zb0) / nz))
    xbw, ybw, zbw = (xb1 - xb0) / nx, (yb1 - yb0) / ny, (zb1 - zb0) / nz
    cart_bin_vol = xbw * ybw * zbw
    if test:
        print('xbinstuff:', xb0, xb1, nx, xbscale)
        print('ybinstuff:', yb0, yb1, ny, ybscale)
        print('zbinstuff:', zb0, zb1, nz, zbscale)

    # Total survival probability in each voxel
    one_minus_survival_prob = np.ones(shape=xyz_shape, dtype=CALC_FTYPE)

    # Overall normalizion applied after looping through all DOMs for avg.
    # photon (i.e., p_x, p_y, p_z components)
    binned_p_survival_prob_by_vol = np.zeros(shape=xyz_shape, dtype=CALC_FTYPE)

    # Average photon direction & length in the bin, encoded as a single vector
    # with x-, y-, and z-components
    avg_photon_x = np.zeros(shape=xyz_shape, dtype=CALC_FTYPE)
    avg_photon_y = np.zeros(shape=xyz_shape, dtype=CALC_FTYPE)
    avg_photon_z = np.zeros(shape=xyz_shape, dtype=CALC_FTYPE)

    # Load detector geometry
    detector_geometry = np.load(geom_file)

    # Phi bins are not defined in the tables, so these are constants that can
    # be defined outside the loops
    phi_edges = np.linspace(0, 2*np.pi, nphi + 1)
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])

    # TODO: function to take spherical and cartesian binnings and output the
    # cartesian indices corresponding to each spherical bin, as well as the
    # volume of overlap. This should have the origins of both Cartesian and
    # spherical binnings at the same location, so that when one wants to shift
    # the spherical binning to a particular DOM, the Cartesian bin indices
    # merely need to be modified by the (integer) amount. Note that this means
    # that the accuracy for a given DOM will be +/- 0.5 Caresian bin lengths
    # (in each dimension). One _could_ upsample the rectangular binning by e.g.
    # a factor of 2, and then the accuracy gets a little better...
    @numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
    def sphbin2cartbin(r_max, r_power, n_rbins, n_thetabins,
                       x_bw, y_bw, z_bw,
                       x_oversample, y_oversample, z_oversample,
                       antialias_per_dim=1):
        """
        Parameters
        ----------
        r_edges_mg, theta_edges_mg, phi_edges_mg : numpy.ndarray
            Meshgrid of r, theta, and phi edges

        x_bw, y_bw, z_bw : float
            Cartesian binwidths in x, y, and z directions

        x_oversample, y_oversample, z_oversample : int
            Oversmapling factors. If oversampling is used, the returned indices
            array will have floating point values. E.g., a bin index with
            oversampling of 2 could have take values 0, 0.5, 1, ...
            Note that this increases the computational cost _and_ increases the
            memory footprint of the produced array(s).

        antialias_factor : int
            The smallest binning unit in each dimension is divided again by
            this factor for more accruately computing the volume of overlap
            (and then the sub-binning for antialiasing is discarded). This
            therefore does not add to the memory footprint, but will increase
            the computational cost.

        Returns
        -------
        indices : list of M shape (N x 3) numpy.ndarrays
            One array per spherical bin. Data type of the arrays is int32 if
            all oversample factors are set to 1; otherwise, dtype is float64.

        overlap_vol : list of M shape (N,) numpy.ndarrays, dtype of float64
            One array per spherical bin

        """
        x_bw_os = x_bw / x_oversample
        y_bw_os = y_bw / y_oversample
        z_bw_os = z_bw / z_oversample

        x_bw_os_aa = x_bw_os / antialias_factor
        y_bw_os_aa = y_bw_os / antialias_factor
        z_bw_os_aa = z_bw_os / antialias_factor

        n_xbins_oct_os_aa = int(np.ceil(r_max / x_bw_os_aa))
        n_ybins_oct_os_aa = int(np.ceil(r_max / y_bw_os_aa))
        n_zbins_oct_os_aa = int(np.ceil(r_max / z_bw_os_aa))

        for x in range(n_xbins):
            x
        pass

    @numba.jit(nopython=True, nogil=True, cache=True, fastmath=True)
    def bin_quantities(r_edges_mg, theta_edges_mg, phi_edges_mg,
                       x_edges_mg, y_edges_mg, z_edges_mg,
                       x_offset, y_offset, z_offset,
                       bin_vols, survival_prob, p_x, p_y, p_z,
                       binned_p_x, binned_p_y, binned_p_z, binned_sp_by_vol,
                       one_minus_survival_prob):
        """Bin various quantities in one step (rather tha multiple calls to
        `numpy.histogramdd`.

        Parameters
        ----------
        r_edges_mg, theta_edges_mg, phi_edges_mg : numpy.ndarray, same shape
            Relative coordinates where the data values lie

        x_edges_mg, y_edges_mg, z_edges_mg : numpy.ndarray, same shape
            Relative coordinates where the data values lie

        x_offset, y_offset, z_offset : float
            Offset for the relative coordinates to translate to detector

        survival_prob : numpy.ndarray, same shape as `x`, `y`, and `z`
            Survival probability of a photon at this coordinate

        p_x, p_y, p_z : numpy.ndarray, same shape as `x`, `y`, and `z`
            Average photon x-, y-, and z-components at this coordinate

        binned_p_x, binned_p_y, binned_p_z : numpy.ndarray of shape (nx, ny, nz)
            Existing arrays into which average photon components are accumulated

        binned_sp_by_vol : numpy.ndarray, same shape as `x`, `y`, and `z`
            Binned photon survival probabilities * volumes, accumulated for all
            DOMs to normalize the average surviving photon info (`binned_p_x`,
            etc.) in the end

        one_minus_survival_prob : numpy.ndarray of shape (nx, ny, nz)
            Existing array to which ``1 - normed_survival_probability`` is
            multiplied (where the normalization factor is not infinite)

        """
        vol_mask = np.zeros((nx, ny, nz), dtype=np.int8)
        binned_vol = np.zeros((nx, ny, nz), dtype=CALC_FTYPE)
        xbshift = xb0 - x_offset
        ybshift = yb0 - y_offset
        zbshift = zb0 - z_offset

        for idx0 in range(nphi):
            slice0 = slice(idx0, idx0 + 2)
            for idx1 in range(n_rbins):
                slice1 = slice(idx1, idx1 + 2)
                for idx2 in range(n_thetabins):
                    slice2 = slice(idx2, idx2 + 2)
                    three_d_slice = (slice0, slice1, slice2)
                    three_d_idx = (idx0, idx1, idx2)

                    x = x_edges_mg[three_d_slice]
                    xidx0 = int((np.min(x) - xbshift) * xbscale)
                    xidx1 = int((np.max(x) - xbshift) * xbscale)
                    if xidx1 < 0 or xidx0 >= nx:
                        continue

                    y = y_edges_mg[three_d_slice]
                    yidx0 = int((np.min(y) - ybshift) * ybscale)
                    yidx1 = int((np.max(y) - ybshift) * ybscale)
                    if yidx1 < 0 or yidx0 >= ny:
                        continue

                    z = z_edges_mg[three_d_slice]
                    zidx0 = int((np.min(z) - zbshift) * zbscale)
                    zidx1 = int((np.max(z) - zbshift) * zbscale)
                    if zidx1 < 0 or zidx0 >= nz:
                        continue

                    vol = min(cart_bin_vol, bin_vols[three_d_idx])
                    sp = survival_prob[three_d_idx]

                    sp_by_vol = sp * vol
                    p_x_ = p_x[three_d_idx]
                    p_y_ = p_y[three_d_idx]
                    p_z_ = p_z[three_d_idx]

                    r_edges = r_edges_mg[three_d_slice]
                    r_lower, r_upper = np.min(r_edges), np.max(r_edges)
                    theta_edges = theta_edges_mg[three_d_slice]
                    theta_lower, theta_upper = np.min(theta_edges), np.max(theta_edges)
                    phi_edges = phi_edges_mg[three_d_slice]
                    phi_lower, phi_upper = np.min(phi_edges), np.max(phi_edges)

                    # Loop through all Cartesian bins in the rectangle
                    # enclosing the spherical volume element, and more
                    # carefully determine which actually overlap
                    xidx0 = max(0, xidx0)
                    yidx0 = max(0, yidx0)
                    zidx0 = max(0, zidx0)
                    xidx1 = min(nx, xidx1 + 1)
                    yidx1 = min(ny, yidx1 + 1)
                    zidx1 = min(nz, zidx1 + 1)
                    # TODO: why e.g. xidx + 0.0 looks good, but xidx + 0.5 looks bad?

                    for xidx in range(xidx0, xidx1):
                        x_rel_center = xbw * xidx + xbshift
                        for yidx in range(yidx0, yidx1):
                            y_rel_center = ybw * yidx + ybshift
                            for zidx in range(zidx0, zidx1):
                                z_rel_center = zbw * zidx + zbshift
                                rho_sq = x_rel_center*x_rel_center + y_rel_center*y_rel_center
                                r_ = np.sqrt(rho_sq + z_rel_center*z_rel_center)
                                if r_ < r_lower or r_ > r_upper:
                                    continue
                                theta_ = np.arccos(z_rel_center / r_)
                                if theta_ < theta_lower or theta_ > theta_upper:
                                    continue
                                phi_ = np.arctan2(y_rel_center, x_rel_center) % (2*np.pi)
                                if phi_ < phi_lower or phi_ > phi_upper:
                                    continue

                                # TODO: more advanced intersection volume
                                # calculation? E.g., antialiasing by subsampling?
                                bin_idx = (xidx, yidx, zidx)
                                vol_mask[bin_idx] = 1
                                binned_vol[bin_idx] += vol
                                binned_sp_by_vol[bin_idx] += sp_by_vol
                                binned_p_x[bin_idx] += p_x_
                                binned_p_y[bin_idx] += p_y_
                                binned_p_z[bin_idx] += p_z_

        flat_vol = binned_vol.flat
        flat_one_minus_sp = one_minus_survival_prob.flat
        flat_sp_by_vol = binned_sp_by_vol.flat
        for idx, mask in enumerate(vol_mask.flat):
            if mask == 1:
                flat_one_minus_sp[idx] *= 1 - flat_sp_by_vol[idx] / flat_vol[idx]

    end_setup_time = time.time()
    print('Setup time: %.3f sec' % (end_setup_time - start_time))

    doms_used = []
    binning_hash = None

    for table_kind in ['ic', 'dc']:
        det_start_time = time.time()
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
            det_depth_start_time = time.time()
            #if test and dom_depth_idx not in [28, 29, 30]:
            #if test and dom_depth_idx not in [48, 49, 50]:
            if test and dom_depth_idx not in [45]:
                continue
            print('table_kind: %s, dom_depth_idx: %s'
                  % (table_kind, dom_depth_idx))

            # Get the (x, y, z) coords for all DOMs of this type and at this
            # depth index
            subdet_depth_dom_coords = detector_geometry[strings_slice, dom_depth_idx, :]

            # Load the 4D table for the DOM type / depth index
            fpath = expand(table_fpath_proto.format(tables_dir=tables_dir, dom=dom_depth_idx))
            photon_info, bin_edges = extract_photon_info(fpath=fpath, dom_depth_index=dom_depth_idx)
            #print('bin_edges:', bin_edges)

            p_survival_prob = photon_info.survival_prob[dom_depth_idx].astype(CALC_FTYPE)
            p_theta = photon_info.theta[dom_depth_idx].astype(CALC_FTYPE)
            p_deltaphi = photon_info.deltaphi[dom_depth_idx].astype(CALC_FTYPE)
            p_length = photon_info.length[dom_depth_idx].astype(CALC_FTYPE)
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
            #p_length *= np.abs(np.cos(p_deltaphi))
            p_deltaphi = np.zeros_like(p_deltaphi)

            orig_p_info_shape = p_survival_prob.shape
            t_indep_p_info_shape = tuple([nphi] + [orig_p_info_shape[d] for d in range(3) if d != RETRO_T_IDX])

            # Marginalize out time for photon survival probabilities; this
            # implements, effectively, the logic
            #   ``Prob(t0 or t1 or ... or tN)``
            t_slice = [slice(None) for _ in orig_p_info_shape]
            t_slice[RETRO_T_IDX] = slice(t_start, t_stop)
            t_indep_p_survival_prob = 1 - (1 - p_survival_prob[t_slice]).prod(axis=RETRO_T_IDX)

            # Marginalize out time for (p_length, p_theta) by converting to
            # Cartesian coordinates and performing an average weighted by
            # survival probability

            p_z, p_rho = np.empty_like(p_length), np.empty_like(p_length)
            pol2cart(r=p_length, theta=p_theta, x=p_z, y=p_rho)
            print('p_length range:', p_length.min(), p_length.max())
            print('p_theta range:', p_theta.min(), p_theta.max())
            print('p_z range:', p_z.min(), p_z.max())
            print('p_rho range:', p_rho.min(), p_rho.max())

            mask = t_indep_p_survival_prob != 0
            scale = 1 / t_indep_p_survival_prob[mask]

            t_indep_p_z = np.zeros_like(t_indep_p_survival_prob)
            t_indep_p_rho = np.zeros_like(t_indep_p_survival_prob)

            t_indep_p_z[mask] = (p_z[t_slice] * p_survival_prob[t_slice]).sum(axis=RETRO_T_IDX)[mask] * scale
            t_indep_p_rho[mask] = (p_rho[t_slice] * p_survival_prob[t_slice]).sum(axis=RETRO_T_IDX)[mask] * scale

            # Broadcast the survival probability and Cartesian components of
            # the average photon # (binned in plane-polar coordinates) to
            # spherical-polar coordinates, computing the x- and y-components
            t_indep_p_survival_prob = np.broadcast_to(t_indep_p_survival_prob, t_indep_p_info_shape)
            t_indep_p_z = np.broadcast_to(t_indep_p_z, t_indep_p_info_shape)
            t_indep_p_x = np.empty(t_indep_p_info_shape, dtype=CALC_FTYPE)
            t_indep_p_y = np.empty(t_indep_p_info_shape, dtype=CALC_FTYPE)
            for phi_idx, phi in enumerate(phi_centers):
                t_indep_p_x[phi_idx, ...] = t_indep_p_rho * np.cos(phi)
                t_indep_p_y[phi_idx, ...] = t_indep_p_rho * np.sin(phi)

            if test:
                print('t_indep_p_survival_prob.shape:', t_indep_p_survival_prob.shape)
                print('t_indep_p_survival_prob.dtype:', t_indep_p_survival_prob.dtype)
                print('t_indep_p_survival_prob range:', t_indep_p_survival_prob.min(), t_indep_p_survival_prob.max())
                print('t_indep_p_x.shape:', t_indep_p_x.shape)
                print('t_indep_p_x range:', t_indep_p_x.min(), t_indep_p_x.max())
                print('t_indep_p_y range:', t_indep_p_y.min(), t_indep_p_y.max())
                print('t_indep_p_z range:', t_indep_p_z.min(), t_indep_p_z.max())
                print('t_indep_p_rho range:', t_indep_p_rho.min(), t_indep_p_rho.max())

            new_binning_hash = 1 #hash(bin_edges)
            if new_binning_hash != binning_hash:
                binning_hash = new_binning_hash

                phi_edges_mg, r_edges_mg, theta_edges_mg = np.meshgrid(
                    phi_edges, bin_edges.r, bin_edges.theta,
                    indexing='ij'
                )

                phi_widths = np.abs(np.diff(phi_edges))
                r_widths = np.abs(np.diff(bin_edges.r))
                costheta_widths = np.abs(np.diff(np.cos(bin_edges.theta)))

                phi_widths_mg, r_widths_mg, costheta_widths_mg = np.meshgrid(phi_widths, r_widths, costheta_widths, indexing='ij')

                bin_vols = spherical_volume(dr=r_widths_mg, dcostheta=costheta_widths_mg, dphi=phi_widths_mg)

                n_rbins = len(bin_edges.r) - 1
                n_thetabins = len(bin_edges.theta) - 1

                x_edges_mg = np.empty_like(r_edges_mg)
                y_edges_mg = np.empty_like(r_edges_mg)
                z_edges_mg = np.empty_like(r_edges_mg)
                sph2cart(r=r_edges_mg, theta=theta_edges_mg, phi=phi_edges_mg,
                         x=x_edges_mg, y=y_edges_mg, z=z_edges_mg)

            # Loop through all of the DOMs in this subdetector index, shifting
            # the coordinates and aggregating the expected survival
            # probabilities  and average photon info in the Cartesian grid
            for str_idx, string_dom_xyz in enumerate(subdet_depth_dom_coords):
                #if test:
                #    if table_kind == 'ic' and str_idx not in [25, 26, 34, 35, 36, 44, 45]:
                if test and str_idx not in [35]:
                        continue
                det_depth_string_start_time = time.time()
                print('table_kind: %s, dom_depth_idx: %s, str_idx: %s'
                      % (table_kind, dom_depth_idx, str_idx))

                string_x, string_y, dom_z = string_dom_xyz
                doms_used.append(string_dom_xyz)

                bin_quantities(
                    r_edges_mg=r_edges_mg, theta_edges_mg=theta_edges_mg,
                    phi_edges_mg=phi_edges_mg,
                    x_edges_mg=x_edges_mg, y_edges_mg=y_edges_mg,
                    z_edges_mg=z_edges_mg,
                    x_offset=string_x, y_offset=string_y, z_offset=dom_z,
                    bin_vols=bin_vols,
                    survival_prob=t_indep_p_survival_prob,
                    p_x=t_indep_p_x, p_y=t_indep_p_y, p_z=t_indep_p_z,
                    binned_p_x=avg_photon_x,
                    binned_p_y=avg_photon_y,
                    binned_p_z=avg_photon_z,
                    binned_sp_by_vol=binned_p_survival_prob_by_vol,
                    one_minus_survival_prob=one_minus_survival_prob
                )

                print('time for det/depth/string (innermost) loop: %.3f sec'
                      % (time.time() - det_depth_string_start_time))
            print('time for det/depth loop: %.3f sec'
                  % (time.time() - det_depth_start_time))
        print('time for det loop: %.3f sec'
              % (time.time() - det_start_time))

    end_loop_time = time.time()
    print('looping took a total of %.3f sec' % (end_loop_time - end_setup_time))

    doms_used = np.array(doms_used)

    if test:
        print('binned_p_survival_prob_by_vol range:',
              binned_p_survival_prob_by_vol.min(),
              binned_p_survival_prob_by_vol.max())
        print('one_minus_survival_prob range:',
              one_minus_survival_prob.min(),
              one_minus_survival_prob.max())
        print('')

    mask = binned_p_survival_prob_by_vol != 0
    scale = 1 / binned_p_survival_prob_by_vol[mask]
    avg_photon_x[mask] *= scale
    avg_photon_y[mask] *= scale
    avg_photon_z[mask] *= scale

    survival_prob = 1 - one_minus_survival_prob

    #if smooth:
    #    gaussian_filter(survival_prob,
    arrays_names = [
        (survival_prob, 'survival_prob'),
        (avg_photon_x, 'avg_photon_x'),
        (avg_photon_y, 'avg_photon_y'),
        (avg_photon_z, 'avg_photon_z')
    ]
    test_str = '_test' if test else ''
    fbasename = ('time_dom_indep_cart_table_%dx%dx%d_nphi%d%s'
                 % (nx, ny, nz, nphi, test_str))
    for array, name in arrays_names:
        fname = '%s_%s.fits' % (fbasename, name)
        fpath = join(tables_dir, fname)
        hdulist = pyfits.HDUList([
            pyfits.PrimaryHDU(array.astype(TABLE_FTYPE)),
            pyfits.ImageHDU(xyz_shape),
            pyfits.ImageHDU(np.array([xlims, ylims, zlims])),
            pyfits.ImageHDU(doms_used)
        ])
        hdulist.writeto(fpath, clobber=True)

    end_save_time = time.time()
    print('saving to disk took %.3f sec' % (end_save_time - end_loop_time))


if __name__ == '__main__':
    generate_time_and_dom_indep_tables(**vars(parse_args()))
