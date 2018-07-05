# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, too-many-nested-blocks, line-too-long, too-many-locals

"""
Convert raw Retro 5D tables (which represent survival probabilities for light
traveling in a particular direction) to tables for Cherenkov emitters with a
particular direction.

Output tables will be in .npy-files-in-a-directory format for easy memory
mapping.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'get_cone_map',
    'convolve_table',
    'survival_prob_from_smeared_cone',
    'survival_prob_from_cone'
]

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

from os.path import abspath, dirname
import sys
import math

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS
from retro.const import SPEED_OF_LIGHT_M_PER_NS


FLOAT_T = np.float32
PI = FLOAT_T(np.pi)
TWO_PI = FLOAT_T(2*np.pi)


# NOTE: dithering the ckv angle appears to do non-representative things to the
# resulting table. Smearing can be done on resulting table if that makes more
# sense.

#@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
#def get_dithered_cone_map(
#        costheta, sintheta, num_phi, axis_costheta, axis_sintheta, axis_cosphi,
#        axis_sinphi, num_costheta_bins, num_deltaphi_bins
#    ):
#    """Get the bin indices and weights for sampling from a Cherenkov cone (or
#    cones) in the binned (costhetadir, deltaphidir) space (the actual sampling
#    is left for a higher-level function to perform).
#
#    Parameters
#    ----------
#    costheta, sintheta : scalar float
#        Cosine and sine of Cherenkov angle (half the cone's opening angle)
#
#    num_phi : scalar int
#        Number of azimuth samples of the circle where the cone intersects the
#        unit sphere. Increase `num_phi` for for higher accuracy.
#
#    axis_costheta, axis_sintheta, axis_cosphi, axis_sinphi : array-like, (n_axes,)
#        Rotate the cone to have axis of symmetry defined by (axis_theta, axis_phi)
#
#    directional_survival_prob : ndarray of shape (N_costhetadir x N_deltaphidir)
#        Note that the binning of the `directional_survival_prob` table slice is
#        expected to be (costhetadir, deltaphidir), and both are assumed to be
#        uniformly gridded in those coordinate spaces.
#
#    num_costheta_bins, num_deltaphi_bins : int
#        Number of bins in costheta and deltaphi dimensions. cosetheta is
#        assumed to be binned from -1 to 1, inclusive, and deltaphi is assumed
#        to be binned from 0 to pi, inclusive.
#
#    Returns
#    -------
#    bin_indices : list of 2-tuples
#
#    weights : array of floats, same len as `bin_indices`
#
#    """
#    costheta_bin_width = 2 / FLOAT_T(num_costheta_bins)
#    deltaphi_bin_width = PI / FLOAT_T(num_deltaphi_bins)
#
#    last_costheta_bin = num_costheta_bins - 1
#    last_deltaphi_bin = num_deltaphi_bins - 1
#
#    bin_indices = []
#    counts = []
#    counts_total = 0
#
#    phi_step = TWO_PI / FLOAT_T(num_phi)
#
#    for phi_idx in range(num_phi):
#        p_phi = phi_idx * phi_step
#        sin_p_phi = math.sin(p_phi)
#        cos_p_phi = math.cos(p_phi)
#
#        dith_ct = costheta[phi_idx]
#        dith_st = sintheta[phi_idx]
#
#        for ax_ct, ax_st, ax_cp, ax_sp in zip(axis_costheta.flat, axis_sintheta.flat,
#                                              axis_cosphi.flat, axis_sinphi.flat):
#            counts_total += 1
#
#            q_costheta = (-dith_st * ax_st * cos_p_phi) + (dith_ct * ax_ct)
#
#            abs_q_phi = abs(math.atan2(
#                (sin_p_phi * dith_st * ax_cp) + (dith_st * ax_sp * cos_p_phi * ax_ct) + (ax_sp * ax_st * dith_ct),
#                (-sin_p_phi * dith_st * ax_sp) + (dith_st * cos_p_phi * ax_cp * ax_ct) + (ax_st * dith_ct * ax_cp)
#            ))
#
#            costheta_bin = int((q_costheta + 1) // costheta_bin_width)
#            if costheta_bin > last_costheta_bin:
#                costheta_bin = last_costheta_bin
#
#            deltaphi_bin = int(abs_q_phi // deltaphi_bin_width)
#            if deltaphi_bin > last_deltaphi_bin:
#                deltaphi_bin = last_deltaphi_bin
#
#            coord = (costheta_bin, deltaphi_bin)
#
#            if coord in bin_indices:
#                counts[bin_indices.index(coord)] += 1
#            else:
#                bin_indices.append(coord)
#                counts.append(1)
#
#    cnt_tot = np.float64(counts_total)
#    weights = np.array([np.float64(c) / cnt_tot for c in counts], dtype=FLOAT_T)
#
#    return bin_indices, weights


@numba_jit(nopython=True, parallel=False, nogil=True, cache=True)
def get_cone_map(
        costheta, sintheta, num_phi, axis_costheta, axis_sintheta, axis_cosphi,
        axis_sinphi, num_costheta_bins, num_deltaphi_bins
    ):
    """Get the bin indices and weights for sampling from a Cherenkov cone (or
    cones) in the binned (costhetadir, deltaphidir) space (the actual sampling
    is left for a higher-level function to perform).

    Parameters
    ----------
    costheta, sintheta : scalar float
        Cosine and sine of Cherenkov angle (half the cone's opening angle)

    num_phi : scalar int
        Number of azimuth samples of the circle where the cone intersects the
        unit sphere. Increase `num_phi` for for higher accuracy.

    axis_costheta, axis_sintheta, axis_cosphi, axis_sinphi : array-like, (n_axes,)
        Rotate the cone to have axis of symmetry defined by (axis_theta, axis_phi)

    directional_survival_prob : ndarray of shape (N_costhetadir x N_deltaphidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, deltaphidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_deltaphi_bins : int
        Number of bins in costheta and deltaphi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and deltaphi is assumed
        to be binned from 0 to pi, inclusive.

    Returns
    -------
    bin_indices : list of 2-tuples

    weights : array of floats, same len as `bin_indices`

    """
    costheta_bin_width = 2 / FLOAT_T(num_costheta_bins)
    deltaphi_bin_width = PI / FLOAT_T(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    bin_indices = []
    counts = []
    counts_total = 0

    phi_step = TWO_PI / FLOAT_T(num_phi)

    assert axis_costheta.shape == axis_sintheta.shape == axis_cosphi.shape == axis_sinphi.shape

    for phi_idx in range(num_phi):
        p_phi = phi_idx * phi_step
        sin_p_phi = math.sin(p_phi)
        cos_p_phi = math.cos(p_phi)

        for ax_ct, ax_st, ax_cp, ax_sp in zip(np.nditer(axis_costheta), np.nditer(axis_sintheta),
                                              np.nditer(axis_cosphi), np.nditer(axis_sinphi)):
            counts_total += 1

            q_costheta = (-sintheta * ax_st * cos_p_phi) + (costheta * ax_ct)

            abs_q_phi = abs(math.atan2(
                (sin_p_phi * sintheta * ax_cp) + (sintheta * ax_sp * cos_p_phi * ax_ct) + (ax_sp * ax_st * costheta),
                (-sin_p_phi * sintheta * ax_sp) + (sintheta * cos_p_phi * ax_cp * ax_ct) + (ax_st * costheta * ax_cp)
            ))

            costheta_bin = int((q_costheta + 1) // costheta_bin_width)
            if costheta_bin > last_costheta_bin:
                costheta_bin = last_costheta_bin

            deltaphi_bin = int(abs_q_phi // deltaphi_bin_width)
            if deltaphi_bin > last_deltaphi_bin:
                deltaphi_bin = last_deltaphi_bin

            coord = (costheta_bin, deltaphi_bin)

            found = False
            for idx, crd in enumerate(bin_indices):
                if coord == crd:
                    counts[idx] += 1
                    found = True
                    break
            if not found:
                bin_indices.insert(0, coord)
                counts.insert(0, 1)

    cnt_tot = np.float64(counts_total)
    weights = np.array([np.float64(c) / cnt_tot for c in counts], dtype=FLOAT_T)
    costheta_indices = np.array([i[0] for i in bin_indices], dtype=np.uint32)
    deltaphi_indices = np.array([i[1] for i in bin_indices], dtype=np.uint32)

    return costheta_indices, deltaphi_indices, weights


@numba_jit(nopython=True, parallel=False, nogil=True, cache=True)
def convolve_table(
        src, dst, cos_ckv, sin_ckv, r_bin_edges, ct_bin_edges, t_bin_edges,
        t_is_dt, ctdir_bin_edges, dpdir_bin_edges, num_cone_samples,
        oversample, n_group
    ):
    """
    Parameters
    ----------
    src : (n_r, n_ct, n_t, n_ctdir, n_dpdir) array

    dst : (n_r, n_ct, n_t, n_ctdir, n_dpdir) array

    cos_ckv, sin_ckv : float

    r_bin_edges
        Radial bin edges, in units of meters.

    ct_bin_edges
        Cosine of theta (zenith angle) bin edges.

    t_bin_edges
        Time bin edges, units of nanoseconds.

    t_is_dt : bool
        Whether time bins represent time residuals (True) or absolute time
        (False).

    ctdir_bin_edges : array
        Cosine-of-direction-theta (zenith angle) bin edges.

    dpdir_bin_edges : array
        Delta-phi (azimuth angle) bin edges, in units of radians.

    num_cone_samples : int > 0

    oversample : int > 0

    n_group : float > 0
        Group refractive index in the medium (use lowest value used for all ice
        simulated).

    """
    n_t = len(t_bin_edges) - 1
    n_ct = len(ct_bin_edges) - 1
    n_ctdir = len(ctdir_bin_edges) - 1
    n_dpdir = len(dpdir_bin_edges) - 1

    ctdir_min = ctdir_bin_edges[0]
    ctdir_max = ctdir_bin_edges[-1]

    dpdir_min = dpdir_bin_edges[0]
    dpdir_max = dpdir_bin_edges[-1]

    ctdir_bw = (ctdir_max - ctdir_min) / n_ctdir
    dpdir_bw = (dpdir_max - dpdir_min) / n_dpdir

    ctdir_samp_step = ctdir_bw / oversample
    dpdir_samp_step = dpdir_bw / oversample

    ctdir_min_samp = ctdir_min + 0.5 * ctdir_samp_step
    dpdir_min_samp = dpdir_min + 0.5 * dpdir_samp_step

    samples_shape = (oversample, oversample)

    # Cosine and sine of thetadir
    ctd_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)
    std_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)

    # Cosine and sine of deltaphidir
    cdpd_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)
    sdpd_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)

    # Max distance from the DOM light could be for each time bin
    if t_is_dt:
        causal = True
    if not t_is_dt:
        tbin_max_dist = np.array(
            [t*SPEED_OF_LIGHT_M_PER_NS/n_group for t in np.nditer(t_bin_edges[1:])],
            dtype=FLOAT_T
        )

    for ctdir_idx in range(n_ctdir):
        ctd0 = ctdir_min_samp + ctdir_idx*ctdir_bw

        for dpdir_idx in range(n_dpdir):
            dpd0 = dpdir_min_samp + dpdir_idx*dpdir_bw

            for ctdir_subidx in range(oversample):
                ctd_samp = ctd0 + ctdir_subidx * ctdir_samp_step
                std_samp = math.sqrt(1 - ctd_samp*ctd_samp)

                for dpdir_subidx in range(oversample):
                    dpd_samp = dpd0 + dpdir_subidx * dpdir_samp_step
                    cdpd_samp = math.cos(dpd_samp)
                    sdpd_samp = math.sqrt(1 - cdpd_samp*cdpd_samp)

                    ctd_samples[ctdir_subidx, dpdir_subidx] = ctd_samp
                    std_samples[ctdir_subidx, dpdir_subidx] = std_samp
                    cdpd_samples[ctdir_subidx, dpdir_subidx] = cdpd_samp
                    sdpd_samples[ctdir_subidx, dpdir_subidx] = sdpd_samp

            ctd_idxs, dpd_idxs, weights = get_cone_map(
                costheta=cos_ckv,
                sintheta=sin_ckv,
                num_phi=num_cone_samples,
                axis_costheta=ctd_samples,
                axis_sintheta=std_samples,
                axis_cosphi=cdpd_samples,
                axis_sinphi=sdpd_samples,
                num_costheta_bins=n_ctdir,
                num_deltaphi_bins=n_dpdir
            )
            num_idxs = len(ctd_idx)
            assert len(dpd_idxs) == len(weights) == num_idxs

            for r_idx, r_lower in enumerate(np.nditer(r_bin_edges[:-1])):
                for t_idx in range(n_t):
                    if not t_is_dt:
                        max_dist = tbin_max_dist[t_idx]
                        causal = r_lower <= max_dist
                    for ct_idx in range(n_ct):
                        avg = 0.0
                        if causal:
                            # Apply the weights to the corresponding entries
                            # (note that weights account for normalization)
                            for i_idx in range(num_idxs):
                                ctd_idx = ctd_idxs[i_idx]
                                dpd_idx = dpd_idxs[i_idx]
                                weight = weights[i_idx]
                                avg += weight * src[r_idx, ct_idx, t_idx, ctd_idx, dpd_idx]
                            #avg = np.sum(
                            #    weights *
                            #    src[r_idx, ct_idx, t_idx, ctd_idxs, dpd_idxs]
                            #)

                        dst[r_idx, ct_idx, t_idx, ctdir_idx, dpdir_idx] = avg


@numba_jit(parallel=False, nogil=False, cache=True) #**DFLT_NUMBA_JIT_KWARGS)
def survival_prob_from_smeared_cone(
        theta, num_phi, rot_costheta, rot_sintheta, rot_cosphi, rot_sinphi,
        directional_survival_prob, num_costheta_bins, num_deltaphi_bins,
        random_delta_thetas
    ):
    """Get a numerical approximation of the expected survival probability for
    photons directed on a cone (as for Cherenkov emission) from Retro table's
    photon-directionality slice.

    Parameters
    ----------
    theta : scalar float
        Cherenkov angle (half the cone's opening angle)

    num_phi : scalar int
        Number of azimuth samples of the circle where the cone intersects the
        unit sphere. Increase `num_phi` for for higher accuracy.

    rot_costheta, rot_sintheta, rot_cosphi, rot_sinphi : scalar float
        Rotate the cone to have axis of symmetry defined by (rot_theta, rot_phi)

    directional_survival_prob : ndarray of shape (N_costhetadir x N_deltaphidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, deltaphidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_deltaphi_bins : int
        Number of bins in costheta and deltaphi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and deltaphi is assumed
        to be binned from 0 to pi, inclusive.

    random_delta_thetas : sequence of length >= num_phi
        Offsets to apply to theta to achieve smearing.

    Returns
    -------
    survival_prob : scalar float
        Numerically-approximated survival probability averaged over the entire
        cone of photons.

    bin_indices

    counts

    """
    # TODO: we can approximate the effects of "large" (space-time width of a
    # bin) by spreading points out off of the circle to simulate the amount of
    # spread expected for photons within the bin. Probably a high-order effect,
    # so no need to do this now, just something to note for later.

    bin_indices = []
    counts = []
    counts_total = 0
    num_indices = 0

    costheta_bin_width = 2 / FLOAT_T(num_costheta_bins)
    deltaphi_bin_width = PI / FLOAT_T(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    for phi_idx in range(num_phi):
        offset_theta = theta + random_delta_thetas[phi_idx]
        costheta = math.cos(offset_theta)
        sintheta = math.sin(offset_theta)

        p_phi = TWO_PI * FLOAT_T(phi_idx) / FLOAT_T(num_phi)

        sin_p_phi = math.sin(p_phi)
        cos_p_phi = math.cos(p_phi)

        q_costheta = ((-sintheta * rot_sintheta * cos_p_phi) + (costheta * rot_costheta))
        abs_q_phi = math.fabs(math.atan2(
            (sin_p_phi * sintheta * rot_cosphi) + (sintheta * rot_sinphi * cos_p_phi * rot_costheta) + (rot_sinphi * rot_sintheta * costheta),
            (-sin_p_phi * sintheta * rot_sinphi) + (sintheta * cos_p_phi * rot_cosphi * rot_costheta) + (rot_sintheta * costheta * rot_cosphi)
        ))

        costheta_bin = int((q_costheta + 1) // costheta_bin_width)
        if costheta_bin > last_costheta_bin:
            costheta_bin = last_costheta_bin

        deltaphi_bin = int(abs_q_phi // deltaphi_bin_width)
        if deltaphi_bin > last_deltaphi_bin:
            deltaphi_bin = last_deltaphi_bin

        coord = (costheta_bin, deltaphi_bin)
        if coord in bin_indices:
            counts[bin_indices.index(coord)] += 1
        else:
            bin_indices.append(coord)
            counts.append(1)
            num_indices += 1
        counts_total += 1

    survival_prob = 0.0
    for i in range(num_indices):
        survival_prob += directional_survival_prob[bin_indices[i]] * counts[i]

    # NOTE: Don't use e.g. /= before return until the following is resolved:
    #   https://github.com/numba/numba/issues/2746
    survival_prob = survival_prob / FLOAT_T(counts_total)

    return survival_prob, bin_indices, counts


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def survival_prob_from_cone(
        costheta, sintheta, num_phi, rot_costheta, rot_sintheta, rot_cosphi,
        rot_sinphi, directional_survival_prob, num_costheta_bins,
        num_deltaphi_bins
    ):
    """Get a numerical approximation of the expected survival probability for
    photons directed on a cone (as for Cherenkov emission) from Retro table's
    photon-directionality slice.

    Parameters
    ----------
    costheta, sintheta : scalar float
        Cherenkov angle (half the cone's opening angle)

    num_phi : scalar int
        Number of azimuth samples of the circle where the cone intersects the
        unit sphere. Increase `num_phi` for for higher accuracy.

    rot_costheta, rot_sintheta, rot_cosphi, rot_sinphi : scalar float
        Rotate the cone to have axis of symmetry defined by (rot_theta, rot_phi)

    directional_survival_prob : ndarray of shape (N_costhetadir x N_deltaphidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, deltaphidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_deltaphi_bins : int
        Number of bins in costheta and deltaphi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and deltaphi is assumed
        to be binned from 0 to pi, inclusive.

    Returns
    -------
    survival_prob : scalar float
        Numerically-approximated survival probability averaged over the entire
        cone of photons.

    bin_indices

    counts

    """
    # TODO: we can approximate the effects of "large" (space-time width of a
    # bin) by spreading points out off of the circle to simulate the amount of
    # spread expected for photons within the bin. Probably a high-order effect,
    # so no need to do this now, just something to note for later.

    bin_indices = []
    counts = []
    counts_total = 0
    num_indices = 0

    costheta_bin_width = 2.0 / FLOAT_T(num_costheta_bins)
    deltaphi_bin_width = PI / FLOAT_T(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    for phi_idx in range(num_phi):
        p_phi = TWO_PI * FLOAT_T(phi_idx) / FLOAT_T(num_phi)
        sin_p_phi = np.sin(p_phi)
        cos_p_phi = np.cos(p_phi)
        q_costheta = ((-sintheta * rot_sintheta * cos_p_phi)
                      + (costheta * rot_costheta))
        abs_q_phi = np.abs(math.atan2(
            (sin_p_phi * sintheta * rot_cosphi)
            + (sintheta * rot_sinphi * cos_p_phi * rot_costheta)
            + (rot_sinphi * rot_sintheta * costheta),

            (-sin_p_phi * sintheta * rot_sinphi)
            + (sintheta * cos_p_phi * rot_cosphi * rot_costheta)
            + (rot_sintheta * costheta * rot_cosphi)
        ))

        costheta_bin = int((q_costheta + 1) // costheta_bin_width)
        if costheta_bin > last_costheta_bin:
            costheta_bin = last_costheta_bin

        deltaphi_bin = int(abs_q_phi // deltaphi_bin_width)
        if deltaphi_bin > last_deltaphi_bin:
            deltaphi_bin = last_deltaphi_bin

        coord = (costheta_bin, deltaphi_bin)
        if coord in bin_indices:
            counts[bin_indices.index(coord)] += 1
        else:
            bin_indices.append(coord)
            counts.append(1)
            num_indices += 1
        counts_total += 1

    survival_prob = 0.0
    for i in range(num_indices):
        survival_prob += directional_survival_prob[bin_indices[i]] * counts[i]

    # NOTE: Don't use e.g. /= before return until the following is resolved:
    #   https://github.com/numba/numba/issues/2746
    survival_prob = survival_prob / FLOAT_T(counts_total)

    return survival_prob, bin_indices, counts
