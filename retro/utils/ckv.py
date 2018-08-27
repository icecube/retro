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
    'convolve_dirmap',
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


FLOAT_T = np.float64
PI = FLOAT_T(np.pi)
TWO_PI = FLOAT_T(2*np.pi)


# NOTE: dithering the ckv angle appears to do non-representative things to the
# resulting table. Smearing can be done on resulting table if that makes more
# sense.

#@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
#def get_dithered_cone_map(
#        ckv_costheta, ckv_sintheta, num_phi, axis_costheta, axis_sintheta, axis_cosphi,
#        axis_sinphi, num_costheta_bins, num_phi_bins
#    ):
#    """Get the bin indices and weights for sampling from a Cherenkov cone (or
#    cones) in the binned (costhetadir, phidir) space (the actual sampling
#    is left for a higher-level function to perform).
#
#    Parameters
#    ----------
#    ckv_costheta, ckv_sintheta : scalar float
#        Cosine and sine of Cherenkov angle (half the cone's opening angle)
#
#    num_phi : scalar int
#        Number of azimuth samples of the circle where the cone intersects the
#        unit sphere. Increase `num_phi` for for higher accuracy.
#
#    axis_costheta, axis_sintheta, axis_cosphi, axis_sinphi : array-like, (n_axes,)
#        Rotate the cone to have axis of symmetry defined by (axis_theta, axis_phi)
#
#    directional_survival_prob : ndarray of shape (N_costhetadir x N_phidir)
#        Note that the binning of the `directional_survival_prob` table slice is
#        expected to be (costhetadir, phidir), and both are assumed to be
#        uniformly gridded in those coordinate spaces.
#
#    num_costheta_bins, num_phi_bins : int
#        Number of bins in costheta and phi dimensions. cosetheta is
#        assumed to be binned from -1 to 1, inclusive, and phi is assumed
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
#    phi_bin_width = PI / FLOAT_T(num_phi_bins)
#
#    last_costheta_bin = num_costheta_bins - 1
#    last_phi_bin = num_phi_bins - 1
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
#        dith_ct = ckv_costheta[phi_idx]
#        dith_st = ckv_sintheta[phi_idx]
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
#            phi_bin = int(abs_q_phi // phi_bin_width)
#            if phi_bin > last_phi_bin:
#                phi_bin = last_phi_bin
#
#            coord = (costheta_bin, phi_bin)
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
    ckv_costheta,
    ckv_sintheta,
    num_phi,
    axis_costheta,
    axis_sintheta,
    axis_cosphi,
    axis_sinphi,
    num_costheta_bins,
    num_phi_bins,
    costheta_min,
    costheta_max,
    phi_min,
    phi_max,
):
    """Get the bin indices and weights for sampling from a Cherenkov cone (or
    cones) in the binned (costhetadir, phidir) space (the actual sampling
    is left for a higher-level function to perform).

    Parameters
    ----------
    ckv_costheta, ckv_sintheta : scalar float
        Cosine and sine of Cherenkov angle (half the cone's opening angle)

    num_phi : scalar int
        Number of azimuth samples of the circle where the cone intersects the
        unit sphere. Increase `num_phi` for for higher accuracy.

    axis_costheta, axis_sintheta, axis_cosphi, axis_sinphi : array-like, (n_axes,)
        Rotate the cone to have axis of symmetry defined by (axis_theta, axis_phi)

    directional_survival_prob : ndarray of shape (N_costhetadir x N_phidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, phidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_phi_bins : int
        Number of bins in costheta and phi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and phi is assumed
        to be binned from 0 to pi, inclusive.

    costheta_min, costheta_max, phi_min, phi_max : float
        Limits of binning in phi and costheta dimensinos

    Returns
    -------
    bin_indices : list of 2-tuples

    weights : array of floats, same len as `bin_indices`

    """
    recip_ctbw = num_costheta_bins / (costheta_max - costheta_min)
    recip_phibw = num_phi_bins / (phi_max - phi_min)

    last_costheta_bin = num_costheta_bins - 1
    last_phi_bin = num_phi_bins - 1

    bin_indices = []
    counts = []
    counts_total = 0

    phi_step = TWO_PI / num_phi

    abs_phidir = phi_min == 0 and phi_max == PI

    assert axis_costheta.shape == axis_sintheta.shape == axis_cosphi.shape == axis_sinphi.shape

    for phi_idx in range(num_phi):
        p_phi = phi_idx * phi_step
        sin_p_phi = math.sin(p_phi)
        cos_p_phi = math.cos(p_phi)

        for ax_ct, ax_st, ax_cp, ax_sp in zip(np.nditer(axis_costheta), np.nditer(axis_sintheta),
                                              np.nditer(axis_cosphi), np.nditer(axis_sinphi)):
            counts_total += 1

            q_costheta = (-ckv_sintheta * ax_st * cos_p_phi) + (ckv_costheta * ax_ct)

            q_phi = math.atan2(
                (sin_p_phi * ckv_sintheta * ax_cp) + (ckv_sintheta * ax_sp * cos_p_phi * ax_ct) + (ax_sp * ax_st * ckv_costheta),
                (-sin_p_phi * ckv_sintheta * ax_sp) + (ckv_sintheta * cos_p_phi * ax_cp * ax_ct) + (ax_st * ckv_costheta * ax_cp)
            )
            if abs_phidir:
                q_phi = abs(q_phi)

            ct_bin = min(max(0, int((q_costheta - costheta_min) * recip_ctbw)), last_costheta_bin)
            phi_bin = min(max(0, int((q_phi - phi_min) * recip_phibw)), last_phi_bin)

            coord = (ct_bin, phi_bin)

            found = False
            for idx, crd in enumerate(bin_indices):
                if coord == crd:
                    counts[idx] += 1
                    found = True
                    break
            if not found:
                bin_indices.insert(0, coord)
                counts.insert(0, 1)

    weights = np.array([np.float64(c) / np.float64(counts_total) for c in counts])
    costheta_indices = np.array([i[0] for i in bin_indices], dtype=np.uint32)
    phi_indices = np.array([i[1] for i in bin_indices], dtype=np.uint32)

    return costheta_indices, phi_indices, weights


@numba_jit(nopython=True, parallel=False, nogil=True, cache=True)
def convolve_dirmap(
    dirmap,
    out,
    cos_ckv,
    sin_ckv,
    costhetadir_bin_edges,
    phidir_bin_edges,
    num_cone_samples,
    oversample,
):
    """Convolve an array of values in a grid of (costhetadir, phidir)
    coordinates (a so-called directional map, or dirmap) with a Cherenkov cone.

    Parameters
    ----------
    dirmap : (n_costhetadir, n_phidir) array
        Directionality map to be convolved with a Cherenkov cone

    out : (n_costhetadir, n_phidir) array
        The result of the convolution is populated to this array

    cos_ckv, sin_ckv : float
        Cosine and sine of Cherenkov angle

    costhetadir_bin_edges : shape (n_costhetadir + 1,) array
        Cosine-of-direction-theta (zenith angle) bin edges.

    phidir_bin_edges : shape (n_phidir + 1,) array
        Direction-phi (azimuth angle) bin edges, in units of radians.

    num_cone_samples : int > 0
        Number of samples around cone (akin to antialiasing; more samples give
        more accurate results)

    oversample : int > 0
        Sample this many times in each dimension in each (costhetadir, phidir)
        bin

    """
    n_costhetadir = len(costhetadir_bin_edges) - 1
    n_phidir = len(phidir_bin_edges) - 1

    costhetadir_min = costhetadir_bin_edges[0]
    costhetadir_max = costhetadir_bin_edges[-1]

    phidir_min = phidir_bin_edges[0]
    phidir_max = phidir_bin_edges[-1]

    costhetadir_bw = (costhetadir_max - costhetadir_min) / n_costhetadir
    phidir_bw = (phidir_max - phidir_min) / n_phidir

    costhetadir_samp_step = costhetadir_bw / oversample
    phidir_samp_step = phidir_bw / oversample

    costhetadir_min_samp = costhetadir_min + 0.5 * costhetadir_samp_step
    phidir_min_samp = phidir_min + 0.5 * phidir_samp_step

    # Cosine and sine of thetadir
    costhetadir_samples = np.empty(shape=(oversample, oversample), dtype=FLOAT_T)
    sinthetadir_samples = np.empty(shape=(oversample, oversample), dtype=FLOAT_T)

    # Cosine and sine of phidir
    cosphidir_samples = np.empty(shape=(oversample, oversample), dtype=FLOAT_T)
    sinphidir_samples = np.empty(shape=(oversample, oversample), dtype=FLOAT_T)

    for costhetadir_idx in range(n_costhetadir):
        costhetadir0 = costhetadir_min_samp + costhetadir_idx*costhetadir_bw

        for phidir_idx in range(n_phidir):
            phidir0 = phidir_min_samp + phidir_idx*phidir_bw

            for costhetadir_subidx in range(oversample):
                costhetadir_samp = costhetadir0 + costhetadir_subidx * costhetadir_samp_step
                sinthetadir_samp = math.sqrt(1 - costhetadir_samp**2)

                for phidir_subidx in range(oversample):
                    phidir_samp = phidir0 + phidir_subidx * phidir_samp_step
                    cosphidir_samp = math.cos(phidir_samp)
                    sinphidir_samp = math.sin(phidir_samp)

                    costhetadir_samples[costhetadir_subidx, phidir_subidx] = costhetadir_samp
                    sinthetadir_samples[costhetadir_subidx, phidir_subidx] = sinthetadir_samp
                    cosphidir_samples[costhetadir_subidx, phidir_subidx] = cosphidir_samp
                    sinphidir_samples[costhetadir_subidx, phidir_subidx] = sinphidir_samp

            costhetadir_idxs, phidir_idxs, weights = get_cone_map(
                ckv_costheta=cos_ckv,
                ckv_sintheta=sin_ckv,
                num_phi=num_cone_samples,
                axis_costheta=costhetadir_samples,
                axis_sintheta=sinthetadir_samples,
                axis_cosphi=cosphidir_samples,
                axis_sinphi=sinphidir_samples,
                num_costheta_bins=n_costhetadir,
                num_phi_bins=n_phidir,
                phi_min=phidir_min,
                phi_max=phidir_max,
                costheta_min=costhetadir_min,
                costheta_max=costhetadir_max,
            )
            num_idxs = len(costhetadir_idxs)
            assert len(phidir_idxs) == len(weights) == num_idxs

            total = 0.0
            for i_idx in range(num_idxs):
                total += weights[i_idx] * dirmap[costhetadir_idxs[i_idx], phidir_idxs[i_idx]]

            out[costhetadir_idx, phidir_idx] = total


@numba_jit(nopython=True, parallel=False, nogil=True, cache=True)
def convolve_table(
    src,
    dst,
    cos_ckv,
    sin_ckv,
    r_bin_edges,
    costheta_bin_edges,
    t_bin_edges,
    t_is_residual_time,
    costhetadir_bin_edges,
    phidir_bin_edges,
    num_cone_samples,
    oversample,
    n_group,
):
    """
    Parameters
    ----------
    src : (n_r, n_costheta, n_t, n_costhetadir, n_phidir) array

    dst : (n_r, n_costheta, n_t, n_costhetadir, n_phidir) array

    cos_ckv, sin_ckv : float

    r_bin_edges
        Radial bin edges, in units of meters.

    costheta_bin_edges
        Cosine of theta (zenith angle) bin edges.

    t_bin_edges
        Time bin edges, units of nanoseconds.

    t_is_residual_time : bool
        Whether time bins represent time residuals (True) or absolute time
        (False).

    costhetadir_bin_edges : array
        Cosine-of-direction-theta (zenith angle) bin edges.

    phidir_bin_edges : array
        Azimuth angle bin edges, in units of radians.

    num_cone_samples : int > 0

    oversample : int > 0

    n_group : float > 0
        Group refractive index in the medium (use lowest value used for all ice
        simulated).

    """
    n_t = len(t_bin_edges) - 1
    n_costheta = len(costheta_bin_edges) - 1
    n_costhetadir = len(costhetadir_bin_edges) - 1
    n_phidir = len(phidir_bin_edges) - 1

    costhetadir_min = costhetadir_bin_edges[0]
    costhetadir_max = costhetadir_bin_edges[-1]

    phidir_min = phidir_bin_edges[0]
    phidir_max = phidir_bin_edges[-1]

    costhetadir_bw = (costhetadir_max - costhetadir_min) / n_costhetadir
    phidir_bw = (phidir_max - phidir_min) / n_phidir

    costhetadir_samp_step = costhetadir_bw / oversample
    phidir_samp_step = phidir_bw / oversample

    costhetadir_min_samp = costhetadir_min + 0.5 * costhetadir_samp_step
    phidir_min_samp = phidir_min + 0.5 * phidir_samp_step

    samples_shape = (oversample, oversample)

    # Cosine and sine of thetadir
    costhetadir_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)
    std_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)

    # Cosine and sine of phidir
    cosphidir_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)
    sinphidir_samples = np.empty(shape=samples_shape, dtype=FLOAT_T)

    # Max distance from the DOM light could be for each time bin
    if t_is_residual_time:
        causal = True
    else:
        tbin_max_dist = np.array(
            [t*SPEED_OF_LIGHT_M_PER_NS/n_group for t in np.nditer(t_bin_edges[1:])],
            dtype=FLOAT_T
        )

    for costhetadir_idx in range(n_costhetadir):
        costhetadir0 = costhetadir_min_samp + costhetadir_idx*costhetadir_bw

        for phidir_idx in range(n_phidir):
            phidir0 = phidir_min_samp + phidir_idx*phidir_bw

            for costhetadir_subidx in range(oversample):
                costhetadir_samp = costhetadir0 + costhetadir_subidx * costhetadir_samp_step
                std_samp = math.sqrt(1 - costhetadir_samp**2)

                for phidir_subidx in range(oversample):
                    phidir_samp = phidir0 + phidir_subidx * phidir_samp_step
                    cosphidir_samp = math.cos(phidir_samp)
                    sinphidir_samp = math.sin(phidir_samp)

                    costhetadir_samples[costhetadir_subidx, phidir_subidx] = costhetadir_samp
                    std_samples[costhetadir_subidx, phidir_subidx] = std_samp
                    cosphidir_samples[costhetadir_subidx, phidir_subidx] = cosphidir_samp
                    sinphidir_samples[costhetadir_subidx, phidir_subidx] = sinphidir_samp

            costhetadir_idxs, phidir_idxs, weights = get_cone_map(
                ckv_costheta=cos_ckv,
                ckv_sintheta=sin_ckv,
                num_phi=num_cone_samples,
                axis_costheta=costhetadir_samples,
                axis_sintheta=std_samples,
                axis_cosphi=cosphidir_samples,
                axis_sinphi=sinphidir_samples,
                num_costheta_bins=n_costhetadir,
                num_phi_bins=n_phidir,
                phi_min=phidir_min,
                phi_max=phidir_max,
                costheta_min=costhetadir_min,
                costheta_max=costhetadir_max,
            )
            num_idxs = len(costhetadir_idxs)
            assert len(phidir_idxs) == len(weights) == num_idxs

            for r_idx, r_lower in enumerate(np.nditer(r_bin_edges[:-1])):
                for t_idx in range(n_t):
                    if not t_is_residual_time:
                        max_dist = tbin_max_dist[t_idx]
                        causal = r_lower <= max_dist
                    for costheta_idx in range(n_costheta):
                        total = 0.0
                        if causal:
                            # Apply the weights to the corresponding entries
                            # (note that weights account for normalization)
                            for i_idx in range(num_idxs):
                                costhetadir_idx = costhetadir_idxs[i_idx]
                                phidir_idx = phidir_idxs[i_idx]
                                weight = weights[i_idx]
                                total += weight * src[r_idx, costheta_idx, t_idx,
                                                      costhetadir_idx, phidir_idx]
                            #total = np.sum(
                            #    weights *
                            #    src[r_idx, costheta_idx, t_idx, costhetadir_idxs, phidir_idxs]
                            #)

                        dst[r_idx, costheta_idx, t_idx, costhetadir_idx, phidir_idx] = total


@numba_jit(parallel=False, nogil=False, cache=True) #**DFLT_NUMBA_JIT_KWARGS)
def survival_prob_from_smeared_cone(
    theta,
    num_phi,
    rot_costheta,
    rot_sintheta,
    rot_cosphi,
    rot_sinphi,
    directional_survival_prob,
    num_costheta_bins,
    num_phi_bins,
    random_delta_thetas,
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

    directional_survival_prob : ndarray of shape (N_costhetadir x N_phidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, phidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_phi_bins : int
        Number of bins in costheta and phi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and phi is assumed
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

    costheta_bin_width = 2 / num_costheta_bins
    phi_bin_width = PI / num_phi_bins

    last_costheta_bin = num_costheta_bins - 1
    last_phi_bin = num_phi_bins - 1

    for phi_idx in range(num_phi):
        offset_theta = theta + random_delta_thetas[phi_idx]
        ckv_costheta = math.cos(offset_theta)
        ckv_sintheta = math.sin(offset_theta)

        p_phi = TWO_PI * phi_idx / num_phi

        sin_p_phi = math.sin(p_phi)
        cos_p_phi = math.cos(p_phi)

        q_costheta = ((-ckv_sintheta * rot_sintheta * cos_p_phi) + (ckv_costheta * rot_costheta))
        abs_q_phi = math.fabs(math.atan2(
            (sin_p_phi * ckv_sintheta * rot_cosphi) + (ckv_sintheta * rot_sinphi * cos_p_phi * rot_costheta) + (rot_sinphi * rot_sintheta * ckv_costheta),
            (-sin_p_phi * ckv_sintheta * rot_sinphi) + (ckv_sintheta * cos_p_phi * rot_cosphi * rot_costheta) + (rot_sintheta * ckv_costheta * rot_cosphi)
        ))

        costheta_bin = int((q_costheta + 1) // costheta_bin_width)
        if costheta_bin > last_costheta_bin:
            costheta_bin = last_costheta_bin

        phi_bin = int(abs_q_phi // phi_bin_width)
        if phi_bin > last_phi_bin:
            phi_bin = last_phi_bin

        coord = (costheta_bin, phi_bin)
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
    survival_prob = survival_prob / counts_total

    return survival_prob, bin_indices, counts


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def survival_prob_from_cone(
    ckv_costheta,
    ckv_sintheta,
    num_phi,
    rot_costheta,
    rot_sintheta,
    rot_cosphi,
    rot_sinphi,
    directional_survival_prob,
    num_costheta_bins,
    num_phi_bins,
):
    """Get a numerical approximation of the expected survival probability for
    photons directed on a cone (as for Cherenkov emission) from Retro table's
    photon-directionality slice.

    Parameters
    ----------
    ckv_costheta, ckv_sintheta : scalar float
        Cherenkov angle (half the cone's opening angle)

    num_phi : scalar int
        Number of azimuth samples of the circle where the cone intersects the
        unit sphere. Increase `num_phi` for for higher accuracy.

    rot_costheta, rot_sintheta, rot_cosphi, rot_sinphi : scalar float
        Rotate the cone to have axis of symmetry defined by (rot_theta, rot_phi)

    directional_survival_prob : ndarray of shape (N_costhetadir x N_phidir)
        Note that the binning of the `directional_survival_prob` table slice is
        expected to be (costhetadir, phidir), and both are assumed to be
        uniformly gridded in those coordinate spaces.

    num_costheta_bins, num_phi_bins : int
        Number of bins in ckv_costheta and phi dimensions. cosetheta is
        assumed to be binned from -1 to 1, inclusive, and phi is assumed
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
    phi_bin_width = PI / FLOAT_T(num_phi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_phi_bin = num_phi_bins - 1

    for phi_idx in range(num_phi):
        p_phi = TWO_PI * FLOAT_T(phi_idx) / FLOAT_T(num_phi)
        sin_p_phi = np.sin(p_phi)
        cos_p_phi = np.cos(p_phi)
        q_costheta = ((-ckv_sintheta * rot_sintheta * cos_p_phi)
                      + (ckv_costheta * rot_costheta))
        abs_q_phi = np.abs(math.atan2(
            (sin_p_phi * ckv_sintheta * rot_cosphi)
            + (ckv_sintheta * rot_sinphi * cos_p_phi * rot_costheta)
            + (rot_sinphi * rot_sintheta * ckv_costheta),

            (-sin_p_phi * ckv_sintheta * rot_sinphi)
            + (ckv_sintheta * cos_p_phi * rot_cosphi * rot_costheta)
            + (rot_sintheta * ckv_costheta * rot_cosphi)
        ))

        costheta_bin = int((q_costheta + 1) // costheta_bin_width)
        if costheta_bin > last_costheta_bin:
            costheta_bin = last_costheta_bin

        phi_bin = int(abs_q_phi // phi_bin_width)
        if phi_bin > last_phi_bin:
            phi_bin = last_phi_bin

        coord = (costheta_bin, phi_bin)
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
