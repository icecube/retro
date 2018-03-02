# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Convert raw Retro 5D tables (which represent survival probabilities for light
traveling in a particular direction) to tables for Cherenkov emitters with a
particular direction.

Output tables will be in .npy-files-in-a-directory format for easy memory
mapping.
"""

from __future__ import absolute_import, division, print_function

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
import retro


__all__ = [
    'get_cone_map', 'survival_prob_from_smeared_cone',
    'survival_prob_from_cone'
]


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
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
    costheta_bin_width = 2 / float(num_costheta_bins)
    deltaphi_bin_width = retro.PI / float(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    bin_indices = []
    counts = []
    counts_total = 0

    for phi_idx in range(num_phi):
        p_phi = retro.TWO_PI * float(phi_idx) / float(num_phi)
        sin_p_phi = math.sin(p_phi)
        cos_p_phi = math.cos(p_phi)

        for ax_ct, ax_st, ax_cp, ax_sp in zip(axis_costheta.flat, axis_sintheta.flat,
                                              axis_cosphi.flat, axis_sinphi.flat):
            counts_total += 1

            q_costheta = -sintheta*ax_st*cos_p_phi + costheta*ax_ct

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

            if coord in bin_indices:
                counts[bin_indices.index(coord)] += 1
            else:
                bin_indices.append(coord)
                counts.append(1)

    cnt_tot = np.float64(counts_total)
    weights = np.array([np.float64(c) / cnt_tot for c in counts], dtype=np.float32)

    return bin_indices, weights



@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
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

    costheta_bin_width = 2 / float(num_costheta_bins)
    deltaphi_bin_width = retro.PI / float(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    for phi_idx in range(num_phi):
        offset_theta = theta + random_delta_thetas[phi_idx]
        costheta = math.cos(offset_theta)
        sintheta = math.sin(offset_theta)

        p_phi = retro.TWO_PI * float(phi_idx) / float(num_phi)

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
    survival_prob = survival_prob / float(counts_total)

    return survival_prob, bin_indices, counts


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
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

    costheta_bin_width = 2 / float(num_costheta_bins)
    deltaphi_bin_width = retro.PI / float(num_deltaphi_bins)

    last_costheta_bin = num_costheta_bins - 1
    last_deltaphi_bin = num_deltaphi_bins - 1

    for phi_idx in range(num_phi):
        p_phi = retro.TWO_PI * float(phi_idx) / float(num_phi)
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
    survival_prob = survival_prob / float(counts_total)

    return survival_prob, bin_indices, counts
