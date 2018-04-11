# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Discrete-time kernels for cascades generating photons, to be used as
hypo_kernels in discrete_hypo/DiscreteHypo class.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'point_cascade',
    'point_ckv_cascade'
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

from math import cos, sin, exp, log, log10
from os.path import abspath, dirname
import sys

import numpy as np
from scipy.stats import gamma, pareto

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS, HYPO_PARAMS_T
from retro.const import (
    PI, COS_CKV, SIN_CKV, THETA_CKV, CASCADE_PHOTONS_PER_GEV, EMPTY_SOURCES,
    SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI, SRC_CKV_BETA1
)
from retro.retro_types import SRC_T, HypoParams8D


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def point_cascade(hypo_params):
    """Point-like cascade.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams8D or HypoParams10D

    Returns
    -------
    sources

    """
    cascade_energy = hypo_params.cascade_energy
    if cascade_energy == 0:
        return EMPTY_SOURCES

    sources = np.empty(shape=(1,), dtype=SRC_T)
    sources[0]['kind'] = SRC_OMNI
    sources[0]['time'] = hypo_params.time
    sources[0]['x'] = hypo_params.x
    sources[0]['y'] = hypo_params.y
    sources[0]['z'] = hypo_params.z
    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * cascade_energy

    return sources


def point_ckv_cascade(hypo_params):
    """Single-point Cherenkov-emitting cascade with axis collinear with the
    track.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams8D or HypoParams10D

    Returns
    -------
    sources : shape (1,) array of dtype retro_types.SRC_T

    """
    cascade_energy = hypo_params.cascade_energy
    if cascade_energy == 0:
        return EMPTY_SOURCES

    if HYPO_PARAMS_T is HypoParams8D:
        opposite_zenith = PI - hypo_params.track_zenith
        opposite_azimuth = PI + hypo_params.track_azimuth
    else:
        opposite_zenith = PI - hypo_params.cascade_zenith
        opposite_azimuth = PI + hypo_params.cascade_azimuth

    dir_costheta = cos(opposite_zenith)
    dir_sintheta = sin(opposite_zenith)

    dir_cosphi = cos(opposite_azimuth)
    dir_sinphi = sin(opposite_azimuth)

    sources = np.empty(shape=(1,), dtype=SRC_T)
    sources[0]['kind'] = SRC_CKV_BETA1
    sources[0]['time'] = hypo_params.time
    sources[0]['x'] = hypo_params.x
    sources[0]['y'] = hypo_params.y
    sources[0]['z'] = hypo_params.z
    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * cascade_energy

    sources[0]['dir_costheta'] = dir_costheta
    sources[0]['dir_sintheta'] = dir_sintheta

    sources[0]['dir_cosphi'] = dir_cosphi
    sources[0]['dir_sinphi'] = dir_sinphi

    sources[0]['ckv_theta'] = THETA_CKV
    sources[0]['ckv_costheta'] = COS_CKV
    sources[0]['ckv_sintheta'] = SIN_CKV

    return sources


# TODO: use quasi-random (low discrepancy) numbers instead of pseudo-random
#       (e.g., Sobol sequence)

# Create angular zenith distribution
np.random.seed(0)

MAX_NUM_SAMPLES = int(1e5)

# Parameterizations from arXiv:1210.5140v2
ZEN_DIST = pareto(b=1.91833423, loc=-22.82924369, scale=22.82924369)
ZEN_SAMPLES = np.deg2rad(np.clip(ZEN_DIST.rvs(size=MAX_NUM_SAMPLES), a_min=0, a_max=180))

# Create angular azimuth distribution
AZI_SAMPLES = np.random.uniform(low=0, high=2*np.pi, size=MAX_NUM_SAMPLES)

PARAM_ALPHA = 2.01849
PARAM_BETA = 1.45469

MIN_CASCADE_ENERGY = np.ceil(10**(-PARAM_ALPHA / PARAM_BETA) * 100) / 100

RAD_LEN = 0.3975
PARAM_B = 0.63207
RAD_LEN_OVER_B = RAD_LEN / PARAM_B


def one_dim_cascade(hypo_params):
    """Cascade with both longitudinal and angular distributions. All emitters
    are located on the shower axis.

    Use as a hypo_kernel with the DiscreteHypo class.

    Note that the nubmer of samples is proportional to the energy of the
    cascade.

    Parameters
    ----------
    hypo_params : HYPO_PARAMS_T

    Returns
    -------
    sources

    """
    cascade_energy = hypo_params.cascade_energy
    if cascade_energy == 0:
        return EMPTY_SOURCES

    # Note that num_samples must be 1 for cascade_energy <= MIN_CASCADE_ENERGY
    # (param_a goes <= 0 at this value and below, causing an exception from
    # gamma distribution)
    if cascade_energy <= MIN_CASCADE_ENERGY:
        num_samples = 1
    else:
        # See `retro/notebooks/energy_dependent_cascade_num_samples.ipynb`
        num_samples = int(np.round(
            np.clip(exp(0.77 * log(cascade_energy) + 2.3), a_min=1, a_max=None)
        ))

    if num_samples == 1:
        return point_ckv_cascade(hypo_params)

    # Assign vertex
    time = hypo_params.time
    x = hypo_params.x
    y = hypo_params.y
    z = hypo_params.z

    # Assign cascade axis direction, works with 8D or 10D
    if HYPO_PARAMS_T is HypoParams8D:
        zenith = PI - hypo_params.track_zenith
        azimuth = PI + hypo_params.track_azimuth
    else:
        zenith = PI - hypo_params.cascade_zenith
        azimuth = PI + hypo_params.cascade_azimuth

    sin_zen = sin(zenith)
    cos_zen = cos(zenith)
    sin_azi = sin(azimuth)
    cos_azi = cos(azimuth)
    dir_x = sin_zen * cos_azi
    dir_y = sin_zen * sin_azi
    dir_z = cos_zen

    # Create rotation matrix
    rot_mat = np.array(
        [[cos_azi * cos_zen, -sin_azi, cos_azi * sin_zen],
         [sin_azi * cos_zen, cos_zen, sin_azi * sin_zen],
         [-sin_zen, 0, cos_zen]]
    )

    # Create longitudinal distribution (from arXiv:1210.5140v2)
    param_a = (
        PARAM_ALPHA
        + PARAM_BETA * log10(max(MIN_CASCADE_ENERGY, cascade_energy))
    )

    np.random.seed(1)
    long_dist = gamma(param_a, scale=RAD_LEN_OVER_B)
    long_samples = long_dist.rvs(size=num_samples)

    # Grab samples from angular zenith distribution
    zen_samples = ZEN_SAMPLES[:num_samples]

    # Grab samples from angular azimuth distribution
    azi_samples = AZI_SAMPLES[:num_samples]

    # Create angular vectors distribution
    sin_zen = np.sin(zen_samples)
    x_ang_dist = sin_zen * np.cos(azi_samples)
    y_ang_dist = sin_zen * np.sin(azi_samples)
    z_ang_dist = np.cos(zen_samples)
    ang_dist = np.concatenate(
        (x_ang_dist[np.newaxis, :],
         y_ang_dist[np.newaxis, :],
         z_ang_dist[np.newaxis, :]),
        axis=0
    )

    final_ang_dist = np.dot(rot_mat, ang_dist)
    final_phi_dist = np.arctan2(final_ang_dist[1], final_ang_dist[0])
    final_theta_dist = np.arccos(final_ang_dist[2])

    # Define photons per sample
    photons_per_sample = CASCADE_PHOTONS_PER_GEV * cascade_energy / num_samples

    # Create photon array
    sources = np.empty(shape=num_samples, dtype=SRC_T)

    sources['kind'] = SRC_CKV_BETA1

    sources['time'] = time + long_samples / SPEED_OF_LIGHT_M_PER_NS
    sources['x'] = x + long_samples * dir_x
    sources['y'] = y + long_samples * dir_y
    sources['z'] = z + long_samples * dir_z

    sources['photons'] = photons_per_sample

    sources['dir_costheta'] = final_ang_dist[2]
    sources['dir_sintheta'] = np.sin(final_theta_dist)

    sources['dir_cosphi'] = np.cos(final_phi_dist)
    sources['dir_sinphi'] = np.sin(final_phi_dist)

    sources['ckv_theta'] = THETA_CKV
    sources['ckv_costheta'] = COS_CKV
    sources['ckv_sintheta'] = SIN_CKV

    return sources
