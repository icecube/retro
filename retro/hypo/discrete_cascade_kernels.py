# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Discrete-time kernels for cascades generating photons, to be used as
hypo_kernels in discrete_hypo/DiscreteHypo class.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['point_cascade']

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

import numpy as np
import math
from scipy.stats import gamma, pareto

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS, HYPO_PARAMS_T
from retro.const import (
    COS_CKV, SIN_CKV, THETA_CKV, CASCADE_PHOTONS_PER_GEV, SPEED_OF_LIGHT_M_PER_NS, PI
    )
from retro.hypo.discrete_hypo import SRC_DTYPE, SRC_OMNI, SRC_CKV_BETA1
from retro.retro_types import HypoParams8D




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
    if hypo_params.cascade_energy == 0:
        return np.empty(shape=0, dtype=SRC_DTYPE)

    sources = np.empty(shape=(1,), dtype=SRC_DTYPE)
    sources[0]['kind'] = SRC_OMNI
    sources[0]['t'] = hypo_params.t
    sources[0]['x'] = hypo_params.x
    sources[0]['y'] = hypo_params.y
    sources[0]['z'] = hypo_params.z
    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy
    #    [(
    #        SRC_OMNI,
    #        hypo_params.t,
    #        hypo_params.x,
    #        hypo_params.y,
    #        hypo_params.z,
    #        CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy,
    #        0.0,
    #        0.0,
    #        0.0
    #    )],
    #    dtype=SRC_DTYPE
    #)
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
    sources : shape (1,) array of dtype retro_types.SRC_DTYPE

    """
    if hypo_params.cascade_energy == 0:
        return EMPTY_SOURCES

    opposite_zenith = PI - hypo_params.track_zenith
    opposite_azimuth = PI + hypo_params.track_azimuth

    dir_costheta = math.cos(opposite_zenith)
    dir_sintheta = math.sin(opposite_zenith)

    dir_cosphi = math.cos(opposite_azimuth)
    dir_sinphi = math.sin(opposite_azimuth)

    sources = np.empty(shape=(1,), dtype=SRC_DTYPE)
    sources[0]['kind'] = SRC_CKV_BETA1
    sources[0]['t'] = hypo_params.t
    sources[0]['x'] = hypo_params.x
    sources[0]['y'] = hypo_params.y
    sources[0]['z'] = hypo_params.z
    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy

    sources[0]['dir_costheta'] = dir_costheta
    sources[0]['dir_sintheta'] = dir_sintheta

    sources[0]['dir_cosphi'] = dir_cosphi
    sources[0]['dir_sinphi'] = dir_sinphi

    sources[0]['ckv_theta'] = THETA_CKV
    sources[0]['ckv_costheta'] = COS_CKV
    sources[0]['ckv_sintheta'] = SIN_CKV

    return sources

def one_dim_cascade(hypo_params, num_samples):
    """
    Cascade with both longitudinal and angular distributions

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : MUST BE HypoParams10D
    num_samples : integer, number of times to sample the distributions

    Returns
    -------
    pinfo_gen
    """

    #assign vertex
    t = hypo_params.t
    x = hypo_params.x
    y = hypo_params.y
    z = hypo_params.z

    #assign cascade axis direction, works with 8D or 10D
    if HYPO_PARAMS_T is HypoParams8D:
        zenith = np.pi - hypo_params.track_zenith
        azimuth = np.pi + hypo_params.track_azimuth
    else:
        zenith = np.pi - hypo_params.cascade_zenith
        azimuth = np.pi + hypo_params.cascade_azimuth
    
    sin_zen = math.sin(zenith)
    cos_zen = math.cos(zenith)
    sin_azi = math.sin(azimuth)
    cos_azi = math.cos(azimuth)
    dir_x = sin_zen * cos_azi
    dir_y = sin_zen * sin_azi
    dir_z = cos_zen

    #create rotation matrix 
    rot_mat = np.array([[cos_azi * cos_zen, -sin_azi, cos_azi * sin_zen],
               [sin_azi * cos_zen, cos_zen, sin_azi * sin_zen],
               [-sin_zen, 0, cos_zen]
               ])

    #define photons per sample
    photons_per_sample = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy / num_samples

    #create longitudinal distribution (from arXiv:1210.5140v2 section 3.3)
    #NOTE: a must be greater than 0, which fails when cascade energy is less than 41 MeV
    alpha = 2.01849
    beta = 1.45469
    a = alpha + beta * np.log10(hypo_params.cascade_energy)
    b = 0.63207
    rad_length = 0.3975

    long_dist = gamma(a, scale=rad_length/b)
    long_samples = long_dist.rvs(size=num_samples)

    #create angular zenith distribution
    zen_dist = pareto(b=1.91833423, loc=-22.82924369, scale=22.82924369)
    zen_samples = zen_dist.rvs(size=num_samples)
    zen_samples[zen_samples > 180] = 0.
    zen_samples = zen_samples * np.pi / 180.

    #create angular azimuth distribution
    azi_samples = np.random.uniform(low=0., high=2*np.pi, size=(num_samples,))

    #create angular vectors distribution
    x_ang_dist = np.sin(zen_samples) * np.cos(azi_samples)
    y_ang_dist = np.sin(zen_samples) * np.sin(azi_samples)
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

    #create photon matrix
    sources = np.empty(shape=num_samples, dtype=SRC_DTYPE)

    sources['kind'] = SRC_CKV_BETA1
    sources['t'] = t + long_samples / SPEED_OF_LIGHT_M_PER_NS
    sources['x'] = x + long_samples * dir_x
    sources['y'] = y + long_samples * dir_y
    sources['z'] = z + long_samples * dir_z
    sources['photons'] = photons_per_sample

    sources['dir_costheta'] = np.cos(final_theta_dist)
    sources['dir_sintheta'] = np.sin(final_theta_dist)

    sources['dir_cosphi'] = np.cos(final_phi_dist)
    sources['dir_sinphi'] = np.sin(final_phi_dist)

    sources['ckv_theta'] = THETA_CKV
    sources['ckv_costheta'] = COS_CKV
    sources['ckv_sintheta'] = SIN_CKV
    sources.sort()
    return sources    
