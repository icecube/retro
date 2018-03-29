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

import math
import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS
from retro.const import (
    PI, COS_CKV, SIN_CKV, THETA_CKV, CASCADE_PHOTONS_PER_GEV
)
from retro.hypo.discrete_hypo import EMPTY_SOURCES, SRC_DTYPE, SRC_OMNI


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
        return EMPTY_SOURCES

    sources = np.empty(shape=(1,), dtype=SRC_DTYPE)
    sources[0]['kind'] = SRC_OMNI
    sources[0]['t'] = hypo_params.t
    sources[0]['x'] = hypo_params.x
    sources[0]['y'] = hypo_params.y
    sources[0]['z'] = hypo_params.z
    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy

    return sources


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
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
    sources[0]['kind'] = SRC_OMNI
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
