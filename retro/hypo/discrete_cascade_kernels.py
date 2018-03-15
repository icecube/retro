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
from scipy.stats import gamma

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS
from retro.const import (
    COS_CKV, CASCADE_PHOTONS_PER_GEV, SPEED_OF_LIGHT_M_PER_NS
    )


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def point_cascade(hypo_params):
    """Point-like cascade.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams8D or HypoParams10D

    Returns
    -------
    pinfo_gen

    """
    pinfo_gen = np.array(
        [
            hypo_params.t,
            hypo_params.x,
            hypo_params.y,
            hypo_params.z,
            CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy,
            0.0,
            0.0,
            0.0
        ],
        dtype=np.float32
    ).reshape((1, 8))
    return pinfo_gen

def one_dim_cascade(hypo_params, n_samples=1000):
    """
    Cascade with both longitudinal and angular distributions

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : MUST BE HypoParams10D
    n_samples : integer, number of times to sample the distributions

    Returns
    -------
    pinfo_gen
    """

    #assign vertex
    t = hypo_params.t
    x = hypo_params.x
    y = hypo_params.y
    z = hypo_params.z

    #assign cascade axis direction
    sin_zen = math.sin(hypo_params.cascade_zenith)
    dir_x = -sin_zen * math.cos(hypo_params.cascade_azimuth)
    dir_y = -sin_zen * math.sin(hypo_params.cascade_azimuth)
    dir_z = -math.cos(hypo_params.cascade_zenith)

    #define photons per sample
    photons_per_sample = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy / n_samples

    #create longitudinal distribution (from arXiv:1210.5140v2)
    alpha = 2.01849
    beta = 1.45469
    a = alpha + beta * np.log10(hypo_params.cascade_energy)
    b = 0.63207
    rad_length = 0.3975

    long_dist = gamma(a, scale=x/b)
    long_samples = l_dist.rvs(size=n_samples)

    #create photon matrix
    pinfo_gen = np.empty((n_samples, 8), dtype=np.float32)

    #NOTE: ask justin about COS_CKV
    pinfo_gen[:, 0] = t + long_samples / SPEED_OF_LIGHT_M_PER_NS
    pinfo_gen[:, 1] = x + long_samples * dir_x
    pinfo_gen[:, 2] = y + long_samples * dir_y
    pinfo_gen[:, 3] = z + long_samples * dir_z
    pinfo_gen[:, 4] = photons_per_samples
    pinfo_gen[:, 5] = dir_x * COS_CKV 
    pinfo_gen[:, 6] = dir_y * COS_CKV
    pinfo_gen[:, 7] = dir_z * COS_CKV

    return pinfo_gen    


