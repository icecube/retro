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

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import CASCADE_PHOTONS_PER_GEV


def point_cascade(hypo_params, limits=None):
    """Point-like cascade.

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : HypoParams8D or HypoParams10D
    limits

    Returns
    -------
    pinfo_gen

    """
    pinfo_gen = np.empty((1, 8), dtype=np.float64)
    pinfo_gen[0, 0] = hypo_params.t
    pinfo_gen[0, 1] = hypo_params.x
    pinfo_gen[0, 2] = hypo_params.y
    pinfo_gen[0, 3] = hypo_params.z
    pinfo_gen[0, 4] = CASCADE_PHOTONS_PER_GEV * hypo_params.cascade_energy
    pinfo_gen[0, 5] = 0
    pinfo_gen[0, 6] = 0
    pinfo_gen[0, 7] = 0
    return pinfo_gen

def long_cascade(hypo_params, limits=None, samples=1000):
    """
    Cascade with both longitudinal and angular distributions

    Use as a hypo_kernel with the DiscreteHypo class.

    Parameters
    ----------
    hypo_params : MUST BE HypoParams10D
    limits : NOT IMPLEMENTED
    samples : integer, number of times to sample the distributions

    Returns
    -------
    pinfo_gen
    """

    #define vertex
    t_v = hypo_params.t
    x_v = hypo_params.x
    y_v = hypo_params.y
    z_v = hypo_params.z

    

