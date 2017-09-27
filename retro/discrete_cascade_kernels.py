# pylint: disable=wrong-import-position

"""
Discrete-time kernels for cascades generating photons, to be used as
hypo_kernels in discrete_hypo/DiscreteHypo class.
"""


from __future__ import absolute_import, division, print_function

import os
from os.path import abspath, dirname

import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import CASCADE_PHOTONS_PER_GEV


__all__ = ['point_cascade']


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
