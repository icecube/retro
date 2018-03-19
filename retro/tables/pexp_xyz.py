# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Function for computing expected number of photons to survive from a
time-independent Cartesian-binned table.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    pexp_xyz
'''.split()

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
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pexp_xyz(sources, x_min, y_min, z_min, nx, ny, nz, binwidth,
             survival_prob, avg_photon_x, avg_photon_y, avg_photon_z,
             use_directionality):
    """Compute the expected number of detected photons in _all_ DOMs at _all_
    times.

    Parameters
    ----------
    sources :
    x_min, y_min, z_min :
    nx, ny, nz :
    binwidth :
    survival_prob :
    avg_photon_x, avg_photon_y, avg_photon_z :
    use_directionality : bool

    """
    expected_photon_count = 0.0
    for source in sources:
        x_idx = int((source['x'] - x_min) // binwidth)
        if x_idx < 0 or x_idx >= nx:
            continue
        y_idx = int((source['y'] - y_min) // binwidth)
        if y_idx < 0 or y_idx >= ny:
            continue
        z_idx = int((source['z'] - z_min) // binwidth)
        if z_idx < 0 or z_idx >= nz:
            continue
        sp = survival_prob[x_idx, y_idx, z_idx]
        surviving_count = source['photons'] * sp

        # TODO: Incorporate photon direction info
        if use_directionality:
            raise NotImplementedError('Directionality cannot be used yet')

        expected_photon_count += surviving_count

    return expected_photon_count
