# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Function for finding expected number of photons from a DOM 3D (time, radius,
costheta) table.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    pexp_t_r_theta
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

import math
from os.path import abspath, dirname
import sys

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def pexp_t_r_theta(sources, hit_time, dom_coord, survival_prob,
                   time_indep_survival_prob, t_min, t_max, n_t_bins, r_min,
                   r_max, r_power, n_r_bins, n_costheta_bins):
    """Compute expected photons in a DOM based on the (t,r,theta)-binned
    Retro DOM tables applied to a the generated photon info `sources`,
    and the total expected photon count (time integrated) -- the normalization
    of the pdf.

    Parameters
    ----------
    sources : shape (N,) numpy ndarray, dtype SRC_T
    hit_time : float
    dom_coord : shape (3,) numpy ndarray, dtype float64
    survival_prob
    time_indep_survival_prob
    t_min : float
    t_max : float
    n_t_bins : int
    r_min : float
    r_max : float
    r_power : float
    n_r_bins : int
    n_costheta_bins : int

    Returns
    -------
    exp_p_at_all_times, exp_p_at_hit_time : (float, float)

    """
    table_dt = (t_max - t_min) / n_t_bins
    table_dcostheta = 2. / n_costheta_bins
    exp_p_at_hit_time = 0.
    exp_p_at_all_times = 0.
    inv_r_power = 1. / r_power
    table_dr_pwr = (r_max-r_min)**inv_r_power / n_r_bins

    rsquared_max = r_max*r_max
    rsquared_min = r_min*r_min

    for source in sources:
        #t, x, y, z, photons, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable

        dx = source['x'] - dom_coord[0]
        dy = source['y'] - dom_coord[1]
        dz = source['z'] - dom_coord[2]

        rsquared = dx**2 + dy**2 + dz**2

        # We can already continue before computing the bin idx
        if rsquared >= rsquared_max or rsquared < rsquared_min:
            continue

        r = math.sqrt(rsquared)
        r_bin_idx = int((r - r_min)**inv_r_power // table_dr_pwr)
        #print('r_bin_idx: ',r_bin_idx)

        costheta_bin_idx = int((1 -(dz / r)) // table_dcostheta)
        if costheta_bin_idx == n_costheta_bins:
            costheta_bin_idx = n_costheta_bins - 1
        #print('costheta_bin_idx: ',costheta_bin_idx)

        photons = source['photons']

        # time indep.
        time_indep_count = (
            photons * time_indep_survival_prob[r_bin_idx, costheta_bin_idx]
        )
        exp_p_at_all_times += time_indep_count

        t = source['time']

        # causally impossible
        if hit_time < t:
            continue

        # A photon that starts immediately in the past (before the DOM was hit)
        # will show up in the Retro DOM tables in the _last_ bin.
        # Therefore, invert the sign of the t coordinate and index sequentially
        # via e.g. -1, -2, ....
        dt = t - hit_time

        t_bin_idx = int((dt - t_min) // table_dt)
        #print('t_bin_idx: ',t_bin_idx)
        if t_bin_idx >= n_t_bins or t_bin_idx < 0:
            #print('t')
            #print('t at ',t,'with idx ',t_bin_idx)
            continue

        #print(t_bin_idx, r_bin_idx, thetabin_idx)
        #raise Exception()
        surviving_count = (
            photons * survival_prob[t_bin_idx, r_bin_idx, costheta_bin_idx]
        )

        #print(surviving_count)

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        exp_p_at_hit_time += surviving_count

    return exp_p_at_all_times, exp_p_at_hit_time
