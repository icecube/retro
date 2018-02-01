#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then for a given hypothesis calculate llh for an event
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


from argparse import ArgumentParser
from collections import OrderedDict, Sequence
from copy import deepcopy
import cPickle as pickle
from itertools import izip, product
from os import makedirs
from os.path import abspath, dirname, isdir, join
import sys
import time

import numba # pylint: disable=unused-import
import numpy as np

from scipy.special import gammaln

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import DC_DOM_JITTER_NS, IC_DOM_JITTER_NS, PI # pylint: disable=unused-import
from retro import FTYPE, HYPO_PARAMS_T, HypoParams10D
from retro import DETECTOR_GEOM_FILE
from retro import (event_to_hypo_params, expand, poisson_llh,
                   get_primary_interaction_str)
from retro.events import Events
from retro.discrete_hypo import DiscreteHypo
from retro.discrete_muon_kernels import const_energy_loss_muon # pylint: disable=unused-import
from retro.discrete_cascade_kernels import point_cascade # pylint: disable=unused-import
from retro.plot_1d_scan import plot_1d_scan
from retro.table_readers import DOMTimePolarTables, TDICartTable # pylint: disable=unused-import

SMALL_P = 1e-8
NOISE_Q = 0.1

def get_neg_llh(pinfo_gen, event, dom_tables):
    """Get log likelihood.

    Parameters
    ----------
    pinfo_gen

    event : retro.Event namedtuple or convertible thereto

    dom_tables

    Returns
    -------
    llh : float
        Negative of the log likelihood

    """
    neg_llh = 0
    small_p_counts = 0
    small_n_counts = 0

    for string_idx in range(86):
        for dom_idx in range(60):
            #print('string %i, DOM %i'%(string_idx+1, dom_idx+1))
            total_observed_ps = 0.
            total_expected_ps = 0.
            try:
                hits = event[string_idx+1][dom_idx+1]
            except KeyError:
                hits = None

            if hits is not None:
                weights = hits['weight']
                total_observed_ps = np.sum(weights)
                times = hits['time']
                for hit_time, weight in zip(times, weights):
                    total_expected_ps, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=hit_time,
                                              string=string_idx+1,
                                              depth_idx=dom_idx,
                                              )
                    #print(total_expected_ps)
                    #print('expected_p %.5f'%expected_p)
                    P = expected_p / total_expected_ps
                    probability = P + SMALL_P - P*SMALL_P
                    neg_llh -= weight * np.log(probability)
                    #print('hit probability %.5f'%probability)
                    # hit ptobability
            else:
                total_expected_ps, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=0,
                                              string=string_idx+1,
                                              depth_idx=dom_idx,
                                              )
            

            #if total_observed_ps > 0. and total_expected_ps > 0.:
            #llh = total_observed_ps * np.log(total_expected_ps) - total_expected_ps - gammaln(total_observed_ps+1)
            #print('expected %.3f, observed %.3f, llh = %.3f'%(total_expected_ps, total_observed_ps, llh))
            #neg_llh -= llh

            N = total_expected_ps + NOISE_Q
            if not (total_observed_ps == 0. and total_expected_ps == 0.):
                neg_llh -= (total_observed_ps * np.log(N) - N)

            ##total_expected_ps = max(total_expected_ps, SMALL_N)
            #if total_observed_ps == 0:
            #    neg_llh -= total_expected_ps
            #else:
            #    if total_expected_ps < SMALL_N:
            #        total_expected_ps = SMALL_N
            #        small_n_counts += 1
            #    #poisson probability (skipping gamma)
            #    #print('DOM %s, %s expected %.5f observed %.5f'%(string_idx, dom_idx, total_expected_ps, total_observed_ps))

    #print('%i small p'%small_p_counts)
    #print('%i small n'%small_n_counts)
    return neg_llh


discrete_hypo = DiscreteHypo(hypo_kernels=[point_cascade, const_energy_loss_muon])


geom_file = DETECTOR_GEOM_FILE
tables_dir = '/data/icecube/retro_tables/full1000/'


# Load detector geometry array
print('Loading detector geometry from "%s"...' % expand(geom_file))
detector_geometry = np.load(expand(geom_file))

# Load tables
print('Loading DOM tables...')
dom_tables = DOMTimePolarTables(
    tables_dir=tables_dir,
    hash_val=None,
    geom=detector_geometry,
    use_directionality=False,
    naming_version=0,
)
dom_tables.load_tables()

#hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=20, cascade_energy=0)
#hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=0, cascade_energy=20)
#hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=-PI, track_energy=20, cascade_energy=0)

with open('benchmarkEMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=-1.0_azimuth=0.0_events.pkl', 'rb') as f:
#f = open('benchmark.pkl', 'rb')
    events = pickle.load(f)

for i, event in enumerate(events[0:1]):
    print('\nevent ',i)

    #for z_pos in np.linspace(-500, -300, 21):
    for x_pos in np.linspace(-100, 100, 21):
    #for t_pos in np.linspace(-1000, 1000, 21):
    #for e_cscd in np.linspace(1, 41, 21):

        #hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=z_pos, track_azimuth=0, track_zenith=0, track_energy=0, cascade_energy=20)
        hypo_params = HYPO_PARAMS_T(t=0, x=x_pos, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=0, cascade_energy=20)
        #hypo_params = HYPO_PARAMS_T(t=t_pos, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=0, cascade_energy=20)
        #hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=0, cascade_energy=e_cscd)
        pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)

        neg_llh = get_neg_llh(pinfo_gen, event, dom_tables)

        print('x = %i, llh = %.2f'%(x_pos,neg_llh))
        #print('z = %i, llh = %.2f'%(z_pos,neg_llh))
        #print('t = %i, llh = %.2f'%(t_pos,neg_llh))
        #print('E = %i, llh = %.2f'%(e_cscd,neg_llh))

    

