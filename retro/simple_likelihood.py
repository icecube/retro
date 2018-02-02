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
            start_t = time.time()
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
                    # normalize to get pdf
                    P = expected_p / total_expected_ps
                    # make sure they're not zero
                    probability = P + SMALL_P - P*SMALL_P
                    # add them number of hit times
                    neg_llh -= weight * np.log(probability)
            else:
                # just want total_expected_ps, but using thr same function right now
                total_expected_ps, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=-1000,
                                              string=string_idx+1,
                                              depth_idx=dom_idx,
                                              )
            

            # add a noise floor
            N = total_expected_ps + NOISE_Q
            # for 0,0 poisson is i think not well defined
            if not (total_observed_ps == 0. and total_expected_ps == 0.):
                # add only non-constant part
                neg_llh -= (total_observed_ps * np.log(N) - N)
            stop_t = time.time()

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

# create scan dimensions disctionary
scan_dims = {}
scan_dims['t'] = {'scan_points':np.linspace(-100, 100, 21)}
scan_dims['x'] = {'scan_points':np.linspace(-100, 100, 21)}
scan_dims['y'] = {'scan_points':np.linspace(-100, 100, 21)}
scan_dims['z'] = {'scan_points':np.linspace(-500, -300, 21)}
scan_dims['track_zenith'] = {'scan_points':np.linspace(0, PI, 21)}
scan_dims['track_azimuth'] = {'scan_points':np.linspace(0, 2*PI, 21)}
scan_dims['track_energy'] = {'scan_points':np.linspace(1, 41, 21)}
scan_dims['cscd_energy'] = {'scan_points':np.linspace(1, 41, 21)}

#open events file:

#with open('benchmarkEMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=-1.0_azimuth=0.0_events.pkl', 'rb') as f:
#with open('benchmark_events.pkl', 'rb') as f:
with open('testMuMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=0.0_azimuth=0.0_events.pkl', 'rb') as f:
    events = pickle.load(f)

hypo = {'t':0, 'x':0, 'y':0, 'z':-400, 'track_zenith':PI, 'track_azimuth':0, 'track_energy':20, 'cascade_energy':0}

scan_dim = 't'

# scan multiple events
llhs = []
for i, event in enumerate(events[0:10]):
    print('\nevent ',i)

    llhs.append([])
    for point in scan_dims[scan_dim]['scan_points']: 
        hypo[scan_dim] = point
        hypo_params = HYPO_PARAMS_T(**hypo)
        pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)
        llhs[-1].append(get_neg_llh(pinfo_gen, event, dom_tables))

llhs = np.array(llhs)

# plot them
for llh in llhs:
    plt.plot(scan_points, llh)
    plt.gca().set_xlabel(scan_dim)
    plt.gca().set_ylabel('-llh')
plt.savefig('scans/%s.png'%scan_dim)
    

