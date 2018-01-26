#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then for a given hypothesis generate the photon pdfs at a DOM
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


discrete_hypo = DiscreteHypo(hypo_kernels=[const_energy_loss_muon])


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

hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=20, cascade_energy=0)

pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)

strings = [79, 81, 86]
doms= range(25,60)


hit_times =  np.linspace(0, 2000, 201)

for dom in doms:
    for string in strings:
        expected_ps = []
        for hit_time in hit_times:
            #print(hit_time)
            expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=-hit_time,
                                              string=string,
                                              depth_idx=dom
                                              )
            expected_ps.append(expected_p)

        expected_ps = np.array(expected_ps)

        plt.clf()
        plt.plot(hit_times, expected_ps)
        plt.savefig('dom_pdfs/retro_%s_%s.png'%(string,dom))
