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


discrete_hypo = DiscreteHypo(hypo_kernels=[point_cascade,const_energy_loss_muon])
#discrete_hypo = DiscreteHypo(hypo_kernels=[point_cascade])


tables_dir = '/data/icecube/retro_tables/full1000/'

with open('../data/gcd_dict.pkl', 'rb') as f:
    gcd_dict = pickle.load(f)


# Load tables
print('Loading DOM tables...')
dom_tables = DOMTimePolarTables(
    tables_dir=tables_dir,
    hash_val=None,
    geom=gcd_dict['geo'],
    use_directionality=False,
    naming_version=0,
)
dom_tables.load_tables()

#hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=0, track_energy=20, cascade_energy=0)
#hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=PI, track_energy=0, cascade_energy=20)
hypo_params = HYPO_PARAMS_T(t=0, x=0, y=0, z=-400, track_azimuth=0, track_zenith=PI, track_energy=20, cascade_energy=0)

#with open('benchmarkEMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=-1.0_azimuth=0.0.pkl', 'rb') as f:
with open('./benchmark.pkl', 'rb') as f:
    photon_histos = pickle.load(f)
#with open('../icetray_processing/cascade_step4_SRTInIcePulses_90_700_1.pkl', 'rb') as f:
with open('../icetray_processing/track_step4_SRTInIcePulses_90_700_1.pkl', 'rb') as f:
    srt_histos = pickle.load(f)
#with open('../icetray_processing/cascade_step4_SplitUncleanedInIcePulses.pkl', 'rb') as f:
with open('../icetray_processing/track_step4_SplitUncleanedInIcePulses.pkl', 'rb') as f:
    uncleaned_histos = pickle.load(f)

# from justin
#with open('/home/justin/dom_pdfs/upgoing_muon/clsim_tables_avgsurfareanorm_dt1.0_sigma20deg_100phi/run_info.pkl', 'rb') as f:
with open('/home/justin/dom_pdfs/upgoing_muon/clsim_tables_pdenorm_dt1.0_sigma5deg_100phi/run_info.pkl', 'rb') as f:
    run_info = pickle.load(f)


pinfo_gen = discrete_hypo.get_pinfo_gen(hypo_params)

strings = [36] + [79, 80, 81, 82, 83, 84, 85, 86] + [26, 27, 35, 37, 45, 46]
#strings = [86]
doms= range(25,60)

norm = False
norm2 = False

hit_times =  np.linspace(0, 2000, 201)
mid_points = 0.5 * (hit_times[1:] + hit_times[:-1])

for dom in doms:
    for string in strings:
        expected_ps = []
        expected_hits = []
        for hit_time in mid_points:
            #print(hit_time)
            total_p, expected_p = dom_tables.get_photon_expectation(
                                              pinfo_gen=pinfo_gen,
                                              hit_time=hit_time,
                                              string=string,
                                              depth_idx=dom-1,
                                              )
            expected_ps.append(expected_p)
            expected_hits.append((expected_p * 0.25 * gcd_dict['rde'][string-1, dom-1]) + gcd_dict['noise'][string-1, dom-1] * 10e-9)

        expected_ps = np.array(expected_ps)
        expected_hits = np.array(expected_hits)
        tot_retro = np.sum(expected_ps)
        if norm:
            expected_ps /= np.sum(expected_ps)


        plt.clf()
        #plt.plot(mid_points, expected_ps, label='retro photons')
        plt.plot(mid_points, expected_hits, label='simplistic retro', c='cyan')
        try:
            #time = histos[string][dom]['time']
            #weight = histos[string][dom]['weight']
            h = np.nan_to_num(uncleaned_histos[string][dom])
            #tot_clsim = np.sum(h)
            #if norm:
            #    h/= np.sum(h)
            #if norm2:
            #    h *= 200
            series =  (run_info['results'][(string, dom)]['pexp_at_hit_times'] * 0.25 * gcd_dict['rde'][string-1, dom-1]) + gcd_dict['noise'][string-1, dom-1] * 10e-9
            plt.plot(mid_points, series, c='red', label='pde_norm justin_dir RDE + noise')

            #plt.plot(mid_points, np.nan_to_num(srt_histos[string][dom]), c='r', label='SRT hits')
            plt.plot(mid_points, h, label='Uncleaned hits', c='orange')
            #plt.plot(mid_points, np.nan_to_num(photon_histos[string][dom]), c='b', label='photons')
            plt.gca().legend(loc='upper right',ncol=1, frameon=False,numpoints=1)
            #a_text = AnchoredText('RETRO = %.5f, CLSIM = %.5f, ratio = %.5f\n total_p = %.2f, sum = %.2f'%(tot_retro, tot_clsim, tot_retro/tot_clsim, total_p, np.sum(expected_ps)), loc=2)
            #print(tot_retro/tot_clsim)
            #plt.gca().add_artist(a_text)
            #plt.plot(time, weight)
            #plt.scatter(time, weight*10, s=1, marker='.', c='r')
        except KeyError:
            pass

        #plt.savefig('dom_pdfs/retro_%s_%s.png'%(string,dom))
        plt.savefig('dom_hit_dists/retro_%s_%s.png'%(string,dom))
