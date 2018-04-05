#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Load retro tables into RAM, then for a given hypothesis generate the photon
pdfs at a DOM
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
from collections import OrderedDict
import cPickle as pickle
import hashlib
from itertools import product
from os.path import abspath, dirname, isdir, isfile, join
import socket
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo.discrete_muon_kernels import const_energy_loss_muon, table_energy_loss_muon
from retro.hypo.discrete_cascade_kernels import point_cascade, one_dim_cascade
from retro.i3info.extract_gcd import extract_gcd
from retro.tables.dom_time_polar_tables import DOMTimePolarTables
#from retro.tables.tdi_cart_tables import TDICartTable
from retro.tables.retro_5d_tables import Retro5DTables
from retro.utils.misc import expand, mkdir
from retro.const import NUM_DOMS_TOT
from retro import init_obj


hostname = socket.gethostname()
run_info = OrderedDict([
    ('datetime', time.strftime('%Y-%m-%d %H:%M:%S')),
    ('hostname', hostname)
])

def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True,
    )
    parser.add_argument(
        '--sim-to-test', required=True,
    )

    dom_tables_kw, hypo_kw, _, pdf_kw = (
        init_obj.parse_args(parser=parser, hits=False)
    )

    return dom_tables_kw, hypo_kw, pdf_kw

dom_tables_kw, hypo_kw, pdf_kw = parse_args()

CODE_TO_TEST = (
    '{tables}_tables_{norm}norm_{no_dir_str}{cone_str}{dedx_str}dt{muon_dt:.1f}'
    .format(
        tables=dom_tables_kw['dom_tables_kind'],
        norm=dom_tables_kw['norm_version'],
        no_dir_str='no_dir_' if not dom_tables_kw['use_directionality'] else '',
        cone_str=(
            'sigma{}deg_{}phi_'.format(dom_tables_kw['ckv_sigma_deg'], dom_tables_kw['ckv_sigma_deg'])
            if dom_tables_kw['use_directionality'] and dom_tables_kw['dom_tables_kind'] in ['raw_uncompr', 'raw_templ_compr']
            else ''
        ),
        dedx_str='dedx_' if hypo_kw['track_kernel'] == 'table_e_loss' else '',
        muon_dt=hypo_kw['track_time_step']
    )
)

# pylint: disable=line-too-long
#CHANGING FROM 8D TO 10D
SIMULATIONS = dict(
    upgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
    downgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=0,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    horizontal_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=np.pi/2,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    upgoing_em_cascade=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
#            cascade_azimuth=0, cascade_zenith=np.pi,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='EMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
)

dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
sim=SIMULATIONS[pdf_kw['sim_to_test']]
outdir = expand(pdf_kw.pop('outdir'))

fwd_sim_dir = '/data/icecube/retro/sims/'
sim['fwd_sim_histo_file'] = join(fwd_sim_dir, sim['fwd_sim_histo_file'])

fwd_sim_histo_file_md5 = None
if sim['fwd_sim_histo_file'] is not None:
    print('Loading and hashing forward simulation histograms from "%s"...'
          % sim['fwd_sim_histo_file'])
    t0 = time.time()
    contents = open(expand(sim['fwd_sim_histo_file']), 'rb').read()
    fwd_sim_histo_file_md5 = hashlib.md5(contents).hexdigest()
    fwd_sim_histos = pickle.loads(contents)
    bin_edges = fwd_sim_histos['bin_edges']
    #print('bin_edges:', bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    del contents
    #print(' ', np.round(time.time() - t0, 3), 'sec\n')
else:
    bin_edges = np.linspace(0, 4000, 401)

run_info['sim'] = OrderedDict([
    ('mc_true_params', sim['mc_true_params']._asdict()),
    ('fwd_sim_histo_file', sim['fwd_sim_histo_file']),
    ('fwd_sim_histo_file_md5', fwd_sim_histo_file_md5)
])
run_info['sim_to_test'] = pdf_kw['sim_to_test']
hit_times = (0.5 * (bin_edges[:-1] + bin_edges[1:])).astype(np.float32)

sources = hypo_handler.get_sources(hypo_params=sim['mc_true_params'])

run_info['sd_indices'] = dom_tables.loaded_sd_indices
run_info['hit_times'] = hit_times
time_window = np.max(bin_edges) - np.min(bin_edges)
run_info['time_window'] = time_window

results = [None] * NUM_DOMS_TOT
#t_start = time.time()
string_counter = 0
#for string, dom in product(unique_strings, unique_doms):
for sd_idx in dom_tables.loaded_sd_indices:
    #TODO: ask justin what this line does
    #t00 = time.time()
    exp_p_at_hit_times = []

    print('string_counter', string_counter)
    string_counter += 1
    hit_counter = 0

    for hit in hit_times: 

        #print('hit_counter', hit_counter)
        hit_counter += 1
        
        exp_p_at_all_times, sum_log_at_hit_times = dom_tables.pexp_func(
            sources,
            np.array([[hit, 1]]).T,
            dom_tables.dom_info[sd_idx],
            time_window,
            *dom_tables.tables[sd_idx]
        )
        exp_p_at_hit_times.append(np.exp(sum_log_at_hit_times))
    #t11 = time.time() - t00
    #pexp_timings.append(t11)
    #hypo_count += hit_times.size
    #pgen_count += hit_times.size * n_source_points

    tot_retro = np.sum(exp_p_at_hit_times)

    results[sd_idx] = OrderedDict([
        ('exp_p_at_all_times', exp_p_at_all_times),
        ('exp_p_at_hit_times', exp_p_at_hit_times)
    ])

    #msg = (
    #    '{:12.0f} ns ({:.2e} hypos computed, w/ total of {:.2e} source points)'
    #    .format(np.round(np.sum(pexp_timings)/pgen_count * 1e9, 3), hypo_count, pgen_count)
    #)

#    if MAKE_PLOTS:
#        plt.clf()
#        plt.plot(hit_times, exp_p_at_hit_times, label='Retro')
#    tot_clsim = 0.0
#    try:
#        fwd_sim_histo = np.nan_to_num(fwd_sim_histos['results'][(string, dom)])
#        tot_clsim = np.sum(fwd_sim_histo)
#        if MAKE_PLOTS:
#            plt.plot(hit_times, fwd_sim_histo, label='CLSim fwd sim')
#    except KeyError:
#        pass
#
#    # Don't plot if both are 0
#    if tot_clsim == 0 and tot_retro == 0:
#        continue
#
#    a_text = AnchoredText(
#        '{sum} Retro t-dep = {retro:.5f}      {sum} Retro / {sum} CLSim = {ratio:.5f}\n'
#        '{sum} CLSim       = {clsim:.5f}\n'
#        'Retro t-indep = {exp_p_at_all_times:.5f}\n'
#        .format(
#            sum=r'$\Sigma$',
#            retro=tot_retro,
#            clsim=tot_clsim,
#            ratio=tot_retro/tot_clsim if tot_clsim != 0 else np.nan,
#            exp_p_at_all_times=exp_p_at_all_times
#        ),
#        loc=2,
#        prop=dict(family='monospace', size=10),
#        frameon=False,
#    )
#    if MAKE_PLOTS:
#        ax = plt.gca()
#        ax.add_artist(a_text)
#
#        ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
#        ax.set_ylim(0, ax.get_ylim()[1])
#        ax.set_title('String {}, DOM {}'.format(string, dom))
#        ax.set_xlabel('time (ns)')
#        ax.legend(loc='center left', frameon=False)
#
#        clsim_code = 'c' if tot_clsim > 0 else ''
#        retro_code = 'r' if tot_retro > 0 else ''
#
#        fname = (
#            'sim_{hypo}_code_{code}_{string}_{dom}_{retro_code}_{clsim_code}'
#            .format(
#                hypo=SIM_TO_TEST, code=CODE_TO_TEST, string=string, dom=dom,
#                retro_code=retro_code, clsim_code=clsim_code
#            )
#        )
#        fpath = join(OUTDIR, fname + '.png')
#        print('Saving "{}"'.format(fpath))
#        plt.savefig(fpath)
#
#print('total of {} unique DOMs'.format(len(loaded_strings_doms)))
#
run_info['results'] = results
#
run_info_fpath = expand(join(outdir, 'run_info.pkl'))
print('Writing run info to "{}"'.format(run_info_fpath))
pickle.dump(run_info, open(run_info_fpath, 'wb'), pickle.HIGHEST_PROTOCOL)

sys.stdout.write('\n\n')
#print(' ', 'Time to compute and plot:')
#print(' ', np.round(time.time() - t0, 3), 'sec\n')

#print('Body of script took {:.3f} sec'.format(time.time() - t_start))
