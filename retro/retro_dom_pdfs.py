#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Load retro tables and, for a given hypothesis, generate the expected photon
distributions at each DOM.
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
import pickle
import hashlib
from os.path import abspath, dirname, isfile, join
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


# pylint: disable=line-too-long
SIMULATIONS = dict(
    upgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
    downgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=0,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    horizontal_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=np.pi/2,
#            cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    upgoing_em_cascade=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
#            cascade_azimuth=0, cascade_zenith=np.pi,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='EMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
)


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True,
    )
    parser.add_argument(
        '--sim-to-test', required=True, choices=sorted(SIMULATIONS.keys())
    )

    return init_obj.parse_args(parser=parser, dom_tables=True, hypo=True)


if __name__ == '__main__':
    kwargs = parse_args()

    run_info = OrderedDict([
        ('datetime', time.strftime('%Y-%m-%d %H:%M:%S')),
        ('hostname', socket.gethostname())
    ])

    CODE_TO_TEST = (
        '{tables}_tables_{norm}norm_{no_dir_str}{cone_str}{dedx_str}dt{muon_dt:.1f}'
    ).format(
        tables=kwargs['dom_tables_kw']['dom_tables_kind'],
        norm=kwargs['dom_tables_kw']['norm_version'],
        no_dir_str='no_dir_' if not kwargs['dom_tables_kw']['use_directionality'] else '',
        cone_str=(
            'sigma{}deg_{}phi_'.format(kwargs['dom_tables_kw']['ckv_sigma_deg'], kwargs['dom_tables_kw']['ckv_sigma_deg'])
            if (kwargs['dom_tables_kw']['use_directionality'] and
                kwargs['dom_tables_kw']['dom_tables_kind'] in ['raw_uncompr', 'raw_templ_compr'])
            else ''
        ),
        dedx_str='dedx_' if kwargs['hypo_kw']['track_kernel'] == 'table_e_loss' else '',
        muon_dt=kwargs['hypo_kw']['track_time_step']
        
    )

    dom_tables = init_obj.setup_dom_tables(**kwargs['dom_tables_kw'])
    hypo_handler = init_obj.setup_discrete_hypo(**kwargs['hypo_kw'])
    sim=SIMULATIONS[kwargs['other_kw']['sim_to_test']]
    outdir = expand(kwargs['other_kw'].pop('outdir'))

    fwd_sim_dir = '/data/icecube/retro/sims/'
    sim['fwd_sim_histo_file'] = join(fwd_sim_dir, sim['fwd_sim_histo_file'])

    fwd_sim_histo_file_md5 = None
    if sim['fwd_sim_histo_file'] is not None:
        print('Loading and hashing forward simulation histograms from "%s"...'
              % sim['fwd_sim_histo_file'])
        contents = open(expand(sim['fwd_sim_histo_file']), 'rb').read()
        fwd_sim_histo_file_md5 = hashlib.md5(contents).hexdigest()
        fwd_sim_histos = pickle.loads(contents)
        bin_edges = fwd_sim_histos['bin_edges']
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        del contents
    else:
        bin_edges = np.linspace(0, 4000, 401)

    run_info['sim'] = OrderedDict([
        ('mc_true_params', sim['mc_true_params']._asdict()),
        ('fwd_sim_histo_file', sim['fwd_sim_histo_file']),
        ('fwd_sim_histo_file_md5', fwd_sim_histo_file_md5)
    ])
    run_info['sim_to_test'] = kwargs['other_kw']['sim_to_test']
    hit_times = (0.5 * (bin_edges[:-1] + bin_edges[1:])).astype(np.float32)

    sources = hypo_handler.get_sources(hypo_params=sim['mc_true_params'])

    run_info['sd_indices'] = dom_tables.loaded_sd_indices
    run_info['hit_times'] = hit_times
    time_window = np.max(bin_edges) - np.min(bin_edges)
    run_info['time_window'] = time_window

    results = [None] * NUM_DOMS_TOT
    for sd_idx in dom_tables.loaded_sd_indices:
        exp_at_hit_times = []

        for hit in hit_times:
            t_indep_exp, sum_log_at_hit_times = dom_tables.pexp(
                sources,
                np.array([[hit, 1]]).T,
                dom_tables.dom_info[sd_idx],
                time_window,
                *dom_tables.tables[sd_idx]
            )
            exp_at_hit_times.append(np.exp(sum_log_at_hit_times))

        tot_retro = np.sum(exp_at_hit_times)

        results[sd_idx] = OrderedDict([
            ('t_indep_exp', t_indep_exp),
            ('exp_at_hit_times', exp_at_hit_times)
        ])

    run_info['results'] = results
    run_info_fpath = expand(join(outdir, 'run_info.pkl'))
    print('Writing run info to "{}"'.format(run_info_fpath))
    pickle.dump(run_info, open(run_info_fpath, 'wb'), pickle.HIGHEST_PROTOCOL)

    sys.stdout.write('\n\n')
