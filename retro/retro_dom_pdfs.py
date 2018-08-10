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

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T
from retro.utils.misc import expand, mkdir
from retro import init_obj, const


# pylint: disable=line-too-long
SIMULATIONS = dict(
    mie_upgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
    mie_downgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=0,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    mie_horizontal_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=np.pi/2,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    mie_upgoing_em_cascade=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            #cascade_azimuth=0, cascade_zenith=np.pi,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='EMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
    lea_upgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_lea_holeice_as.9_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-5000ns_1000bins.pkl',
    ),
    lea_downgoing_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=0,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-300_cz+1.0_az0_ice_spice_lea_holeice_as.9_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-5000ns_1000bins.pkl',
    ),
    lea_horizontal_muon=dict(
        mc_true_params=dict(
            time=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=np.pi/2,
            #cascade_azimuth=0, cascade_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_lea_holeice_as.9_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-5000ns_1000bins.pkl',
    ),
    #lea_upgoing_em_cascade=dict(
    #    mc_true_params=dict(
    #        time=0, x=0, y=0, z=-400,
    #        track_azimuth=0, track_zenith=np.pi,
    #        #cascade_azimuth=0, cascade_zenith=np.pi,
    #        track_energy=0, cascade_energy=20
    #    ),
    #    fwd_sim_histo_file='EMinus_energy20_x0_y0_z-400_cz-1.0_az0_ice_spice_mie_holeice_as.9_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos_0-5000ns_500bins.pkl'
    #),
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

    #CODE_TO_TEST = (
    #    '{tables}_tables_{norm}norm_{no_dir_str}{cone_str}{dedx_str}dt{muon_dt:.1f}'
    #).format(
    #    tables=kwargs['dom_tables_kw']['dom_tables_kind'],
    #    norm=kwargs['dom_tables_kw']['norm_version'],
    #    no_dir_str='no_dir_' if not kwargs['dom_tables_kw']['use_directionality'] else '',
    #    cone_str=(
    #        'sigma{}deg_{}phi_'.format(kwargs['dom_tables_kw']['ckv_sigma_deg'], kwargs['dom_tables_kw']['ckv_sigma_deg'])
    #        if (kwargs['dom_tables_kw']['use_directionality'] and
    #            kwargs['dom_tables_kw']['dom_tables_kind'] in ['raw_uncompr', 'raw_templ_compr'])
    #        else ''
    #    ),
    #    dedx_str='dedx_' if kwargs['hypo_kw']['track_kernel'] == 'table_e_loss' else '',
    #    muon_dt=kwargs['hypo_kw']['track_time_step']
    #
    #)

    dom_tables = init_obj.setup_dom_tables(**kwargs['dom_tables_kw'])
    hypo_handler = init_obj.setup_discrete_hypo(**kwargs['hypo_kw'])
    sim = SIMULATIONS[kwargs['other_kw']['sim_to_test']]
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
        print('%i time bins'%len(bin_centers))
        del contents
    else:
        bin_edges = np.linspace(0, 4000, 401)

    run_info['sim'] = OrderedDict([
        ('mc_true_params', sim['mc_true_params']),
        ('fwd_sim_histo_file', sim['fwd_sim_histo_file']),
        ('fwd_sim_histo_file_md5', fwd_sim_histo_file_md5)
    ])
    run_info['sim_to_test'] = kwargs['other_kw']['sim_to_test']
    hit_times = (0.5 * (bin_edges[:-1] + bin_edges[1:])).astype(np.float32)

    print('Obtaining light sources from hypothesis')
    generic_sources = hypo_handler.get_generic_sources(sim['mc_true_params'])
    pegleg_sources = hypo_handler.get_pegleg_sources(sim['mc_true_params'])
    scaling_sources = hypo_handler.get_scaling_sources(sim['mc_true_params'])

    # TODO: enable using pegleg and scaling sources (with "ideal" choices of
    # `pegleg_idx` and `scalefactor`) to replicate the exact behavior of the
    # actual code were optimizer to fit an event perfectly
    assert len(pegleg_sources) == 0
    assert len(scaling_sources) == 0

    pegleg_idx = len(pegleg_sources)
    scalefactor = 1

    pegleg_sources = pegleg_sources[:pegleg_idx]
    scaling_sources['photons'] *= scalefactor

    all_sources = np.concatenate((
        generic_sources,
        pegleg_sources,
        scaling_sources
    ))

    print('Finding expected light in all operational DOMs at all times for those sources')
    run_info['sd_indices'] = dom_tables.loaded_sd_indices
    run_info['hit_times'] = hit_times
    time_window = np.max(bin_edges) - np.min(bin_edges)
    run_info['time_window'] = time_window

    num_doms_loaded = len(dom_tables.loaded_sd_indices)
    num_hit_times = len(hit_times)

    # Construct event hit & DOM arrays as if hits came at all times for all
    # loaded & operational DOMs
    event_hit_info = np.empty(shape=num_doms_loaded * num_hit_times,
                              dtype=EVT_HIT_INFO_T)
    event_dom_info = np.empty(shape=num_doms_loaded, dtype=EVT_DOM_INFO_T)

    copy_fields = ['sd_idx', 'x', 'y', 'z', 'quantum_efficiency', 'noise_rate_per_ns']

    hits_start_idx = 0
    event_dom_idx = 0
    for sd_idx in dom_tables.loaded_sd_indices:
        table_idx = dom_tables.sd_idx_table_indexer[sd_idx]

        hits_stop_idx = hits_start_idx + num_hit_times
        this_event_hits_info = event_hit_info[hits_start_idx:hits_stop_idx]

        this_event_hits_info['time'] = hit_times
        this_event_hits_info['charge'] = 1 # any value > 0 should be ok
        this_event_hits_info['event_dom_idx'] = event_dom_idx

        # Note we need to keep a slice of length 1 to be able to update the
        # contents of the array
        this_event_dom_info = event_dom_info[event_dom_idx:event_dom_idx+1]

        # Copy info from the dom_info array...
        this_dom_info = dom_tables.dom_info[sd_idx]

        this_event_dom_info[copy_fields] = this_dom_info[copy_fields]
        this_event_dom_info['sd_idx'] = sd_idx
        this_event_dom_info['table_idx'] = dom_tables.sd_idx_table_indexer[sd_idx]
        this_event_dom_info['hits_start_idx'] = hits_start_idx
        this_event_dom_info['hits_stop_idx'] = hits_stop_idx
        this_event_dom_info['total_observed_charge'] = num_hit_times

        hits_start_idx += num_hit_times
        event_dom_idx += 1

    dom_exp = np.zeros(shape=event_dom_info.shape)
    hit_exp = np.zeros(shape=event_hit_info.shape)

    dom_tables._pexp( # pylint: disable=protected-access
        sources=all_sources,
        sources_start=0,
        sources_stop=len(all_sources),
        event_dom_info=event_dom_info,
        event_hit_info=event_hit_info,
        tables=dom_tables.tables,
        table_norms=dom_tables.table_norms,
        t_indep_tables=dom_tables.t_indep_tables,
        t_indep_table_norms=dom_tables.t_indep_table_norms,
        dom_exp=dom_exp,
        hit_exp=hit_exp,
    )

    run_info['dom_exp'] = dom_exp
    run_info['hit_exp'] = hit_exp.reshape(num_doms_loaded, num_hit_times)
    run_info_fpath = expand(join(outdir, 'run_info.pkl'))
    print('Writing run info to "{}"'.format(run_info_fpath))
    pickle.dump(run_info, open(run_info_fpath, 'wb'), pickle.HIGHEST_PROTOCOL)

    sys.stdout.write('\n\n')
