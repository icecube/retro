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
from retro.hypo.discrete_cascade_kernels import point_cascade
from retro.i3info.extract_gcd import extract_gcd
from retro.tables.dom_time_polar_tables import DOMTimePolarTables
from retro.tables.tdi_cart_tables import TDICartTable
from retro.tables.retro_5d_tables import Retro5DTables
from retro.utils.misc import expand, mkdir


hostname = socket.gethostname()
run_info = OrderedDict([
    ('datetime', time.strftime('%Y-%m-%d %H:%M:%S')),
    ('hostname', hostname)
])

if hostname in ['schwyz', 'uri', 'unterwalden', 'luzern']:
    fwd_sim_dir = '/data/icecube/retro/sims/'
    dom_time_polar_tables_basedir = '/data/icecube/retro_tables/full1000'
    single_table_path = '/data/icecube/retro_tables/large_5d_notilt_string_dc_depth_0-59'
    orig_table_basedir = None
    combined_tables_basedir = '/data/icecube/retro_tables/large_5d_notilt_combined'
    if hostname == 'luzern':
        ckv_tables_basedir = '/fastio2/icecube/retro/tables'
    elif hostname in ['schwyz', 'uri', 'unterwalden']:
        ckv_tables_basedir = '/data/icecube/retro_tables/large_5d_notilt_combined'
elif 'aci.ics.psu.edu' in hostname:
    fwd_sim_dir = '/gpfs/group/dfc13/default/sim/retro'
    dom_time_polar_tables_basedir = None
    single_table_path = None
    orig_table_basedir = None
    combined_tables_basedir = '/gpfs/scratch/jll1062/retro_tables'
    ckv_tables_basedir = None
else:
    raise ValueError('Unhandled HOSTNAME="{}" for using CLSimTables'
                     .format(hostname))


# One of the keys from SIMULATIONS dict, below
SIM_TO_TEST = 'upgoing_muon'

# One of {'raw_uncompr', 'ckv_uncompr', 'ckv_templ_compr', 'dom_time_polar'}
TABLE_KIND = 'ckv_uncompr'

# One of {'pde', 'binvol', 'binvol2', 'avgsurfarea', 'wtf', 'wtf2', ...}
NORM_VERSION = 'wtf'

# Whether to use directionality from tables
USE_DIRECTIONALITY = True

# Use CKV_SIGMA_DEG and NUM_PHI_SAMPLES if USE_DIRECTIONALITY and TABLE_KIND is raw
CKV_SIGMA_DEG = 10
NUM_PHI_SAMPLES = 100

# Use dE/dX discrete muon kernel?
MUON_DEDX = False

# Time step (ns) for discrete muon kernel (whether or not dEdX)
MUON_DT = 1.0

ANGULAR_ACCEPTANCE_FRACT = 0.338019664877
STEP_LENGTH = 1.0
MMAP = True
MAKE_PLOTS = False

retro.DEBUG = 0

CODE_TO_TEST = (
    '{tables}_tables_{norm}norm_{no_dir_str}{cone_str}{dedx_str}dt{muon_dt:.1f}'
    .format(
        tables=TABLE_KIND,
        norm=NORM_VERSION,
        no_dir_str='no_dir_' if not USE_DIRECTIONALITY else '',
        cone_str=(
            'sigma{}deg_{}phi_'.format(CKV_SIGMA_DEG, NUM_PHI_SAMPLES)
            if USE_DIRECTIONALITY and TABLE_KIND in ['raw_uncompr', 'raw_templ_compr']
            else ''
        ),
        dedx_str='dedx_' if MUON_DEDX else '',
        muon_dt=MUON_DT
    )
)

OUTDIR = expand(join('~/', 'dom_pdfs', SIM_TO_TEST, CODE_TO_TEST))

run_info['sim_to_test'] = SIM_TO_TEST

# pylint: disable=line-too-long
SIMULATIONS = dict(
    upgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl'
    ),
    downgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    horizontal_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=np.pi/2,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos_0-4000ns_400bins.pkl',
    ),
    em_cascade=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=0,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='cascade_step4_SplitUncleanedInIcePulses.pkl'
    ),
)

sim = SIMULATIONS[SIM_TO_TEST]

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
    print('bin_edges:', bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    del contents
    print(' ', np.round(time.time() - t0, 3), 'sec\n')
else:
    bin_edges = np.linspace(0, 4000, 401)

run_info['sim'] = OrderedDict([
    ('mc_true_params', sim['mc_true_params']._asdict()),
    ('fwd_sim_histo_file', sim['fwd_sim_histo_file']),
    ('fwd_sim_histo_file_md5', fwd_sim_histo_file_md5)
])

strings = [86] + [36] + [79, 80, 81, 82, 83, 84, 85] + [26, 27, 35, 37, 45, 46]
doms = list(range(1, 60+1))
loaded_strings_doms = list(product(strings, doms))

hit_times = (0.5 * (bin_edges[:-1] + bin_edges[1:])).astype(np.float32)

run_info['strings'] = strings
run_info['doms'] = doms
run_info['hit_times'] = hit_times
time_window = np.max(bin_edges) - np.min(bin_edges)
run_info['time_window'] = time_window

t_start = time.time()

# Load detector GCD
t0 = time.time()
if sim['fwd_sim_histo_file'] is not None:
    try:
        gcd_file = fwd_sim_histos['gcd_info']['source_gcd_name']
    except (IndexError, TypeError):
        gcd_file = fwd_sim_histos['gcd_info'][0]
print('Loading detector geometry, calibration, and noise from "{}"...'
      .format(gcd_file))
gcd_info = extract_gcd(gcd_file)

copy_keys = ['source_gcd_name', 'source_gcd_md5', 'source_gcd_i3_md5']
run_info['gcd_info'] = OrderedDict()
for key in copy_keys:
    run_info['gcd_info'][key] = gcd_info[key]
geom, rde, noise_rate_hz = gcd_info['geo'], gcd_info['rde'], gcd_info['noise']
print(' {:.3f} sec\n'.format(np.round(time.time() - t0, 3)))

t0 = time.time()
if TABLE_KIND == 'dom_time_polar':
    print('Instantiating DOMTimePolarTables...')
    assert isdir(dom_time_polar_tables_basedir), str(dom_time_polar_tables_basedir)

    assert NORM_VERSION == 'pde'
    retro_tables = DOMTimePolarTables(
        tables_dir=dom_time_polar_tables_basedir,
        hash_val=None,
        geom=geom,
        use_directionality=USE_DIRECTIONALITY,
        naming_version=0,
    )
    print('Loading tables...')
    retro_tables.load_tables()

    run_info['tables_class'] = 'DOMTimePolarTables'
    run_info['tables_dir'] = dom_time_polar_tables_basedir
    run_info['norm_version'] = NORM_VERSION

elif TABLE_KIND == 'raw_uncompr':
    if not USE_DIRECTIONALITY:
        print('Instantiating CLSimTables (NOT using directionality), norm={}...'
              .format(NORM_VERSION))
        CKV_SIGMA_DEG = None
        NUM_PHI_SAMPLES = None
    else:
        print(
            'Instantiating CLSimTables using directionality;'
            ' CKV_SIGMA_DEG={} deg'
            ' and {} phi_dir samples; norm={}...'
            .format(CKV_SIGMA_DEG, NUM_PHI_SAMPLES, NORM_VERSION))

    retro_tables = Retro5DTables(
        table_kind=TABLE_KIND,
        geom=geom,
        rde=rde,
        noise_rate_hz=noise_rate_hz,
        compute_t_indep_exp=True,
        use_directionality=USE_DIRECTIONALITY,
        norm_version=NORM_VERSION,
        num_phi_samples=NUM_PHI_SAMPLES,
        ckv_sigma_deg=CKV_SIGMA_DEG
    )

    run_info['tables_class'] = 'Retro5DTables'
    run_info['table_kind'] = TABLE_KIND
    run_info['use_directionality'] = USE_DIRECTIONALITY
    run_info['num_phi_samples'] = NUM_PHI_SAMPLES
    run_info['ckv_sigma_deg'] = CKV_SIGMA_DEG
    run_info['norm_version'] = NORM_VERSION

    if 'single_table' in CODE_TO_TEST:
        print('Loading single table for all DOMs...')
        assert (isfile(single_table_path) or isdir(single_table_path)), str(single_table_path)
        retro_tables.load_table(
            fpath=single_table_path,
            string='all',
            dom='all',
            angular_acceptance_fract=ANGULAR_ACCEPTANCE_FRACT,
            mmap=MMAP
        )

        run_info['tables'] = OrderedDict([
            (('all', 'all'),
             OrderedDict([
                 ('fpath', single_table_path),
                 ('step_length', STEP_LENGTH),
                 ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
                 ('mmap', MMAP)
             ])
            )
        ])

    else:
        print('Loading {} tables...'.format(2 * len(doms)))
        tables = OrderedDict()
        loaded_strings_doms = []
        for string, dom in product(('dc', 'ic'), doms):
            depth_idx = dom - 1

            if string == 'ic':
                subdet_strings = list(range(1, 79))
            elif string == 'dc':
                subdet_strings = list(range(79, 86 + 1))

            if 'orig' in CODE_TO_TEST:
                assert isdir(orig_table_basedir), str(orig_table_basedir)
                table_path = join(
                    orig_table_basedir,
                    'full1000_{}{}'.format(string, depth_idx)
                )
            else:
                assert isdir(combined_tables_basedir), str(combined_tables_basedir)
                table_path = join(
                    combined_tables_basedir,
                    'large_5d_notilt_string_{:s}_depth_{:d}'.format(string, depth_idx)
                )

            try:
                retro_tables.load_table(
                    fpath=table_path,
                    string=string,
                    dom=dom,
                    step_length=STEP_LENGTH,
                    angular_acceptance_fract=ANGULAR_ACCEPTANCE_FRACT,
                    mmap=MMAP
                )
            except (AssertionError, ValueError) as err:
                print(err)
                tables[(string, dom)] = None
            else:
                loaded_strings_doms.extend(
                    [(s, dom) for s in subdet_strings if s in strings]
                )
                tables[(string, dom)] = OrderedDict([
                    ('fpath', table_path),
                    ('step_length', STEP_LENGTH),
                    ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
                    ('mmap', MMAP)
                ])

        run_info['tables'] = tables

elif TABLE_KIND in ['ckv_uncompr', 'ckv_templ_compr']:
    assert isdir(ckv_tables_basedir), str(ckv_tables_basedir)

    retro_tables = Retro5DTables(
        table_kind=TABLE_KIND,
        geom=geom,
        rde=rde,
        noise_rate_hz=noise_rate_hz,
        compute_t_indep_exp=True,
        use_directionality=USE_DIRECTIONALITY,
        norm_version=NORM_VERSION
    )

    run_info['tables_class'] = 'Retro5DTables'
    run_info['table_kind'] = TABLE_KIND
    run_info['use_directionality'] = USE_DIRECTIONALITY
    run_info['norm_version'] = NORM_VERSION

    print('Loading {} tables...'.format(2 * len(doms)))
    tables = OrderedDict()
    loaded_strings_doms = []
    for string, dom in product(('dc', 'ic'), doms):
        if string == 'ic':
            subdet_strings = list(range(1, 79))
        elif string == 'dc':
            subdet_strings = list(range(79, 86 + 1))

        depth_idx = dom - 1
        table_path = join(
            ckv_tables_basedir,
            'large_5d_notilt_string_{:s}_depth_{:d}'.format(string, depth_idx)
        )

        try:
            retro_tables.load_table(
                fpath=table_path,
                string=string,
                dom=dom,
                step_length=STEP_LENGTH,
                angular_acceptance_fract=ANGULAR_ACCEPTANCE_FRACT,
                mmap=MMAP
            )
        except (AssertionError, ValueError) as err:
            #print('Could not load table for ({}, {}), skipping.'
            #      .format(string, dom))
            print(err)
            #print('')
            tables[(string, dom)] = None
        else:
            loaded_strings_doms.extend(
                [(s, dom) for s in subdet_strings if s in strings]
            )
            tables[(string, dom)] = OrderedDict([
                ('fpath', table_path),
                ('step_length', STEP_LENGTH),
                ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
                ('mmap', MMAP)
            ])

    run_info['tables'] = tables

else:
    raise ValueError(TABLE_KIND)

# Sort loaded_strings_doms first by subdet (dc then ic); then by depth index
# (descending index); then by string (ascending index)
loaded_strings_doms.sort(key=lambda sd: (sd[0] < 79, -sd[1], sd[0]))

print(' ', np.round(time.time() - t0, 3), 'sec\n')


if MUON_DEDX:
    muon_kernel = table_energy_loss_muon
    muon_kernel_label = 'table_energy_loss_muon'
else:
    muon_kernel = const_energy_loss_muon
    muon_kernel_label = 'const_energy_loss_muon'

print('Generating source photons from "point_cascade" + "{}" kernels'.format(muon_kernel_label))
print('  fed with MC-true parameters:\n ', sim['mc_true_params'])
t0 = time.time()

print('Generating track hypo (if present) with dt={}'.format(MUON_DT))

kernel_kwargs = [dict(), dict(dt=MUON_DT)]

discrete_hypo = DiscreteHypo(
    hypo_kernels=[point_cascade, muon_kernel],
    kernel_kwargs=kernel_kwargs
)
mc_true_params = tuple(np.float32(val) for val in sim['mc_true_params'])
print(mc_true_params)
mc_true_params = retro.HYPO_PARAMS_T(*mc_true_params)
print(mc_true_params)
sources = discrete_hypo.get_sources(mc_true_params)

run_info['hypo_class'] = 'DiscreteHypo'
run_info['hypo_kernels'] = ['point_cascade', muon_kernel_label]
run_info['kernel_kwargs'] = kernel_kwargs

print(' ', np.round(time.time() - t0, 3), 'sec\n')

msg = 'Running test "{}" on "{}" sim'.format(CODE_TO_TEST, SIM_TO_TEST)
print('\n' + '='*len(msg))
print(msg)
print('='*len(msg) + '\n')

print('Getting expectations for {} loaded DOMs: {}'.format(len(loaded_strings_doms), loaded_strings_doms))
t0 = time.time()

results = OrderedDict()

pexp_timings = []
pgen_count = 0
hypo_count = 0
total_p = 0
prev_string = -1
n_source_points = sources.shape[0]

mkdir(OUTDIR)

#for string, dom in product(unique_strings, unique_doms):
for string, dom in loaded_strings_doms:
    sys.stdout.write('OMKey: ({:2s}, {:2s})'.format(str(string), str(dom)))
    t00 = time.time()
    exp_p_at_all_times, exp_p_at_hit_times = retro_tables.get_expected_det(
        sources=sources,
        hit_times=hit_times,
        string=string,
        dom=dom,
        include_noise=True,
        time_window=time_window
    )
    t11 = time.time() - t00
    pexp_timings.append(t11)
    hypo_count += hit_times.size
    pgen_count += hit_times.size * n_source_points

    tot_retro = np.sum(exp_p_at_hit_times)

    results[(string, dom)] = OrderedDict([
        ('exp_p_at_all_times', exp_p_at_all_times),
        ('exp_p_at_hit_times', exp_p_at_hit_times)
    ])

    msg = (
        '{:12.0f} ns ({:.2e} hypos computed, w/ total of {:.2e} source points)'
        .format(np.round(np.sum(pexp_timings)/pgen_count * 1e9, 3), hypo_count, pgen_count)
    )
    sys.stdout.write('  (running avg time per source point per hit DOM: {})\n'.format(msg))
    #sys.stdout.write('  ' + '\b'*len(msg) + msg)

    if MAKE_PLOTS:
        plt.clf()
        plt.plot(hit_times, exp_p_at_hit_times, label='Retro')
    tot_clsim = 0.0
    try:
        fwd_sim_histo = np.nan_to_num(fwd_sim_histos['results'][(string, dom)])
        tot_clsim = np.sum(fwd_sim_histo)
        if MAKE_PLOTS:
            plt.plot(hit_times, fwd_sim_histo, label='CLSim fwd sim')
    except KeyError:
        pass

    # Don't plot if both are 0
    if tot_clsim == 0 and tot_retro == 0:
        continue

    a_text = AnchoredText(
        '{sum} Retro t-dep = {retro:.5f}      {sum} Retro / {sum} CLSim = {ratio:.5f}\n'
        '{sum} CLSim       = {clsim:.5f}\n'
        'Retro t-indep = {exp_p_at_all_times:.5f}\n'
        .format(
            sum=r'$\Sigma$',
            retro=tot_retro,
            clsim=tot_clsim,
            ratio=tot_retro/tot_clsim if tot_clsim != 0 else np.nan,
            exp_p_at_all_times=exp_p_at_all_times
        ),
        loc=2,
        prop=dict(family='monospace', size=10),
        frameon=False,
    )
    if MAKE_PLOTS:
        ax = plt.gca()
        ax.add_artist(a_text)

        ax.set_xlim(np.min(bin_edges), np.max(bin_edges))
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title('String {}, DOM {}'.format(string, dom))
        ax.set_xlabel('time (ns)')
        ax.legend(loc='center left', frameon=False)

        clsim_code = 'c' if tot_clsim > 0 else ''
        retro_code = 'r' if tot_retro > 0 else ''

        fname = (
            'sim_{hypo}_code_{code}_{string}_{dom}_{retro_code}_{clsim_code}'
            .format(
                hypo=SIM_TO_TEST, code=CODE_TO_TEST, string=string, dom=dom,
                retro_code=retro_code, clsim_code=clsim_code
            )
        )
        fpath = join(OUTDIR, fname + '.png')
        print('Saving "{}"'.format(fpath))
        plt.savefig(fpath)

print('total of {} unique DOMs'.format(len(loaded_strings_doms)))

run_info['results'] = results

run_info_fpath = expand(join(OUTDIR, 'run_info.pkl'))
print('Writing run info to "{}"'.format(run_info_fpath))
pickle.dump(run_info, open(run_info_fpath, 'wb'), pickle.HIGHEST_PROTOCOL)

sys.stdout.write('\n\n')
print(' ', 'Time to compute and plot:')
print(' ', np.round(time.time() - t0, 3), 'sec\n')

print('Body of script took {:.3f} sec'.format(time.time() - t_start))
