#!/usr/bin/env python
# pylint: disable=wrong-import-position, invalid-name

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


from collections import OrderedDict
import cPickle as pickle
import hashlib
from itertools import product
from os.path import abspath, dirname, join
import re
import socket
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
import retro
from retro.discrete_hypo import DiscreteHypo
from retro.discrete_muon_kernels import const_energy_loss_muon, table_energy_loss_muon
from retro.discrete_cascade_kernels import point_cascade
from retro.table_readers import ( # pylint: disable=unused-import
    DOMTimePolarTables, TDICartTable, CLSimTables, CKVTables
)


run_info = OrderedDict([
    ('datetime', time.strftime('%Y-%m-%d %H:%M:%S')),
    ('hostname', socket.gethostname())
])


retro.DEBUG = 0
SIM_TO_TEST = 'upgoing_muon'
#CODE_TO_TEST = 'dom_time_polar_tables'
#CODE_TO_TEST = 'clsim_tables_no_dir_pdenorm'
CODE_TO_TEST = 'ckv_tables_pdenorm_dt1.0'
#CODE_TO_TEST = 'clsim_tables_no_dir_pdenorm_dedx_dt0.1'
GCD_FILE = retro.expand(retro.DETECTOR_GCD_DICT_FILE)
ANGULAR_ACCEPTANCE_FRACT = 0.338019664877
STEP_LENGTH = 1.0
MMAP = True
TIME_WINDOW = 2e3 # ns
MAKE_PLOTS = True

outdir = retro.expand(join('~/', 'dom_pdfs', SIM_TO_TEST, CODE_TO_TEST))
retro.mkdir(outdir)


run_info['sim_to_test'] = SIM_TO_TEST
run_info['gcd_file'] = GCD_FILE
run_info['gcd_file_md5'] = hashlib.md5(open(GCD_FILE, 'rb').read()).hexdigest()


# pylint: disable=line-too-long
SIMULATIONS = dict(
    upgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-400_cz-1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims10000000_step1_photon_histos.pkl'
    ),
    cascade=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=0,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='/data/icecube/retro/sims/cascade_step4_SplitUncleanedInIcePulses.pkl'
    ),
    downgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-300,
            track_azimuth=0, track_zenith=-retro.PI,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-300_cz+1_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims1000000_step1_photon_histos.pkl',
    ),
    horizontal_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-350,
            track_azimuth=0, track_zenith=0,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='/data/icecube/retro/sims/MuMinus_energy20_x0_y0_z-350_cz0_az0_ice_spice_mie_holeice_as.h2-50cm_gcd_md5_14bd15d0_geant_false_nsims100000_step1_photon_histos.pkl',
    )
)


sim = SIMULATIONS[SIM_TO_TEST]

fwd_sim_histo_file_md5 = None
if sim['fwd_sim_histo_file'] is not None:
    print('Loading and hashing forward simulation histograms from "%s"...'
          % sim['fwd_sim_histo_file'])
    t0 = time.time()
    contents = open(retro.expand(sim['fwd_sim_histo_file']), 'rb').read()
    fwd_sim_histo_file_md5 = hashlib.md5(contents).hexdigest()
    fwd_sim_histos = pickle.loads(contents)
    bin_edges = fwd_sim_histos['bin_edges']
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    print(' ', np.round(time.time() - t0, 3), 'sec\n')


run_info['sim'] = OrderedDict([
    ('mc_true_params', sim['mc_true_params']._asdict()),
    ('fwd_sim_histo_file', sim['fwd_sim_histo_file']),
    ('fwd_sim_histo_file_md5', fwd_sim_histo_file_md5)
])


strings = [86] + [36] + [79, 80, 81, 82, 83, 84, 85] + [26, 27, 35, 37, 45, 46]

doms = list(range(25, 60+1))
loaded_strings_doms = list(product(strings, doms))

hit_times = np.linspace(0, 2000, 201)

sample_hit_times = 0.5 * (hit_times[:-1] + hit_times[1:])


run_info['strings'] = strings
run_info['doms'] = doms
run_info['hit_times'] = hit_times
run_info['sample_hit_times'] = sample_hit_times
run_info['time_window'] = TIME_WINDOW


t_start = time.time()

# Load detector GCD
print(
    'Loading detector geometry, calibration, and RDE from "%s"...'
    % retro.expand(GCD_FILE)
)
t0 = time.time()
gcd = np.load(retro.expand(GCD_FILE))
geom, rde, noise_rate_hz = gcd['geo'], gcd['rde'], gcd['noise']
print(' ', np.round(time.time() - t0, 3), 'sec\n')

t0 = time.time()
if CODE_TO_TEST == 'dom_time_polar_tables':
    print('Instantiating DOMTimePolarTables...')
    norm_version = 'pde'
    tables_dir = '/data/icecube/retro_tables/full1000'
    retro_tables = DOMTimePolarTables(
        tables_dir=tables_dir,
        hash_val=None,
        geom=geom,
        use_directionality=False,
        naming_version=0,
    )
    print('Loading tables...')
    retro_tables.load_tables()


    run_info['tables_class'] = 'DOMTimePolarTables'
    run_info['tables_dir'] = tables_dir
    run_info['norm_version'] = norm_version


elif 'clsim_tables' in CODE_TO_TEST:
    use_directionality = False
    num_phi_samples = 1
    ckv_sigma_deg = 0
    norm_version = 'pde'

    try:
        norm_version = re.findall(r'(pde|avgsurfarea)norm', CODE_TO_TEST)[0]
    except (ValueError, IndexError):
        pass

    if 'no_dir' in CODE_TO_TEST:
        print('Instantiating CLSimTables (NOT using directionality), norm={}...'
              .format(norm_version))
    else:
        use_directionality = True

        try:
            ckv_sigma_deg = float(re.findall(r'sigma([0-9.]+)deg', CODE_TO_TEST)[0])
        except (ValueError, IndexError):
            pass

        try:
            num_phi_samples = int(re.findall(r'([0-9]+)phi', CODE_TO_TEST)[0])
        except (ValueError, IndexError):
            pass

        print(
            'Instantiating CLSimTables using directionality;'
            ' ckv_sigma_deg={} deg'
            ' and {} phi_dir samples; norm={}...'
            .format(ckv_sigma_deg, num_phi_samples, norm_version))

    retro_tables = CLSimTables(
        geom=geom,
        rde=rde,
        noise_rate_hz=noise_rate_hz,
        use_directionality=use_directionality,
        num_phi_samples=num_phi_samples,
        ckv_sigma_deg=ckv_sigma_deg,
        norm_version=norm_version
    )


    run_info['tables_class'] = 'CLSimTables'
    run_info['use_directionality'] = use_directionality
    run_info['num_phi_samples'] = num_phi_samples
    run_info['ckv_sigma_deg'] = ckv_sigma_deg
    run_info['norm_version'] = norm_version


    if 'single_table' in CODE_TO_TEST:
        print('Loading single table for all DOMs...')
        table_path = '/fastio/justin/retro_tables/large_5d_notilt_string_dc_depth_0-59'
        retro_tables.load_table(
            fpath=table_path,
            string='all',
            dom='all',
            step_length=STEP_LENGTH,
            angular_acceptance_fract=ANGULAR_ACCEPTANCE_FRACT,
            mmap=MMAP
        )

        run_info['tables'] = OrderedDict([
            (('all', 'all'),
             OrderedDict([
                 ('fpath', table_path),
                 ('step_length', STEP_LENGTH),
                 ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
                 ('mmap', MMAP)
             ])
            )
        ])

    else:
        print('Loading {} tables...'.format(2 * len(doms)))
        tables = OrderedDict()
        for string, dom in product(('dc', 'ic'), doms):
            depth_idx = dom - 1
            if 'orig' in CODE_TO_TEST:
                table_path = join(
                    '/fastio/justin/retro_tables/full1000_npy',
                    'full1000_{}{}'.format(string, depth_idx)
                )
            else:
                table_path = join(
                    '/data/icecube/retro_tables/large_5d_notilt_combined',
                    'large_5d_notilt_string_{:s}_depth_{:d}'.format(string, depth_idx)
                )

            retro_tables.load_table(
                fpath=table_path,
                string=string,
                dom=dom,
                step_length=STEP_LENGTH,
                angular_acceptance_fract=ANGULAR_ACCEPTANCE_FRACT,
                mmap=MMAP
            )

            tables[(string, dom)] = OrderedDict([
                ('fpath', table_path),
                ('step_length', STEP_LENGTH),
                ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
                ('mmap', MMAP)
            ])

        run_info['tables'] = tables

elif 'ckv_tables' in CODE_TO_TEST:
    norm_version = 'pde'
    try:
        norm_version = re.findall(r'(pde|avgsurfarea)norm', CODE_TO_TEST)[0]
    except (ValueError, IndexError):
        pass

    use_directionality = True
    if 'no_dir' in CODE_TO_TEST:
        use_directionality = False
        print('Instantiating CLSimTables (NOT using directionality), norm={}...'
              .format(norm_version))

    retro_tables = CKVTables(
        geom=geom,
        rde=rde,
        noise_rate_hz=noise_rate_hz,
        use_directionality=use_directionality,
        norm_version=norm_version
    )

    run_info['tables_class'] = 'CKVTables'
    run_info['use_directionality'] = use_directionality
    run_info['norm_version'] = norm_version

    print('Loading {} tables...'.format(2 * len(doms)))
    tables = OrderedDict()
    loaded_strings_doms = []
    for string, dom in product(('dc', 'ic'), doms):
        if string == 'ic':
            subdet_strings = list(range(79))
        elif string == 'dc':
            subdet_strings = list(range(79, 86 + 1))

        depth_idx = dom - 1
        table_path = join(
            '/data/icecube/retro_tables/large_5d_notilt_combined',
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
        except ValueError:
            print('Could not load table for ({}, {}), skipping.'
                  .format(string, dom))
        else:
            loaded_strings_doms.extend([(s, dom) for s in subdet_strings if s in strings])

        tables[(string, dom)] = OrderedDict([
            ('fpath', table_path),
            ('step_length', STEP_LENGTH),
            ('angular_acceptance_fract', ANGULAR_ACCEPTANCE_FRACT),
            ('mmap', MMAP)
        ])

    run_info['tables'] = tables

else:
    raise ValueError(CODE_TO_TEST)

print(' ', np.round(time.time() - t0, 3), 'sec\n')


if 'dedx' in CODE_TO_TEST:
    muon_kernel = table_energy_loss_muon
    muon_kernel_label = 'table_energy_loss_muon'
else:
    muon_kernel = const_energy_loss_muon
    muon_kernel_label = 'const_energy_loss_muon'

print('Generating source photons from "point_cascade" + "{}" kernels'.format(muon_kernel_label))
print('  fed with MC-true parameters:\n ', sim['mc_true_params'])
t0 = time.time()

dt = 1.0
try:
    dt = float(re.findall(r'dt([0-9.]+)', CODE_TO_TEST)[0])
except (ValueError, IndexError):
    pass
print('Generating track hypo (if present) with dt={}'.format(dt))

kernel_kwargs = [dict(), dict(dt=dt)]

discrete_hypo = DiscreteHypo(
    hypo_kernels=[point_cascade, muon_kernel],
    kernel_kwargs=kernel_kwargs
)
pinfo_gen = discrete_hypo.get_pinfo_gen(sim['mc_true_params'])


run_info['hypo_class'] = 'DiscreteHypo'
run_info['hypo_kernels'] = ['point_cascade', muon_kernel_label]
run_info['kernel_kwargs'] = kernel_kwargs


print(' ', np.round(time.time() - t0, 3), 'sec\n')


msg = 'Running test "{}" on "{}" sim'.format(CODE_TO_TEST, SIM_TO_TEST)
print('\n' + '='*len(msg))
print(msg)
print('='*len(msg) + '\n')

unique_strings = sorted(set(sd[0] for sd in loaded_strings_doms))
unique_doms = sorted(set(sd[1] for sd in loaded_strings_doms))
print('Getting expectations for {} strings: {}'.format(len(unique_strings), unique_strings))
print('  ... and {} DOMs: {}'.format(len(unique_doms), unique_doms))
t0 = time.time()



results = OrderedDict()


pexp_timings = []
pgen_count = 0
hypo_count = 0
total_p = 0
prev_string = -1
n_source_points = pinfo_gen.shape[0]
#for string, dom in product(unique_strings, unique_doms):
for dom, string in product(unique_doms, unique_strings):
    #if string != prev_string:
    #    prev_string = string
    #    print('String {} ({} DOMs)'.format(string, len(doms)))
    sys.stdout.write('OMKey: ({:2s}, {:2s})'.format(str(string), str(dom)))
    #sys.stdout.write('  DOM {}'.format(dom))
    t00 = time.time()

    pexp_at_hit_times = []
    for hit_time in np.nditer(sample_hit_times):
        exp_p_at_all_t, exp_p_at_hit_t = retro_tables.get_photon_expectation(
            pinfo_gen=pinfo_gen,
            hit_time=hit_time,
            time_window=TIME_WINDOW,
            string=string,
            dom=dom,
        )
        pexp_at_hit_times.append(exp_p_at_hit_t)
    t11 = time.time() - t00
    pexp_timings.append(t11)
    hypo_count += sample_hit_times.size
    pgen_count += sample_hit_times.size * n_source_points

    pexp_at_hit_times = np.array(pexp_at_hit_times)
    tot_retro = np.sum(pexp_at_hit_times)

    results[(string, dom)] = OrderedDict([
        ('exp_p_at_all_t', exp_p_at_all_t),
        ('pexp_at_hit_times', pexp_at_hit_times)
    ])

    msg = (
        '{:12.3f} ns ({:.2e} hypos computed, w/ total of {:.2e} source points)'
        .format(np.round(np.sum(pexp_timings)/pgen_count * 1e9, 3), hypo_count, pgen_count)
    )
    sys.stdout.write('  (running avg time per source point per hit DOM: {})\n'.format(msg))
    #sys.stdout.write('  ' + '\b'*len(msg) + msg)

    if MAKE_PLOTS:
        plt.clf()
        plt.plot(sample_hit_times, pexp_at_hit_times, label='Retro')
    tot_clsim = 0.0
    try:
        fwd_sim_histo = np.nan_to_num(fwd_sim_histos['results'][string][dom])
        tot_clsim = np.sum(fwd_sim_histo)
        if MAKE_PLOTS:
            plt.plot(sample_hit_times, fwd_sim_histo, label='CLSim fwd sim')
    except KeyError:
        pass

    # Don't plot if both are 0
    if tot_clsim == 0 and tot_retro == 0:
        continue

    a_text = AnchoredText(
        '{sum} Retro t-dep = {retro:.5f}      {sum} Retro / {sum} CLSim = {ratio:.5f}\n'
        '{sum} CLSim       = {clsim:.5f}\n'
        'Retro t-indep = {exp_p_at_all_t:.5f}\n'
        .format(
            sum=r'$\Sigma$',
            retro=tot_retro,
            clsim=tot_clsim,
            ratio=tot_retro/tot_clsim if tot_clsim != 0 else np.nan,
            exp_p_at_all_t=exp_p_at_all_t
        ),
        loc=2,
        prop=dict(family='monospace', size=10),
        frameon=False,
    )
    if MAKE_PLOTS:
        ax = plt.gca()
        ax.add_artist(a_text)

        ax.set_xlim(np.min(hit_times), np.max(hit_times))
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
        fpath = join(outdir, fname + '.png')
        print('Saving "{}"'.format(fpath))
        plt.savefig(fpath)

print(
    'total of {} unique DOMs'
    .format(len(list(product(unique_doms, unique_strings))))
)

run_info['results'] = results


run_info_fpath = retro.expand(join(outdir, 'run_info.pkl'))
pickle.dump(run_info, open(run_info_fpath, 'wb'), pickle.HIGHEST_PROTOCOL)


sys.stdout.write('\n\n')
print(' ', 'Time to compute and plot:')
print(' ', np.round(time.time() - t0, 3), 'sec\n')

print('Body of script took {:.3f} sec'.format(time.time() - t_start))
