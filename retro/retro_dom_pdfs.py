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


import cPickle as pickle
from itertools import product
from os.path import abspath, dirname, join
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
from retro.discrete_muon_kernels import const_energy_loss_muon
from retro.discrete_cascade_kernels import point_cascade
from retro.table_readers import DOMTimePolarTables, TDICartTable, CLSimTables # pylint: disable=unused-import


retro.DEBUG = 1

# pylint: disable=line-too-long
SIMULATIONS = dict(
    upgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=np.pi,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file='~/src/retro/data/benchmark.pkl'
        #fwd_sim_histo_file='/home/peller/retro/retro/testMuMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=0.0_azimuth=0.0_events.pkl'
    ),
    cascade=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=0,
            track_energy=0, cascade_energy=20
        ),
        fwd_sim_histo_file='benchmarkEMinus_E=20.0_x=0.0_y=0.0_z=-400.0_coszen=-1.0_azimuth=0.0.pkl'
    ),
    downgoing_muon=dict(
        mc_true_params=retro.HYPO_PARAMS_T(
            t=0, x=0, y=0, z=-400,
            track_azimuth=0, track_zenith=-retro.PI,
            track_energy=20, cascade_energy=0
        ),
        fwd_sim_histo_file=None
    )
)

SIM_TO_TEST = 'upgoing_muon'
#CODE_TO_TEST = 'dom_time_polar_tables'
#CODE_TO_TEST = 'clsim_tables_no_dir'
CODE_TO_TEST = 'orig_clsim_tables_my_norm_rndtheta10deg'
NUM_PHI_SAMPLES = 100


sim = SIMULATIONS[SIM_TO_TEST]
strings = [36] + [79, 80, 81, 82, 83, 84, 85, 86] + [26, 27, 35, 37, 45, 46]
#strings = [35, 36]
doms = list(range(25, 60))
norm = False
norm2 = False
hit_times = np.linspace(0, 2000, 201)

t_start = time.time()

# Load detector geometry array
print('Loading detector geometry from "%s"...' % retro.expand(retro.DETECTOR_GEOM_FILE))
t0 = time.time()
detector_geometry = np.load(retro.expand(retro.DETECTOR_GEOM_FILE))
print(' ', np.round(time.time() - t0, 3), 'sec\n')

t0 = time.time()
if CODE_TO_TEST == 'dom_time_polar_tables':
    print('Instantiating DOMTimePolarTables...')
    retro_tables = DOMTimePolarTables(
        tables_dir='/data/icecube/retro_tables/full1000/',
        hash_val=None,
        geom=detector_geometry,
        use_directionality=False,
        naming_version=0,
    )
    print('Loading tables...')
    retro_tables.load_tables()

elif 'clsim_tables' in CODE_TO_TEST:
    use_directionality = False if 'no_dir' in CODE_TO_TEST else True
    if use_directionality:
        print('Instantiating CLSimTables (using directionality)...')
    else:
        print('Instantiating CLSimTables (NOT using directionality)...')

    retro_tables = CLSimTables(
        geom=detector_geometry,
        use_directionality=use_directionality,
        num_phi_samples=NUM_PHI_SAMPLES
    )

    if 'single_table' in CODE_TO_TEST:
        print('Loading single table...')
        table_path = '/fastio/justin/retro_tables/large_5d_notilt_string_dc_depth_0-59'
        retro_tables.load_table(
            fpath=table_path,
            string='all',
            depth_idx='all',
            step_length=1,
            angular_acceptance_fract=0.338019664877,
            quantum_efficiency=1,
            mmap=True
        )
    else:
        print('Loading {} tables...'.format(2 * len(doms)))
        for string, dom in product(('dc', 'ic'), doms):
            depth_idx = dom - 1
            if 'orig' in CODE_TO_TEST:
                table_path = join(
                    '/fastio/justin//retro_tables/full1000_npy',
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
                depth_idx=depth_idx,
                step_length=1,
                angular_acceptance_fract=0.338019664877,
                quantum_efficiency=1,
                mmap=True
            )
else:
    raise ValueError(CODE_TO_TEST)
print(' ', np.round(time.time() - t0, 3), 'sec\n')


print('Loading forward simulation histograms from "%s"...' % sim['fwd_sim_histo_file'])
t0 = time.time()
fwd_sim_histos = pickle.load(open(retro.expand(sim['fwd_sim_histo_file']), 'rb'))
print(' ', np.round(time.time() - t0, 3), 'sec\n')

print('Generating source photons from "point_cascade" + "const_energy_loss_muon" kernels')
print('  fed with MC-true parameters:\n ', sim['mc_true_params'])
t0 = time.time()
discrete_hypo = DiscreteHypo(hypo_kernels=[point_cascade, const_energy_loss_muon])
pinfo_gen = discrete_hypo.get_pinfo_gen(sim['mc_true_params'])
print(' ', np.round(time.time() - t0, 3), 'sec\n')

print('Getting expectations for {} strings: {}'.format(len(strings), strings))
print('  ... and {} DOMs: {}'.format(len(doms), doms))
t0 = time.time()

sample_hit_times = 0.5 * (hit_times[:-1] + hit_times[1:])

pexp_timings = []
pgen_count = 0
total_p = 0
print('Average time to compute expected photons at DOM, per hit:')
sys.stdout.write(' '*(12+3))
for string, dom in product(strings, doms):
    t00 = time.time()
    depth_idx = dom - 1
    pexp_at_hit_times = []
    for hit_time in sample_hit_times.flat:
        exp_p_at_all_t, exp_p_at_hit_t = retro_tables.get_photon_expectation(
            pinfo_gen=pinfo_gen,
            hit_time=hit_time,
            string=string,
            depth_idx=depth_idx,
        )
        pexp_at_hit_times.append(exp_p_at_hit_t)
    pexp_timings.append(time.time() - t00)
    pgen_count += sample_hit_times.size

    pexp_at_hit_times = np.array(pexp_at_hit_times)
    tot_retro = np.sum(pexp_at_hit_times)
    if norm:
        pexp_at_hit_times /= tot_retro

    msg = '{:12.3f} ms'.format(np.round(np.mean(pexp_timings) * 1e3, 3))
    sys.stdout.write('\b'*len(msg) + msg)

    plt.clf()
    plt.plot(sample_hit_times, pexp_at_hit_times, label='Retro')
    tot_clsim = 0.0
    try:
        fwd_sim_histo = np.nan_to_num(fwd_sim_histos[string][dom])
        tot_clsim = np.sum(fwd_sim_histo)
        if norm:
            fwd_sim_histo /= np.sum(fwd_sim_histo)
        #if norm2:
        #    fwd_sim_histo *= 200
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
    ax = plt.gca()
    ax.add_artist(a_text)

    ax.set_xlim(np.min(hit_times), np.max(hit_times))
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_title('String {}, DOM {}'.format(string, dom))
    ax.set_xlabel('time (ns)')
    ax.legend(loc='center left', frameon=False)

    clsim_code = 'c' if tot_clsim > 0 else ''
    retro_code = 'r' if tot_retro > 0 else ''

    outdir = join('dom_pdfs', SIM_TO_TEST)
    retro.mkdir(outdir)
    fname = (
        'retro_{code}_{string}_{dom}_{retro_code}_{clsim_code}'
        .format(code=CODE_TO_TEST, string=string, dom=dom,
                retro_code=retro_code, clsim_code=clsim_code)
    )
    plt.savefig(join(outdir, fname + '.png'))

sys.stdout.write('\n\n')
print(' ', 'Time to compute and plot:')
print(' ', np.round(time.time() - t0, 3), 'sec\n')

print('Body of script took {:.3f} sec'.format(time.time() - t_start))
