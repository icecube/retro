#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Instantiate Retro tables and find the max over the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['run_multinest', 'reco', 'parse_args']

__author__ = 'J.L. Lanfranchi'
__license__ = '''Copyright 2017 Justin L. Lanfranchi

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
from copy import deepcopy
from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np
import pymultinest

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, LLHP_T, const
from retro.const import PI, TWO_PI
from retro.utils.misc import expand, mkdir
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo.discrete_cascade_kernels import (
    point_cascade
)
from retro.hypo.discrete_muon_kernels import (
    const_energy_loss_muon, table_energy_loss_muon
)
from retro.i3info.extract_gcd import extract_gcd
from retro.likelihood import get_neg_llh
from retro.tables.retro_5d_tables import (
    NORM_VERSIONS, TABLE_KINDS, Retro5DTables
)


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )
    parser.add_argument(
        '--n-events', type=int, default=None
    )
    parser.add_argument(
        '--start-event-idx', type=int, default=0
    )

    #for dim in HYPO_PARAMS_T._fields:
    #    parser.add_argument(
    #        '--{}'.format(dim.replace('_', '-')), nargs='+', required=True,
    #        help='''Hypothses will take this(these) value(s) for dimension
    #        {dim_hr}. Specify a single value to not scan over this dimension;
    #        specify a human-readable string of values, e.g. '0, 0.5, 1-10:0.2'
    #        scans 0, 0.5, and from 1 to 10 (inclusive of both endpoints) with
    #        stepsize of 0.2.'''.format(dim_hr=dim.replace('_', ' '))
    #    )

    parser.add_argument(
        '--hits', required=True,
    )
    parser.add_argument(
        '--hits-are-photons', action='store_true',
    )
    #parser.add_argument(
    #    '--time-window', type=float, required=True,
    #)
    parser.add_argument(
        '--angsens-model',
        choices='nominal  h1-100cm  h2-50cm  h3-30cm'.split()
    )

    parser.add_argument(
        '--cascade-kernel', choices=['point', 'one_dim'], required=True,
    )
    parser.add_argument(
        '--cascade-samples', type=int, default=1,
    )
    parser.add_argument(
        '--track-kernel', required=True,
        choices=['const_e_loss', 'nonconst_e_loss'],
    )
    parser.add_argument(
        '--track-time-step', type=float, required=True,
    )

    parser.add_argument(
        '--dom-tables-fname-proto', required=True,
        help='''Must have one of the brace-enclosed fields "{string}" or
        "{subdet}", and must have one of "{dom}" or "{depth_idx}". E.g.:
        "my_tables_{subdet}_{depth_idx}"'''
    )
    parser.add_argument(
        '--step-length', type=float, default=1.0,
        help='''Step length used in the CLSim table generator.'''
    )
    parser.add_argument(
        '--force-no-mmap', action='store_true',
        help='''Specify to NOT memory map the tables. If not specified, a
        sensible default is chosen for the type of tables being used.'''
    )
    parser.add_argument(
        '--dom-table-kind', choices=TABLE_KINDS, required=True,
        help='''Kind of single-DOM table to use.'''
    )
    parser.add_argument(
        '--gcd', required=True,
        help='''IceCube GCD file; can either specify an i3 file, or the
        extracted pkl file used in Retro.'''
    )
    parser.add_argument(
        '--norm-version', choices=NORM_VERSIONS, required=True,
        help='''Norm version.'''
    )
    parser.add_argument(
        '--no-dir', action='store_true',
        help='''Do NOT use source photon directionality'''
    )
    parser.add_argument(
        '--num-phi-samples', type=int, default=None,
    )
    parser.add_argument(
        '--ckv-sigma-deg', type=float, default=None,
    )
    parser.add_argument(
        '--tdi-table', default=None
    )
    parser.add_argument(
        '--template-library', default=None
    )

    return parser.parse_args()


def run_multinest(
        outdir,
        event_idx,
        neg_llh_kw,
        t_lims,
        #x_lims=(-600, 600),
        #y_lims=(-550, 550),
        #z_lims=(-550, 550),
        x_lims=(-150, 250),
        y_lims=(-200, 150),
        z_lims=(-550, 0),
    ):

    priors = OrderedDict([
        ('t', ('uniform', t_lims)),
        ('x', ('uniform', x_lims)),
        ('y', ('uniform', y_lims)),
        ('z', ('uniform', z_lims)),
        ('z', ('uniform', z_lims)),
        ('track_zenith', ('uniform', (0, np.pi))),
        ('track_azimuth', ('uniform', (0, 2*np.pi))),
        ('total_energy', ('log_uniform', (0, 3))),
        ('track_fraction', ('uniform', (0, 1)))
    ])

    t_min = np.min(t_lims)
    x_min = np.min(x_lims)
    y_min = np.min(y_lims)
    z_min = np.min(z_lims)

    t_range = np.max(t_lims) - t_min
    x_range = np.max(x_lims) - x_min
    y_range = np.max(y_lims) - y_min
    z_range = np.max(z_lims) - z_min

    param_values = []
    log_likelihoods = []

    def prior(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function for pymultinest to translate the hypercube MultiNest uses
        (each value is in [0, 1]) into the dimensions of the parameter space.

        """
        # Note: track_fraction needs no xform, as it is in [0, 1]

        # t, x, y, z
        cube[0] = cube[0] * t_range + t_min
        cube[1] = cube[1] * x_range + x_min
        cube[2] = cube[2] * y_range + y_min
        cube[3] = cube[3] * z_range + z_min

        # zenith
        cube[4] *= PI

        # azimuth
        cube[5] *= TWO_PI

        # energy: log uniform prior between 10^0 and 10^3
        cube[6] = 10**(cube[6]*3)

    t_start = []

    def loglike(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function pymultinest calls to get llh values.

        Note that this is called _after_ `prior` has been called, so `cube`
        alsready contains the parameter values scaled to be in their physical
        ranges.

        """
        if not t_start:
            t_start.append(time.time())

        t0 = time.time()
        hypo_params = HYPO_PARAMS_T(
            t=cube[0],
            x=cube[1],
            y=cube[2],
            z=cube[3],
            track_zenith=cube[4],
            track_azimuth=cube[5],
            cascade_energy=cube[6]*(1 - cube[7]),
            track_energy=cube[6]*cube[7]
        )
        llh = -get_neg_llh(hypo_params, **neg_llh_kw)
        t1 = time.time()

        param_values.append(hypo_params)
        log_likelihoods.append(llh)

        n_calls = len(log_likelihoods)

        if n_calls % 200 == 0:
            t_now = time.time()
            best_idx = np.argmax(log_likelihoods)
            best_llh = log_likelihoods[best_idx]
            best_p = param_values[best_idx]
            print(
                'best llh = {:.3f} @ (t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f}, zen={:.1f} deg, az={:.1f} deg, Etrk={:.1f}, Ecscd={:.1f})'
                .format(
                    best_llh, best_p.t, best_p.x, best_p.y, best_p.z,
                    np.rad2deg(best_p.track_zenith),
                    np.rad2deg(best_p.track_azimuth),
                    best_p.track_energy,
                    best_p.cascade_energy
                )
            )
            print('{} LLH computed'.format(n_calls))
            print('avg time per llh: {:.3f} ms'.format((t_now - t_start[0])/n_calls*1000))
            print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
            print('')

        return llh

    n_dims = len(HYPO_PARAMS_T._fields)

    mn_kw = OrderedDict([
        ('n_clustering_params', n_dims),
        ('n_live_points', 160),
        ('evidence_tolerance', 0.1),
        ('sampling_efficiency', 0.8),
        ('max_modes', 10),
        ('seed', 0),
        ('max_iter', 100000),
    ])

    mn_meta = OrderedDict([
        ('params', HYPO_PARAMS_T._fields),
        ('priors', priors),
        ('kwargs', mn_kw),
    ])

    outdir = expand(outdir)
    mkdir(outdir)
    outf = join(outdir, 'evt{}-multinest_meta.pkl'.format(event_idx))
    print('Saving MultiNest metadata to "{}"'.format(outf))
    pickle.dump(
        mn_meta,
        open(outf, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    outname = '%s/evt%i-'%(outdir, event_idx)
    pymultinest.run(
        LogLikelihood=loglike,
        Prior=prior,
        n_dims=n_dims,
        n_params=n_dims,
        verbose=False,
        outputfiles_basename=outname,
        resume=False,
        **mn_kw
    )

    return param_values, log_likelihoods


def reco():
    """Script "main" function"""
    t00 = time.time()

    args = parse_args()
    kwargs = vars(args)
    orig_kwargs = deepcopy(kwargs)

    # -- Instantiate hypo class with kernels -- #
    print('Instantiating hypo object & kernels')
    t0 = time.time()

    hypo_kernels = []
    kernel_kwargs = []

    cascade_kernel = kwargs.pop('cascade_kernel')
    cascade_samples = kwargs.pop('cascade_samples')
    if cascade_kernel == 'point':
        hypo_kernels.append(point_cascade)
        kernel_kwargs.append(dict())
    else:
        raise NotImplementedError('{} cascade not implemented yet.'
                                  .format(cascade_kernel))
        #hypo_kernels.append(one_dim_cascade)
        #kernel_kwargs.append(dict(num_samples=cascade_samples))

    track_kernel = kwargs.pop('track_kernel')
    if track_kernel == 'const_e_loss':
        hypo_kernels.append(const_energy_loss_muon)
    else:
        hypo_kernels.append(table_energy_loss_muon)
    kernel_kwargs.append(dict(dt=kwargs.pop('track_time_step')))

    hypo_handler = DiscreteHypo(
        hypo_kernels=hypo_kernels,
        kernel_kwargs=kernel_kwargs
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # -- Instantiate tables class and load tables -- #
    print('Instantiating tables object')
    t0 = time.time()

    dom_table_kind = kwargs.pop('dom_table_kind')
    angsens_model = kwargs.pop('angsens_model')
    norm_version = kwargs.pop('norm_version')
    num_phi_samples = kwargs.pop('num_phi_samples')
    ckv_sigma_deg = kwargs.pop('ckv_sigma_deg')
    dom_tables_fname_proto = kwargs.pop('dom_tables_fname_proto')
    step_length = kwargs.pop('step_length')
    force_no_mmap = kwargs.pop('force_no_mmap')
    if force_no_mmap:
        mmap = False
    else:
        mmap = 'uncompr' in dom_table_kind

    use_directionality = not kwargs.pop('no_dir')

    tdi_table = kwargs.pop('tdi_table')
    if tdi_table is not None:
        raise NotImplementedError('TDI table not handled yet')

    if dom_table_kind in ['raw_templ_compr', 'ckv_templ_compr']:
        template_library = np.load(args.template_library)
    else:
        template_library = None

    compute_t_indep_exp = tdi_table is None

    gcd = extract_gcd(kwargs.pop('gcd'))

    # Instantiate single-DOM tables
    dom_tables = Retro5DTables(
        table_kind=dom_table_kind,
        geom=gcd['geo'],
        rde=gcd['rde'],
        noise_rate_hz=gcd['noise'],
        angsens_model=angsens_model,
        compute_t_indep_exp=compute_t_indep_exp,
        use_directionality=use_directionality,
        norm_version=norm_version,
        num_phi_samples=num_phi_samples,
        ckv_sigma_deg=ckv_sigma_deg,
        template_library=template_library,
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # Load single-DOM tables
    print('Loading single-DOM tables')
    t0 = time.time()

    if '{subdet' in dom_tables_fname_proto:
        for subdet in ['ic', 'dc']:
            if subdet == 'ic':
                subdust_doms = const.IC_SUBDUST_DOMS
                strings = const.DC_IC_STRS
            else:
                subdust_doms = const.DC_SUBDUST_DOMS
                strings = const.DC_STRS

            for dom in subdust_doms:
                fpath = dom_tables_fname_proto.format(
                    subdet=subdet, dom=dom, depth_idx=dom-1
                )
                sd_indices = [const.get_sd_idx(string, dom) for string in strings]

                dom_tables.load_table(
                    fpath=fpath,
                    sd_indices=sd_indices,
                    step_length=step_length,
                    mmap=mmap
                )
    elif '{string' in dom_tables_fname_proto:
        raise NotImplementedError()
        #for string, dom in product(range(1, 86+1), range(1, 60+1)):
        #    fpath = dom_tables_fname_proto.format(
        #        string=string, string_idx=string - 1,
        #        dom=dom, depth_idx=dom - 1
        #    )
        #    dom_tables.load_table(
        #        fpath=fpath,
        #        string=string,
        #        dom=dom,
        #        **common_kw
        #    )

    table_t_max = dom_tables.pexp_meta['binning_info']['t_max']

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    # -- Load hits -- #
    print('Loading hits')
    t0 = time.time()

    hits_are_photons = kwargs.pop('hits_are_photons')

    hits_file = expand(kwargs['hits'])
    with open(hits_file, 'rb') as f:
        hits = pickle.load(f)

    outdir = kwargs.pop('outdir')
    mkdir(outdir)

    start_event_idx = kwargs.pop('start_event_idx')
    n_events = kwargs.pop('n_events')
    stop_event_idx = None if n_events is None else start_event_idx + n_events
    events_slice = slice(start_event_idx, stop_event_idx)

    # Keyword args for the `metric` callable (get_neg_llh)
    neg_llh_kw = dict(
        time_window=None,
        hypo_handler=hypo_handler,
        dom_tables=dom_tables,
        tdi_table=tdi_table
    )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Scanning paramters')
    t0 = time.time()

    best_llh_vals = []
    best_llh_params = []
    n_points = 0

    for event_ofst, event_hits in enumerate(hits[events_slice]):
        event_idx = start_event_idx + event_ofst
        t1 = time.time()
        first_hit_t = np.inf
        last_hit_t = -np.inf
        if hits_are_photons:
            # For photons, we assign a "charge" from their weight, which comes
            # from angsens model.
            event_photons = event_hits
            # DEBUG: set back to EMPTY_HITS when not debugging!
            event_hits = [EMPTY_HITS]*const.NUM_DOMS_TOT
            #event_hits = [None]*const.NUM_DOMS_TOT
            for str_dom, pinfo in event_photons.items():
                sd_idx = const.get_sd_idx(string=str_dom[0], dom=str_dom[1])
                t = pinfo[0, :]
                coszen = pinfo[4, :]
                weight = np.float32(dom_tables.angsens_poly(coszen))
                event_hits[sd_idx] = np.concatenate(
                    (t[np.newaxis, :], weight[np.newaxis, :]),
                    axis=0
                )
                first_hit_t = min(first_hit_t, t.min())
                last_hit_t = max(last_hit_t, t.max())

        neg_llh_kw['hits'] = event_hits

        t_lims = (first_hit_t - table_t_max + 100, last_hit_t)
        #t_lims = (-100, 100)
        t_range = last_hit_t - first_hit_t + table_t_max
        neg_llh_kw['time_window'] = t_range

        param_values, log_likelihoods = run_multinest(
            outdir=outdir,
            event_idx=event_idx,
            neg_llh_kw=neg_llh_kw,
            t_lims=t_lims
        )

        log_likelihoods = np.array(log_likelihoods)
        max_idx = np.argmax(log_likelihoods)

        bllh = log_likelihoods[max_idx]
        bparam = param_values[max_idx]

        best_llh_vals.append(bllh)
        best_llh_params.append(bparam)

        #for llh, p in zip(log_likelihoods, param_values):
        #    print('llh={}, {}'.format(llh, HYPO_PARAMS_T(*p)))

        print(bllh, bparam)

        #llhp = np.concatenate(
        #    [log_likelihoods[:, np.newaxis], param_values],
        #    axis=1
        #).astype(np.float16)
        llhp = np.empty(shape=len(param_values), dtype=LLHP_T)
        for idx, (llh, p) in enumerate(zip(log_likelihoods, param_values)):
            llhp[idx] = (llh,) + p

        outfpath = join(outdir, 'evt{:d}-llhp.npy'.format(event_idx))
        print('Saving llhp to "{}"...'.format(outfpath))
        np.save(outfpath, llhp)

        dt = time.time() - t1
        n_points += log_likelihoods.size
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt/n_points*1e3))

    kwargs.pop('hits')
    info = OrderedDict([
        ('hypo_params', HYPO_PARAMS_T._fields),
        ('metric_name', 'neg_llh'),
        ('best_params', best_llh_params),
        ('best_metric_vals', best_llh_vals),
    ])

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    t0 = time.time()
    outfpath = join(outdir, 'minimization_info.pkl')
    print('Saving results to pickle file at "{}"'.format(outfpath))
    pickle.dump(info, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Total script run time is {:.3f} s'.format(time.time() - t00))

    return best_llh_params, best_llh_vals, orig_kwargs


if __name__ == '__main__':
    best_llh_params, best_llh_vals, orig_kwargs = reco() # pylint: disable=invalid-name
