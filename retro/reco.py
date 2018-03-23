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
from retro import HYPO_PARAMS_T, LLHP_T, const, init_obj
from retro.const import PI, TWO_PI
from retro.utils.misc import expand, mkdir, sort_dict
from retro.likelihood import get_llh


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )

    group = parser.add_argument_group(
        title='Parameter priors'
    )

    group.add_argument(
        '--spatial-lims', choices=['dc', 'dc_subdust', 'ic'],
        help='''Choose a volume for limiting spatial samples'''
    )
    group.add_argument(
        '--energy-lims', nargs='+',
        help='''Lower and upper energy limits, in GeV. E.g.: --energy-lims=1,100'''
    )
    group.add_argument(
        '--energy-prior', choices=['log-uniform', 'uniform'],
        help='''Prior to put on _total_ event energy'''
    )

    group = parser.add_argument_group(
        title='MultiNest parameters'
    )

    group.add_argument(
        '--n-live-points', type=int,
    )
    group.add_argument(
        '--evidence-tolerance', type=float,
    )
    group.add_argument(
        '--sampling-efficiency', type=float,
    )
    group.add_argument(
        '--max-modes', type=int,
    )
    group.add_argument(
        '--seed', type=int,
    )
    group.add_argument(
        '--max-iter', type=int,
    )

    dom_tables_kw, hypo_kw, hits_kw, reco_kw = (
        init_ob.parse_args(parser=parser)
    )

    reco_kw['energy_lims'] = ''.join(reco_kw['energy_lims'])


def run_multinest(
        outdir,
        event_idx,
        llh_kw,
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
        llh = get_llh(hypo_params, **llh_kw)
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


def reco(dom_tables_kw, hypo_kw, hits_kw, reco_kw):
    """Script "main" function"""
    t00 = time.time()

    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
    hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
    hits_generator = init_obj.get_hits(**hits_kw)

    print('Running reconstructions...')
    t0 = time.time()

    metric_kw = dict(
        sd_indices=dom_tables.loaded_sd_indices,
        time_window=None,
        hypo_handler=hypo_handler,
        dom_tables=dom_tables,
        tdi_table=None
    )

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

        llh_kw['hits'] = event_hits

        t_lims = (first_hit_t - table_t_max + 100, last_hit_t)
        #t_lims = (-100, 100)
        t_range = last_hit_t - first_hit_t + table_t_max
        llh_kw['time_window'] = t_range

        param_values, log_likelihoods = run_multinest(
            outdir=outdir,
            event_idx=event_idx,
            llh_kw=llh_kw,
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
        ('metric_name', 'llh'),
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
