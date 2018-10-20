#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating, too-many-locals

"""
Reco class for performing reconstructions
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'METHODS',
    'CRS_STOP_FLAGS',
    'REPORT_AFTER',
    'APPEND_FILE',
    'Reco',
    'get_multinest_meta',
    'parse_args',
]

__author__ = 'J.L. Lanfranchi, P. Eller'
__license__ = '''Copyright 2017-2018 Justin L. Lanfranchi and Philipp Eller

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
import os
from os.path import abspath, dirname, isdir, join
import pickle
from shutil import rmtree
import sys
from tempfile import mkdtemp
import time

import numpy as np
import xarray as xr

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import init_obj
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T, SPHER_T
from retro.utils.geom import (
    rotate_points, add_vectors, fill_from_spher, fill_from_cart, reflect
)
from retro.utils.misc import expand, mkdir, sort_dict
from retro.utils.stats import estimate_from_llhp
from retro.priors import get_prior_fun
from retro.hypo.discrete_muon_kernels import pegleg_eval
from retro.tables.pexp_5d import generate_pexp_and_llh_functions
from retro.hypo.discrete_cascade_kernels import SCALING_CASCADE_ENERGY


METHODS = set([
    "multinest",
    "crs",
    "crs_prefit_mn",
    "nlopt",
    "scipy",
    "skopt",
    "experimental_trackfit",
    "fast",
    "test",
    "truth",
])

CRS_STOP_FLAGS = {
    0: 'max iterations reached',
    1: 'stddev below threshold',
    2: 'no improvement',
    3: 'vertex stddev below threshold'
}

# TODO: make following args to `__init__` or `run`
REPORT_AFTER = 100
APPEND_FILE = True


class Reco(object):
    """
    Setup tables, get events, run reconstructons on them, and optionally store
    results to disk.

    Note that "recipes" for different reconstructions are defined in the
    `Reco.run` method.

    Parameters
    ----------
    events_kw, dom_tables_kw, tdi_tables_kw : mappings
        As returned by `retro.init_obj.parse_args`; `other_kw` must contain
        key "outdir".

    outdir : string
        Directory in which to save any generated files

    save_llhp : bool, optional
        Whether to save llhp (within 30 LLH of max-LLH) to disk; default is
        False

    """
    def __init__(
        self,
        events_kw,
        dom_tables_kw,
        tdi_tables_kw,
        outdir,
        save_llhp=False,
    ):
        self.events_kw = events_kw
        self.dom_tables_kw = dom_tables_kw
        self.tdi_tables_kw = tdi_tables_kw
        self.attrs = sort_dict(dict(
            events_kw=sort_dict(self.events_kw),
            dom_tables_kw=sort_dict(self.dom_tables_kw),
            tdi_tables_kw=sort_dict(self.tdi_tables_kw),
        ))
        self._get_events = init_obj.get_events(**events_kw)
        self.outdir = outdir
        self.save_llhp = save_llhp

        # Replace None values for `start` and `step` for fewer branches in
        # subsequent logic (i.e., these will always be integers)
        self.events_start = 0 if events_kw['start'] is None else events_kw['start']
        self.events_step = 1 if events_kw['step'] is None else events_kw['step']
        # Nothing we can do about None for `stop` since we don't know how many
        # events there are in total.
        self.events_stop = events_kw['stop']

        self.slice_prefix = join(
            outdir,
            'slc{start}:{stop}:{step}.'.format(
                start=self.events_start,
                stop='' if self.events_stop is None else self.events_stop,
                step=self.events_step,
            )
        )
        """Slice-notation string to append to *estimate filenames if
        APPEND_FILE is True"""

        self.outdir = expand(self.outdir)
        mkdir(self.outdir)
        self.dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
        self.tdi_tables, self.tdi_metas = init_obj.setup_tdi_tables(**tdi_tables_kw)
        self.pexp, self.get_llh, _ = generate_pexp_and_llh_functions(
            dom_tables=self.dom_tables,
            tdi_tables=self.tdi_tables,
            tdi_metas=self.tdi_metas,
        )
        self.event_prefix = None
        self.current_event = None
        self.current_event_idx = -1
        self.event_counter = -1
        self.hypo_handler = None
        self.prior = None
        self.priors_used = None
        self.loglike = None
        self.n_params = None
        self.n_opt_params = None

    @property
    def events(self):
        """Iterator over events which sets class variables `event_prefix`,
        `current_event`, `current_event_idx`, and `event_counter` for each
        event retrieved."""
        for event_idx, event in self._get_events:
            print('Operating on event: "{}"'.format(event_idx))
            self.event_prefix = join(self.outdir, 'evt{}.'.format(event_idx))
            self.current_event = event
            self.current_event_idx = event_idx
            self.event_counter += 1
            yield self.current_event

    def setup_hypo(self, **kwargs):
        """Setup hypothesis and record `n_params` and `n_opt_params`
        corresponding to the hypothesis.

        Parameters
        ----------
        **kwargs
            Passed to `retro.init_obj.setup_discrete_hypo`

        """
        self.hypo_handler = init_obj.setup_discrete_hypo(**kwargs)
        self.n_params = self.hypo_handler.n_params
        self.n_opt_params = self.hypo_handler.n_opt_params

    def run(self, method):
        """Run reconstructions on events.

        This method collects many recipes for performing different kinds of
        reconstructions.

        Parameters
        ----------
        method : string
            One of {}

        """.format(sorted(METHODS))
        if method not in METHODS:
            raise ValueError(
                'Unrecognized `method` "{}"; must be one of {}'.format(method, METHODS)
            )

        print('Running "{}" reconstruction...'.format(method))
        t00 = time.time()

        for _ in self.events:
            if method in (
                'multinest',
                'test',
                'truth',
                'crs',
                'scipy',
                'nlopt',
                'skopt',
            ):
                t0 = time.time()
                self.setup_hypo(
                    cascade_kernel='scaling_aligned_one_dim',
                    track_kernel='pegleg',
                    track_time_step=1.,
                )

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(kind='SPEFit2', extent='tight')),
                        ('y', dict(kind='SPEFit2', extent='tight')),
                        ('z', dict(kind='SPEFit2', extent='tight')),
                        ('time', dict(kind='SPEFit2', extent='tight')),
                    ])
                )

                param_values = []
                log_likelihoods = []
                t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=t_start,
                )

                if method == 'test':
                    run_info = self.run_test(seed=0)
                if method == 'truth':
                    run_info = self.run_with_truth()
                elif method == 'crs':
                    run_info = self.run_crs(
                        n_live=250,
                        max_iter=20000,
                        max_noimprovement=5000,
                        min_fn_std=0.1,
                        min_vertex_std=(1, 1, 1, 3),
                        use_priors=False,
                        use_sobol=True,
                        seed=0,
                    )
                elif method == 'multinest':
                    run_info = self.run_multinest(
                        importance_sampling=True,
                        max_modes=1,
                        const_eff=True,
                        n_live=160,
                        evidence_tol=0.5,
                        sampling_eff=0.3,
                        max_iter=10000,
                        seed=0,
                    )
                elif method == 'scipy':
                    run_info = self.run_scipy(
                        method='differential_evolution',
                        eps=0.02
                    )
                elif method == 'nlopt':
                    run_info = self.run_nlopt()
                elif method == 'skopt':
                    run_info = self.run_skopt()

                t1 = time.time()
                run_info['run_time'] = t1 - t0

                if self.save_llhp:
                    llhp_fname = '{}.llhp'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                self.make_estimate(
                    llhp=llhp,
                    remove_priors=True,
                    run_info=run_info,
                    fname='{}.estimate'.format(method),
                )

            elif method == 'fast':
                t0 = time.time()
                self.setup_hypo(
                     cascade_kernel='scaling_aligned_point_ckv',
                     track_kernel='pegleg',
                     track_time_step=3.,
                 )

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(kind='SPEFit2', extent='tight')),
                        ('y', dict(kind='SPEFit2', extent='tight')),
                        ('z', dict(kind='SPEFit2', extent='tight')),
                        ('time', dict(kind='SPEFit2', extent='tight')),
                    ])
                )

                param_values = []
                log_likelihoods = []
                t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=t_start,
                )

                run_info = self.run_crs(
                    n_live=160,
                    max_iter=10000,
                    max_noimprovement=1000,
                    min_fn_std=0.5,
                    min_vertex_std=(5, 5, 5, 15),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                )

                t1 = time.time()
                run_info['run_time'] = t1 - t0

                if self.save_llhp:
                    llhp_fname = '{}.llhp'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                self.make_estimate(
                    llhp=llhp,
                    remove_priors=False,
                    run_info=run_info,
                    fname='{}.estimate'.format(method),
                )

            elif method == 'crs_prefit_mn':
                t0 = time.time()

                print('--- Track-only CRS prefit ---')

                self.setup_hypo(
                    cascade_kernel='scaling_aligned_point_ckv',
                    track_kernel='pegleg',
                    track_time_step=3.,
                )

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(kind='SPEFit2', extent='tight')),
                        ('y', dict(kind='SPEFit2', extent='tight')),
                        ('z', dict(kind='SPEFit2', extent='tight')),
                        ('time', dict(kind='SPEFit2', extent='tight')),
                    ])
                )

                param_values = []
                log_likelihoods = []
                prefit_t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=prefit_t_start,
                )

                prefit_run_info = self.run_crs(
                    n_live=160,
                    max_iter=10000,
                    max_noimprovement=1000,
                    min_fn_std=0.5,
                    min_vertex_std=(5, 5, 5, 15),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                )

                t1 = time.time()
                prefit_run_info['run_time'] = t1 - t0

                if self.save_llhp:
                    llhp_fname = '{}.llhp_prefit'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                prefit_estimate = self.make_estimate(
                    llhp=llhp,
                    remove_priors=False,
                    run_info=prefit_run_info,
                    fname='{}.estimate_prefit'.format(method),
                )

                print('--- MultiNest fit including aligned 1D cascade ---')

                self.setup_hypo(
                    cascade_kernel='scaling_aligned_one_dim',
                    track_kernel='pegleg',
                    track_time_step=1.,
                )

                # Setup prior

                pft_est = prefit_estimate.sel(kind='mean')
                pft_x = float(pft_est.sel(param='x'))
                pft_y = float(pft_est.sel(param='y'))
                pft_z = float(pft_est.sel(param='z'))
                pft_time = float(pft_est.sel(param='time'))

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(
                            kind='cauchy',
                            loc=pft_x,
                            scale=15,
                            low=pft_x - 300,
                            high=pft_x + 300,
                        )),
                        ('y', dict(
                            kind='cauchy',
                            loc=pft_y,
                            scale=15,
                            low=pft_y - 300,
                            high=pft_y + 300,
                        )),
                        ('z', dict(
                            kind='cauchy',
                            loc=pft_z,
                            scale=10,
                            low=pft_z - 200,
                            high=pft_z + 200,
                        )),
                        ('time', dict(
                            kind='cauchy',
                            loc=pft_time,
                            scale=40,
                            low=pft_time - 800,
                            high=pft_time + 800,
                        )),
                    ])
                )

                param_values = []
                log_likelihoods = []
                t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=t_start,
                )

                run_info = self.run_multinest(
                    importance_sampling=True,
                    max_modes=1,
                    const_eff=True,
                    n_live=250,
                    evidence_tol=0.02,
                    sampling_eff=0.5,
                    max_iter=10000,
                    seed=0,
                )

                t2 = time.time()
                run_info['run_time'] = t2 - t1

                if self.save_llhp:
                    llhp_fname = '{}.llhp'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                self.make_estimate(
                    llhp=llhp,
                    remove_priors=True,
                    run_info=run_info,
                    fname='{}.estimate'.format(method),
                )

                #print('--- MN 10d fit ---')

                #self.setup_hypo(
                #    cascade_kernel='scaling_one_dim',
                #    track_kernel='pegleg',
                #    track_time_step=1.,
                #)

                #self.hypo_handler.fixed_params = OrderedDict()
                #self.hypo_handler.fixed_params['x'] = estimate['mean']['x']
                #self.hypo_handler.fixed_params['y'] = estimate['mean']['y']
                #self.hypo_handler.fixed_params['z'] = estimate['mean']['z']
                #self.hypo_handler.fixed_params['time'] = estimate['mean']['time']

                #param_values = []
                #log_likelihoods = []

                ## Setup prior (none)
                #prior_defs = OrderedDict()
                #self.generate_prior_method(prior_defs)

                #self.generate_loglike_method(
                #    param_values=param_values,
                #    log_likelihoods=log_likelihoods,
                #    t_start=t_start,
                #)

                #run_info_10d = self.run_multinest(
                #    importance_sampling=True,
                #    max_modes=1,
                #    const_eff=True,
                #    n_live=160,
                #    evidence_tol=0.5,
                #    sampling_eff=0.3,
                #    max_iter=10000,
                #    seed=0,
                #)

                #t3 = time.time()
                #run_info_10d['run_time'] = t3 - t2

                #if self.save_llhp:
                #    llhp_fname = '{}.llhp_10d'.format(method)
                #else:
                #    llhp_fname = None
                #llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                #self.make_estimate(
                #    llhp=llhp,
                #    remove_priors=True,
                #    run_info=run_info_10d,
                #    fname='{}.estimate_10d'.format(method),
                #)

            elif method == 'experimental_trackfit':
                print('--- track-only prefit ---')

                t0 = time.time()
                self.setup_hypo(track_kernel='pegleg', track_time_step=1.)

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(kind='SPEFit2', extent='tight')),
                        ('y', dict(kind='SPEFit2', extent='tight')),
                        ('z', dict(kind='SPEFit2', extent='tight')),
                        ('time', dict(kind='SPEFit2', extent='tight')),
                    ])
                )

                param_values = []
                log_likelihoods = []
                prefit_t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=t_start,
                )

                prefit_run_info = self.run_crs(
                    n_live=160,
                    max_iter=20000,
                    max_noimprovement=2000,
                    min_fn_std=0.1,
                    min_vertex_std=(5, 5, 5, 15),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                )

                t1 = time.time()
                prefit_run_info['run_time'] = t1 - t0

                if self.save_llhp:
                    llhp_fname = '{}.llhp_prefit'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                prefit_estimate = self.make_estimate(
                    llhp=llhp,
                    remove_priors=False,
                    run_info=prefit_run_info,
                    fname='{}.estimate_prefit'.format(method),
                )

                print('--- hybrid fit ---')

                t2 = time.time()
                # track AND cascade to hypo
                self.setup_hypo(
                    cascade_kernel='scaling_aligned_one_dim',
                    #track_kernel='pegleg',
                    track_kernel='table_energy_loss',
                    track_time_step=1.,
                )

                pft_est = prefit_estimate.sel(kind='median')
                pft_x = float(pft_est.sel(param='x'))
                pft_y = float(pft_est.sel(param='y'))
                pft_z = float(pft_est.sel(param='z'))
                pft_time = float(pft_est.sel(param='time'))
                pft_track_energy = float(pft_est.sel(param='track_energy'))

                self.hypo_handler.fixed_params = OrderedDict([
                    ('track_energy', pft_track_energy)
                ])

                self.generate_prior_method(
                    prior_defs=OrderedDict([
                        ('x', dict(kind='cauchy', loc=pft_x, scale=12)),
                        ('y', dict(kind='cauchy', loc=pft_y, scale=13)),
                        ('z', dict(kind='cauchy', loc=pft_z, scale=7.5)),
                        ('time', dict(
                            kind='cauchy',
                            loc=pft_time,
                            scale=40,
                            low=pft_time - 2000,
                            high=pft_time + 2000,
                        )),
                    ])
                )

                param_values = []
                log_likelihoods = []
                t_start = []
                self.generate_loglike_method(
                    param_values=param_values,
                    log_likelihoods=log_likelihoods,
                    t_start=t_start,
                )

                run_info = self.run_crs(
                    n_live=160,
                    max_iter=20000,
                    max_noimprovement=2000,
                    min_fn_std=0.1,
                    min_vertex_std=(5, 5, 5, 15),
                    use_priors=False,
                    use_sobol=True,
                    seed=0,
                )

                t3 = time.time()
                run_info['run_time'] = t3 - t2

                if self.save_llhp:
                    llhp_fname = '{}.llhp'.format(method)
                else:
                    llhp_fname = None
                llhp = self.make_llhp(log_likelihoods, param_values, fname=llhp_fname)
                self.make_estimate(
                    llhp=llhp,
                    remove_priors=False,
                    run_info=run_info,
                    fname='{}.estimate'.format(method),
                )
            else:
                raise ValueError('Unknown `Method` {}'.format(method))

        print('Total script run time is {:.3f} s'.format(time.time() - t00))

    def generate_prior_method(self, prior_defs):
        """Generate the prior transform method `self.prior` and info
        `self.priors_used` for a given event.

        Parameters
        ----------
        prior_defs : dict

        """
        prior_funcs = []
        self.priors_used = OrderedDict()

        for dim_num, dim_name in enumerate(self.hypo_handler.opt_param_names):
            if prior_defs.has_key(dim_name):
                kwargs = prior_defs[dim_name]
            else:
                kwargs = {}

            prior_fun, prior_def = get_prior_fun(
                dim_num=dim_num,
                dim_name=dim_name,
                event=self.current_event,
                **kwargs
            )
            prior_funcs.append(prior_fun)
            self.priors_used[dim_name] = prior_def

        def prior(cube, ndim=None, nparams=None): # pylint: disable=unused-argument
            """Apply `prior_funcs` to the hypercube to map values from the unit
            hypercube onto values in the physical parameter space.

            The result overwrites the values in `cube`.

            Parameters
            ----------
            cube
            ndim
            nparams

            """
            for prior_func in prior_funcs:
                prior_func(cube)

        self.prior = prior

    def generate_loglike_method(self, param_values, log_likelihoods, t_start):
        """Generate the LLH callback method `self.loglike` for a given event.

        Parameters
        ----------
        param_values : list
        log_likelihoods : list
        t_start : list
            Needs to be a list for start time to be passed by reference and
            therefore universally accessible within all methods that require
            knowing the start time

        """
        # -- Variables to be captured by `loglike` closure -- #

        all_param_names = self.hypo_handler.all_param_names
        opt_param_names = self.hypo_handler.opt_param_names
        n_opt_params = self.hypo_handler.n_opt_params
        fixed_params = self.hypo_handler.fixed_params
        event = self.current_event
        hits = event['hits']
        hits_indexer = event['hits_indexer']
        hypo_handler = self.hypo_handler
        pegleg_muon_dt = hypo_handler.pegleg_kernel_kwargs.get('dt')
        pegleg_muon_const_e_loss = False
        dom_info = self.dom_tables.dom_info
        sd_idx_table_indexer = self.dom_tables.sd_idx_table_indexer
        truth_info = OrderedDict([
            ('x', event['truth']['x']),
            ('y', event['truth']['y']),
            ('z', event['truth']['z']),
            ('time', event['truth']['time']),
            ('zenith', np.arccos(event['truth']['coszen'])),
            ('azimuth', event['truth']['azimuth']),
            ('track_azimuth', event['truth']['longest_daughter_azimuth']),
            ('track_zenith', np.arccos(event['truth']['longest_daughter_coszen'])),
            ('track_energy', event['truth']['longest_daughter_energy']),
            ('cascade_azimuth', event['truth']['cascade_azimuth']),
            ('cascade_zenith', np.arccos(event['truth']['cascade_coszen'])),
            ('cascade_energy', event['truth']['cascade_energy']),
            ('neutrino_energy', event['truth']['energy']),
        ])
        num_operational_doms = np.sum(dom_info['operational'])

        # Array containing only DOMs operational during the event & info
        # relevant to the hits these DOMs got (if any)
        event_dom_info = np.zeros(shape=num_operational_doms, dtype=EVT_DOM_INFO_T)

        # Array containing all relevant hit info for the event, including a
        # pointer back to the index of the DOM in the `event_dom_info` array
        event_hit_info = np.zeros(shape=hits.size, dtype=EVT_HIT_INFO_T)

        # Copy 'time' and 'charge' over directly; add 'event_dom_idx' below
        event_hit_info[['time', 'charge']] = hits[['time', 'charge']]

        copy_fields = ['sd_idx', 'x', 'y', 'z', 'quantum_efficiency', 'noise_rate_per_ns']

        print('all noise rate %.5f' % np.sum(dom_info['noise_rate_per_ns']))
        print('DOMs with zero noise %i' % np.sum(dom_info['noise_rate_per_ns'] == 0))

        # Fill `event_{hit,dom}_info` arrays only for operational DOMs
        for dom_idx, this_dom_info in enumerate(dom_info[dom_info['operational']]):
            this_event_dom_info = event_dom_info[dom_idx:dom_idx+1]
            this_event_dom_info[copy_fields] = this_dom_info[copy_fields]
            sd_idx = this_dom_info['sd_idx']
            this_event_dom_info['table_idx'] = sd_idx_table_indexer[sd_idx]

            # Copy any hit info from `hits_indexer` and total charge from
            # `hits` into `event_hit_info` and `event_dom_info` arrays
            this_hits_indexer = hits_indexer[hits_indexer['sd_idx'] == sd_idx]
            if len(this_hits_indexer) == 0:
                this_event_dom_info['hits_start_idx'] = 0
                this_event_dom_info['hits_stop_idx'] = 0
                this_event_dom_info['total_observed_charge'] = 0
                continue

            start = this_hits_indexer[0]['offset']
            stop = start + this_hits_indexer[0]['num']
            event_hit_info[start:stop]['event_dom_idx'] = dom_idx
            this_event_dom_info['hits_start_idx'] = start
            this_event_dom_info['hits_stop_idx'] = stop
            this_event_dom_info['total_observed_charge'] = (
                np.sum(hits[start:stop]['charge'])
            )

        print('this evt. noise rate %.5f'%np.sum(event_dom_info['noise_rate_per_ns']))
        print('DOMs with zero noise: %i'%np.sum(event_dom_info['noise_rate_per_ns'] == 0))
        # settings those to minimum noise
        noise = event_dom_info['noise_rate_per_ns']
        mask = noise < 1e-7
        noise[mask] = 1e-7
        print('this evt. noise rate %.5f'%np.sum(event_dom_info['noise_rate_per_ns']))
        print('DOMs with zero noise: %i'%np.sum(event_dom_info['noise_rate_per_ns'] == 0))
        print('min noise: ', np.min(noise))
        print('mean noise: ', np.mean(noise))

        assert np.sum(event_dom_info['quantum_efficiency'] <= 0) == 0, 'negative QE'
        assert np.sum(event_dom_info['total_observed_charge']) > 0, 'no charge'
        assert np.isfinite(np.sum(event_dom_info['total_observed_charge'])), 'inf charge'

        def loglike(cube, ndim=None, nparams=None): # pylint: disable=unused-argument
            """Get log likelihood values.

            Defined as a closure to capture particulars of the event and priors
            without having to pass these as parameters to the function.

            Note that this is called _after_ `prior` has been called, so `cube`
            already contains the parameter values scaled to be in their
            physical ranges.

            Parameters
            ----------
            cube
            ndim : int, optional
            nparams : int, optional

            Returns
            -------
            llh : float

            """
            t0 = time.time()
            if len(t_start) == 0:
                t_start.append(time.time())

            hypo = OrderedDict(list(zip(opt_param_names, cube)))

            generic_sources = hypo_handler.get_generic_sources(hypo)
            pegleg_sources = hypo_handler.get_pegleg_sources(hypo)
            scaling_sources = hypo_handler.get_scaling_sources(hypo)

            llh, pegleg_idx, scalefactor = self.get_llh(
                generic_sources=generic_sources,
                pegleg_sources=pegleg_sources,
                scaling_sources=scaling_sources,
                event_hit_info=event_hit_info,
                event_dom_info=event_dom_info,
                pegleg_stepsize=1,
            )

            assert np.isfinite(llh), 'LLH not finite'
            assert llh < 0, 'LLH positive'

            additional_results = []

            if self.hypo_handler.pegleg_kernel:
                pegleg_result = pegleg_eval(
                    pegleg_idx=pegleg_idx,
                    dt=pegleg_muon_dt,
                    const_e_loss=pegleg_muon_const_e_loss,
                    mmc=True,
                )
                additional_results.append(pegleg_result)

            if self.hypo_handler.scaling_kernel:
                additional_results.append(scalefactor*SCALING_CASCADE_ENERGY)

            result = (
                tuple(cube[:n_opt_params])
                + tuple(fixed_params.values())
                + tuple(additional_results)
            )
            param_values.append(result)

            log_likelihoods.append(llh)
            n_calls = len(log_likelihoods)
            t1 = time.time()

            if n_calls % REPORT_AFTER == 0:
                print('')
                msg = 'truth:                '
                for key, val in zip(all_param_names, result):
                    try:
                        msg += ' %s=%.1f'%(key, truth_info[key])
                    except KeyError:
                        pass
                print(msg)
                t_now = time.time()
                best_idx = np.argmax(log_likelihoods)
                best_llh = log_likelihoods[best_idx]
                best_p = param_values[best_idx]
                msg = 'best llh = {:.3f} @ '.format(best_llh)
                for key, val in zip(all_param_names, best_p):
                    msg += ' %s=%.1f'%(key, val)
                print(msg)
                msg = 'this llh = {:.3f} @ '.format(llh)
                for key, val in zip(all_param_names, result):
                    msg += ' %s=%.1f'%(key, val)
                print(msg)
                print('{} LLH computed'.format(n_calls))
                print('avg time per llh: {:.3f} ms'.format(
                    (t_now - t_start[0])/n_calls*1000)
                )
                print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
                print('')

            return llh

        self.loglike = loglike

    def make_llhp(self, log_likelihoods, param_values, fname=None):
        """Create a structured numpy array containing the reco information;
        also add derived dimensions, and optionally save to disk.

        Parameters
        ----------
        log_likelihoods : array

        param_values : array

        fname : str, optional
            If provided, llhp for the event reco are saved to file at path
              {self.outdir}/evt{event_idx}.{fname}.npy

        Returns
        -------
        llhp : length-n_llhp array of dtype llhp_t
            Note that llhp_t is derived from the defined parameter names.

        """
        # Setup LLHP dtype
        dim_names = list(self.hypo_handler.all_param_names)

        # add derived quantities
        derived_dim_names = ['energy', 'azimuth', 'zenith']
        if 'cascade_d_zenith' in dim_names and 'cascade_d_azimuth' in dim_names:
            derived_dim_names += ['cascade_zenith', 'cascade_azimuth']

        all_dim_names = dim_names + derived_dim_names

        llhp_t = np.dtype([(field, np.float32) for field in ['llh'] + all_dim_names])

        # dump
        llhp = np.zeros(shape=len(param_values), dtype=llhp_t)
        llhp['llh'] = log_likelihoods
        llhp[dim_names] = param_values

        # create derived dimensions
        if 'energy' in derived_dim_names:
            if 'track_energy' in dim_names:
                llhp['energy'] += llhp['track_energy']
            if 'cascade_energy' in dim_names:
                llhp['energy'] += llhp['cascade_energy']

        if 'cascade_d_zenith' in dim_names and 'cascade_d_azimuth' in dim_names:
            # create cascade angles from delta angles
            rotate_points(
                p_theta=llhp['cascade_d_zenith'],
                p_phi=llhp['cascade_d_azimuth'],
                rot_theta=llhp['track_zenith'],
                rot_phi=llhp['track_azimuth'],
                q_theta=llhp['cascade_zenith'],
                q_phi=llhp['cascade_azimuth'],
            )

        if 'track_zenith' in all_dim_names and 'track_azimuth' in all_dim_names:
            if 'cascade_zenith' in all_dim_names and 'cascade_azimuth' in all_dim_names:
                # this resulting radius we won't need, but need to supply an array to
                # the function
                r_out = np.empty(shape=llhp.shape, dtype=np.float32)
                # combine angles:
                add_vectors(
                    r1=llhp['track_energy'],
                    theta1=llhp['track_zenith'],
                    phi1=llhp['track_azimuth'],
                    r2=llhp['cascade_energy'],
                    theta2=llhp['cascade_zenith'],
                    phi2=llhp['cascade_azimuth'],
                    r3=r_out,
                    theta3=llhp['zenith'],
                    phi3=llhp['azimuth'],
                )
            else:
                # in this case there is no cascade angles
                llhp['zenith'] = llhp['track_zenith']
                llhp['azimuth'] = llhp['track_azimuth']

        elif 'cascade_zenith' in all_dim_names and 'cascade_azimuth' in all_dim_names:
            # in this case there are no track angles
            llhp['zenith'] = llhp['cascade_zenith']
            llhp['azimuth'] = llhp['cascade_azimuth']

        if fname:
            # NOTE: since each array can have different length and numpy
            # doesn't handle "ragged" arrays nicely, forcing each llhp to be
            # saved to its own file even if APPEND_FILE is True.
            llhp_outf = '{}{}.npy'.format(self.event_prefix, fname)
            llh = llhp['llh']
            cut_llhp = llhp[llh > np.max(llh) - 30]
            print(
                'Saving llhp within 30 LLH of max ({} llhp) to "{}"'
                .format(len(cut_llhp), llhp_outf)
            )
            np.save(llhp_outf, cut_llhp)

        return llhp

    def make_estimate(
        self,
        llhp,
        remove_priors,
        run_info=None,
        fname=None,
    ):
        """Create estimate from llhp and optionally save to disk.

        Parameters
        ----------
        llhp : length-n_llhp array of dtype llhp_t
        remove_priors : bool
            Remove effect of priors
        run_info : mapping, optional
        fname : string, optional
            * If not provided, estimate is not written to disk.
            * If provided and APPEND_FILE is True, an `xarray.Dataset` where
              each contained `xarray.DataArray` is one event's reconstruction
              (plus run and estimate metadata); the name of each DataArray is
              the event index, and Reco metadata is stored to `Dataset.attrs`;
              output file is
                {outdir}/slc{start}:{stop}:{step}.{fname}.pkl
            * If provided and APPEND_FILE is False, event and run_info metadata
              is written as an `xarray.DataArray` containing estimate and
              metadata to file at
                {outdir}/evt{event_idx}.{fname}.pkl

        Returns
        -------
        estimate : xarray.DataArray

        """
        estimate = estimate_from_llhp(
            llhp=llhp,
            treat_dims_independently=False,
            use_prob_weights=True,
            priors_used=self.priors_used if remove_priors else None,
        )
        attrs = estimate.attrs
        attrs['event_idx'] = self.current_event_idx
        attrs['params'] = list(self.hypo_handler.all_param_names)
        attrs['priors_used'] = self.priors_used
        if run_info is None:
            attrs['run_info'] = OrderedDict()
        else:
            attrs['run_info'] = run_info
        estimate.attrs = sort_dict(attrs)
        estimate.name = self.current_event_idx
        if not fname:
            return estimate

        if APPEND_FILE:
            estimate_outf = os.path.join(
                self.outdir, '{}{}.pkl'.format(self.slice_prefix, fname)
            )
            file_exists = os.path.isfile(estimate_outf)
            if self.event_counter == 0:
                if file_exists:
                    raise IOError('File already exists at "{}"'.format(estimate_outf))
                # create new Dataset
                dataset = xr.Dataset()
                dataset.attrs = self.attrs
            else:
                if not file_exists:
                    raise IOError(
                        'Output file with previous events does not exist at "{}"'
                        .format(estimate_outf)
                    )
                dataset = pickle.load(open(estimate_outf, 'rb'))
            dataset[estimate.name] = estimate
            pickle.dump(
                obj=dataset,
                file=open(estimate_outf, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        else: # save each event reco estimate as its own pickle file
            estimate_outf = '{}{}.pkl'.format(self.event_prefix, fname)
            print('Saving estimate to "{}"'.format(estimate_outf))
            pickle.dump(
                obj=estimate,
                file=open(estimate_outf, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return estimate

    def run_test(self, seed):
        """Random sampling instead of an actual minimizer"""
        raise NotImplementedError('`run_test` not implemented') # TODO
        kwargs = sort_dict(dict(seed=seed))
        rand = np.random.RandomState(seed=seed)
        for i in range(100):
            param_vals = rand.uniform(0, 1, self.n_opt_params)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
        run_info = sort_dict(dict(method='run_test', kwargs=kwargs))
        return run_info

    def run_with_truth(self, rand_dims=None, n_samples=10000, seed=0):
        """Run with all params set to truth except for the dimensions defined,
        which will be randomized.

        Parameters
        ----------
        rand_dims : list, optional
            Dimensions to randomly sample; all not specified are set to truth

        n_samples : int
            Number of samples to draw

        """
        raise NotImplementedError('`run_with_truth` not implemented') # TODO
        if rand_dims is None:
            rand_dims = []

        truth = self.current_event['truth']
        true_params = np.zeros(self.n_opt_params)

        for i, name in enumerate(self.hypo_handler.opt_param_names):
            if name in ('x', 'y', 'z', 'time'):
                true_params[i] = truth[name]
            elif name == 'track_zenith':
                true_params[i] = np.arccos(truth['coszen'])
            elif name == 'track_azimuth':
                true_params[i] = truth['azimuth']
            else:
                raise NotImplementedError()

        rand = np.random.RandomState(seed=seed)
        if len(rand_dims) > 1:
            for i in range(n_samples):
                rand_params = rand.uniform(0, 1, self.n_opt_params)
                self.prior(rand_params)
                param_vals = np.zeros(self.n_opt_params)
                param_vals[:] = true_params[:]
                param_vals[rand_dims] = rand_params[rand_dims]
                llh = self.loglike(param_vals)
        else:
            llh = self.loglike(true_params)

        run_info = sort_dict(dict(method='run_with_truth', kwargs=kwargs))
        return run_info

    def run_crs(
        self,
        n_live,
        max_iter,
        max_noimprovement,
        min_fn_std,
        min_vertex_std,
        use_priors,
        use_sobol,
        seed,
    ):
        """Implementation of the CRS2 algorithm, adapted to work with spherical
        coordinates (correct centroid calculation, reflection, and mutation).

        At the moment Cartesian (standard) parameters and spherical parameters
        are assumed to have particular names (i.e., spherical coordinates start
        with "az" and "zen"). Furthermore, all Cartesian coordinates must come
        first followed by the pairs of (azimuth, zenith) spherical coordinates;
        e.g., "az_1", "zen_1", "az_2", "zen_2", etc.

        Parameters
        ----------
        n_live : int
            Number of live points
        max_iter : int
            Maximum iterations
        max_noimprovement : int
            Maximum iterations with no improvement of best point
        min_fn_std : float
            Break if stddev of function values across all livepoints drops
            below this threshold
        min_vertex_std : sequence
            Break condition on stddev of vertex
        use_priors : bool
            Use priors during minimization; if `False`, priors are only used
            for sampling the initial distributions
        use_sobol : bool
            Use a Sobol sequence instead of numpy pseudo-random numbers
        seed : int
            Random seed

        Returns
        -------
        run_info : OrderedDict

        Notes
        -----
        CRS2 [1] is a variant of controlled random search (CRS, a global
        optimizer) with faster convergence than CRS.

        Refrences
        ---------
        .. [1] P. Kaelo, M.M. Ali, "Some variants of the controlled random
           search algorithm for global optimization," J. Optim. Theory Appl.,
           130 (2) (2006), pp. 253-264.

        """
        if use_sobol:
            from sobol import i4_sobol

        kwargs = sort_dict(dict(
            n_live=n_live,
            max_iter=max_iter,
            max_noimprovement=max_noimprovement,
            min_fn_std=min_fn_std,
            min_vertex_std=min_vertex_std,
            use_priors=use_priors,
            use_sobol=use_sobol,
            seed=seed,
        ))

        rand = np.random.RandomState(seed=seed)

        n_opt_params = self.n_opt_params
        # absolute minimum number of points necessary
        assert n_live > n_opt_params + 1

        # figure out which variables are Cartesian and which spherical
        opt_param_names = self.hypo_handler.opt_param_names
        cart_param_names = set(opt_param_names) & set(['time', 'x', 'y', 'z'])
        n_cart = len(cart_param_names)
        assert set(opt_param_names[:n_cart]) == cart_param_names
        n_spher_param_pairs = int((n_opt_params - n_cart)/2)
        for sph_pair_idx in range(n_spher_param_pairs):
            az_param = opt_param_names[n_cart + sph_pair_idx*2]
            zen_param = opt_param_names[n_cart + sph_pair_idx*2 + 1]
            assert 'az' in az_param, '"{}" not azimuth param'.format(az_param)
            assert 'zen' in zen_param, '"{}" not zenith param'.format(zen_param)

        # setup arrays to store points
        s_cart = np.zeros(shape=(n_live, n_cart))
        s_spher = np.zeros(shape=(n_live, n_spher_param_pairs), dtype=SPHER_T)
        fx = np.zeros(shape=(n_live,))

        def fun(x):
            """Callable for minimizer"""
            if use_priors:
                param_vals = np.zeros_like(x)
                param_vals[:n_cart] = x[:n_cart]
                self.prior(param_vals)
                param_vals[n_cart:] = x[n_cart:]
            else:
                param_vals = x
            llh = self.loglike(param_vals)
            return -llh

        def create_x(x_cart, x_spher):
            """Patch Cartesian and spherical coordinates into one array"""
            # TODO: make proper
            x = np.empty(shape=n_opt_params)
            x[:n_cart] = x_cart
            x[n_cart+1::2] = x_spher['zen']
            x[n_cart::2] = x_spher['az']
            return x

        # generate initial population
        for i in range(n_live):
            if use_sobol:
                # sobol seems to do slightly better
                x, _ = i4_sobol(n_opt_params, i+1)
            else:
                x = rand.uniform(0, 1, n_opt_params)
            param_vals = np.copy(x)
            self.prior(param_vals)
            # always transform angles!
            x[n_cart:] = param_vals[n_cart:]
            if not use_priors:
                x[:n_cart] = param_vals[:n_cart]

            # break up into cartesiand and spherical coordinates
            s_cart[i] = x[:n_cart]
            s_spher[i]['zen'] = x[n_cart+1::2]
            s_spher[i]['az'] = x[n_cart::2]
            fill_from_spher(s_spher[i])
            fx[i] = fun(x)

        best_llh = np.min(fx)
        no_improvement_counter = -1

        # optional bookkeeping
        num_simplex_successes = 0
        num_mutation_successes = 0
        num_failures = 0
        stopping_flag = 0

        # minimizer loop
        iter_num = 0
        for iter_num in range(max_iter):
            if iter_num % REPORT_AFTER == 0:
                print(
                    'simplex: %i, mutation: %i, failed: %i'
                    % (num_simplex_successes, num_mutation_successes, num_failures)
                )

            # break condition 1
            if np.std(fx) < min_fn_std:
                stopping_flag = 1
                break

            # break condition 2
            if no_improvement_counter > max_noimprovement:
                stopping_flag = 2
                break

            # break condition 3
            done = []
            stds = []
            for dim, cond in zip(('x', 'y', 'z', 'time'), min_vertex_std):
                if dim in opt_param_names:
                    std = np.std(s_cart[:, opt_param_names.index(dim)])
                    stds.append(std)
                    done.append(std < cond)
            if len(done) > 0 and all(done):
                stopping_flag = 3
                break

            new_best_llh = np.min(fx)

            if new_best_llh < best_llh:
                best_llh = new_best_llh
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            worst_idx = np.argmax(fx)
            best_idx = np.argmin(fx)

            # choose n_opt_params random points but not best
            choice = rand.choice(n_live - 1, n_opt_params, replace=False)
            choice[choice >= best_idx] += 1

            # Cartesian centroid
            centroid_cart = (
                (np.sum(s_cart[choice[:-1]], axis=0) + s_cart[best_idx]) / n_opt_params
            )

            # reflect point
            new_x_cart = 2*centroid_cart - s_cart[choice[-1]]

            # spherical centroid
            centroid_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            centroid_spher['x'] = (
                np.sum(s_spher['x'][choice[:-1]], axis=0) + s_spher['x'][best_idx]
            ) / n_opt_params
            centroid_spher['y'] = (
                np.sum(s_spher['y'][choice[:-1]], axis=0) + s_spher['y'][best_idx]
            ) / n_opt_params
            centroid_spher['z'] = (
                np.sum(s_spher['z'][choice[:-1]], axis=0) + s_spher['z'][best_idx]
            ) / n_opt_params
            fill_from_cart(centroid_spher)

            # reflect point
            new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            reflect(s_spher[choice[-1]], centroid_spher, new_x_spher)

            if use_priors:
                outside = np.any(new_x_cart < 0) or np.any(new_x_cart > 1)
            else:
                outside = False

            if not outside:
                new_fx = fun(create_x(new_x_cart, new_x_spher))

                if new_fx < fx[worst_idx]:
                    # found better point
                    s_cart[worst_idx] = new_x_cart
                    s_spher[worst_idx] = new_x_spher
                    fx[worst_idx] = new_fx
                    num_simplex_successes += 1
                    continue

            # mutation
            w = rand.uniform(0, 1, n_cart)
            new_x_cart2 = (1 + w) * s_cart[best_idx] - w * new_x_cart

            # first reflect at best point
            reflected_new_x_spher = np.zeros(n_spher_param_pairs, dtype=SPHER_T)
            reflect(new_x_spher, s_spher[best_idx], reflected_new_x_spher)

            new_x_spher2 = np.zeros_like(new_x_spher)

            # now do a combination of best and reflected point with weight w
            for dim in ('x', 'y', 'z'):
                w = rand.uniform(0, 1, n_spher_param_pairs)
                new_x_spher2[dim] = (
                    (1 - w) * s_spher[best_idx][dim]
                    + w * reflected_new_x_spher[dim]
                )
            fill_from_cart(new_x_spher2)

            if use_priors:
                outside = np.any(new_x_cart2 < 0) or np.any(new_x_cart2 > 1)
            else:
                outside = False

            if not outside:
                new_fx = fun(create_x(new_x_cart2, new_x_spher2))

                if new_fx < fx[worst_idx]:
                    # found better point
                    s_cart[worst_idx] = new_x_cart2
                    s_spher[worst_idx] = new_x_spher2
                    fx[worst_idx] = new_fx
                    num_mutation_successes += 1
                    continue

            # if we get here no method was successful in replacing worst
            # point -> start over
            num_failures += 1

        print(CRS_STOP_FLAGS[stopping_flag])

        fit_meta = sort_dict(dict(
            stopping_flag=stopping_flag,
            stopping_message=CRS_STOP_FLAGS[stopping_flag],
            num_simplex_successes=num_simplex_successes,
            num_mutation_successes=num_mutation_successes,
            num_failures=num_failures,
            iterations=iter_num,
        ))

        run_info = sort_dict(dict(
            method='run_crs',
            method_description='CRS2spherical+lm+sampling',
            kwargs=kwargs,
            fit_meta=fit_meta,
        ))
        return run_info

    def run_scipy(self, method, eps):
        from scipy import optimize

        kwargs = sort_dict(dict(
            method=method,
            eps=eps,
        ))

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def fun(x, *args): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(eps, 1 - eps)]*self.n_opt_params
        settings = OrderedDict()
        settings['eps'] = eps

        if method == 'differential_evolution':
            optimize.differential_evolution(fun, bounds=bounds, popsize=100)
        else:
            optimize.minimize(fun, x0, method=method, bounds=bounds, options=settings)

        run_info = sort_dict(dict(method='run_scipy', kwargs=kwargs))
        return run_info

    def run_skopt(self):
        from skopt import gp_minimize #, forest_minimize

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def fun(x, *args): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            self.prior(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(0, 1)]*self.n_opt_params
        settings = sort_dict(dict(
            acq_func='EI',      # acquisition function
            n_calls=1000,       # number of evaluations of f
            n_random_starts=5,  # number of random initialization
        ))

        _ = gp_minimize(
            fun,                # function to minimize
            bounds,             # bounds on each dimension of x
            x0=list(x0),
            **settings
        )

        run_info = sort_dict(dict(method='run_skopt', settings=settings))
        return run_info

    def run_nlopt(self):
        import nlopt

        def fun(x, grad): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            #print(param_vals)
            self.prior(param_vals)
            #print(param_vals)
            llh = self.loglike(param_vals)
            del param_vals
            return -llh

        # bounds
        lower_bounds = np.zeros(shape=self.n_opt_params)
        upper_bounds = np.ones(shape=self.n_opt_params)

        # for angles make bigger
        for i, name in enumerate(self.hypo_handler.opt_param_names):
            if 'azimuth' in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5
            if 'zenith' in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        # stepsize
        dx = np.zeros(shape=self.n_opt_params)
        for i in range(self.n_opt_params):
            if 'azimuth' in self.hypo_handler.opt_param_names[i]:
                dx[i] = 0.001
            elif 'zenith' in self.hypo_handler.opt_param_names[i]:
                dx[i] = 0.001
            elif self.hypo_handler.opt_param_names[i] in ('x', 'y'):
                dx[i] = 0.005
            elif self.hypo_handler.opt_param_names[i] == 'z':
                dx[i] = 0.002
            elif self.hypo_handler.opt_param_names[i] == 'time':
                dx[i] = 0.01

        # seed from several angles
        #opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
        opt = nlopt.opt(nlopt.GN_CRS2_LM, self.n_opt_params)
        #opt = nlopt.opt(nlopt.LN_PRAXIS, self.n_opt_params)
        opt.set_lower_bounds([0.]*self.n_opt_params)
        opt.set_upper_bounds([1.]*self.n_opt_params)
        opt.set_min_objective(fun)
        opt.set_ftol_abs(0.1)

        # initial guess

        angles = np.linspace(0, 1, 3)
        angles = 0.5 * (angles[1:] + angles[:-1])

        for zen in angles:
            for az in angles:
                x0 = 0.5 * np.ones(shape=self.n_opt_params)

                for i in range(self.n_opt_params):
                    if 'az' in self.hypo_handler.opt_param_names[i]:
                        x0[i] = az
                    elif 'zen' in self.hypo_handler.opt_param_names[i]:
                        x0[i] = zen
                x = opt.optimize(x0) # pylint: disable=unused-variable

        #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
        #local_opt.set_lower_bounds([0.]*self.n_opt_params)
        #local_opt.set_upper_bounds([1.]*self.n_opt_params)
        #local_opt.set_min_objective(fun)
        ##local_opt.set_ftol_abs(0.5)
        ##local_opt.set_ftol_abs(100)
        ##local_opt.set_xtol_rel(10)
        #local_opt.set_ftol_abs(1)
        # global
        #opt = nlopt.opt(nlopt.G_MLSL, self.n_opt_params)
        #opt.set_lower_bounds([0.]*self.n_opt_params)
        #opt.set_upper_bounds([1.]*self.n_opt_params)
        #opt.set_min_objective(fun)
        #opt.set_local_optimizer(local_opt)
        #opt.set_ftol_abs(10)
        #opt.set_xtol_rel(1)
        #opt.set_maxeval(1111)

        #opt = nlopt.opt(nlopt.GN_ESCH, self.n_opt_params)
        #opt = nlopt.opt(nlopt.GN_ISRES, self.n_opt_params)
        #opt = nlopt.opt(nlopt.GN_CRS2_LM, self.n_opt_params)
        #opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND_NOSCAL, self.n_opt_params)
        #opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)

        #opt.set_lower_bounds(lower_bounds)
        #opt.set_upper_bounds(upper_bounds)
        #opt.set_min_objective(fun)
        #opt.set_ftol_abs(0.1)
        #opt.set_population([x0])
        #opt.set_initial_step(dx)

        #local_opt.set_maxeval(10)

        #x = opt.optimize(x0) # pylint: disable=unused-variable

        # polish it up
        #print('***************** polishing ******************')

        #dx = np.ones(shape=self.n_opt_params) * 0.001
        #dx[0] = 0.1
        #dx[1] = 0.1

        #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
        #lower_bounds = np.clip(np.copy(x) - 0.1, 0, 1)
        #upper_bounds = np.clip(np.copy(x) + 0.1, 0, 1)
        #lower_bounds[0] = 0
        #lower_bounds[1] = 0
        #upper_bounds[0] = 0
        #upper_bounds[1] = 0

        #local_opt.set_lower_bounds(lower_bounds)
        #local_opt.set_upper_bounds(upper_bounds)
        #local_opt.set_min_objective(fun)
        #local_opt.set_ftol_abs(0.1)
        #local_opt.set_initial_step(dx)
        #x = opt.optimize(x)

        settings = sort_dict(dict(
            method=opt.get_algorithm_name(),
            ftol_abs=opt.get_ftol_abs(),
            ftol_rel=opt.get_ftol_rel(),
            xtol_abs=opt.get_xtol_abs(),
            xtol_rel=opt.get_xtol_rel(),
            maxeval=opt.get_maxeval(),
            maxtime=opt.get_maxtime(),
            stopval=opt.get_stopval(),
        ))

        run_info = sort_dict(dict(method='run_nlopt', settings=settings))
        return run_info

    def run_multinest(
        self,
        importance_sampling,
        max_modes,
        const_eff,
        n_live,
        evidence_tol,
        sampling_eff,
        max_iter,
        seed,
    ):
        """Setup and run MultiNest on an event.

        See the README file from MultiNest for greater detail on parameters
        specific to to MultiNest (parameters from `importance_sampling` on).

        Parameters
        ----------
        importance_sampling
        max_modes
        const_eff
        n_live
        evidence_tol
        sampling_eff
        max_iter
            Note that this limit is the maximum number of sample replacements
            and _not_ max number of likelihoods evaluated. A replacement only
            occurs when a likelihood is found that exceeds the minimum
            likelihood among the live points.
        seed

        Returns
        -------
        run_info : OrderedDict
            Metadata dict containing MultiNest settings used and extra info returned by
            MultiNest

        """
        # Import pymultinest here; it's a less common dependency, so other
        # functions / constants in this module will still be import-able w/o it.
        import pymultinest

        kwargs = sort_dict(dict(
            importance_sampling=importance_sampling,
            max_modes=max_modes,
            const_eff=const_eff,
            n_live=n_live,
            evidence_tol=evidence_tol,
            sampling_eff=sampling_eff,
            max_iter=max_iter,
            seed=seed,
        ))

        mn_kwargs = sort_dict(dict(
            n_dims=self.n_opt_params,
            n_params=self.n_params,
            n_clustering_params=self.n_opt_params,
            wrapped_params=[
                'az' in p.lower() for p in self.hypo_handler.all_param_names
            ],
            importance_nested_sampling=importance_sampling,
            multimodal=max_modes > 1,
            const_efficiency_mode=const_eff,
            n_live_points=n_live,
            evidence_tolerance=evidence_tol,
            sampling_efficiency=sampling_eff,
            null_log_evidence=-1e90,
            max_modes=max_modes,
            mode_tolerance=-1e90,
            seed=seed,
            log_zero=-1e100,
            max_iter=max_iter,
        ))

        print('Runing MultiNest...')

        fit_meta = {}
        tmpdir = mkdtemp()
        outputfiles_basename = join(tmpdir, '')
        try:
            pymultinest.run(
                LogLikelihood=self.loglike,
                Prior=self.prior,
                verbose=True,
                outputfiles_basename=outputfiles_basename,
                resume=False,
                write_output=True,
                n_iter_before_update=REPORT_AFTER,
                **mn_kwargs
            )
            fit_meta = get_multinest_meta(outputfiles_basename=outputfiles_basename)
        finally:
            rmtree(tmpdir)

        run_info = sort_dict(dict(
            method='run_multinest',
            kwargs=kwargs,
            mn_kwargs=mn_kwargs,
            fit_meta=sort_dict(fit_meta),
        ))
        return run_info


def get_multinest_meta(outputfiles_basename):
    """Get metadata from files that MultiNest writes to disk.

    Parameters
    ----------
    outputfiles_basename : str

    Returns
    -------
    fit_meta : OrderedDict
        Contains "logZ", "logZ_err" and, if importance nested sampling was run,
        "ins_logZ" and "ins_logZ_err"

    """
    fit_meta = OrderedDict()
    if isdir(outputfiles_basename):
        stats_fpath = join(outputfiles_basename, 'stats.dat')
    else:
        stats_fpath = outputfiles_basename + 'stats.dat'

    with open(stats_fpath, 'r') as stats_f:
        stats = stats_f.readlines()

    logZ, logZ_err = None, None
    ins_logZ, ins_logZ_err = None, None

    for line in stats:
        if logZ is None and line.startswith('Nested Sampling Global Log-Evidence'):
            logZ, logZ_err = [float(x) for x in line.split(':')[1].split('+/-')]
        elif ins_logZ is None and line.startswith('Nested Importance Sampling Global Log-Evidence'):
            ins_logZ, ins_logZ_err = [float(x) for x in line.split(':')[1].split('+/-')]

    if logZ is not None:
        fit_meta['logZ'] = logZ
        fit_meta['logZ_err'] = logZ_err
    if ins_logZ is not None:
        fit_meta['ins_logZ'] = ins_logZ
        fit_meta['ins_logZ_err'] = ins_logZ_err

    return fit_meta


def parse_args(description=__doc__):
    """Parse command-line arguments.

    Returns
    -------
    split_kwargs : dict of dicts
        Contains keys "dom_tables_kw", "hypo_kw", "events_kw", and "reco_kw",
        where values are kwargs dicts usable to instantiate or call each of the
        corresponding objects or functions.

    """
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )
    parser.add_argument(
        '--method', required=True, choices=METHODS,
        help='Method to use for performing reconstructions'
    )
    parser.add_argument(
        '--save-llhp', action='store_true',
        help='Whether to save LLHP within 30 LLH of max-LLH to disk'
    )

    split_kwargs = init_obj.parse_args(
        dom_tables=True, tdi_tables=True, events=True, parser=parser
    )

    return split_kwargs


if __name__ == '__main__':
    # pylint: disable=invalid-name
    kwargs = parse_args()
    other_kw = kwargs.pop('other_kw')
    method = other_kw.pop('method')
    kwargs.update(other_kw)
    my_reco = Reco(**kwargs)
    my_reco.run(method=method)
