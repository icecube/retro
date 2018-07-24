#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating, too-many-locals

"""
Instantiate Retro tables and find the max over the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['RetroReco', 'parse_args']

__author__ = 'J.L. Lanfranchi, P. Eller'
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

from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import init_obj
from retro.const import PEGLEG_PARAM_NAMES, SCALING_PARAM_NAMES
from retro.retro_types import EVT_DOM_INFO_T, EVT_HIT_INFO_T
from retro.utils.misc import expand, mkdir, sort_dict
from retro.priors import (
    get_prior_def,
    get_prior_fun,
    PRI_UNIFORM,
    PRI_LOG_NORMAL,
    PRI_LOG_UNIFORM,
)
from retro.hypo.discrete_muon_kernels import pegleg_eval


class RetroReco(object):

    def __init__(self, dom_tables_kw, hypo_kw, events_kw, reco_kw):
        self.dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)

        # check tables are finite:
        for tbl in self.dom_tables.tables:
            assert np.sum(~np.isfinite(tbl['weight'])) == 0, 'table not finite!'
            assert np.sum(tbl['weight'] < 0) == 0, 'table is negative!'
            assert np.min(tbl['index']) >= 0, 'table has negative index'
            assert np.max(tbl['index']) < self.dom_tables.template_library.shape[0], 'table too large index'
        assert np.sum(~np.isfinite(self.dom_tables.template_library)) == 0, 'templates not finite!'
        assert np.sum(self.dom_tables.template_library < 0) == 0, 'templates not finite!'

        self.hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
        self.events_iterator = init_obj.get_events(**events_kw)
        self.reco_kw = reco_kw

        self.n_params = self.hypo_handler.n_params
        self.n_opt_params = self.hypo_handler.n_opt_params

        self.out_prefix = None

        # Setup priors
        self.prior_defs = OrderedDict()
        for param in self.hypo_handler.opt_param_names:
            self.prior_defs[param] = get_prior_def(param, reco_kw)

        # Remove unused reco kwargs
        reco_kw.pop('spatial_prior')
        reco_kw.pop('cascade_angle_prior')

    def run(self, method):
        """Run reconstructions.

        Parameters
        ----------
        method : str
            One of "multinest", "nlopt", "scipy", or "skopt".

        """
        print('Running reconstructions...')
        t00 = time.time()
        outdir = self.reco_kw.pop('outdir')
        outdir = expand(outdir)
        mkdir(outdir)

        # Setup LLHP dtype
        dim_names = list(self.hypo_handler.all_param_names)
        llhp_t = np.dtype([(field, np.float32) for field in ['llh'] + dim_names])

        for event_idx, event in self.events_iterator:
            t0 = time.time()

            self.out_prefix = join(outdir, 'evt{}-'.format(event_idx))
            print('Output files prefix: "{}"\n'.format(self.out_prefix))

            prior, priors_used = self.generate_prior(event)
            param_values = []
            log_likelihoods = []
            t_start = []
            loglike = self.generate_loglike(
                event=event,
                param_values=param_values,
                log_likelihoods=log_likelihoods,
                t_start=t_start,
            )

            if method == 'multinest':
                settings = self.run_multinest(
                    prior=prior,
                    loglike=loglike,
                    **self.reco_kw
                )
            elif method == 'scipy':
                settings = self.run_scipy(
                    prior=prior,
                    loglike=loglike,
                    method='differential_evolution',
                    eps=0.02
                )
            elif method == 'nlopt':
                settings = self.run_nlopt(
                    prior=prior,
                    loglike=loglike,
                )
            elif method == 'skopt':
                settings = self.run_skopt(
                    prior=prior,
                    loglike=loglike,
                )
            else:
                raise ValueError('Unknown `Method` {}'.format(method))

            t1 = time.time()

            # dump
            llhp = np.empty(shape=len(param_values), dtype=llhp_t)
            llhp['llh'] = log_likelihoods
            llhp[dim_names] = param_values

            llhp_outf = self.out_prefix + 'llhp.npy'
            print('Saving llhp to "{}"...'.format(llhp_outf))
            np.save(llhp_outf, llhp)

            opt_meta = OrderedDict([
                ('params', dim_names),
                ('original_prior_specs', self.prior_defs),
                ('priors_used', priors_used),
                ('settings', sort_dict(settings)),
            ])

            opt_meta['num_llhp'] = len(param_values)
            opt_meta['run_time'] = t1 - t0
            opt_meta_outf = self.out_prefix + 'opt_meta.pkl'
            print('Saving metadata to "{}"'.format(opt_meta_outf))
            pickle.dump(
                opt_meta,
                open(opt_meta_outf, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL
            )

            dt = time.time() - t1
            n_points = llhp.size
            print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
                  .format(dt, n_points, dt / n_points * 1e3))

        print('Total script run time is {:.3f} s'.format(time.time() - t00))

    def generate_prior(self, event):
        """Generate the prior transform functions + info for a given event.

        Parameters
        ----------
        event

        Returns
        -------
        prior : callable
        priors_used : OrderedDict

        """
        prior_funcs = []
        priors_used = OrderedDict()

        for dim_num, dim_name in enumerate(self.hypo_handler.all_param_names):
            if dim_name in PEGLEG_PARAM_NAMES + SCALING_PARAM_NAMES:
                continue
            prior_fun, prior_def = get_prior_fun(
                dim_num=dim_num,
                dim_name=dim_name,
                prior_def=self.prior_defs[dim_name],
                event=event,
            )
            prior_funcs.append(prior_fun)
            priors_used[dim_name] = prior_def

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
            #print(cube)
            #print([cube[i] for i in range(6)])

        return prior, priors_used

    def generate_loglike(self, event, param_values, log_likelihoods, t_start):
        """Generate the LLH callback function for a given event

        Parameters
        ----------
        event
        param_values : list
        log_likelihoods : list
        t_start : list

        Returns
        -------
        loglike : callable

        """
        # -- Variables to be captured by `loglike` closure -- #

        report_after = 100

        all_param_names = self.hypo_handler.all_param_names
        n_opt_params = self.n_opt_params

        hits = event['hits']
        hits_indexer = event['hits_indexer']
        hypo_handler = self.hypo_handler
        pegleg_muon_dt = hypo_handler.pegleg_kernel_kwargs.get('dt')
        pegleg_muon_const_e_loss = False

        get_llh = self.dom_tables._get_llh # pylint: disable=protected-access
        dom_info = self.dom_tables.dom_info
        tables = self.dom_tables.tables
        table_norms = self.dom_tables.table_norms
        t_indep_tables = self.dom_tables.t_indep_tables
        t_indep_table_norms = self.dom_tables.t_indep_table_norms
        sd_idx_table_indexer = self.dom_tables.sd_idx_table_indexer
        time_window = (
            event['hits_summary']['time_window_stop']
            - event['hits_summary']['time_window_start']
        )

        truth_info = OrderedDict()
        truth_info['x'] = event['truth']['x']
        truth_info['y'] = event['truth']['y']
        truth_info['z'] = event['truth']['z']
        truth_info['time'] = event['truth']['time']
        truth_info['zenith'] = np.arccos(event['truth']['coszen'])
        truth_info['azimuth'] = event['truth']['azimuth']
        truth_info['track_azimuth'] = event['truth']['longest_daughter_azimuth']
        truth_info['track_zenith'] = np.arccos(event['truth']['longest_daughter_coszen'])
        truth_info['track_energy'] = event['truth']['longest_daughter_energy']
        truth_info['cascade_azimuth'] = event['truth']['cascade_azimuth']
        truth_info['cascade_zenith'] = np.arccos(event['truth']['cascade_coszen'])
        truth_info['cascade_energy'] = event['truth']['cascade_energy']
        truth_info['neutrino_energy'] = event['truth']['energy']

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

        print('all noise rate %.5f'%np.sum(dom_info['noise_rate_per_ns']))
        print('DOMs with zero noise %i'%np.sum(dom_info['noise_rate_per_ns'] == 0))

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
        print('min noise: ',np.min(noise))
        print('mean noise: ',np.mean(noise))

        assert np.sum(event_dom_info['quantum_efficiency'] <= 0) == 0, 'negative QE'
        assert np.sum(event_dom_info['total_observed_charge']) > 0, 'no charge'
        assert np.isfinite(np.sum(event_dom_info['total_observed_charge'])), 'inf charge'


        def loglike(cube, ndim=None, nparams=None): # pylint: disable=unused-argument
            """Get log likelihood values.

            Defined as a closure to capture particulars of the event and priors without
            having to pass these as parameters to the function.

            Note that this is called _after_ `prior` has been called, so `cube`
            already contains the parameter values scaled to be in their physical
            ranges.

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
            if not t_start:
                t_start.append(time.time())

            hypo = OrderedDict(list(zip(all_param_names, cube)))

            generic_sources = hypo_handler.get_generic_sources(hypo)
            pegleg_sources = hypo_handler.get_pegleg_sources(hypo)
            scaling_sources = hypo_handler.get_scaling_sources(hypo)

            llh, pegleg_idx, scalefactor = get_llh(
                generic_sources=generic_sources,
                pegleg_sources=pegleg_sources,
                scaling_sources=scaling_sources,
                event_hit_info=event_hit_info,
                time_window=time_window,
                event_dom_info=event_dom_info,
                tables=tables,
                table_norms=table_norms,
                t_indep_tables=t_indep_tables,
                t_indep_table_norms=t_indep_table_norms,
            )
            assert np.isfinite(llh), 'LLH not finite'
            assert llh < 0, 'LLH positive'

            pegleg_result = pegleg_eval(
                pegleg_idx=pegleg_idx,
                dt=pegleg_muon_dt,
                const_e_loss=pegleg_muon_const_e_loss,
            )
            result = tuple(cube[:n_opt_params]) + (pegleg_result, scalefactor)
            param_values.append(result)

            log_likelihoods.append(llh)
            n_calls = len(log_likelihoods)
            t1 = time.time()

            if n_calls % report_after == 0:
                print('')
                msg = 'truth:                '
                for key, val in zip(all_param_names, result):
                    msg += ' %s=%.1f'%(key, truth_info[key])
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
                print('avg time per llh: {:.3f} ms'.format((t_now - t_start[0])/n_calls*1000))
                print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
                print('')

            return llh

        return loglike

    def run_scipy(self, prior, loglike, method, eps):
        from scipy import optimize

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def fun(x, *args): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            prior(param_vals)
            llh = loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(eps, 1 - eps)]*self.n_opt_params
        settings = OrderedDict()
        settings['eps'] = eps

        if method == 'differential_evolution':
            optimize.differential_evolution(fun, bounds=bounds, popsize=100)
        else:
            optimize.minimize(fun, x0, method=method, bounds=bounds, options=settings)

        return settings

    def run_skopt(self, prior, loglike):
        from skopt import gp_minimize #, forest_minimize

        # initial guess
        x0 = 0.5 * np.ones(shape=self.n_opt_params)

        def fun(x, *args): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            prior(param_vals)
            llh = loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(0, 1)]*self.n_opt_params
        settings = OrderedDict()

        _ = gp_minimize(
            fun,                # function to minimize
            bounds,             # bounds on each dimension of x
            acq_func="EI",      # acquisition function
            n_calls=1000,       # number of evaluations of f
            n_random_starts=5,  # number of random initialization
            x0=list(x0),
        )

        return settings

    def run_nlopt(self, prior, loglike):
        import nlopt

        def fun(x, grad): # pylint: disable=unused-argument, missing-docstring
            param_vals = np.copy(x)
            #print(param_vals)
            prior(param_vals)
            #print(param_vals)
            llh = loglike(param_vals)
            del param_vals
            return -llh

        # bounds
        lower_bounds = np.zeros(shape=self.n_opt_params)
        upper_bounds = np.ones(shape=self.n_opt_params)
        # for angles make bigger
        for i, name in enumerate(self.hypo_handler.generic_param_names):
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
            if 'azimuth' in self.hypo_handler.generic_param_names[i]:
                dx[i] = 0.001
            elif 'zenith' in self.hypo_handler.generic_param_names[i]:
                dx[i] = 0.001
            elif self.hypo_handler.generic_param_names[i] in ('x', 'y'):
                dx[i] = 0.005
            elif self.hypo_handler.generic_param_names[i] == 'z':
                dx[i] = 0.002
            elif self.hypo_handler.generic_param_names[i] == 'time':
                dx[i] = 0.01

        # does not converge :/
        local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_params)
        local_opt.set_lower_bounds([0.]*self.n_opt_params)
        local_opt.set_upper_bounds([1.]*self.n_opt_params)
        local_opt.set_min_objective(fun)
        #local_opt.set_ftol_abs(0.5)
        #local_opt.set_ftol_abs(100)
        #local_opt.set_xtol_rel(10)
        local_opt.set_ftol_abs(1)
        opt = nlopt.opt(nlopt.G_MLSL, self.n_opt_params)
        opt.set_lower_bounds([0.]*self.n_opt_params)
        opt.set_upper_bounds([1.]*self.n_opt_params)
        opt.set_min_objective(fun)
        opt.set_local_optimizer(local_opt)
        opt.set_ftol_abs(10)
        opt.set_xtol_rel(1)
        opt.set_maxeval(1111)

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

        x = opt.optimize(x0) # pylint: disable=unused-variable

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

        settings = OrderedDict()
        settings['method'] = opt.get_algorithm_name()
        settings['ftol_abs'] = opt.get_ftol_abs()
        settings['ftol_rel'] = opt.get_ftol_rel()
        settings['xtol_abs'] = opt.get_xtol_abs()
        settings['xtol_rel'] = opt.get_xtol_rel()
        settings['maxeval'] = opt.get_maxeval()
        settings['maxtime'] = opt.get_maxtime()
        settings['stopval'] = opt.get_stopval()

        return settings

    def run_multinest(
            self,
            prior,
            loglike,
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
        prior
        loglike
        importance_sampling
        max_modes
        const_eff
        n_live
        evidence_tol
        sampling_eff
        max_iter
            Note that this limit is the maximum number of sample replacements and
            _not_ max number of likelihoods evaluated. A replacement only occurs
            when a likelihood is found that exceeds the minimum likelihood among
            the live points.
        seed

        Returns
        -------
        settings : OrderedDict
            Metadata used for running MultiNest, i.e.
            the keyword args used to invoke the `pymultinest.run` function.

        """
        # Import pymultinest here; it's a less common dependency, so other
        # functions / constants in this module will still be import-able w/o it.
        import pymultinest

        settings = OrderedDict([
            ('n_dims', self.n_opt_params),
            ('n_params', self.n_params),
            ('n_clustering_params', self.n_opt_params),
            ('wrapped_params', [
                ('azimuth' in p.lower()) for p in self.hypo_handler.all_param_names
            ]),
            ('importance_nested_sampling', importance_sampling),
            ('multimodal', max_modes > 1),
            ('const_efficiency_mode', const_eff),
            ('n_live_points', n_live),
            ('evidence_tolerance', evidence_tol),
            ('sampling_efficiency', sampling_eff),
            ('null_log_evidence', -1e90),
            ('max_modes', max_modes),
            ('mode_tolerance', -1e90),
            ('seed', seed),
            ('log_zero', -1e100),
            ('max_iter', max_iter),
        ])

        print('Runing MultiNest...')
        pymultinest.run(
            LogLikelihood=loglike,
            Prior=prior,
            verbose=True,
            outputfiles_basename=self.out_prefix,
            resume=False,
            write_output=False,
            n_iter_before_update=5000,
            **settings
        )

        return settings


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

    group = parser.add_argument_group(
        title='Hypothesis parameter priors',
    )

    group.add_argument(
        '--spatial-prior',
        choices='dc dc_subdust ic SPEFit2'.split(),
        required=True,
        help='''Choose a prior for choosing spatial samples. "dc", "dc_subdust"
        and "ic" are uniform priors with hard cut-offs at the extents of the
        respective volumes, while "SPEFit2" samples from Cauchy distributions
        around the SPEFit2 (x, y, z) best-fit values.'''
    )
    group.add_argument(
        '--temporal-prior',
        choices='uniform SPEFit2'.split(),
        required=True,
        help='''Choose a prior for choosing temporal samples. "uniform" chooses
        uniformly from 4000 ns prior to the first hit up to the last hit, while
        "SPEFit2" samples from a Cauchy distribution near (not *at* due to
        bias) the SPEFit2 time best-fit value.'''
    )
    group.add_argument(
        '--cascade-angle-prior',
        choices=[PRI_UNIFORM, PRI_LOG_NORMAL],
        required=False,
        help='''Prior to put on opening angle between track and cascade.'''
    )
    group.add_argument(
        '--cascade-energy-prior',
        choices=[PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL],
        required=False,
        help='''Prior to put on _total_ event cascade-energy. Must specify
        --cascade-energy-lims.'''
    )
    group.add_argument(
        '--cascade-energy-lims', nargs='+',
        required=False,
        help='''Lower and upper cascade-energy limits, in GeV. E.g.: --cascade-energy-lims=1,100
        Required if --cascade-energy-prior is {}, {}, or {}'''
        .format(PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL)
    )

    group.add_argument(
        '--track-energy-prior',
        choices=[PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL],
        required=False,
        help='''Lower and upper cascade-energy limits, in GeV. E.g.: --cascade-energy-lims=1,100
        Required if --cascade-energy-prior is {}, {}, or {}'''
        .format(PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL)
    )

    group.add_argument(
        '--track-energy-lims', nargs='+',
        required=False,
        help='''Lower and upper track-energy limits, in GeV. E.g.: --track-energy-lims=1,100
        Required if --track-energy-prior is {}, {}, or {}'''
        .format(PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL)
    )

    group = parser.add_argument_group(
        title='MultiNest parameters',
    )

    group.add_argument(
        '--importance-sampling', action='store_true',
        help='''Importance nested sampling (INS) mode. Could be more efficient,
        but also can be unstable. Does not work with multimodal.'''
    )
    group.add_argument(
        '--max-modes', type=int, required=True,
        help='''Set to 1 to disable multi-modal search. Must be 1 if --importance-sampling is
        specified.'''
    )
    group.add_argument(
        '--const-eff', action='store_true',
        help='''Constant efficiency mode.'''
    )
    group.add_argument(
        '--n-live', type=int, required=True
    )
    group.add_argument(
        '--evidence-tol', type=float, required=True
    )
    group.add_argument(
        '--sampling-eff', type=float, required=True
    )
    group.add_argument(
        '--max-iter', type=int, required=True,
        help='''Note that iterations of the MultiNest algorithm are _not_ the
        number of likelihood evaluations. An iteration comes when one live
        point is discarded by finding a sample with higher likelihood than at
        least one other live point. Such a point can take many likelihood
        evaluatsions to find.'''
    )
    group.add_argument(
        '--seed', type=int, required=True,
        help='''Integer seed for MultiNest's random number generator.'''
    )

    split_kwargs = init_obj.parse_args(
        dom_tables=True, hypo=True, events=True, parser=parser
    )

    split_kwargs['reco_kw'] = reco_kw = split_kwargs.pop('other_kw')

    if reco_kw['cascade_energy_prior'] in [PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL]:
        assert reco_kw['cascade_energy_lims'] is not None
        elims = ''.join(reco_kw['cascade_energy_lims'])
        elims = [float(l) for l in elims.split(',')]
        reco_kw['cascade_energy_lims'] = elims
    elif reco_kw['cascade_energy_prior'] is None:
        reco_kw.pop('cascade_energy_prior')
        reco_kw.pop('cascade_energy_lims')
    elif reco_kw['cascade_energy_lims'] is not None:
        raise ValueError('--cascade-energy-lims not used with cascade_energy prior {}'
                         .format(reco_kw['cascade_energy_prior']))

    if reco_kw['track_energy_prior'] in [PRI_UNIFORM, PRI_LOG_UNIFORM, PRI_LOG_NORMAL]:
        assert reco_kw['track_energy_lims'] is not None
        elims = ''.join(reco_kw['track_energy_lims'])
        elims = [float(l) for l in elims.split(',')]
        reco_kw['track_energy_lims'] = elims
    elif reco_kw['track_energy_prior'] is None:
        reco_kw.pop('track_energy_prior')
        reco_kw.pop('track_energy_lims')
    elif reco_kw['track_energy_lims'] is not None:
        raise ValueError('--track-energy-lims not used with track_energy prior {}'
                         .format(reco_kw['track_energy_prior']))

    return split_kwargs


if __name__ == '__main__':
    my_reco = RetroReco(**parse_args()) # pylint: disable=invalid-name
    my_reco.run(method='multinest')
