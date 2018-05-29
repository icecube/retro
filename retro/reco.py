#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating, too-many-locals

"""
Instantiate Retro tables and find the max over the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'reco',
    'parse_args'
]

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
from math import acos, exp

from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np
from scipy import stats

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import init_obj
from retro.const import TWO_PI, ALL_STRS_DOMS, EMPTY_SOURCES, SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV
from retro.retro_types import PARAM_NAMES, EVT_DOM_INFO_T, EVT_HIT_INFO_T
from retro.utils.misc import expand, mkdir, sort_dict
from retro.priors import *
from retro.hypo.discrete_muon_kernels import pegleg_eval


class retro_reco(object):

    def __init__(self, dom_tables_kw, hypo_kw, events_kw, reco_kw):

        self.dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
        self.hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
        self.events_iterator = init_obj.get_events(**events_kw)
        self.reco_kw = reco_kw
        self.opt_params = self.hypo_handler.params
        self.n_opt_dim = len(self.opt_params)

        # setup priors
        self.prior_defs = OrderedDict()
        for param in self.opt_params:
            self.prior_defs[param] = get_prior_def(param, reco_kw)
        # keyword fuckery
        reco_kw.pop('spatial_prior')
        reco_kw.pop('cascade_angle_prior')


    def run(self):
        print('Running reconstructions...')
        t00 = time.time()
        outdir = self.reco_kw.pop('outdir')
        outdir = expand(outdir)
        mkdir(outdir)

        #setup LLHP dtype
        hypo_params = self.opt_params + self.hypo_handler.pegleg_params + self.hypo_handler.scaling_params
        hypo_params_sorted = ['llh'] + [dim for dim in PARAM_NAMES if dim in hypo_params]
        LLHP_T = np.dtype([(n, np.float32) for n in hypo_params_sorted])

        for event_idx, event in self.events_iterator: # pylint: disable=unused-variable
            t0 = time.time()

            self.out_prefix = join(outdir, 'evt{}-'.format(event_idx))
            print('Output files prefix: "{}"\n'.format(self.out_prefix))

            prior, priors_used = self.generate_prior(event)
            param_values = []
            log_likelihoods = []
            t_start = []
            loglike = self.generate_loglike(event, param_values, log_likelihoods, t_start)

            #settings = self.run_multinest(
            #    event_idx=event_idx,
            #    event=event,
            #    prior=prior,
            #    loglike=loglike,
            #    **self.reco_kw
            #)
            #settings = self.run_scipy(
            #    event_idx=event_idx,
            #    event=event,
            #    prior=prior,
            #    loglike=loglike,
            #    method='differential_evolution',
            #    eps=0.02
            #)
            settings = self.run_nlopt(
                event_idx=event_idx,
                event=event,
                prior=prior,
                loglike=loglike,
            )
            #settings = self.run_skopt(
            #    event_idx=event_idx,
            #    event=event,
            #    prior=prior,
            #    loglike=loglike,
            #)

            t1 = time.time()

            # dump
            llhp = np.empty(shape=len(param_values), dtype=LLHP_T)
            llhp['llh'] = log_likelihoods
            llhp[hypo_params] = param_values

            llhp_outf = self.out_prefix + 'llhp.npy'
            print('Saving llhp to "{}"...'.format(llhp_outf))
            np.save(llhp_outf, llhp)

            opt_meta = OrderedDict([
                ('params', hypo_params),
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
        '''
        generate the prior transform functions + info for a given events

        '''
        prior_funcs = []
        priors_used = OrderedDict()

        for dim_num, dim_name in enumerate(self.opt_params):
            prior_fun, prior_def = get_prior_fun(dim_num, dim_name, self.prior_defs[dim_name], event)
            prior_funcs.append(prior_fun)
            priors_used[dim_name] = prior_def

        def prior(cube, ndim=None, nparams=None): # pylint: disable=unused-argument
            """Function for pymultinest to translate the hypercube MultiNest uses
            (each value is in [0, 1]) into the dimensions of the parameter space.

            Note that the cube dimension names are defined in module variable
            `CUBE_DIMS` for reference elsewhere.
            """
            for prior_func in prior_funcs:
                prior_func(cube)

        return prior, priors_used


    def generate_loglike(self, event, param_values, log_likelihoods, t_start):
        '''
        generate the LLH callback function for a given event

        Parameters:
        -----------

        event

        param_values : list
        log_likelihoods : list
        t_start : list


        '''
        report_after = 1

        hypo_params = self.opt_params + self.hypo_handler.pegleg_params + self.hypo_handler.scaling_params
        opt_params = self.opt_params

        # --- define here stuff for closure ---
        hits = event['hits']
        hits_indexer = event['hits_indexer']
        hypo_handler = self.hypo_handler
        get_llh = self.dom_tables._get_llh # pylint: disable=protected-access
        dom_info = self.dom_tables.dom_info
        tables = self.dom_tables.tables
        table_norm = self.dom_tables.table_norm
        t_indep_tables = self.dom_tables.t_indep_tables
        t_indep_table_norm = self.dom_tables.t_indep_table_norm
        sd_idx_table_indexer = self.dom_tables.sd_idx_table_indexer
        time_window = np.float32(
            event['hits_summary']['time_window_stop'] - event['hits_summary']['time_window_start']
        )

        n_operational_doms = np.sum(dom_info['operational'])
        # array containing all relevant DOMs for the event and the hit information
        event_dom_info = np.zeros(shape=(n_operational_doms,), dtype=EVT_DOM_INFO_T)
        event_hit_info = np.zeros(shape=hits.shape, dtype=EVT_HIT_INFO_T)

        event_hit_info['time'][:] = hits['time']
        event_hit_info['charge'][:] = hits['charge']

        #loop through all DOMs to fill array:
        position = 0
        for sd_idx in ALL_STRS_DOMS:
            if not dom_info[sd_idx]['operational']:
                continue
            event_dom_info[position]['x'] = dom_info[sd_idx]['x']
            event_dom_info[position]['y'] = dom_info[sd_idx]['y']
            event_dom_info[position]['z'] = dom_info[sd_idx]['z']
            event_dom_info[position]['quantum_efficiency'] = dom_info[sd_idx]['quantum_efficiency']
            event_dom_info[position]['noise_rate_per_ns'] = dom_info[sd_idx]['noise_rate_per_ns']
            event_dom_info[position]['table_idx'] = sd_idx_table_indexer[sd_idx]


            # add hits indices
            # super shitty way at the moment, due to legacy way of doing things
            if sd_idx in event['hits_indexer']['sd_idx']:
                hit_idx = list(event['hits_indexer']['sd_idx']).index(sd_idx)
                start = event['hits_indexer'][hit_idx]['offset']
                stop = start + event['hits_indexer'][hit_idx]['num']
                event_dom_info[position]['hits_start_idx'] = start
                event_dom_info[position]['hits_stop_idx'] = stop
                for h_idx in range(start, stop):
                    event_hit_info[h_idx]['dom_idx'] = position
                    event_dom_info[position]['total_observed_charge'] += hits[h_idx]['charge']

            position += 1

        #print('evt dom info: ', event_dom_info)
        #print('tot charge: ', np.sum(event_dom_info['total_observed_charge']))

        # --------------------------------------------


        def loglike(cube, ndim=None, nparams=None): # pylint: disable=unused-argument
            """Function pymultinest calls to get llh values.

            Note that this is called _after_ `prior` has been called, so `cube`
            already contains the parameter values scaled to be in their physical
            ranges.

            """
            if not t_start:
                t_start.append(time.time())

            hypo = dict(zip(opt_params, cube))

            sources = hypo_handler.get_sources(hypo)
            pegleg_sources = hypo_handler.get_pegleg_sources(hypo)
            scaling_sources = hypo_handler.get_scaling_sources(hypo)

            t0 = time.time()
            llh, pegleg_idx, scalefactor = get_llh(
                sources=sources,
                pegleg_sources=pegleg_sources,
                scaling_sources=scaling_sources,
                event_hit_info=event_hit_info,
                time_window=time_window,
                event_dom_info=event_dom_info,
                tables=tables,
                table_norm=table_norm,
                t_indep_tables=t_indep_tables,
                t_indep_table_norm=t_indep_table_norm,
            )
            t1 = time.time()

            # ToDo, this is just for testing
            pegleg_result = pegleg_eval(pegleg_idx)
            result = tuple([float(cube[i]) for i in range(len(opt_params))] + [pegleg_result] + [scalefactor])

            param_values.append(result)
            log_likelihoods.append(llh)

            n_calls = len(log_likelihoods)

            if n_calls % report_after == 0:
                print('')
                t_now = time.time()
                best_idx = np.argmax(log_likelihoods)
                best_llh = log_likelihoods[best_idx]
                best_p = param_values[best_idx]
                msg = 'best llh = {:.3f} @ '.format(best_llh)
                for key, val in zip(hypo_params, best_p):
                    msg += ' %s=%.1f'%(key, val)
                print(msg)
                msg = 'this llh = {:.3f} @ '.format(llh)
                for key, val in zip(hypo_params, result):
                    msg += ' %s=%.1f'%(key, val)
                print(msg)
                print('{} LLH computed'.format(n_calls))
                print('avg time per llh: {:.3f} ms'.format((t_now - t_start[0])/n_calls*1000))
                print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
                print('')

            return llh

        return loglike


    def run_scipy(
            self,
            event_idx,
            event,
            prior,
            loglike,
            method,
            eps):

        from scipy import optimize

        # initial guess
        x0 = 0.5 * np.ones(shape=(self.n_opt_dim,))

        def fun(x, *args):
            param_vals = np.copy(x)
            prior(param_vals)    
            llh = loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(eps,1-eps)]*self.n_opt_dim
        settings = OrderedDict()
        settings['eps'] = eps

        if method == 'differential_evolution':
            optimize.differential_evolution(fun, bounds=bounds, popsize=100)
        else:
            optimize.minimize(fun, x0, method=method, bounds=bounds, options=settings)
        return settings

    def run_skopt(
            self,
            event_idx,
            event,
            prior,
            loglike,
            ):

        from skopt import gp_minimize, forest_minimize

        # initial guess
        x0 = 0.5 * np.ones(shape=(self.n_opt_dim,))

        def fun(x, *args):
            param_vals = np.copy(x)
            prior(param_vals)    
            llh = loglike(param_vals)
            del param_vals
            return -llh

        bounds = [(0,1)]*self.n_opt_dim
        settings = OrderedDict()

        res = gp_minimize(fun,                  # the function to minimize
                          bounds,      # the bounds on each dimension of x
                          acq_func="EI",      # the acquisition function
                          n_calls=1000,         # the number of evaluations of f 
                          n_random_starts=5,  # the number of random initialization 
                          x0=list(x0),
                          )

        return settings

    def run_nlopt(
            self,
            event_idx,
            event,
            prior,
            loglike,
            ):

        import nlopt

        def fun(x, grad):
            param_vals = np.copy(x)
            #print(param_vals)
            prior(param_vals)
            #print(param_vals)
            llh = loglike(param_vals)
            del param_vals
            return -llh

        # bounds
        lower_bounds = np.zeros(shape=(self.n_opt_dim,))
        upper_bounds = np.ones(shape=(self.n_opt_dim,))
        # for angles make bigger
        for i, name in enumerate(self.opt_params):
            if 'azimuth' in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5
            if 'zenith' in name:
                lower_bounds[i] = -0.5
                upper_bounds[i] = 1.5

        # initial guess
        x0 = 0.5 * np.ones(shape=(self.n_opt_dim,))

        # stepsize
        dx = np.zeros(shape=(self.n_opt_dim,))
        for i in range(self.n_opt_dim):
            if 'azimuth' in self.hypo_handler.params[i]:
                dx[i] = 0.001
            elif 'zenith' in self.hypo_handler.params[i]:
                dx[i] = 0.001
            elif self.hypo_handler.params[i] in ['x','y']:
                dx[i] = 0.005
            elif self.hypo_handler.params[i] == 'z':
                dx[i] = 0.002
            elif self.hypo_handler.params[i] == 'time':
                dx[i] = 0.01


        # does not converge :/
        local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_dim)
        local_opt.set_lower_bounds([0.]*self.n_opt_dim)
        local_opt.set_upper_bounds([1.]*self.n_opt_dim)
        local_opt.set_min_objective(fun)
        #local_opt.set_ftol_abs(0.5)
        #local_opt.set_ftol_abs(100)
        #local_opt.set_xtol_rel(10)
        local_opt.set_ftol_abs(1)
        opt = nlopt.opt(nlopt.G_MLSL, self.n_opt_dim)
        opt.set_lower_bounds([0.]*self.n_opt_dim)
        opt.set_upper_bounds([1.]*self.n_opt_dim)
        opt.set_min_objective(fun)
        opt.set_local_optimizer(local_opt)
        opt.set_ftol_abs(10)
        opt.set_xtol_rel(1)
        opt.set_maxeval(1111)

        #opt = nlopt.opt(nlopt.GN_ESCH, self.n_opt_dim)
        #opt = nlopt.opt(nlopt.GN_ISRES, self.n_opt_dim)
        #opt = nlopt.opt(nlopt.GN_CRS2_LM, self.n_opt_dim)
        #opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND_NOSCAL, self.n_opt_dim)
        #opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_dim)

        #opt.set_lower_bounds(lower_bounds)
        #opt.set_upper_bounds(upper_bounds)
        #opt.set_min_objective(fun)
        #opt.set_ftol_abs(0.1)
        #opt.set_population([x0])
        #opt.set_initial_step(dx)

        #local_opt.set_maxeval(10)

        x = opt.optimize(x0)

        # polish it up
        #print('***************** polishing ******************')

        #dx = np.ones(shape=(self.n_opt_dim,)) * 0.001
        #dx[0] = 0.1
        #dx[1] = 0.1

        #local_opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.n_opt_dim)
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
            event_idx,
            event,
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
        event_idx
        event
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
        llhp : shape (num_llh,) structured array of dtype retro.LLHP_T
            LLH and the corresponding parameter values.

        settings : OrderedDict
            Metadata used for running MultiNest, i.e.
            the keyword args used to invoke the `pymultinest.run` function.

        """
        # pylint: disable=missing-docstring
        # Import pymultinest here; it's a less common dependency, so other
        # functions / constants in this module will still be import-able w/o it.
        import pymultinest

        settings = OrderedDict([
            ('n_dims', self.n_opt_dim),
            ('n_params', self.n_opt_dim),
            ('n_clustering_params', self.n_opt_dim),
            ('wrapped_params', [int('azimuth' in p.lower()) for p in self.opt_params]),
            ('importance_nested_sampling', importance_sampling),
            ('multimodal', max_modes > 1),
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
    my_reco = retro_reco(**parse_args())
    my_reco.run()
