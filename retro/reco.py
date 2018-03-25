#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Instantiate Retro tables and find the max over the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['CUBE_DIMS', 'PRI_UNIFORM', 'PRI_LOG_UNIFORM', 'run_multinest',
           'reco', 'parse_args']

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
import math

from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, LLHP_T, init_obj
from retro.const import TWO_PI
from retro.utils.misc import expand, mkdir, sort_dict
from retro.likelihood import get_llh


CUBE_DIMS = ['x', 'y', 'z', 't', 'track_zenith', 'track_azimuth', 'energy', 'track_fraction']
PRI_UNIFORM = 0
PRI_LOG_UNIFORM = 1


def run_multinest(
        outdir,
        event_idx,
        llh_kw,
        t_lims,
        spatial_lims,
        energy_lims,
        energy_prior,
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
    outdir
    event_idx
    llh_kw
    t_lims
    spatial_lims
    energy_lims
    energy_prior
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
    llhp : shape (N_llh,) structured array of dtype retro.LLHP_T
        LLH and the corresponding parameter values.

    mn_meta : OrderedDict
        Metadata used for running MultiNest, including priors, parameters, and
        the keyword args used to invoke the `pymultinest.run` function.

    """
    # Import pymultinest here since it's a "difficult" dependency
    import pymultinest

    if energy_prior == 'uniform':
        energy_prior = PRI_UNIFORM
        e_min = np.min(energy_lims)
        e_range = np.max(energy_lims) - e_min
    elif energy_prior == 'log_uniform':
        energy_prior = PRI_LOG_UNIFORM
        log_e_min = np.log(np.min(energy_lims))
        log_e_range = np.log(np.max(energy_lims) / np.min(energy_lims))
    else:
        raise ValueError(str(energy_prior))

    energy_lims = (np.min(energy_lims), np.max(energy_lims))

    if isinstance(spatial_lims, basestring):
        spatial_lims = spatial_lims.strip().lower()
        if spatial_lims == 'ic':
            x_lims = [-860., 870.]
            y_lims = [-780., 770.]
            z_lims = [-780., 790.]
        elif spatial_lims == 'dc':
            x_lims = [-150., 270.]
            y_lims = [-210., 150.]
            z_lims = [-770., 760.]
        elif spatial_lims == 'dc_subdust':
            x_lims = [-150., 270.]
            y_lims = [-210., 150.]
            z_lims = [-610., -60.]
        else:
            raise ValueError(spatial_lims)
    else:
        raise ValueError(str(spatial_lims))

    priors = OrderedDict([
        ('x', ('uniform', x_lims)),
        ('y', ('uniform', y_lims)),
        ('z', ('uniform', z_lims)),
        ('t', ('uniform', t_lims)),
        ('track_cosen', ('uniform', (-1, 1))),
        ('track_azimuth', ('uniform', (0, 2*np.pi))),
        ('energy', (energy_prior, energy_lims)),
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
    t_start = []

    report_after = 500

    def prior(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function for pymultinest to translate the hypercube MultiNest uses
        (each value is in [0, 1]) into the dimensions of the parameter space.

        Note that the cube dimension names are defined in module variable
        `CUBE_DIMS` for reference elsewhere.

        """
        # Note: cube[7], track_fraction, needs no xform, as it is in [0, 1]

        # x, y, z, t
        cube[0] = cube[0] * x_range + x_min
        cube[1] = cube[1] * y_range + y_min
        cube[2] = cube[2] * z_range + z_min
        cube[3] = cube[3] * t_range + t_min

        # cube 0 -> 1 maps to coszen -1 -> +1 and this maps to zenith pi -> 0
        cube[4] = math.acos(cube[4] * 2 - 1)

        # azimuth
        cube[5] *= TWO_PI

        # energy: either uniform or log uniform prior
        if energy_prior is PRI_UNIFORM:
            cube[6] = cube[6] * e_range + e_min
        elif energy_prior is PRI_LOG_UNIFORM:
            cube[6] = np.exp(cube[6] * log_e_range + log_e_min)

    def loglike(cube, ndim, nparams): # pylint: disable=unused-argument
        """Function pymultinest calls to get llh values.

        Note that this is called _after_ `prior` has been called, so `cube`
        alsready contains the parameter values scaled to be in their physical
        ranges.

        """
        if not t_start:
            t_start.append(time.time())

        t0 = time.time()

        total_energy = cube[6]
        track_fraction = cube[7]

        hypo_params = HYPO_PARAMS_T(
            t=cube[3],
            x=cube[0],
            y=cube[1],
            z=cube[2],
            track_zenith=cube[4],
            track_azimuth=cube[5],
            cascade_energy=total_energy * (1 - track_fraction),
            track_energy=total_energy * track_fraction
        )
        llh = get_llh(hypo_params, **llh_kw)

        t1 = time.time()

        param_values.append(hypo_params)
        log_likelihoods.append(llh)

        n_calls = len(log_likelihoods)

        if n_calls % report_after == 0:
            t_now = time.time()
            best_idx = np.argmax(log_likelihoods)
            best_llh = log_likelihoods[best_idx]
            best_p = param_values[best_idx]
            print('')
            print(('best llh = {:.3f} @ '
                   '(t={:+.1f}, x={:+.1f}, y={:+.1f}, z={:+.1f},'
                   ' zen={:.1f} deg, az={:.1f} deg, Etrk={:.1f}, Ecscd={:.1f})')
                  .format(
                      best_llh, best_p.t, best_p.x, best_p.y, best_p.z,
                      np.rad2deg(best_p.track_zenith),
                      np.rad2deg(best_p.track_azimuth),
                      best_p.track_energy,
                      best_p.cascade_energy))
            print('{} LLH computed'.format(n_calls))
            print('avg time per llh: {:.3f} ms'.format((t_now - t_start[0])/n_calls*1000))
            print('this llh took:    {:.3f} ms'.format((t1 - t0)*1000))
            print('')

        return llh

    n_dims = len(HYPO_PARAMS_T._fields)
    mn_kw = OrderedDict([
        ('n_dims', n_dims),
        ('n_params', n_dims),
        ('n_clustering_params', n_dims),
        ('wrapped_params', [int('azimuth' in p) for p in HYPO_PARAMS_T._fields]),
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

    mn_meta = OrderedDict([
        ('params', HYPO_PARAMS_T._fields),
        ('priors', priors),
        ('kwargs', sort_dict(mn_kw)),
    ])

    outdir = expand(outdir)
    mkdir(outdir)

    out_prefix = join(outdir, 'evt{}-'.format(event_idx))
    print('Output files prefix: "{}"\n'.format(out_prefix))

    mn_meta_outf = out_prefix + 'multinest_meta.pkl'
    print('Saving MultiNest metadata to "{}"'.format(mn_meta_outf))
    pickle.dump(
        mn_meta,
        open(mn_meta_outf, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

    print('Runing MultiNest...')
    pymultinest.run(
        LogLikelihood=loglike,
        Prior=prior,
        verbose=True,
        outputfiles_basename=out_prefix,
        resume=False,
        write_output=True,
        n_iter_before_update=5000,
        **mn_kw
    )

    llhp = np.empty(shape=len(param_values), dtype=LLHP_T)
    llhp['llh'] = log_likelihoods
    llhp[[f for f in llhp.dtype.fields.keys() if f != 'llh']] = param_values

    llhp_outf = out_prefix + 'llhp.npy'
    print('Saving llhp to "{}"...'.format(llhp_outf))
    np.save(llhp_outf, llhp)

    return llhp, mn_meta


def reco(dom_tables_kw, hypo_kw, hits_kw, reco_kw):
    """Script "main" function"""
    t00 = time.time()

    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
    hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
    hits_iterator = init_obj.get_hits(**hits_kw)

    table_t_max = dom_tables.pexp_meta['binning_info']['t_max']

    print('Running reconstructions...')

    llh_kw = dict(
        sd_indices=dom_tables.loaded_sd_indices,
        time_window=None,
        hypo_handler=hypo_handler,
        dom_tables=dom_tables,
        tdi_table=None
    )

    for event_idx, hits, time_window, hit_tmin, hit_tmax in hits_iterator: # pylint: disable=unused-variable
        t1 = time.time()

        llh_kw['hits'] = hits
        llh_kw['time_window'] = time_window #t_range

        t_lims = (hit_tmin - table_t_max - 1e3, hit_tmax)

        llhp, _ = run_multinest(
            event_idx=event_idx,
            llh_kw=llh_kw,
            t_lims=t_lims,
            **reco_kw
        )

        dt = time.time() - t1
        n_points = llhp.size
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt / n_points * 1e3))

    print('Total script run time is {:.3f} s'.format(time.time() - t00))


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )

    group = parser.add_argument_group(
        title='Hypothesis parameter priors',
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
        '--energy-prior', choices=['log_uniform', 'uniform'],
        help='''Prior to put on _total_ event energy'''
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
        '--max-modes', type=int,
        help='''Set to 1 to disable multi-modal search. Must be 1 if --importance-sampling is
        specified.'''
    )
    group.add_argument(
        '--const-eff', action='store_true',
        help='''Constant efficiency mode.'''
    )
    group.add_argument(
        '--n-live', type=int,
    )
    group.add_argument(
        '--evidence-tol', type=float,
    )
    group.add_argument(
        '--sampling-eff', type=float,
    )
    group.add_argument(
        '--max-iter', type=int,
    )
    group.add_argument(
        '--seed', type=int,
    )

    dom_tables_kw, hypo_kw, hits_kw, reco_kw = (
        init_obj.parse_args(parser=parser)
    )

    elims = ''.join(reco_kw['energy_lims'])
    elims = [float(l) for l in elims.split(',')]
    reco_kw['energy_lims'] = elims

    return dom_tables_kw, hypo_kw, hits_kw, reco_kw


if __name__ == '__main__':
    reco(*parse_args())
