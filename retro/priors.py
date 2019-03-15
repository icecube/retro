#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Prior definition generator and prior funcion generator to use for multinest
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'PRI_UNIFORM',
    'PRI_LOG_UNIFORM',
    'PRI_LOG_NORMAL',
    'PRI_COSINE',
    'PRI_GAUSSIAN',
    'PRI_INTERP',
    'PRI_AZ_INTERP',
    'PRI_SPEFIT2',
    'PRI_SPEFIT2TIGHT',
    'PRI_OSCNEXT_L5_V1',
    'OSCNEXT_L5_V1_PRIORS',
    'define_oscnext_l5_v1_prior',
    'define_generic_prior',
    'get_prior_fun',
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

from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, dirname, join
import sys

import numpy as np
from scipy import interpolate, stats

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import GarbageInputError
from retro.const import TWO_PI
from retro.retro_types import FitStatus
#from retro.utils.lerp import generate_lerp
from retro.utils.misc import LazyLoader


PRI_UNIFORM = 'uniform'
PRI_LOG_UNIFORM = 'log_uniform'
PRI_LOG_NORMAL = 'log_normal'
PRI_COSINE = 'cosine'
PRI_GAUSSIAN = 'gaussian'
PRI_INTERP = 'interp'
PRI_AZ_INTERP = 'az_interp'

PRI_SPEFIT2 = 'spefit2'
"""From fits to DRAGON (GRECO?) i.e. pre-oscNext MC"""

PRI_SPEFIT2TIGHT = 'spefit2tight'
"""From fits to DRAGON (GRECO?) i.e. pre-oscNext MC"""

PRI_OSCNEXT_L5_V1 = 'oscnext_l5_v1'
"""Priors from best-fits to oscNext level 5 (first version of processing, or
v1) events.

Priority is to use L5_SPEFit11 fit as prior, but this fails on some events, so
fall back to using LineFit_DC for these (LineFit_DC did not fail in any MC
events).

L5_SPEFit11 priors are derived for each dimension from:
    * time : splined-sampled VBWKDE fit to (negative) error dists
    * x : splined-sampled VBWKDE fit to (negative) error dists
    * y : splined-sampled VBWKDE fit to (negative) error dists
    * z : splined-sampled VBWKDE fit to (negative) error dists
    * azimuth : splined-sampled VBWKDE fit to (negative) error dists divided by coszen
    * coszen : splined-sampled VBWKDE fit to (negative) error dists divided by coszen
    * zenith : splined-sampled VBWKDE fit to (negative) error dists divided by coszen

LineFit_DC priors are derived for each dimension from:
    * time : splined-sampled VBWKDE fit to (negative) error dists
    * x : splined-sampled VBWKDE fit to (negative) error dists
    * y : splined-sampled VBWKDE fit to (negative) error dists
    * z : splined-sampled VBWKDE fit to (negative) error dists
    * azimuth : splined-sampled VBWKDE fit to (negative) error dists divided by coszen
    * coszen : splined-sampled VBWKDE fit to (negative) error dists divided by coszen
    * zenith : splined-sampled VBWKDE fit to (negative) error dists divided by coszen

All distributions (names from scipy.stats.distributions) are fit to event with
weights, where the weights are taken from the `weight` field in each event's
I3MCWeightDict.

See retro/notebooks/plot_prior_reco_candidates.ipynb for the fitting process.
"""

OSCNEXT_L5_V1_PRIORS = dict(
    time=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_time_neg_error.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_time_neg_error.pkl',
            ),
        ),
    ),
    x=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_x_neg_error.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_x_neg_error.pkl',
            ),
        ),
    ),
    y=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_y_neg_error.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_y_neg_error.pkl',
            ),
        ),
    ),
    z=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_z_neg_error.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_z_neg_error.pkl',
            ),
        ),
    ),
    azimuth=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_azimuth_neg_error_splitby_reco_coszen_10.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_azimuth_neg_error_splitby_reco_coszen_10.pkl',
            ),
        ),
    ),
    coszen=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_coszen_neg_error_splitby_reco_coszen_16.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_coszen_neg_error_splitby_reco_coszen_16.pkl',
            ),
        ),
    ),
    zenith=dict(
        L5_SPEFit11=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'L5_SPEFit11_zenith_neg_error_splitby_reco_coszen_16.pkl',
            ),
        ),
        LineFit_DC=LazyLoader(
            datasource=join(
                RETRO_DIR,
                'data',
                'priors',
                'LineFit_DC_zenith_neg_error_splitby_reco_coszen_16.pkl',
            ),
        ),
    ),
)


def define_oscnext_l5_v1_prior(dim_name, event, low, high, extent=None):
    """Construct (first-stage) prior def for OSCNEXT_L5_V1_PRIORS"""
    reco = 'L5_SPEFit11'
    if event['recos'][reco]['fit_status'] != FitStatus.OK:
        reco = 'LineFit_DC'
        assert event['recos'][reco]['fit_status'] == FitStatus.OK

    # Remove "track_*", etc prefixes
    for prefix in ('track', 'cascade'):
        if dim_name.startswith(prefix):
            dim_name = dim_name[len(prefix):].lstrip('_')
            break

    if dim_name == 'coszen':
        reco_val = np.cos(event['recos'][reco]['zenith'])
    else:

        reco_val = event['recos'][reco][dim_name]

    if not np.isfinite(reco_val):
        raise GarbageInputError(
            'dim_name "{}", reco "{}": reco val = {}'
            .format(dim_name, reco, reco_val)
        )

    prior_info = OSCNEXT_L5_V1_PRIORS[dim_name][reco].data
    prior_sha256 = OSCNEXT_L5_V1_PRIORS[dim_name][reco].sha256
    prior_fname = basename(OSCNEXT_L5_V1_PRIORS[dim_name][reco].datasource)

    split_by_reco_param = prior_info['metadata']['split_by_reco_param']
    if split_by_reco_param is None:
        split_val = None
    else:
        if split_by_reco_param == 'coszen':
            split_val = np.cos(event['recos'][reco]['zenith'])
        else:
            split_val = event['recos'][reco][split_by_reco_param]

        if not np.isfinite(split_val):
            raise GarbageInputError(
                'Reco "{}", split val "{}" = {}'
                .format(reco, split_by_reco_param, split_val)
            )

    pri = None
    for edges, pri_ in prior_info['dists'].items():
        if split_by_reco_param is None:
            pri = pri_
            break

        if edges[0] <= split_val <= edges[1]:
            pri = pri_
            break

    if pri is None:
        raise ValueError(
            '`split_by_reco_param` "{}" value={} outside binned ranges?: {}'
            .format(split_by_reco_param, split_val, prior_info['dists'].keys())
        )

    if dim_name == 'azimuth':
        pri_kind = PRI_AZ_INTERP
    else:
        pri_kind = PRI_INTERP

    # 'tight' (low, high) extents are relative to the fit value
    if extent is not None and extent.lower() == 'tight':
        low += reco_val
        high += reco_val
        if reco == 'LineFit_DC' and dim_name == 'z':
            low -= 15
            high -= 15

    prior_def = (pri_kind, (pri['x'] + reco_val, pri['pdf'], low, high))

    misc = deepcopy(prior_info['metadata'])
    misc['prior_file_name'] = prior_fname
    misc['prior_file_sha256'] = prior_sha256[:10]
    misc['reco_val'] = reco_val
    misc['split_val'] = split_val

    return prior_def, misc


def define_generic_prior(kind, kwargs, low, high, extent=None):  # pylint: disable=unused-argument
    """Create prior definition for a `kind` that exists in `scipy.stats.distributions`.

    Parameters
    ----------
    kind : str
        Must be a continuous distribution in `scipy.stats.distributions`
    kwargs : Mapping
        Must contain keys for any `shapes` (shape parameters) taken by the
        distribution as well as "loc" and "scale" (which are required for all
        distributions).
    low : finite scalar
        Range of values after prior is applied is clipped from below at `low`
    high : finite scalar
        Range of values after prior is applied is clipped from above at `high`
    extent : str or None

    Returns
    -------
    prior_def : tuple
        As defined/used in `retro.priors.get_prior_fun`; i.e., formatted as ::

            (kind, (arg0, arg1, ..., argN, low, high)

    """
    kind = kind.lower()
    loc = kwargs['loc']
    scale = kwargs['scale']
    dist = getattr(stats.distributions, kind)
    if dist.shapes:
        args = []
        for shape_param in dist.shapes:
            args.append(kwargs[shape_param])
        args = tuple(args)
    else:
        args = tuple()
    #if extent is not None and extent.lower() == 'tight':
    #    low -= fit
    prior_def = (kind, args + (loc, scale, low, high))
    return prior_def


def get_prior_fun(dim_num, dim_name, event, **kwargs):
    """Generate prior function given a prior definition and the actual event

    Parameters
    ----------
    dim_num : int
        the cube dimension number from multinest
    dim_name : str
        parameter name
    event : event
    kwargs : any additional arguments

    Returns
    -------
    prior_fun : callable
    prior_def : tuple

    """
    hits_summary = event['hits_summary']

    # -- setup default values and prior defintions -- #

    misc = OrderedDict()
    prior_def = None

    # Note in the following, invalid or unhandled cases are treated by
    # checking after if/else nested branches here, `prior_def` is still `None`.

    if 'zenith' in  dim_name:
        kind = kwargs.get('kind', PRI_COSINE).lower()
        low = kwargs.get('low', 0)
        high = kwargs.get('high', np.pi)

        if kind == "PRI_COSINE":
            prior_def = (kind, (low, high))

        elif kind == PRI_LOG_NORMAL and dim_name == 'cascade_d_zenith':
            # scipy.stats.lognorm 3 dim_nameters (from fit to data)
            shape = 0.6486628230670546
            loc = -0.1072667784813348
            scale = 0.6337073562137334
            prior_def = ('lognorm', (shape, loc, scale, low, high))

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif 'coszen' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        low = kwargs.get('low', -1)
        high = kwargs.get('high', 1)

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif 'azimuth' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 2 * np.pi)

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif dim_name == 'x':
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        extent = kwargs.get('extent', 'ic').lower()
        if extent == 'ic':
            low = kwargs.get('low', -860)
            high = kwargs.get('high', 870)
        elif extent in ['dc', 'dc_subdust']:
            low = kwargs.get('low', -150)
            high = kwargs.get('high', 270)
        elif extent == 'tight':
            low = -200
            high = 200
        else:
            raise ValueError('invalid extent "{}"'.format(extent))

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_SPEFIT2:
            reco = 'SPEFit2'
            fit_status = FitStatus(event['recos'][reco]['fit_status'])
            if fit_status is not FitStatus.OK:
                raise GarbageInputError(
                    'dim_name "{}", reco "{}": fit status = {!r}'
                    .format(dim_name, reco, fit_status)
                )
            fit_val = event['recos'][reco][dim_name]
            loc = -0.19687812829978152 + fit_val
            scale = 14.282171566308806
            if extent == 'tight':
                low += loc
                high += loc
            prior_def = ('cauchy', (loc, scale, low, high))
            misc = OrderedDict([('reco', reco), ('fit_val', fit_val)])

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high, extent)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif dim_name == 'y':
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        extent = kwargs.get('extent', 'ic').lower()
        if extent == 'ic':
            low = kwargs.get('low', -780)
            high = kwargs.get('high', 770)
        elif extent in ['dc', 'dc_subdust']:
            low = kwargs.get('low', -210)
            high = kwargs.get('high', 150)
        elif extent == 'tight':
            low = -200
            high = 200
        else:
            raise ValueError('invalid extent "{}"'.format(extent))

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_SPEFIT2:
            reco = 'SPEFit2'
            fit_status = FitStatus(event['recos'][reco]['fit_status'])
            if fit_status is not FitStatus.OK:
                raise GarbageInputError(
                    'dim_name "{}", reco "{}": fit val = {}'
                    .format(dim_name, reco, fit_val)
                )
            fit_val = event['recos'][reco][dim_name]
            loc = -0.2393645701205161 + fit_val
            scale = 15.049528023495354
            if extent == 'tight':
                low += loc
                high += loc
            prior_def = ('cauchy', (loc, scale, low, high))
            misc = OrderedDict([('reco', reco), ('fit_val', fit_val)])

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high, extent)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif dim_name == 'z':
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        extent = kwargs.get('extent', 'ic').lower()
        if extent is None or extent.lower() == 'ic':
            low = kwargs.get('low', -780)
            high = kwargs.get('high', 790)
        elif extent.lower() == 'dc':
            # are these number sensible?
            low = kwargs.get('low', -770)
            high = kwargs.get('high', 760)
        elif extent.lower() == 'dc_subdust':
            low = kwargs.get('low', -610)
            high = kwargs.get('high', 60)
        elif extent.lower() == 'tight':
            low = -100
            high = 100
        else:
            raise ValueError('invalid extent "{}"'.format(extent))

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_SPEFIT2:
            reco = 'SPEFit2'
            fit_status = FitStatus(event['recos'][reco]['fit_status'])
            if fit_status is not FitStatus.OK:
                raise GarbageInputError(
                    'dim_name "{}", reco "{}": fit val = {}'
                    .format(dim_name, reco, fit_val)
                )
            fit_val = event['recos'][reco][dim_name]
            loc = -5.9170661027492546 + fit_val
            scale = 12.089399308036718
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = ('cauchy', (loc, scale, low, high))
            misc = OrderedDict([('reco', reco), ('fit_val', fit_val)])

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high, extent)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif dim_name == 'time':
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        low = kwargs.get('low', -4e3)
        high = kwargs.get('high', 0)
        extent = kwargs.get('extent', 'none')
        if extent.lower() == 'hits_window':
            low += hits_summary['earliest_hit_time']
            high += hits_summary['latest_hit_time']
        elif extent.lower() == 'tight':
            low = -1000
            high = 1000
        else:
            raise ValueError()

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))

        elif kind == PRI_SPEFIT2:
            reco = 'SPEFit2'
            fit_status = FitStatus(event['recos'][reco]['fit_status'])
            if fit_status is not FitStatus.OK:
                raise GarbageInputError(
                    'dim_name "{}", reco "{}": fit val = {}'
                    .format(dim_name, reco, fit_val)
                )
            fit_val = event['recos'][reco][dim_name]
            loc = -82.631395081663754 + fit_val
            scale = 75.619895703067343
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = ('cauchy', (loc, scale, low, high))
            misc = OrderedDict([('reco', reco), ('fit_val', fit_val)])

        elif kind == PRI_OSCNEXT_L5_V1:
            prior_def, misc = define_oscnext_l5_v1_prior(dim_name, event, low, high, extent)

        elif hasattr(stats.distributions, kind):
            prior_def = define_generic_prior(kind=kind, kwargs=kwargs, low=low, high=high)

    elif 'energy' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM).lower()
        low = kwargs.get('low', 0.1 if kind == PRI_LOG_UNIFORM else 0.0)
        high = kwargs.get('high', 1e3)

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))
        elif kind == PRI_LOG_NORMAL:
            shape = 0.96251341305506233
            loc = 0.4175592980195757
            scale = 17.543915051586644
            prior_def = ('lognorm', (shape, loc, scale, low, high))

    if prior_def is None:
        kind = kwargs.get('kind', None)
        raise ValueError(
            'Unhandled or invalid prior "{}" for dim_name "{}"'.format(kind, dim_name)
        )

    # create prior function

    if not prior_def:
        raise ValueError("no prior def?")

    kind, args = prior_def

    #if not np.all(np.isfinite(args)):
    #    raise ValueError(
    #        'Dimension "{}" got non-finite arg(s) for prior kind "{}"; args = {}'
    #        .format(dim_name, kind, args)
    #    )

    if kind == PRI_UNIFORM:
        if args == (0, 1):
            def prior_fun(cube): # pylint: disable=unused-argument, missing-docstring
                pass
        elif np.min(args[0]) == 0:
            maxval = np.max(args)
            def prior_fun(cube, n=dim_num, maxval=maxval): # pylint: disable=missing-docstring
                cube[n] = cube[n] * maxval
        else:
            minval = np.min(args)
            width = np.max(args) - minval
            def prior_fun(cube, n=dim_num, width=width, minval=minval): # pylint: disable=missing-docstring
                cube[n] = cube[n] * width + minval

    elif kind == PRI_LOG_UNIFORM:
        log_min = np.log(np.min(args))
        log_width = np.log(np.max(args) / np.min(args))
        def prior_fun(cube, n=dim_num, log_width=log_width, log_min=log_min): # pylint: disable=missing-docstring
            cube[n] = np.exp(cube[n] * log_width + log_min)

    elif kind == PRI_COSINE:
        assert args[-2:] == (0, np.pi)
        def prior_fun(cube, n=dim_num): # pylint: disable=missing-docstring
            x = (2 * cube[n]) - 1
            cube[n] = np.arccos(x)

    elif kind == PRI_GAUSSIAN:
        raise NotImplementedError('limits not correctly working') # TODO
        mean, stddev, low, high = args
        norm = 1 / (stddev * np.sqrt(TWO_PI))
        def prior_fun(cube, n=dim_num, norm=norm, mean=mean, stddev=stddev): # pylint: disable=missing-docstring
            cube[n] = norm * np.exp(-((cube[n] - mean) / stddev)**2)

    elif kind in (PRI_INTERP, PRI_AZ_INTERP):
        x, pdf, low, high = args

        if (
            kind == PRI_AZ_INTERP and not np.isclose(x.max() - x.min(), high - low)
            or kind == PRI_INTERP and (x.min() > low or x.max() < high)
        ):
            print(
                'Dim "{}", prior kind "{}" `x` range = [{}, {}] does not cover'
                " [low, high] range = [{}, {}]"
                .format(dim_name, kind, x.min(), x.max(), low, high)
            )

        if kind == PRI_AZ_INTERP:
            if not (np.isclose(low, 0) and np.isclose(high, 2*np.pi)):
                raise ValueError("az range [low, high) must be [0, 2pi)")

            # Ensure x covers exactly the same distance as (low, high) defines
            highlow_range = high - low
            x = x.min() + (x - x.min()) * highlow_range/(x.max() - x.min())

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            # Create smooth spline interpolator for isf (inverse of cdf)
            isf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext='raise', s=0)

            def prior_fun(  # pylint: disable=missing-docstring
                cube,
                n=dim_num,
                isf_interp=isf_interp,
            ):
                cube[n] = isf_interp(cube[n]) % (2*np.pi)

        else:
            # If x covers _more_ than the allowed [low, high] range, resample the
            # pdf in the allowed range (expected to occur for binned zenith and
            # coszen error distributions)
            if dim_name != 'time' and (x.min() < low or x.max() > high):
                xp = x
                x = np.linspace(low, high, len(x))
                pdf = np.interp(x=x, xp=xp, fp=pdf)
                pdf /= np.trapz(x=x, y=pdf)

            # Compute cumulative distribution function (cdf) via trapezoidal-rule
            # integration
            cdf = np.array([np.trapz(x=x[:n], y=pdf[:n]) for n in range(1, len(x) + 1)])
            # Ensure first value in cdf is exactly 0
            cdf -= cdf[0]
            # Ensure last value in cdf is exactly 1
            cdf /= cdf[-1]

            ## Create linear interpolator for isf (inverse of cdf)
            #_, isf_interp = generate_lerp(
            #    x=cdf,
            #    y=x,
            #    low_behavior='error',
            #    high_behavior='error',
            #)

            # Create smooth spline interpolator for isf (inverse of cdf)
            isf_interp = interpolate.UnivariateSpline(x=cdf, y=x, ext='raise', s=0)

            def prior_fun(cube, n=dim_num, isf_interp=isf_interp): # pylint: disable=missing-docstring
                cube[n] = isf_interp(cube[n])

    elif hasattr(stats.distributions, kind):
        dist_args = args[:-2]
        low, high = args[-2:]
        frozen_dist_isf = getattr(stats.distributions, kind)(*dist_args).isf
        def prior_fun(
            cube,
            frozen_dist_isf=frozen_dist_isf,
            dim_num=dim_num,
            low=low,
            high=high,
        ): # pylint: disable=missing-docstring
            cube[dim_num] = np.clip(
                frozen_dist_isf(cube[dim_num]),
                a_min=low,
                a_max=high,
            )

    else:
        raise NotImplementedError('Prior "{}" not implemented.'.format(kind))

    return prior_fun, prior_def, misc
