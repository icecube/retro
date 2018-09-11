#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-return-statements

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
    'PRI_SPEFIT2',
    'PRI_CAUCHY',
    'get_prior_def',
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

from math import acos, exp

from os.path import abspath, dirname
import sys

import numpy as np
from scipy import stats

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import TWO_PI


PRI_UNIFORM = 'uniform'
PRI_LOG_UNIFORM = 'log_uniform'
PRI_LOG_NORMAL = 'log_normal'
PRI_COSINE = 'cosine'
PRI_GAUSSIAN = 'gaussian'
PRI_SPEFIT2 = 'spefit2'
PRI_CAUCHY = 'cauchy'


def get_prior_def(param, reco_kw):
    """Generate prior definitions from keyword args passed to `reco`.

    Parameters
    ----------
    params : str
        name of parameter
    reco_kw : dict
        dict from arg parse containing the prior info

    Returns
    -------
    prior_kind : tuple
    prior_params : tuple

    """
    if param == 'cascade_d_zenith':
        cascade_angle_prior_name = reco_kw.pop('cascade_angle_prior')
        if cascade_angle_prior_name == PRI_UNIFORM:
            return (PRI_UNIFORM, (0, np.pi))
        elif cascade_angle_prior_name == PRI_LOG_NORMAL:
            return (
                PRI_LOG_NORMAL,
                (
                    # scipy.stats.lognorm 3 paramters (from fit to data)
                    0.6486628230670546, -0.1072667784813348, 0.6337073562137334,
                    # hard limits
                    0., np.pi
                )
            )
        else:
            raise ValueError(str(cascade_angle_prior_name))

    elif 'zenith' in param:
        return (PRI_COSINE, (0, np.pi))

    elif 'coszen' in param:
        return (PRI_UNIFORM, (-1, 1))

    elif 'azimuth' in param:
        return (PRI_UNIFORM, (0, TWO_PI))

    elif param == 'x':
        spatial_prior_orig = reco_kw.get('spatial_prior').strip()
        spatial_prior_name = spatial_prior_orig.lower()
        if spatial_prior_name == 'ic':
            return (PRI_UNIFORM, (-860, 870))
        elif spatial_prior_name == 'dc':
            return (PRI_UNIFORM, (-150, 270))
        elif spatial_prior_name == 'dc_subdust':
            return (PRI_UNIFORM, (-150, 270))
        elif spatial_prior_name == 'spefit2':
            return (
                PRI_SPEFIT2,
                (
                    # scipy.stats.cauchy loc, scale parameters
                    -0.19687812829978152, 14.282171566308806,
                    # Hard limits
                    #-600, 750, # original limits
                    -590, 600, # TDI table limits
                )
            )
        else:
            raise ValueError('Spatial prior "{}" not recognized'
                             .format(spatial_prior_orig))

    elif param == 'y':
        spatial_prior_orig = reco_kw.get('spatial_prior').strip()
        spatial_prior_name = spatial_prior_orig.lower()
        if spatial_prior_name == 'ic':
            return (PRI_UNIFORM, (-780, 770))
        elif spatial_prior_name == 'dc':
            return (PRI_UNIFORM, (-210, 150))
        elif spatial_prior_name == 'dc_subdust':
            return (PRI_UNIFORM, (-210, 150))
        elif spatial_prior_name == 'spefit2':
            return (
                PRI_SPEFIT2,
                (
                    # scipy.stats.cauchy loc, scale parameters
                    -0.2393645701205161, 15.049528023495354,
                    # Hard limits
                    #-750, 650, # original limits
                    -540, 530, # TDI table limits
                )
            )
        else:
            raise ValueError('Spatial prior "{}" not recognized'
                             .format(spatial_prior_orig))

    elif param == 'z':
        spatial_prior_orig = reco_kw.get('spatial_prior').strip()
        spatial_prior_name = spatial_prior_orig.lower()
        if spatial_prior_name == 'ic':
            return (PRI_UNIFORM, (-780, 790))
        elif spatial_prior_name == 'dc':
            return (PRI_UNIFORM, (-770, 760))
        elif spatial_prior_name == 'dc_subdust':
            return (PRI_UNIFORM, (-610, -60))
        elif spatial_prior_name == 'spefit2':
            return (
                PRI_SPEFIT2,
                (
                    # scipy.stats.cauchy loc, scale parameters
                    -5.9170661027492546, 12.089399308036718,
                    # Hard limits
                    #-1200, 200, # original limits
                    -530, 540, # TDI table limits
                )
            )
        else:
            raise ValueError('Spatial prior "{}" not recognized'
                             .format(spatial_prior_orig))

    elif param == 'time':
        temporal_prior_orig = reco_kw.pop('temporal_prior').strip()
        temporal_prior_name = temporal_prior_orig.lower()
        if temporal_prior_name == PRI_UNIFORM:
            return (PRI_UNIFORM, (-4e3, 0.0))
        elif temporal_prior_name == PRI_SPEFIT2:
            return (
                PRI_SPEFIT2,
                (
                    # scipy.stats.cauchy loc (rel to SPEFit2 time), scale
                    -82.631395081663754, 75.619895703067343,
                    # Hard limits (relative to left, right edges of window,
                    # respectively)
                    -4e3, 0.0,
                )
            )
        else:
            raise ValueError('Temporal prior "{}" not recognized'
                             .format(temporal_prior_orig))

    elif param == 'cascade_energy':
        cascade_energy_prior_name = reco_kw.pop('cascade_energy_prior')
        cascade_energy_lims = reco_kw.pop('cascade_energy_lims')
        if cascade_energy_prior_name == PRI_UNIFORM:
            return (PRI_UNIFORM, (np.min(cascade_energy_lims), np.max(cascade_energy_lims)))
        elif cascade_energy_prior_name == PRI_LOG_UNIFORM:
            return (PRI_LOG_UNIFORM, (np.min(cascade_energy_lims), np.max(cascade_energy_lims)))
        elif cascade_energy_prior_name == PRI_LOG_NORMAL:
            return (
                PRI_LOG_NORMAL,
                (
                    # scipy.stats.lognorm 3 paramters
                    0.96251341305506233, 0.4175592980195757, 17.543915051586644,
                    # hard limits
                    np.min(cascade_energy_lims), np.max(cascade_energy_lims),
                )
            )
        else:
            raise ValueError(str(cascade_energy_prior_name))

    elif param == 'track_energy':
        track_energy_prior_name = reco_kw.pop('track_energy_prior')
        track_energy_lims = reco_kw.pop('track_energy_lims')
        if track_energy_prior_name == PRI_UNIFORM:
            return (PRI_UNIFORM, (np.min(track_energy_lims), np.max(track_energy_lims)))
        elif track_energy_prior_name == PRI_LOG_UNIFORM:
            return (PRI_LOG_UNIFORM, (np.min(track_energy_lims), np.max(track_energy_lims)))
        elif track_energy_prior_name == PRI_LOG_NORMAL:
            return (
                PRI_LOG_NORMAL,
                (
                    # scipy.stats.lognorm 3 paramters
                    0.96251341305506233, 0.4175592980195757, 17.543915051586644,
                    # hard limits
                    np.min(track_energy_lims), np.max(track_energy_lims),
                )
            )
        else:
            raise ValueError(str(track_energy_prior_name))

    else:
        raise ValueError('Dimension %s unknown for priors'%param)


def get_prior_fun(dim_num, dim_name, prior_def, event):
    """Generate prior function given a prior definition and the actual event

    Parameters
    ----------
    dim_num : int
        the cube dimension number from multinest
    dim_name : str
        parameter name
    prior_def : tuple
    event : event

    Returns
    -------
    prior_func : callable
    prior_def : tuple

    """
    hits_summary = event['hits_summary']
    prior_kind, prior_params = prior_def

    if prior_kind is PRI_UNIFORM:
        # Time is special since prior is relative to hits in the event
        if dim_name == 'time':
            prior_params = (
                hits_summary['earliest_hit_time'] + prior_params[0],
                hits_summary['latest_hit_time'] + prior_params[1]
            )
        prior_def = (prior_kind, prior_params)

        if prior_params == (0, 1):
            def prior_func(cube): # pylint: disable=unused-argument, missing-docstring
                pass
        elif np.min(prior_params[0]) == 0:
            maxval = np.max(prior_params)
            if 'azimuth' in dim_name:
                # minval must be 0 and maxval must be 2pi for wraparound logic to work
                assert maxval == 2*np.pi
                def prior_func(cube, n=dim_num, maxval=maxval): # pylint: disable=missing-docstring
                    # cyclic quantity
                    x = cube[n]
                    x = x%1
                    cube[n] = x * maxval
            elif 'zenith' in dim_name:
                def prior_func(cube, n=dim_num, maxval=maxval): # pylint: disable=missing-docstring
                    x = cube[n]
                    while x < 0 or x > 1:
                        if x < 0:
                            x = -x
                        else:
                            x = 1 - x
                    cube[n] = x * maxval
            else:
                def prior_func(cube, n=dim_num, maxval=maxval): # pylint: disable=missing-docstring
                    cube[n] = cube[n] * maxval
        else:
            minval = np.min(prior_params)
            width = np.max(prior_params) - minval
            def prior_func(cube, n=dim_num, width=width, minval=minval): # pylint: disable=missing-docstring
                cube[n] = cube[n] * width + minval

    elif prior_kind == PRI_LOG_UNIFORM:
        prior_def = (prior_kind, prior_params)
        log_min = np.log(np.min(prior_params))
        log_width = np.log(np.max(prior_params) / np.min(prior_params))
        def prior_func(cube, n=dim_num, log_width=log_width, log_min=log_min): # pylint: disable=missing-docstring
            cube[n] = exp(cube[n] * log_width + log_min)

    elif prior_kind == PRI_COSINE:
        prior_def = (prior_kind, prior_params)
        def prior_func(cube, n=dim_num): # pylint: disable=missing-docstring
            x = (2 * cube[n]) - 1
            cube[n] = acos(x)

    elif prior_kind == PRI_GAUSSIAN:
        prior_def = (prior_kind, prior_params)
        mean, stddev = prior_params
        norm = 1 / (stddev * np.sqrt(TWO_PI))
        def prior_func(cube, n=dim_num, norm=norm, mean=mean, stddev=stddev): # pylint: disable=missing-docstring
            cube[n] = norm * exp(-((cube[n] - mean) / stddev)**2)

    elif prior_kind == PRI_LOG_NORMAL:
        prior_def = (prior_kind, prior_params)
        shape, loc, scale, low, high = prior_params
        lognorm = stats.lognorm(shape, loc, scale)
        def prior_func(cube, lognorm=lognorm, n=dim_num, low=low, high=high): # pylint: disable=missing-docstring
            cube[n] = np.clip(lognorm.isf(cube[n]), a_min=low, a_max=high)

    elif prior_kind == PRI_SPEFIT2:
        spe_fit_val = event['recos']['SPEFit2'][dim_name]
        rel_loc, scale, low, high = prior_params
        loc = spe_fit_val + rel_loc
        cauchy = stats.cauchy(loc=loc, scale=scale)
        if dim_name == 'time':
            #low = spe_fit_val - 3000
            #high = spe_fit_val + 3000
            low += hits_summary['time_window_start']
            high += hits_summary['time_window_stop']
        prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        def prior_func(cube, cauchy=cauchy, n=dim_num, low=low, high=high): # pylint: disable=missing-docstring
            cube[n] = np.clip(cauchy.isf(cube[n]), a_min=low, a_max=high)

    else:
        raise NotImplementedError('Prior "{}" not implemented.'
                                  .format(prior_kind))

    return prior_func, prior_def
