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
    'PRI_SPEFIT2TIGHT',
    'PRI_CAUCHY',
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
PRI_SPEFIT2TIGHT = 'spefit2tight'
PRI_CAUCHY = 'cauchy'



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
    prior_func : callable
    prior_def : tuple

    """
    hits_summary = event['hits_summary']

    # setup default values and prior defintions

    if 'zenith' in  dim_name:
        kind = kwargs.get('kind', PRI_COSINE)
        kind = kind.lower()
        low = kwargs.get('low', 0)
        high = kwargs.get('high', np.pi)

        if kind == PRI_LOG_NORMAL and dim_name == 'cascade_d_zenith':
            # scipy.stats.lognorm 3 dim_nameters (from fit to data)
            shape = 0.6486628230670546
            loc = -0.1072667784813348
            scale = 0.6337073562137334
            prior_def = (PRI_LOG_NORMAL, (shape, loc, scale, low, high))
        else:
            prior_def = (kind, (low, high))

    elif 'azimuth' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 2 * np.pi)
        prior_def = (kind, (low, high))

    elif 'coszen' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        low = kwargs.get('low', -1)
        high = kwargs.get('high', 1)
        prior_def = (kind, (low, high))

    elif dim_name == 'x':
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        extent = kwargs.get('extent', 'ic')
        if extent.lower() == 'ic':
            low = kwargs.get('low', -860)
            high = kwargs.get('high', 870)
        elif extent.lower() in ['dc', 'dc_subdust']:
            low = kwargs.get('low', -150)
            high = kwargs.get('high', 270)
        elif extent.lower() == 'tight':
            low = -200
            high = 200

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))
        elif kind == PRI_SPEFIT2:
            spe_fit_val = event['recos']['SPEFit2'][dim_name]
            loc = -0.19687812829978152
            loc += spe_fit_val
            scale = 14.282171566308806
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        elif kind == PRI_CAUCHY:
            loc = kwargs.get('loc')
            scale = kwargs.get('scale')
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        else:
            raise ValueError()


    elif dim_name == 'y':
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        extent = kwargs.get('extent', 'ic')
        if extent.lower() == 'ic':
            low = kwargs.get('low', -780)
            high = kwargs.get('high', 770)
        elif extent.lower() in ['dc', 'dc_subdust']:
            low = kwargs.get('low', -210)
            high = kwargs.get('high', 150)
        elif extent.lower() == 'tight':
            low = -200
            high = 200

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))
        elif kind == PRI_SPEFIT2:
            loc = -0.2393645701205161
            loc += event['recos']['SPEFit2'][dim_name]
            scale = 15.049528023495354
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        elif kind == PRI_CAUCHY:
            loc = kwargs.get('loc')
            scale = kwargs.get('scale')
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        else:
            raise ValueError()

    elif dim_name == 'z':
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        extent = kwargs.get('extent', 'ic')
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

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))
        elif kind == PRI_SPEFIT2:
            loc = -5.9170661027492546
            loc += event['recos']['SPEFit2'][dim_name]
            scale = 12.089399308036718
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        elif kind == PRI_CAUCHY:
            loc = kwargs.get('loc')
            scale = kwargs.get('scale')
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        else:
            raise ValueError()

    elif dim_name == 'time':
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        low = kwargs.get('low', -4e3)
        high = kwargs.get('high', 0)
        extent = kwargs.get('extent', 'none')
        if extent.lower() == 'hits_window':
            low += hits_summary['earliest_hit_time']
            high += hits_summary['latest_hit_time']
        elif extent.lower() == 'tight':
            low = -1000
            high = 1000

        if kind == PRI_UNIFORM:
            prior_def = (kind, (low, high))
        elif kind == PRI_SPEFIT2:
            loc = -82.631395081663754
            loc += event['recos']['SPEFit2'][dim_name]
            scale = 75.619895703067343
            if extent.lower() == 'tight':
                low += loc
                high += loc
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        elif kind == PRI_CAUCHY:
            loc = kwargs.get('loc')
            scale = kwargs.get('scale')
            prior_def = (PRI_CAUCHY, (loc, scale, low, high))
        else:
            raise ValueError()

    elif 'energy' in dim_name:
        kind = kwargs.get('kind', PRI_UNIFORM)
        kind = kind.lower()
        if kind == PRI_LOG_UNIFORM:
            low = kwargs.get('low', 0.1)
        else:
            low = kwargs.get('low', 0)
        high = kwargs.get('high', 1000)

        if kind == PRI_LOG_NORMAL:
            shape = 0.96251341305506233
            loc = 0.4175592980195757
            scale = 17.543915051586644
            prior_def = (PRI_LOG_NORMAL, (shape, loc, scale, low, high))
        else:
            prior_def = (kind, (low, high))

    else:
        kind = kwargs.get('kind', None)
        raise ValueError('Unknown prior %s for dim %s'%(kind,dim_name))



    # create prior function

    kind, args = prior_def

    if kind == PRI_UNIFORM:
        if args == (0, 1):
            def prior_func(cube): # pylint: disable=unused-argument, missing-docstring
                pass
        elif np.min(args[0]) == 0:
            maxval = np.max(args)
            def prior_func(cube, n=dim_num, maxval=maxval): # pylint: disable=missing-docstring
                cube[n] = cube[n] * maxval
        else:
            minval = np.min(args)
            width = np.max(args) - minval
            def prior_func(cube, n=dim_num, width=width, minval=minval): # pylint: disable=missing-docstring
                cube[n] = cube[n] * width + minval

    elif kind == PRI_LOG_UNIFORM:
        log_min = np.log(np.min(args))
        log_width = np.log(np.max(args) / np.min(args))
        def prior_func(cube, n=dim_num, log_width=log_width, log_min=log_min): # pylint: disable=missing-docstring
            cube[n] = exp(cube[n] * log_width + log_min)

    elif kind == PRI_COSINE:
        assert args == (0, np.pi)
        def prior_func(cube, n=dim_num): # pylint: disable=missing-docstring
            x = (2 * cube[n]) - 1
            cube[n] = acos(x)

    elif kind == PRI_GAUSSIAN:
        raise NotImplementedError('not correctly working limits, ToDo')
        mean, stddev, low, high = args
        norm = 1 / (stddev * np.sqrt(TWO_PI))
        def prior_func(cube, n=dim_num, norm=norm, mean=mean, stddev=stddev): # pylint: disable=missing-docstring
            cube[n] = norm * exp(-((cube[n] - mean) / stddev)**2)

    elif kind == PRI_LOG_NORMAL:
        shape, loc, scale, low, high = args
        lognorm = stats.lognorm(shape, loc, scale)
        def prior_func(cube, lognorm=lognorm, n=dim_num, low=low, high=high): # pylint: disable=missing-docstring
            cube[n] = np.clip(lognorm.isf(cube[n]), a_min=low, a_max=high)

    elif kind == PRI_CAUCHY:
        loc, scale, low, high = args
        cauchy = stats.cauchy(loc=loc, scale=scale)
        def prior_func(cube, cauchy=cauchy, n=dim_num, low=low, high=high): # pylint: disable=missing-docstring
            cube[n] = np.clip(cauchy.isf(cube[n]), a_min=low, a_max=high)
    else:
        raise NotImplementedError('Prior "{}" not implemented.'
                                  .format(kind))

    return prior_func, prior_def
