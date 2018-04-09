# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Statistics
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    poisson_llh
    partial_poisson_llh
    weighted_average
    estimate_from_llhp
'''.split()

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

from collections import OrderedDict
from os.path import abspath, dirname
import sys

import numpy as np
from scipy.special import gammaln
from scipy import stats

RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro


def poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - {\rm expected} \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    llh = observed * np.log(expected) - expected - gammaln(observed + 1)
    return llh


def partial_poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood _excluding_ subtracting off
    expected. This part, which constitutes an expected-but-not-observed
    penalty, is intended to be taken care of outside this function.

    .. math::
        {\rm observed} \cdot \log {\rm expected} - \log \Gamma({\rm observed})

    Parameters
    ----------
    expected
        Expected value(s)

    observed
        Observed value(s)

    Returns
    -------
    llh
        Log likelihood(s)

    """
    llh = observed * np.log(expected) - expected - gammaln(observed)
    return llh


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def weighted_average(x, w):
    """Average of elements in `x` weighted by `w`.

    Parameters
    ----------
    x : numpy.ndarray
        Values to average

    w : numpy.ndarray
        Weights, same shape as `x`

    Returns
    -------
    avg : numpy.ndarray
        Weighted average, same shape as `x`

    """
    sum_xw = 0.0
    sum_w = 0.0
    for x_i, w_i in zip(x, w):
        sum_xw += x_i * w_i
        sum_w += w_i
    return sum_xw / sum_w

def weighted_percentile(data, percentile, weights=None):
    '''
    percenttile (0..100)
    weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1 * w.cumsum() / w.sum() * 100
    y = np.interp(percentile, p, d)
    return y


def estimate_from_llhp(llhp, priors=None, percentile=0.95):
    """Evaluate estimator for reconstruction quantities given the MultiNest
    points of LLH space exploration.

    Paranters
    ---------
    llhp : shape (num_llh,) array of dtype retro_types.LLHP8D_T, LLHP10_T, etc.
        Fields of the structured array must contain 'llh' and any reconstructed
        quantities

    priors : mapping or None
        If specified, "no_prior_*" estimates will also be returned.

    percentile : float
        On what percentile of llh values to base the calculation

    Returns
    -------
    estimate : OrderedDict
        Keys are dimension names and values are "mean", "median", "low", and
        "high", where the latter two come are the `percentile` bounds. If
        `priors` is specified, then values are estimated by _removing_ the
        effect of the prior from the llh values.

    """
    columns = list(llhp.dtype.names)
    assert 'llh' in columns, 'llh not in %s'%columns
    columns.remove('llh')

    num_dims = len(columns)

    estimate = OrderedDict()

    # cut away upper and lower 13.35% to arrive at 1 sigma
    cut = llhp['llh'] >= np.nanmax(llhp['llh']) - stats.chi2.ppf(percentile, num_dims)
    percentile_nd = (percentile - 0.682689492137086) / 2 * 100

    cut_llhp = llhp[cut]

    if priors is None:
        weights = np.ones(shape=len(cut_llhp))
    else:
        raise NotImplementedError()

    for col in columns:
        estimate[col] = OrderedDict()
        var = cut_llhp[col]
        if 'azimuth' in col.lower():
            # azimuth is a cyclic function, so need some special treatement to
            # get correct mean
            mean = stats.circmean(var)
            shifted = (var - mean + np.pi) % (2*np.pi)
            #median = np.median(shifted)
            low = (np.percentile(shifted, percentile_nd) + mean - np.pi) % (2*np.pi)
            high = (np.percentile(shifted, 100 - percentile_nd) + mean - np.pi) % (2*np.pi)
        else:
            mean = var.mean()
            low = np.percentile(var, percentile_nd)
            high = np.percentile(var, 100 - percentile_nd)

        estimate[col]['mean'] = mean
        #estimate[col]['median'] = mean
        estimate[col]['low'] = low
        estimate[col]['high'] = high

    if not priors:
        return estimate

    for col in columns:
        if 'azimuth' in col.lower():
            weighted_mean = (np.average(shifted, weights=weights) + mean - np.pi) % (2*np.pi)
            median = (weighted_percentile(shifted, 50, weights) + mean - np.pi) % (2*np.pi)
            low = (weighted_percentile(shifted, percentile_nd, weights) + mean - np.pi) % (2*np.pi)
            high = (weighted_percentile(shifted, 100-percentile_nd, weights) + mean - np.pi) % (2*np.pi)
        else:
            mean = np.mean(var)
            weighted_mean = np.average(var, weights=weights)
            median = weighted_percentile(var, 50, weights)
            low = weighted_percentile(var, percentile_nd, weights)
            high = weighted_percentile(var, 100-percentile_nd, weights)
        estimate[col]['noprior_mean'] = mean
        estimate[col]['noprior_median'] = median
        estimate[col]['noprior_low'] = low
        estimate[col]['noprior_high'] = high

    return estimate
