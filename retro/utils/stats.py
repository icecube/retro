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
        return np.percentile(data, percentile)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = w.cumsum() / w.sum() * 100
    return np.interp(percentile, p, d)


def estimate_from_llhp(llhp, meta=None, per_dim=False, prob_weights=True):
    """Evaluate estimate for reconstruction quantities given the MultiNest
    points of LLH space exploration.

    Paranters
    ---------
    llhp : shape (num_llh,) array of dtype retro_types.LLHP8D_T, LLHP10_T, etc.
        Fields of the structured array must contain 'llh' and any reconstructed
        quantities

    meta : dict or None
        meta information from the minimization, including priors
        If specified, "no_prior_*" estimates will also be returned.

    per_dim : boolean
        treat each dimension individually (not yet sure how much sense that makes)

    prob_weights : boolean
        use LLH weights

    Returns
    -------
    estimate : OrderedDict
        Keys are dimension names and values are "mean", "median", "low", and
        "high", where the latter two come are the `percentile` bounds. If
        `meta` is specified, then values are estimated by _removing_ the
        effect of the prior from the llh values.

    """
    columns = list(llhp.dtype.names)
    assert 'llh' in columns, 'llh not in %s'%columns
    columns.remove('llh')

    num_dims = len(columns)

    # cut away extremely low llh (30 or more below best point)
    cut = llhp['llh'] >= np.nanmax(llhp['llh']) - 30
    if np.sum(cut) == 0:
        raise IndexError('no points')

    # can throw rest of points away
    llhp = llhp[cut]
    #print(llhp)
    
    # calculate weights from probabilities
    if prob_weights:
        prob_weights = np.exp(llhp['llh'] - np.max(llhp['llh']))
        #prob_weights = 1./np.square((1. + np.max(llhp['llh']) - llhp['llh']))
    else:
        prob_weights = None


    # calculate the prior weights from the used priors
    if meta is None:
        prior_weights = None
    else:
        if per_dim:
            prior_weights = {}
        else:
            prior_weights = np.ones(len(llhp))
        if not meta is None:
            priors = meta['priors_used']
            for dim in priors.keys():
                prior = priors[dim]
                if prior[0] == 'uniform':
                    w = np.ones(len(llhp))
                elif prior[0] in ['cauchy', 'spefit2']:
                    w = 1./stats.cauchy.pdf(llhp[dim], *prior[1][:2])
                    #w = np.ones(len(llhp))
                elif prior[0] == 'log_normal' and dim == 'energy':
                    w = 1./stats.lognorm.pdf(llhp['track_energy'] + llhp['cascade_energy'], *prior[1][:3])
                    #w = np.ones(len(llhp))
                elif prior[0] == 'log_uniform' and dim == 'energy':
                    w = llhp['track_energy'] + llhp['cascade_energy']
                    #w = np.ones(len(llhp))
                elif prior[0] == 'log_uniform' and dim == 'cascade_energy':
                    w = llhp['cascade_energy']
                elif prior[0] == 'log_uniform' and dim == 'track_energy':
                    w = llhp['track_energy']
                elif prior[0] == 'cosine':
                    w = 1./np.clip(np.sin(llhp[dim]), 0.01, None)
                    #w = np.ones(len(llhp))
                else:
                    raise NotImplementedError('prior %s for dimension %s unknown'%(prior[0], dim))

                if per_dim:
                    prior_weights[dim] = w
                else:
                    prior_weights *= w

    if per_dim:
        prior_weights['track_energy'] = llhp['track_energy'] / (llhp['track_energy'] + llhp['cascade_energy']) * prior_weights['energy']
        prior_weights['cascade_energy'] = llhp['cascade_energy'] / (llhp['track_energy'] + llhp['cascade_energy']) * prior_weights['energy']


    estimate = OrderedDict()

    estimate['mean'] = OrderedDict()
    estimate['best'] = OrderedDict()
    estimate['median'] = OrderedDict()
    estimate['low'] = OrderedDict()
    estimate['high'] = OrderedDict()
    if prior_weights is not None or prob_weights is not None:
        estimate['weighted_mean'] = OrderedDict()
        estimate['weighted_median'] = OrderedDict()
        estimate['weighted_best'] = OrderedDict()

    # cut away upper and lower 13.35% to arrive at 1 sigma
    percentile = (1 - 0.682689492137086) / 2. * 100.
    #print(percentile)
    #percentile = 13

    for col in columns:
        if prob_weights is None and prior_weights is None:
            weights = None
        else:
            if prob_weights is None:
                weights = np.ones(len(llhp))
            else:
                weights = np.copy(prob_weights)
        if prior_weights is not None:
            if per_dim:
                weights *= prior_weights[col]
            else:
                weights *= prior_weights

        var = llhp[col]
        post_llh = np.log(weights)
        sigma = post_llh > np.max(post_llh) - 15.5
        #weights = weights ** 1./8.
        #s_idx = np.argsort(weights)[::-1]
        #print(col, weights[s_idx][:10], var[s_idx][:10], post_llh[s_idx][:10])
        #print('low %s, high %s'%(np.min(var[sigma]), np.max(var[sigma])))
        sigma_vals = var[sigma]
        #sigma_weights = None
        sigma_weights = weights[sigma]
        if 'azimuth' in col:
            # azimuth is a cyclic function, so need some special treatement to get correct mean
            mean = stats.circmean(var)
            shift_mean = stats.circmean(sigma_vals)
            shifted = (var - shift_mean + np.pi)%(2*np.pi)
            median = (np.median(shifted) + shift_mean - np.pi)%(2*np.pi)
            sigma_shifted = (sigma_vals - shift_mean + np.pi)%(2*np.pi)
            low = (weighted_percentile(sigma_shifted, 13.35, sigma_weights) + shift_mean - np.pi)%(2*np.pi)
            high = (weighted_percentile(sigma_shifted, 86.65, sigma_weights) + shift_mean - np.pi)%(2*np.pi)

            #low = (weighted_percentile(shifted, percentile, weights) + shift_mean - np.pi)%(2*np.pi)
            #high = (weighted_percentile(shifted, 100-percentile, weights) + shift_mean - np.pi)%(2*np.pi)
            if weights is not None:
                weighted_mean = (np.average(shifted, weights=weights) + shift_mean - np.pi)%(2*np.pi)
                weighted_median = (weighted_percentile(shifted, 50, weights) + shift_mean - np.pi)%(2*np.pi)
        else:
            mean = np.mean(var)
            median = np.median(var)
            low = weighted_percentile(sigma_vals, 13.35, sigma_weights)
            high = weighted_percentile(sigma_vals, 86.65, sigma_weights)
            #low = weighted_percentile(var, percentile, weights)
            #high = weighted_percentile(var, 100-percentile, weights)
            if weights is not None:
                weighted_mean = np.average(var, weights=weights)
                weighted_median = weighted_percentile(var, 50, weights)
        estimate['mean'][col] = mean
        estimate['median'][col] = median
        estimate['low'][col] = low
        estimate['high'][col] = high

        best_idx = np.argmax(llhp['llh'])
        estimate['best'][col] = var[best_idx] 

        if weights is not None:
            estimate['weighted_mean'][col] = weighted_mean
            estimate['weighted_median'][col] = weighted_median
            best_idx = np.argmax(post_llh)
            estimate['weighted_best'][col] = var[best_idx] 

    return estimate
