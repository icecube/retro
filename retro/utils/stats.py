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
    estimate
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


def estimate(llhp, percentile_nd=0.95):
    """Evaluate estimator for reconstruction quantities given the MultiNest
    points of LLH space exploration.

    Paranters
    ---------
    llhp : shape (num_llh,) array of dtype retro_types.LLHP8D, LLHP10, etc.
        Fields of the structured array must contain 'llh' and any reconstructed
        quantities

    percentile_nd : float
        On what percentile of llh values to base the calculation

    Returns
    -------
    estimator : OrderedDict
        Containing estimated points incluing uncertainties.

    """
    columns = list(llhp.dtype.names)
    assert 'llh' in columns, 'llh not in %s'%columns
    columns.remove('llh')

    nd = len(columns)

    # keep best LLHs
    cut = llhp['llh'] >= np.nanmax(llhp['llh']) - stats.chi2.ppf(percentile_nd, nd)

    estimator = OrderedDict()

    # cut away upper and lower 13.35% to arrive at 1 sigma
    percentile = (percentile_nd - 0.682689492137086) / 2 * 100

    cut_llhp = llhp[cut]

    for col in columns:
        estimator[col] = OrderedDict()
        var = cut_llhp[col]
        if 'azimuth' in col.lower():
            # azimuth is a cyclic function, so need some special treatement to
            # get correct mean
            mean = stats.circmean(var)
            shifted = (var - mean + np.pi)%(2*np.pi)
            low = (np.percentile(shifted, percentile) + mean - np.pi) % (2*np.pi)
            high = (np.percentile(shifted, 100 - percentile) + mean - np.pi) % (2*np.pi)
        else:
            mean = var.mean()
            low = np.percentile(var, percentile)
            high = np.percentile(var, 100 - percentile)

        estimator[col]['mean'] = mean
        estimator[col]['low'] = low
        estimator[col]['high'] = high

    return estimator
