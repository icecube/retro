# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Statistics
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'DELTA_LLH_CUTOFF',
    'EST_KINDS',
    'poisson_llh',
    'partial_poisson_llh',
    'weighted_average',
    'weighted_percentile',
    'test_weighted_percentile',
    'estimate_from_llhp',
    'fit_cdf',
]

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
from copy import copy, deepcopy
from os.path import abspath, dirname
import sys
import time
import traceback
import warnings

import numpy as np
from scipy import interpolate, optimize, special, stats
from six import string_types

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro
from retro.priors import PRI_INTERP, PRI_AZ_INTERP


DELTA_LLH_CUTOFF = 15.5
"""What values of the llhp space to include relative to the max-LLH point"""

EST_KINDS = ['lower_bound', 'mean', 'median', 'max', 'upper_bound']


def poisson_llh(expected, observed):
    u"""Compute the log Poisson likelihood.

    .. math::
        observed × log(expected) - expected - log Γ(observed + 1)

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
    return observed * np.log(expected) - expected - special.gammaln(observed + 1)


def partial_poisson_llh(expected, observed):
    u"""Compute the log Poisson likelihood _excluding_ subtracting off
    expected. This part, which constitutes an expected-but-not-observed
    penalty, is intended to be taken care of outside this function.

    .. math::
        observed × log(expected) - log Γ(observed + 1)

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
    return observed * np.log(expected) - special.gammaln(observed + 1)


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
    for i in range(x.size):
        sum_xw += x[i] * w[i]
        sum_w += w[i]
    return sum_xw / sum_w


def weighted_percentile(a, q, weights=None):
    """Compute percentile(s) of data, optionally using weights.

    Parameters
    ----------
    a : array
        Data
    q : scalar or array-like in [0, 100]
        Percentile(s) to compute for data
    weights : array, same shape as `data`
        Frequencies (counts) of data

    Returns
    -------
    percentile : scalar or ndarray

    """
    if weights is None:
        return np.percentile(a=a, q=q)

    sort_indices = np.argsort(a)
    sorted_data = a[sort_indices]
    sorted_weights = weights[sort_indices]

    # Samples from unnormed cdf via cumulative sum of sorted samples of pdf
    probs = sorted_weights.cumsum()

    # Total (norm factor) can be read off as last value from cumsum
    tot = probs[-1]

    # Scale `percentile` to be a fraction of the unnormed total of `probs` to
    # avoid re-scaling the entire `probs` array
    return np.interp(np.asarray(q) * (tot / 100), probs, sorted_data)


def test_weighted_percentile():
    """Unit test for `weighted_percentile` function."""
    rand = np.random.RandomState(0)
    data = rand.normal(size=1000)
    weights = rand.uniform(size=1000)
    pctiles = [0, 10, 25, 50, 75, 90, 100]
    vals = weighted_percentile(a=data, q=pctiles, weights=weights)
    ref_vals = np.array([
        -3.046143054799927,
        -1.231954827314273,
        -0.659949597686861,
        -0.041388356155877,
        0.645261111851618,
        1.322480276194482,
        2.759355114021582,
    ])
    assert np.allclose(vals, ref_vals, atol=0, rtol=1e-12)
    print("<< PASS : test_weighted_percentile >>")


def estimate_from_llhp(
    llhp,
    treat_dims_independently,
    use_prob_weights,
    priors_used=None,
    meta=None,
):
    """Evaluate estimate for reconstruction quantities given the MultiNest
    points of LLH space exploration. .

    Paranters
    ---------
    llhp : shape (num_llh,) array of custom dtype llhp_t
        Fields of the structured array must contain 'llh' and any reconstructed
        quantities (aka parameters or dimensions)

    treat_dims_independently : boolean
        treat each dimension individually (not yet sure how much sense that
        makes)

    use_prob_weights : boolean
        use LLH weights in computing estimates

    priors_used : dict, optional
        Specify the priors used to remove their effects on the posterior LLH
        distributions; if not specified, effects of priors will not be removed

    meta : OrderedDict, optional

    Returns
    -------
    estimate : numpy struct array
    estimation_settings : OrderedDict

    """
    if meta is None:
        meta = OrderedDict()

    # currently spherical averages are not supported if dimensions are treated
    # independently (how would this even work?)
    averages_spherically_aware = not treat_dims_independently
    remove_priors = bool(priors_used)

    names = list(llhp.dtype.names)
    #for name in ('llh', 'track_energy', 'cascade_energy'):
    #    if name not in names:
    #        raise ValueError(
    #            '"{}" not a field in `llhp.dtype.names` = {}'
    #            .format(name, names)
    #        )

    params = copy(names)
    params.remove('llh')

    num_params = len(params)
    num_llh = len(llhp)

    # cut away extremely low llh
    max_llh = np.nanmax(llhp['llh'])
    llhp = llhp[llhp['llh'] >= (max_llh - 30)]
    if len(llhp) == 0:
        raise ValueError('no points')
    llh = llhp['llh']

    if use_prob_weights:
        # weight points by their likelihood (_not_ log likelihood) relative to
        # max; keep prob_weights around for later use
        prob_weights = np.exp(llh - max_llh)
        weights = deepcopy(prob_weights)
    else:
        prob_weights = None
        weights = np.ones(shape=len(llh))

    if treat_dims_independently:
        weights = {d: deepcopy(weights) for d in priors_used.keys()}

    if remove_priors:
        # calculate the prior weights from the priors used
        for dim, (prior_kind, prior_params) in priors_used.items():
            if prior_kind == 'uniform':
                w = None
            elif prior_kind in ('cauchy', 'spefit2'):
                w = 1 / stats.cauchy.pdf(llhp[dim], *prior_params[:2])
            elif prior_kind == 'log_normal' and dim == 'energy':
                w = 1 / stats.lognorm.pdf(
                    llhp['track_energy'] + llhp['cascade_energy'],
                    *prior_params[:3]
                )
            elif prior_kind == 'log_uniform' and dim == 'energy':
                w = llhp['track_energy'] + llhp['cascade_energy']
            elif prior_kind == 'log_uniform' and dim == 'cascade_energy':
                w = llhp['cascade_energy']
            elif prior_kind == 'log_uniform' and dim == 'track_energy':
                w = llhp['track_energy']
            elif prior_kind == 'cosine':
                w = None
            elif prior_kind == 'log_normal' and dim == 'cascade_d_zenith':
                w = None
            elif prior_kind in (PRI_AZ_INTERP, PRI_INTERP):
                x, pdf, low, high = prior_params[-4:]
                pdf_interp = interpolate.UnivariateSpline(x=x, y=pdf, ext='raise', s=0)
                w = 1 / pdf_interp(llhp[dim])
            else:
                raise NotImplementedError(
                    'Prior "{}" for dimension/param "{}" is unhandled'
                    .format(prior_kind, dim)
                )

            if w is not None:
                if treat_dims_independently:
                    weights[dim] *= w
                else:
                    weights *= w

        if treat_dims_independently:
            if 'energy' in weights:
                w = prob_weights if use_prob_weights else 1
                weights['track_energy'] = w * (
                    llhp['track_energy']
                    / (llhp['track_energy'] + llhp['cascade_energy'])
                    * weights['energy']
                )
                weights['cascade_energy'] = w * (
                    llhp['cascade_energy']
                    / (llhp['track_energy'] + llhp['cascade_energy'])
                    * weights['energy']
                )

    if treat_dims_independently:
        postproc_llh = {}
        for dim, weights in weights.items():
            if use_prob_weights or remove_priors:
                postproc_llh[dim] = np.log(weights)
            else:
                postproc_llh[dim] = llh
        # simply report `max_llh` for `max_postproc_llh` since each dimension
        # will have a different max since each gets weighted independently
        max_postproc_llh = max_llh
    else:
        postproc_llh = max_llh + np.log(weights)
        max_idx = np.nanargmax(postproc_llh)
        max_postproc_llh = postproc_llh[max_idx]
        params_at_max_llh = llhp[max_idx]
        cut = postproc_llh > max_postproc_llh - DELTA_LLH_CUTOFF
        cut_llhp = llhp[cut]
        cut_postproc_llh = postproc_llh[cut]
        cut_weights = weights[cut]

    # -- Construct struct array for storing estimates and per-event metadata -- #

    sub_dtype = np.dtype([(kind, np.float32) for kind in EST_KINDS], align=False)
    est_dtype = np.dtype(
        [(dim, sub_dtype) for dim in params]
        + [('max_llh', np.float32), ('max_postproc_llh', np.float32), ('num_llh', np.uint32)]
        + [(key, val.dtype) for key, val in meta.items()],
        align=False,
    )

    # Note that filling with nan does not fail but stuffs undefined (?) values
    # to int/uint fields
    estimate = np.full(shape=1, fill_value=np.nan, dtype=est_dtype)

    estimate['num_llh'] = num_llh
    estimate['max_llh'] = max_llh
    estimate['max_postproc_llh'] = max_postproc_llh
    for key, val in meta.items():
        estimate[key] = val

    # -- Calculate each kind of estimate for each param -- #

    one_sigma_range = 100 * 0.682689492137086
    qth_percentiles = np.array([
        50.0 - one_sigma_range / 2,
        50.0,
        50.0 + one_sigma_range / 2,
    ])

    for param_idx, param in enumerate(params):
        if treat_dims_independently:
            this_postproc_llh = postproc_llh[param]
            max_idx = np.nanargmax(this_postproc_llh)
            max_postproc_llh = this_postproc_llh[max_idx]
            param_vals = llhp[param]
            param_at_max_llh = param_vals[max_idx]
            cut = this_postproc_llh > max_postproc_llh - DELTA_LLH_CUTOFF
            this_postproc_llh = this_postproc_llh[cut]
            this_weights = weights[param][cut]
            param_vals = param_vals[cut]
        else:
            param_at_max_llh = params_at_max_llh[param]
            this_postproc_llh = cut_postproc_llh
            this_weights = cut_weights
            param_vals = cut_llhp[param]

        estimate[param]['max'] = param_at_max_llh

        if 'azimuth' in param:
            # azimuth is a cyclic function, so need some special treatment to
            # get correct mean: shift everything such that the best-fit point is
            # in the middle (pi)
            shift = param_at_max_llh
            shifted_vals = (param_vals - shift + np.pi) % (2*np.pi)

            mean = (stats.circmean(shifted_vals) + shift - np.pi) % (2*np.pi)
            vals_at_q = weighted_percentile(
                a=shifted_vals,
                q=qth_percentiles,
                weights=this_weights,
            )
            lower, median, upper = (vals_at_q + shift - np.pi) % (2*np.pi)
        else:
            mean = np.average(param_vals, weights=this_weights)
            lower, median, upper = weighted_percentile(
                a=param_vals,
                q=qth_percentiles,
                weights=this_weights,
            )

        estimate[param]['mean'] = mean
        estimate[param]['median'] = median
        estimate[param]['lower_bound'] = lower
        estimate[param]['upper_bound'] = upper

    # -- Construct estimates array & metadata dict -- #

    estimation_settings = OrderedDict(
        [
            ('treat_dims_independently', np.bool8(treat_dims_independently)),
            ('use_prob_weights', np.bool8(use_prob_weights)),
            ('remove_priors', np.bool8(remove_priors)),
            ('averages_spherically_aware', np.bool8(averages_spherically_aware)),
        ]
    )

    if not averages_spherically_aware:
        return estimate, estimation_settings

    # Idea: calculate the meanns on the sphere for az and zen combined
    #
    # currently the below duplicates work done above but aware of spherical
    # coords, but just allowing this inefficiency for now since we're still
    # testing what's best
    for angle in ['', 'track_', 'cascade_']:
        az_name = angle + 'azimuth'
        zen_name = angle + 'zenith'

        if not (az_name in params and zen_name in params):
            continue

        az_idx = params.index(az_name)
        zen_idx = params.index(zen_name)

        az = cut_llhp[az_name]
        zen = cut_llhp[zen_name]

        # calculate the average of Cartesian coords
        # first need to create (x,y,z) array
        cart = np.empty(shape=(3, len(cut_llhp)))
        cart[0, :] = np.cos(az) * np.sin(zen)
        cart[1, :] = np.sin(az) * np.sin(zen)
        cart[2, :] = np.cos(zen)

        if use_prob_weights or remove_priors:
            cart_mean = np.average(cart, axis=1, weights=cut_weights)
        else:
            cart_mean = np.average(cart, axis=1)

        # normalize if r > 0
        r = np.sqrt(np.sum(np.square(cart_mean)))
        if r == 0:
            estimate[zen_name]['mean'] = 0
            estimate[az_name]['mean'] = 0
        else:
            estimate[zen_name]['mean'] = np.arccos(cart_mean[2] / r)
            estimate[az_name]['mean'] = np.arctan2(cart_mean[1], cart_mean[0]) % (2 * np.pi)

    return estimate, estimation_settings


def fit_cdf(x, cdf, distribution, x_is_data, verbosity=0):
    """Fit a distribution to a supplied cumulative distribution function (cdf).
    This allows fitting a distribution to weighted data.

    Parameters
    ----------
    x : array of same len as `cdf`
        `cdf` is sampled at `x` (sorted datapoints)

    cdf : array of same len as `x`, values in [0, 1]
        Values of the cdf at each `x`

    distribution : scipy.stats.distributions
        Continuous distribution with same interface as scipy's distributions

    x_is_data : bool
        If the `x` values are datapoint values, use these for getting a simple
        first guess fit

    Returns
    -------
    best_fit_params
    first_guess_params
    mse

    """
    if not x_is_data:
        raise NotImplementedError()

    if isinstance(distribution, string_types):
        distribution = getattr(stats.distributions, distribution)

    t0 = time.time()
    if distribution.shapes:
        shape_param_names = [d.strip() for d in distribution.shapes.split(',')]
    else:
        shape_param_names = []

    all_param_names = shape_param_names + ['loc', 'scale']

    failed_params = OrderedDict([(p, np.nan) for p in all_param_names])
    retval = OrderedDict(
        [
            ('name', distribution.name),
            ('success', False),
            ('total_time', np.nan),

            ('first_guess_params', failed_params),
            ('first_guess_mse', np.nan),
            ('first_guess_fit_time', np.nan),
            ('first_guess_exception', None),

            ('best_fit_params', failed_params),
            ('best_fit_mse', np.nan),
            ('best_fit_time', np.nan),
            ('best_fit_exception', None),
        ]
    )
    key_valfmts = OrderedDict([
        ('name', 's'),
        ('success', ''),
        ('total_time', '.3f'),

        ('first_guess_params', ''),
        ('first_guess_mse', '.3e'),
        ('first_guess_fit_time', '.1f'),
        ('first_guess_exception', 's'),

        ('best_fit_params', ''),
        ('best_fit_mse', '.3e'),
        ('best_fit_time', '.1f'),
        ('best_fit_exception', 's'),
    ])
    def print_info(retval, keys=None, stream=sys.stderr):
        """print info about fits"""
        if keys is None:
            keys = retval.keys()
        string = ''
        for key in keys:
            val = retval[key]
            if key.endswith('_params'):
                val = ', '.join('{} = {:.15e}'.format(*xx) for xx in val.items())
            elif key.endswith('_exception') and val is not None:
                val = '\n' + val
            string += ('{} : {:%s}\n' % key_valfmts[key]).format(key.rjust(25), val)
        string += '\n'
        stream.write(string)
        stream.flush()

    # Get first-guess param values by fitting unweighted data
    t1 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            first_guess = distribution.fit(x)
    except Exception:
        exc_info = sys.exc_info()
        try:
            retval['first_guess_fit_time'] = time.time() - t1
            retval['first_guess_exception'] = ''.join(traceback.format_exception(*exc_info))
            retval['total_time'] = time.time() - t0
        finally:
            del exc_info
        sys.stderr.write('ERROR: Dist "{}" first guess fit failed.\n'.format(distribution.name))
        if verbosity > 0:
            print_info(retval)
        return retval

    first_guess_params = OrderedDict([(p, v) for p, v in zip(all_param_names, first_guess)])
    first_guess_cdf = distribution.cdf(x, **first_guess_params)
    first_guess_mse = np.mean(np.square(first_guess_cdf - cdf))
    t2 = time.time()
    retval['first_guess_params'] = first_guess_params
    retval['first_guess_mse'] = first_guess_mse
    retval['first_guess_fit_time'] = t2 - t1

    if verbosity > 0:
        print_info(
            retval,
            keys=['name', 'first_guess_params', 'first_guess_mse', 'first_guess_fit_time'],
        )

    # Use weighted average as first-guess for 'loc' param instead?
    #weighted_average = np.average(a=data, weights=weights)

    # -- Perform fit to weighted data via CDF -- #

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            best_fit, _ = optimize.curve_fit(distribution.cdf, x, cdf, p0=first_guess)
    except Exception:
        exc_info = sys.exc_info()
        try:
            retval['best_fit_time'] = time.time() - t2
            retval['best_fit_exception'] = ''.join(traceback.format_exception(*exc_info))
            retval['total_time'] = time.time() - t0
        finally:
            del exc_info
        sys.stderr.write('ERROR: Dist "{}" best fit failed.\n'.format(distribution.name))
        if verbosity > 0:
            print_info(retval)
        return retval

    best_fit_params = OrderedDict([(p, v) for p, v in zip(all_param_names, best_fit)])
    best_fit_cdf = distribution.cdf(x, **best_fit_params)
    best_fit_mse = np.mean(np.square(best_fit_cdf - cdf))
    t3 = time.time()
    if np.all(np.isfinite(best_fit_params.values())) and np.isfinite(best_fit_mse):
        retval['success'] = True
    retval['best_fit_params'] = best_fit_params
    retval['best_fit_mse'] = best_fit_mse
    retval['best_fit_time'] = t3 - t2
    retval['total_time'] = t3 - t0

    if verbosity > 0:
        print_info(retval)

    return retval


if __name__ == '__main__':
    test_weighted_percentile()
