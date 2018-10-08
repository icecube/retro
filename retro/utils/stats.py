# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Statistics
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'poisson_llh',
    'partial_poisson_llh',
    'weighted_average',
    'estimate_from_llhp',
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
import copy
from os.path import abspath, dirname
import sys

import numpy as np
from scipy.special import gammaln
from scipy import stats
import xarray

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro

DELTA_LLH_CUTOFF = 15.5
"""What values of the llhp space to include relative to the max-LLH point"""

def poisson_llh(expected, observed):
    r"""Compute the log Poisson likelihood.

    .. math::
        observed \times \log expected - expected \log \Gamma(observed)

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
    """

    Parameters
    ----------
    percenttile : scalar in [0, 100]
    weights
        Frequency (count) of data

    Returns
    -------
    wtd_pct : scalar

    """
    if weights is None:
        return np.percentile(data, percentile)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = w.cumsum() / w.sum() * 100
    return np.interp(percentile, p, d)


def estimate_from_llhp(
    llhp,
    treat_dims_independently,
    use_prob_weights,
    remove_priors,
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
        treat each dimension individually (not yet sure how much sense that makes)

    use_prob_weights : boolean
        use LLH weights

    remove_priors : boolean
        Whether to remove the effects of priors; if specified, `meta` must be
        passed (see help on that for more details)

    meta : dict, required if `remove_priors` is True
        Metadata from the minimization; must include priors if `remove_priors`
        is True; will be attached to returned `estimate` as `estimate.attrs`

    Returns
    -------
    estimate : OrderedDict of OrderedDicts
        Format is .. ::
            {
                "max": {"llh": llh_val, "x": x_val, ...},
                "mean": {"llh": llh_val, "x": x_val, ...},
                "median": {"llh": llh_val, "x": x_val, ...},
                "lower_bound": {"llh": llh_val, "x": x_val, ...},
                "upper_bound": {"llh": llh_val, "x": x_val, ...},
                "weighted_max": {"llh": llh_val, "x": x_val, ...},
                "weighted_mean": {"llh": llh_val, "x": x_val, ...},
                "weighted_median": {"llh": llh_val, "x": x_val, ...},
            }
        where the "weighted_*" keys (and corresponding dicts) will only be
        present if `use_prob_weights` is True. Keys "lower_bound" and "upper_bound" come from
        the `percentile` bounds. If `meta` is specified, then values are
        estimated by _removing_ the effect of the prior from the llh values.
        Values are themselves OrderedDict with keys being "llh" and the
        parameter (dimension) names, and values are the value of each
        parameter.

    """
    # currently spherical averages are not supported if dimensions are treated
    # independently (how would this even work?)
    averages_spherically_aware = not treat_dims_independently
    if remove_priors and 'priors_used' not in meta:
        raise KeyError('`meta` must contain "priors_used"')

    names = list(llhp.dtype.names)
    for name in ('llh', 'track_energy', 'cascade_energy'):
        if name not in names:
            raise ValueError(
                '"{}" not a field in `llhp.dtype.names` = {}'
                .format(name, names)
            )

    params = copy.copy(names)
    params.remove('llh')

    num_params = len(params)
    num_llhp = len(llhp)

    # cut away extremely low llh (30 or more below max llh)
    max_llh = np.nanmax(llhp['llh'])
    llhp = llhp[llhp['llh'] >= max_llh - 30]
    if len(llhp) == 0:
        raise IndexError('no points')

    max_llh_idx = np.nanargmax(llhp['llh'])

    # calculate weights from probabilities
    if use_prob_weights:
        prob_weights = np.exp(llhp['llh'] - max_llh)
        #prob_weights = 1./np.square((1. + max_llh - llhp['llh']))

    # calculate the prior weights from the priors used
    if remove_priors:
        if treat_dims_independently:
            prior_weights = {}
        else:
            prior_weights = np.ones(shape=num_llhp)

        for dim, (prior_kind, prior_params) in meta['priors_used'].items():
            if prior_kind == 'uniform':
                # for testing
                #if 'zenith' in dim:
                #    w = np.clip(np.abs(np.sin(llhp[dim])), 0.01, None)
                #else:
                w = np.ones(shape=num_llhp)
            elif prior_kind in ['cauchy', 'spefit2']:
                w = 1 / stats.cauchy.pdf(llhp[dim], *prior_params[:2])
                #w = np.ones(shape=num_llhp)
            elif prior_kind == 'log_normal' and dim == 'energy':
                w = 1 / stats.lognorm.pdf(
                    llhp['track_energy'] + llhp['cascade_energy'],
                    *prior_params[:3]
                )
                #w = np.ones(shape=num_llhp)
            elif prior_kind == 'log_uniform' and dim == 'energy':
                w = llhp['track_energy'] + llhp['cascade_energy']
                #w = np.ones(shape=num_llhp)
            elif prior_kind == 'log_uniform' and dim == 'cascade_energy':
                w = llhp['cascade_energy']
            elif prior_kind == 'log_uniform' and dim == 'track_energy':
                w = llhp['track_energy']
            elif prior_kind == 'cosine':
                # we don;t want to unweight that!
                #w = 1./np.clip(np.sin(llhp[dim]), 0.01, None)
                w = np.ones(shape=num_llhp)
            elif prior_kind == 'log_normal' and dim == 'cascade_d_zenith':
                w = 1.
            else:
                raise NotImplementedError(
                    'Prior %s for dimension %s unknown' % (prior_kind, dim)
                )

            if treat_dims_independently:
                prior_weights[dim] = w
            else:
                prior_weights *= w

    if remove_priors and treat_dims_independently and 'energy' in prior_weights:
        prior_weights['track_energy'] = (
            llhp['track_energy']
            / (llhp['track_energy'] + llhp['cascade_energy'])
            * prior_weights['energy']
        )
        prior_weights['cascade_energy'] = (
            llhp['cascade_energy']
            / (llhp['track_energy'] + llhp['cascade_energy'])
            * prior_weights['energy']
        )

    # -- Construct xarray for storing estimates & metadata -- #

    if meta is None:
        attrs = OrderedDict()
    else:
        # Copy `meta` and ensure `attrs` is OrderedDict
        attrs = OrderedDict(meta)

    # Save metadata for how estimate is calculated & max_llh value
    for var_name in (
        'treat_dims_independently',
        'use_prob_weights',
        'remove_priors',
        'averages_spherically_aware',
        'max_llh',
    ):
        val = eval(var_name) # pylint: disable=eval-used
        if var_name in attrs and attrs[var_name] != val:
            raise ValueError('key "{}" in `meta` with contradictory value')
        attrs[var_name] = val

    # Note that xarray requires list (tuple fails) for `coords`
    est_kinds = ['max', 'mean', 'median', 'lower_bound', 'upper_bound']
    compute_weighted_estimates = False
    if remove_priors or use_prob_weights:
        compute_weighted_estimates = True
        est_kinds += ['weighted_' + est_kind for est_kind in est_kinds]

    estimate = xarray.DataArray(
        data=np.full(
            fill_value=np.nan,
            shape=(len(est_kinds), num_params),
            dtype=np.float32,
        ),
        dims=('kind', 'param'),
        coords=dict(kind=est_kinds, param=params),
        attrs=attrs,
        name=attrs.get('event_idx'),
    )

    for param in params:
        param_vals = llhp[param]
        estimate.loc[dict(kind='max', param=param)] = param_vals[max_llh_idx]

        if not (use_prob_weights or remove_priors):
            continue

        if use_prob_weights:
            weights = np.copy(prob_weights)
        else:
            weights = np.ones(shape=num_llhp)

        if remove_priors:
            if treat_dims_independently:
                weights *= prior_weights[param]
            else:
                weights *= prior_weights

        # prirem_* attempt to remove the effect of priors seen in the "raw" llh values
        prirem_llh = np.log(weights)
        prirem_llh_max_llh_idx = np.nanargmax(prirem_llh)
        prirem_max_llh = prirem_llh[prirem_llh_max_llh_idx]
        estimate.loc[dict(kind='weighted_max', param=param)] = prirem_max_llh

        prirem_cut = prirem_llh > prirem_max_llh - DELTA_LLH_CUTOFF
        prirem_vals = param_vals[prirem_cut]
        prirem_weights = weights[prirem_cut]

        # now that we calculated values with effects of priors removed, we can
        # cut tighter
        llh_cut = llhp['llh'] > np.max(llhp['llh']) - DELTA_LLH_CUTOFF
        param_vals = param_vals[llh_cut]
        weights = weights[llh_cut]

        if 'azimuth' in param:
            # azimuth is a cyclic function, so need some special treatment to
            # get correct mean shift everything such that the best-fit point is
            # in the middle (pi)
            shift = estimate.loc[dict(kind='max', param=param)]
            var_shifted = (param_vals - shift + np.pi) % (2*np.pi)
            prirem_shifted = (prirem_vals - shift + np.pi) % (2*np.pi)

            median = (np.median(var_shifted) + shift - np.pi) % (2*np.pi)
            mean = (stats.circmean(var_shifted) + shift - np.pi) % (2*np.pi)
            lower_bound = (
                weighted_percentile(prirem_shifted, 13.35, prirem_weights)
                + shift - np.pi
            ) % (2*np.pi)
            upper_bound = (
                weighted_percentile(prirem_shifted, 86.65, prirem_weights)
                + shift - np.pi
            ) % (2*np.pi)
            if compute_weighted_estimates:
                weighted_mean = (
                    np.average(var_shifted, weights=weights) + shift - np.pi
                ) % (2*np.pi)
                weighted_median = (
                    weighted_percentile(var_shifted, 50, weights) + shift - np.pi
                ) % (2*np.pi)
        else:
            mean = np.mean(param_vals)
            median = np.median(param_vals)
            lower_bound = weighted_percentile(prirem_vals, 13.35, prirem_weights)
            upper_bound = weighted_percentile(prirem_vals, 86.65, prirem_weights)
            #lower_bound = weighted_percentile(param_vals, percentile, weights)
            #upper_bound = weighted_percentile(param_vals, 100-percentile, weights)
            if compute_weighted_estimates:
                weighted_mean = np.average(param_vals, weights=weights)
                weighted_median = weighted_percentile(param_vals, 50, weights)

        estimate.loc[dict(kind='mean', param=param)] = mean
        estimate.loc[dict(kind='median', param=param)] = median
        estimate.loc[dict(kind='lower_bound', param=param)] = lower_bound
        estimate.loc[dict(kind='upper_bound', param=param)] = upper_bound

        if compute_weighted_estimates:
            estimate.loc[dict(kind='weighted_mean', param=param)] = weighted_mean
            estimate.loc[dict(kind='weighted_median', param=param)] = weighted_median

    if not averages_spherically_aware:
        return estimate

    # Idea: calculate the medians on the sphere for az and zen combined
    #
    # currently the below duplicates work done above but aware of spherical
    # coords, but just allowing this inefficiency for now since we're still
    # testing what's best
    for angle in ['', 'track_', 'cascade_']:
        if angle + 'zenith' not in params or angle + 'azimuth' not in params:
            continue

        if prob_weights is None and prior_weights is None:
            weights = None
        else:
            if prob_weights is None:
                weights = np.ones(shape=num_llhp)
            else:
                weights = np.copy(prob_weights)
        if prior_weights is not None:
            weights *= prior_weights

        az = llhp[angle + 'azimuth']
        zen = llhp[angle + 'zenith']
        llh_cut = llhp['llh'] > np.max(llhp['llh']) - DELTA_LLH_CUTOFF
        az = az[llh_cut]
        zen = zen[llh_cut]
        weights = weights[llh_cut]

        # calculate the average and weighted average on sphere:
        # first need to create (x,y,z) array
        cart = np.zeros(shape=(3, num_llhp))

        cart[0] = np.cos(az) * np.sin(zen)
        cart[1] = np.sin(az) * np.sin(zen)
        cart[2] = np.cos(zen)

        cart_mean = np.average(cart, axis=1)
        cart_weighted_mean = np.average(cart, axis=1, weights=weights)

        # normalize
        r = np.sqrt(np.sum(np.square(cart_mean)))
        r_weighted = np.sqrt(np.sum(np.square(cart_weighted_mean)))

        if r == 0:
            estimate.loc[dict(kind='mean', param=angle + 'zenith')] = 0
            estimate.loc[dict(kind='mean', param=angle + 'azimuth')] = 0
        else:
            estimate.loc[dict(kind='mean', param=angle + 'zenith')] = (
                np.arccos(cart_mean[2] / r)
            )
            estimate.loc[dict(kind='mean', param=angle + 'azimuth')] = np.arctan2(
                cart_mean[1], cart_mean[0]
            ) % (2 * np.pi)

        if r_weighted == 0:
            estimate.loc[dict(kind='weighted_mean', param=angle + 'zenith')] = 0
            estimate.loc[dict(kind='weighted_mean', param=angle + 'azimuth')] = 0
        else:
            estimate.loc[dict(kind='weighted_mean', param=angle + 'zenith')] = np.arccos(
                cart_weighted_mean[2] / r_weighted
            )
            estimate.loc[dict(kind='weighted_mean', param=angle + 'azimuth')] = np.arctan2(
                cart_weighted_mean[1], cart_weighted_mean[0]
            ) % (2 * np.pi)

    return estimate
