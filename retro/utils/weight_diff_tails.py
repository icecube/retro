# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Define a function for compensating for inherent "tails" when differencing two
numbers drawn from the same finite range.

Useful for, e.g., zenith or cos(zenith) error distributions.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["weight_diff_tails", "test_weight_diff_tails"]

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2019 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from os.path import abspath, dirname
import sys

import numpy as np
from numpy import inf # pylint: disable=unused-import

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS


@numba_jit(**DFLT_NUMBA_JIT_KWARGS)
def weight_diff_tails(
    diff,
    weights,
    inbin_lower,
    inbin_upper,
    range_lower,
    range_upper,
    max_weight=np.inf,
):
    """Calculate weights that compensate for fewer points in the inherent tails
    of the difference between two values drawn from a limited range (e.g.
    coszen- or zenith-error).

    Parameters
    ----------
    diff : array
        Differences

    weights : None or array of same size as `diff`
        Existing weights that are to be multiplied by the tail weights to
        arrive at an overall weight for each event. If provided, must have same
        shape as `diff`.

    inbin_lower, inbin_upper : floats

    range_lower, range_upper : floats

    max_weight : float, optional

    Returns
    -------
    weights : array
    diff_limits : tuple of two scalars
        (diff_lower_lim, diff_upper_lim)

    """
    new_weights = np.empty_like(diff)
    num_elements = np.float64(len(diff))

    # Identify limits of possible diff distribution
    diff_lower_lim = range_lower - inbin_upper
    diff_upper_lim = range_upper - inbin_lower
    diff_limits = (diff_lower_lim, diff_upper_lim)

    # Identify inner limits of the tails
    lower_tail_upper_lim = range_lower - inbin_lower
    upper_tail_lower_lim = range_upper - inbin_upper

    # Identify tail widths
    lower_tail_width = lower_tail_upper_lim - diff_lower_lim
    upper_tail_width = diff_upper_lim - upper_tail_lower_lim

    max_nonweight = min(1.0, max_weight)

    total = 0.0
    if len(weights) > 0:
        for n, orig_weight in enumerate(weights):
            if diff[n] > upper_tail_lower_lim:
                new_weight = (
                    orig_weight * min(max_weight, upper_tail_width / (diff_upper_lim - diff[n]))
                )
            elif diff[n] < lower_tail_upper_lim:
                new_weight = (
                    orig_weight * min(max_weight, lower_tail_width / (diff[n] - diff_lower_lim))
                )
            else:
                new_weight = orig_weight * max_nonweight
            total += new_weight
            new_weights[n] = new_weight
    else:
        for n, diff[n] in enumerate(diff):
            if diff[n] > upper_tail_lower_lim:
                new_weight = min(max_weight, upper_tail_width / (diff_upper_lim - diff[n]))
            elif diff[n] < lower_tail_upper_lim:
                new_weight = min(max_weight, lower_tail_width / (diff[n] - diff_lower_lim))
            else:
                new_weight = max_nonweight
            total += new_weight
            new_weights[n] = new_weight

    norm_factor = num_elements / total

    for n, wt in enumerate(new_weights):
        new_weights[n] = wt * norm_factor

    return new_weights, diff_limits


def test_weight_diff_tails():
    """unit tests for `weight_diff_tails` function"""

    # -- Check that zenith error dist is approx. flat after weighting -- #

    n_samp = int(1e7)
    n_split_bins = 10
    n_hist_bins = 10

    rand = np.random.RandomState(0)
    true_zen = rand.uniform(0, np.pi, n_samp)
    reco_zen = rand.uniform(0, np.pi, n_samp)

    err = reco_zen - true_zen
    w0 = np.array([])
    w1 = np.ones_like(true_zen)

    true_zen_edges = np.linspace(0, np.pi, n_split_bins + 1)

    for bin_num in range(n_split_bins):
        le, ue = true_zen_edges[bin_num:bin_num+2]
        mask = (true_zen >= le) & (true_zen <= ue)
        for w in [w0, w1]:
            if len(w) > 0:
                w = w[mask]
            new_weights, (dl0, dl1) = weight_diff_tails(
                diff=err[mask],
                weights=w,
                inbin_lower=le,
                inbin_upper=ue,
                range_lower=0.,
                range_upper=np.pi,
            )
            bins = np.linspace(dl0, dl1, n_hist_bins + 1)
            hvals, _ = np.histogram(err[mask], weights=new_weights, bins=bins, density=True)
            bin_width = (dl1 - dl0) / n_hist_bins
            # Setting `density=True`,
            #   np.sum(bin_width * hvals) = 1
            # assuming all hvals are equal,
            #   bin_width * uniform_hval * n_hist_bins = 1
            # solving for uniform_hval gives:
            uniform_hval = 1 / (bin_width * n_hist_bins)
            relmaxabsdiff = np.max(np.abs((hvals/uniform_hval - 1)))
            assert relmaxabsdiff < 0.05, str(relmaxabsdiff)
    print("<< PASS : test_weight_diff_tails >>")


if __name__ == "__main__":
    test_weight_diff_tails()
