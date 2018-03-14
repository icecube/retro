# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Perform N-dimensional parameter scans of the likelihood space.
"""

from __future__ import absolute_import, division, print_function

__author__ = 'J.L. Lanfranchi'
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

from collections import Sequence
from copy import deepcopy
from itertools import izip, product
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import FTYPE, HYPO_PARAMS_T


def scan(hypo_generator, event, metric, dims, scan_values, nominal_params=None,
         metric_kwargs=None):
    """Scan likelihoods for hypotheses changing one parameter dimension.

    Parameters
    ----------
    hypo_generator
        Object e.g. retro.hypo.discrete_hypo.DiscreteHypo

    event : Event
        Event for which to get likelihoods

    metric : callable
        Function used to compute e.g. a likelihood. Must take ``sources`` and
        ``event`` as first two arguments, where ``sources`` is (...) and
        ``event`` is the argument passed here. Function must return just one
        value (e.g., ``-llh``)

    dims : string or iterable thereof
        One of 't', 'x', 'y', 'z', 'azimuth', 'zenith', 'cascade_energy',
        or 'track_energy'.

    scan_values : iterable of floats, or iterable thereof
        Values to set for the dimension being scanned.

    nominal_params : None or HYPO_PARAMS_T namedtuple
        Nominal values for all param values. The value for the params being
        scanned are irrelevant, as this is replaced with each value from
        `scan_values`. Therefore this is optional if _all_ parameters are
        subject to the scan.

    metric_kwargs : mapping or None
        Keyword arguments to pass to `get_neg_llh` function

    Returns
    -------
    metric_vals : numpy.ndarray (len(scan_values[0]) x len(scan_values[1]) x ...)
        Likelihoods corresponding to each value in product(*scan_values).

    """
    if metric_kwargs is None:
        metric_kwargs = {}

    all_params = HYPO_PARAMS_T._fields

    # Need list of strings (dim names). If we just have a string, make it the
    # first element of a single-element tuple.
    if isinstance(dims, basestring):
        dims = (dims,)

    # Need iterable-of-iterables-of-floats. If we have just an iterable of
    # floats (e.g. for 1D scan), then make it the first element of a
    # single-element tuple.
    if np.isscalar(next(iter(scan_values))):
        scan_values = (scan_values,)

    scan_sequences = []
    shape = []
    for sv in scan_values:
        if not isinstance(sv, Sequence):
            sv = list(sv)
        scan_sequences.append(sv)
        shape.append(len(sv))

    if nominal_params is None:
        #assert len(dims) == len(all_params)
        nominal_params = HYPO_PARAMS_T(*([np.nan]*len(all_params)))

    # Make nominal into a list so we can mutate its values as we scan
    nominal_params = list(nominal_params)

    # Get indices for each param that we'll be changing, in the order they will
    # be specified
    param_indices = []
    for dim in dims:
        param_indices.append(all_params.index(dim))

    metric_vals = []
    for param_values in product(*scan_sequences):
        params = deepcopy(nominal_params)
        for pidx, pval in izip(param_indices, param_values):
            params[pidx] = pval

        hypo_params = HYPO_PARAMS_T(*params)

        sources = hypo_generator.get_sources(hypo_params=hypo_params)
        metric_val = metric(sources, event, **metric_kwargs)
        metric_vals.append(metric_val)

    metric_vals = np.array(metric_vals, dtype=FTYPE)
    metric_vals.reshape(shape)

    return metric_vals
