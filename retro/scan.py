# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position


"""
Perform 1D and 2D parameter scans of the likelihood space.
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
from itertools import izip, product
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro


def scan(hypo_obj, event, neg_llh_func, dims, scan_values, nominal_params=None,
         neg_llh_func_kwargs=None):
    """Scan likelihoods for hypotheses changing one parameter dimension.

    Parameters
    ----------
    hypo_obj

    event : Event
        Event for which to get likelihoods

    neg_llh_func : callable
        Function used to compute a likelihood. Must take ``pinfo_gen`` and
        ``event`` as first two arguments, where ``pinfo_gen`` is (...) and
        ``event`` is the argument passed here. Function must return just one
        value (the ``llh``)

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

    neg_llh_func_kwargs : mapping or None
        Keyword arguments to pass to `get_neg_llh` function

    Returns
    -------
    all_llh : numpy.ndarray (len(scan_values[0]) x len(scan_values[1]) x ...)
        Likelihoods corresponding to each value in product(*scan_values).

    """
    if neg_llh_func_kwargs is None:
        neg_llh_func_kwargs = {}

    all_params = retro.HYPO_PARAMS_T._fields

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
        nominal_params = retro.HYPO_PARAMS_T(*([np.nan]*len(all_params)))

    # Make nominal into a list so we can mutate its values as we scan
    params = list(nominal_params)

    # Get indices for each param that we'll be changing, in the order they will
    # be specified
    param_indices = []
    for dim in dims:
        param_indices.append(all_params.index(dim))

    all_neg_llh = []
    for param_values in product(*scan_sequences):
        for pidx, pval in izip(param_indices, param_values):
            params[pidx] = pval

        hypo_params = retro.HYPO_PARAMS_T(*params)

        pinfo_gen = hypo_obj.get_pinfo_gen(hypo_params=hypo_params)
        neg_llh = neg_llh_func(pinfo_gen, event, **neg_llh_func_kwargs)
        all_neg_llh.append(neg_llh)

    all_neg_llh = np.array(all_neg_llh, dtype=retro.FTYPE)
    all_neg_llh.reshape(shape)

    return all_neg_llh
