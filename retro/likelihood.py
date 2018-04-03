# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Define likelihood functions used in Retro with the various kinds of tables we
have generated.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_llh']

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

from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import EMPTY_HITS # pylint: disable=unused-import


def get_llh(hypo, hits, time_window, hypo_handler, dom_tables, sd_indices=None,
            tdi_table=None):
    """Get the negative of the log likelihood of `event` having come from
    hypothesis `hypo` (whose light detection expectation is computed by
    `hypo_handler`).

    Parameters
    ----------
    hypo : HYPO_PARAMS_T
        Hypothesized event parameters

    hits : sequence of length NUM_DOMS_TOT
        Keys are (string, dom) tuples, values are `retro_types.Hits`
        namedtuples, where ``val.times`` and ``val.charges`` are arrays of
        shape (n_dom_hits_i,).

    time_window : FTYPE
        Time window pertinent to the event's reconstruction. Used for
        computing expected noise hits.

    hypo_handler : hypo.discrete_hypo.DiscreteHypo, etc.
        Object with method `get_sources` able to produce light sources expected
        to be produced by a hypothesized event

    dom_tables : tables.retro_5d_tables.Retro5DTables, etc.
        Instantiated object able to take light sources and convert into
        expected detections in each DOM.

    sd_indices : None or iterable of shape (2,) arrays
        Only use this subset of loaded doms. If None, all loaded DOMs will be
        used for computing the LLH.

    tdi_table : tables.tdi_table.TDITable, optional
        If provided, this is used to compute total expected hits, independent
        of time _and_ DOM. Instantiate the `dom_tables` object with
        `compute_t_indep_exp=False` to avoid unnecessary computations. If
        `tdi_table` is not provided, then the time-independent
        expecations must be computed for each DOM in the detector individually
        using the `dom_tables` object; be sure to instantiate this with
        `compute_t_indep_exp=True`.

    Returns
    -------
    llh : float
        Log likelihood

    """
    hypo_light_sources = hypo_handler.get_sources(hypo)

    llh = 0.0
    if tdi_table is not None:
        raise NotImplementedError()
        #llh = - tdi_table.get_expected_det(
        #    sources=hypo_light_sources
        #)

    if sd_indices is None:
        sd_indices = dom_tables.loaded_sd_indices

    #print('t_indep_tables dtypes:')
    #print([t.dtype for t in dom_tables.grouped_tuples['t_indep_tables']])

    #llh += dom_tables.get_llh(
    #    hypo_light_sources=hypo_light_sources,
    #    hits=hits,
    #    time_window=np.float32(time_window),
    #    dom_info=dom_tables.dom_info,
    #    sd_indices=sd_indices,
    #    tables=dom_tables.grouped_tuples['tables'],
    #    table_norms=dom_tables.grouped_tuples['table_norms'],
    #    t_indep_tables=dom_tables.grouped_tuples['t_indep_tables'],
    #    t_indep_table_norms=dom_tables.grouped_tuples['t_indep_table_norms'],
    #)

    pexp_func = dom_tables.pexp_func
    dom_info = dom_tables.dom_info
    tables = dom_tables.table_tups

    for sd_idx in sd_indices:
        # DEBUG: remove the below if / continue when no longer debugging!
        #if this_hits is EMPTY_HITS:
        #    continue
        exp_p_at_all_times, sum_log_at_hit_times = pexp_func(
            hypo_light_sources,
            hits[sd_idx],
            dom_info[sd_idx],
            np.float32(time_window),
            *tables[sd_idx]
        )
        llh += sum_log_at_hit_times - exp_p_at_all_times

    return llh
