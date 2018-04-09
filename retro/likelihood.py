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
from retro.const import ALL_STRS_DOMS, EMPTY_HITS # pylint: disable=unused-import


def get_llh(sources, hits, hits_indexer, unhit_sd_indices, time_window,
            dom_tables, tdi_table=None):
    """Get the negative of the log likelihood of `event` having come from
    hypothesis `hypo` (whose light detection expectation is computed by
    `hypo_handler`).

    Parameters
    ----------
    sources

    hits : shape (n_hits_total,) array of dtype HIT_T
        Keys are (string, dom) tuples, values are `retro_types.Hits`
        namedtuples, where ``val.times`` and ``val.charges`` are arrays of
        shape (n_dom_hits_i,).

    hits_indexer : shape (n_hit_dims,) array of dtype SD_INDEXER_T

    unhit_sd_indices : shape (n_unhit_doms,) array of dtype uint32

    time_window : FTYPE
        Time window pertinent to the event's reconstruction. Used for
        computing expected noise hits.

    dom_tables : tables.retro_5d_tables.Retro5DTables, etc.
        Instantiated object able to take light sources and convert into
        expected detections in each DOM.

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
    llh = 0.0
    if tdi_table is not None:
        raise NotImplementedError()
        #llh = - tdi_table.get_expected_det(
        #    sources=hypo_light_sources
        #)

    pexp_func = dom_tables._pexp
    dom_info = dom_tables.dom_info
    tables = dom_tables.tables
    table_norms = dom_tables.table_norms
    t_indep_tables = dom_tables.t_indep_tables
    t_indep_table_norms = dom_tables.t_indep_table_norms
    sd_idx_table_indexer = dom_tables.sd_idx_table_indexer

    for sd_idx in unhit_sd_indices:
        table_idx = sd_idx_table_indexer[sd_idx]
        exp_p_at_all_times, sum_log_at_hit_times = pexp_func(
            sources=sources,
            hits=EMPTY_HITS,
            dom_info=dom_info[sd_idx],
            time_window=time_window,
            table=tables[table_idx],
            table_norm=table_norms[table_idx],
            t_indep_table=t_indep_tables[table_idx],
            t_indep_table_norm=t_indep_table_norms[table_idx]
        )
        llh += sum_log_at_hit_times - exp_p_at_all_times

    for indexer_entry in hits_indexer:
        sd_idx = indexer_entry['sd_idx']
        table_idx = sd_idx_table_indexer[sd_idx]
        start = indexer_entry['offset']
        stop = start + indexer_entry['num']
        exp_p_at_all_times, sum_log_at_hit_times = pexp_func(
            sources=sources,
            hits=hits[start : stop],
            dom_info=dom_info[sd_idx],
            time_window=time_window,
            table=tables[table_idx],
            table_norm=table_norms[table_idx],
            t_indep_table=t_indep_tables[table_idx],
            t_indep_table_norm=t_indep_table_norms[table_idx]
        )
        llh += sum_log_at_hit_times - exp_p_at_all_times

    return llh
