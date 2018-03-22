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
from pisa.utils.profiler import line_profile, profile
from pisa.utils import log
log.set_verbosity(2)

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import const


def get_llh(
        hypo, hits, sd_indices, time_window, hypo_handler, dom_tables,
        tdi_table=None
    ):
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

    sd_indices

    time_window : float
        Time window pertinent to the event's reconstruction. Used for
        computing expected noise hits.

    hypo_handler : hypo.discrete_hypo.DiscreteHypo, etc.
        Object with method `get_sources` able to produce light sources expected
        to be produced by a hypothesized event

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
    hypo_light_sources = hypo_handler.get_sources(hypo)

    #sum_at_all_times_computed = False
    if tdi_table is None:
        sum_at_all_times = 0.0
    else:
        raise NotImplementedError()
        #sum_at_all_times = tdi_table.get_expected_det(
        #    sources=hypo_light_sources
        #)
        #sum_at_all_times_computed = True

    llh = 0.0
    for sd_idx in sd_indices:
        this_hits = hits[sd_idx]
        # DEBUG: remove the below if / continue when no longer debugging!
        #if this_hits is None:
        #    continue
        exp_p_at_all_times, sum_log_at_hit_times = dom_tables.pexp_func(
            hypo_light_sources,
            this_hits,
            dom_tables.dom_info[sd_idx],
            time_window,
            *dom_tables.tables[sd_idx]
        )
        llh += tot_sum_log_at_hit_times - exp_p_at_all_times

    llh = tot_sum_log_at_hit_times - sum_at_all_times

    return llh
