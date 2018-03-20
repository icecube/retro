# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Define likelihood functions used in Retro with the various kinds of tables we
have generated.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_neg_llh']

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

from itertools import product
from os.path import abspath, dirname
import sys
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)


DC_STRS = [79, 80, 81, 82, 83, 84, 85, 86]
DC_IC_STRS = [26, 27, 35, 36, 37, 45, 46]

DC_SUBDUST_DOMS = list(range(11, 60+1))
IC_SUBDUST_DOMS = list(range(25, 60+1))

DC_SUBDUST_STRS_DOMS = list(product(DC_STRS, DC_SUBDUST_DOMS))
DC_IC_SUBDUST_STRS_DOMS = list(product(DC_IC_STRS, IC_SUBDUST_DOMS))

DC_ALL_SUBDUST_STRS_DOMS = DC_SUBDUST_STRS_DOMS + DC_IC_SUBDUST_STRS_DOMS

ALL_STRS = list(range(1, 86+1))
ALL_DOMS = list(range(1, 60+1))
ALL_STRS_DOMS = list(product(ALL_STRS, ALL_DOMS))

#EMPTY_HITS = Hits(
#    times=np.empty(shape=(0, 1), dtype=np.float32),
#    charges=np.empty(shape=(0, 1), dtype=np.float32)
#)
EMPTY_HITS = np.empty(shape=(2, 0), dtype=np.float32)


def get_neg_llh(
        hypo, hits, time_window, hypo_handler, dom_tables, tdi_table=None
    ):
    """Get the negative of the log likelihood of `event` having come from
    hypothesis `hypo` (whose light detection expectation is computed by
    `hypo_handler`).

    Parameters
    ----------
    hypo : HYPO_PARAMS_T
        Hypothesized event parameters

    hits : mapping
        Keys are (string, dom) tuples, values are `retro_types.Hits`
        namedtuples, where ``val.times`` and ``val.charges`` are arrays of
        shape (n_dom_hits_i,).

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
    neg_llh : float
        Negative of the log likelihood

    """
    start_tot_t = time.time()
    hypo_light_sources = hypo_handler.get_sources(hypo)

    sum_at_all_times_computed = False
    if tdi_table is None:
        if not dom_tables.compute_t_indep_exp:
            print('*'*79)
            print('WARNING! Time-independent expectation will not be computed')
            print('*'*79)
        sum_at_all_times = 0.0
    else:
        sum_at_all_times = tdi_table.get_expected_det(
            sources=hypo_light_sources
        )
        sum_at_all_times_computed = True

    sum_log_at_hit_times = np.float64(0.0)
    table_t = 0.
    for str_dom in DC_ALL_SUBDUST_STRS_DOMS:
        string = str_dom[0]
        dom = str_dom[1]

        this_hits = hits.get((string, dom), EMPTY_HITS)

        start_t = time.time()
        exp_p_at_all_times, exp_p_at_hit_times = dom_tables.get_expected_det(
            sources=hypo_light_sources,
            hit_times=this_hits[0, :].astype(np.float32),
            string=string,
            dom=dom,
            include_noise=True,
            time_window=time_window
        )
        table_t += (time.time() - start_t)

        if not sum_at_all_times_computed:
            sum_at_all_times += exp_p_at_all_times

        sum_log_at_hit_times += np.sum(
            this_hits[1, :] * np.log(exp_p_at_hit_times)
        )

    neg_llh = sum_at_all_times - sum_log_at_hit_times
    
    print('total time : %.5f'%(time.time() - start_tot_t))
    print('table time : %.5f'%(table_t))


    return neg_llh
