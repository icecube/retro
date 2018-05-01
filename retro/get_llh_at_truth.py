#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Instantiate Retro tables and scan the negative log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['scan_neg_llh', 'parse_args']

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

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, const
from retro.utils.misc import expand, mkdir, hrlist2list
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo.discrete_cascade_kernels import (
    point_cascade
)
from retro.hypo.discrete_muon_kernels import (
    const_energy_loss_muon, table_energy_loss_muon
)
from retro.i3info.extract_gcd import extract_gcd
from retro.likelihood import get_neg_llh
from retro.scan import scan
from retro.tables.retro_5d_tables import (
    NORM_VERSIONS, TABLE_KINDS, Retro5DTables
)


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument(
        '--outdir', required=True
    )
    parser.add_argument(
        '--n-events', type=int, default=None
    )
    parser.add_argument(
        '--start-event-idx', type=int, default=0
    )

    parser.add_argument(
        '--hits', required=True,
    )
    parser.add_argument(
        '--hits-are-photons', action='store_true',
    )
    parser.add_argument(
        '--time-window', type=float, required=True,
    )
    parser.add_argument(
        '--events', required=True,
        help='''Events file corresponding the hits file specified'''
    )
    parser.add_argument(
        '--angsens-model',
        choices='nominal  h1-100cm  h2-50cm  h3-30cm'.split()
    )

    parser.add_argument(
        '--cascade-kernel', choices=['point', 'one_dim'], required=True,
    )
    parser.add_argument(
        '--cascade-samples', type=int, default=1,
    )
    parser.add_argument(
        '--track-kernel', required=True,
        choices=['const_e_loss', 'nonconst_e_loss'],
    )
    parser.add_argument(
        '--track-time-step', type=float, required=True,
    )

    parser.add_argument(
        '--dom-tables-fname-proto', required=True,
        help='''Must have one of the brace-enclosed fields "{string}" or
        "{subdet}", and must have one of "{dom}" or "{depth_idx}". E.g.:
        "my_tables_{subdet}_{depth_idx}"'''
    )
    parser.add_argument(
        '--step-length', type=float, default=1.0,
        help='''Step length used in the CLSim table generator.'''
    )
    parser.add_argument(
        '--force-no-mmap', action='store_true',
        help='''Specify to NOT memory map the tables. If not specified, a
        sensible default is chosen for the type of tables being used.'''
    )
    parser.add_argument(
        '--dom-table-kind', choices=TABLE_KINDS, required=True,
        help='''Kind of single-DOM table to use.'''
    )
    parser.add_argument(
        '--gcd', required=True,
        help='''IceCube GCD file; can either specify an i3 file, or the
        extracted pkl file used in Retro.'''
    )
    parser.add_argument(
        '--norm-version', choices=NORM_VERSIONS, required=True,
        help='''Norm version.'''
    )
    parser.add_argument(
        '--no-dir', action='store_true',
        help='''Do NOT use source photon directionality'''
    )
    parser.add_argument(
        '--num-phi-samples', type=int, default=None,
    )
    parser.add_argument(
        '--ckv-sigma-deg', type=float, default=None,
    )
    parser.add_argument(
        '--tdi-table', default=None
    )
    parser.add_argument(
        '--template-library', default=None
    )

    return parser.parse_args()


def get_llh_at_truth(outdir, hypo_kw, dom_tables_kw, hits_kw, num_events=None):
    """Script "main" function

    Parameters
    ----------
    outdir : string

    hypo_kw : mapping
        Keyword args to pass to ``setup_discrete_hypo(**hypo_kw)``.

    dom_tables_kw : mapping
        Keyword args to pass to ``setup_dom_tables(**dom_tables_kw)``.

    hits_kw : mapping
        Keyword args to pass to ``get_hits(**dom_tables_kw)``.

    num_events : int or None
        If None, process all events returned by the hits generator.
    
    """
    hypo_handler = setup_discrete_hypo(**hypo_kw)
    dom_tables = setup_dom_tables(**dom_tables_kw)
    tdi_table = None
    hits_generator = get_hits(**hits_kw)

    outdir = kwargs.pop('outdir')
    mkdir(outdir)

    for count, (event_idx, event_hits) in enumerate(hits_generator):
        if count >= num_events:
            break
        min_t = np.inf
        max_t = -np.inf
        for hits in event_hits.values():
            min_t = min(min_t, hits[
        t_lims = (first_hit_t - table_t_max + 100, last_hit_t)
        t_range = last_hit_t - first_hit_t + table_t_max
        time_window = t_range

        # Keyword args for the `metric` callable (get_neg_llh)
        metric_kw = dict(
            time_window=time_window,
            hypo_handler=hypo_handler,
            dom_tables=dom_tables,
            tdi_table=tdi_table
        )

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Scanning paramters')
    t0 = time.time()

    metrics = []
    for event_ofst, event_hits in enumerate(hits[events_slice]):
        event_idx = start_event_idx + event_ofst
        t1 = time.time()
        if hits_are_photons:
            # For photons, we assign a "charge" from their weight, which comes
            # from angsens model.
            event_photons = event_hits
            # DEBUG: set back to EMPTY_HITS when not debugging!
            event_hits = [const.EMPTY_HITS]*const.NUM_DOMS_TOT
            #event_hits = [None]*const.NUM_DOMS_TOT
            for str_dom, pinfo in event_photons.items():
                sd_idx = const.get_sd_idx(string=str_dom[0], dom=str_dom[1])
                t = pinfo[0, :]
                coszen = pinfo[4, :]
                weight = np.float32(dom_tables.angsens_poly(coszen))
                event_hits[sd_idx] = np.concatenate(
                    (t[np.newaxis, :], weight[np.newaxis, :]),
                    axis=0
                )

        metric_kw['hits'] = event_hits

        # Perform the actual scan
        metric_vals = scan(
            scan_values=scan_values,
            metric=get_neg_llh,
            metric_kw=metric_kw
        )

        metrics.append(metric_vals)

        dt = time.time() - t1
        n_points = metric_vals.size
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt/n_points*1e3))

    kwargs.pop('hits')
    info = OrderedDict([
        ('hypo_params', HYPO_PARAMS_T._fields),
        ('scan_values', scan_values),
        ('kwargs', OrderedDict([(k, orig_kwargs[k]) for k in sorted(orig_kwargs.keys())])),
        ('metric_name', 'neg_llh'),
        ('metrics', metrics)
    ])

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    t0 = time.time()
    outfpath = join(outdir, 'scan.pkl')
    print('Saving results to pickle file at "{}"'.format(outfpath))
    pickle.dump(info, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    print('Total script run time is {:.3f} s'.format(time.time() - t00))

    return metrics, orig_kwargs


if __name__ == '__main__':
    metric_vals, orig_kwargs = scan_neg_llh() # pylint: disable=invalid-name
