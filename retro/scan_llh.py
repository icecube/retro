#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, range-builtin-not-iterating

"""
Scan the log-likelihood space.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['scan_llh', 'parse_args']

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
from os.path import abspath, dirname, join
import pickle
import sys
import time

import numpy as np

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, init_obj
from retro import likelihood
from retro.const import ALL_STRS_DOMS_SET
from retro.scan import scan
from retro.utils.misc import expand, mkdir, sort_dict


def parse_args(description=__doc__):
    """Parse command-line arguments"""
    parser = ArgumentParser(description=description)

    parser.add_argument('--outdir', required=True)

    group = parser.add_argument_group(title='Scan parameters')
    for dim in HYPO_PARAMS_T._fields:
        group.add_argument(
            '--{}'.format(dim.replace('_', '-')), nargs='+', required=True,
            help='''Hypothses will take this(these) value(s) for dimension
            {dim_hr}. Specify a single value to not scan over this dimension;
            specify a human-readable string of values, e.g. '0, 0.5, 1-10:0.2'
            scans 0, 0.5, and from 1 to 10 (inclusive of both endpoints) with
            stepsize of 0.2.'''.format(dim_hr=dim.replace('_', ' '))
        )

    split_kwargs = init_obj.parse_args(dom_tables=True, hypo=True, events=True,
                                       parser=parser)
    split_kwargs['scan_kw'] = split_kwargs.pop('other_kw')

    if split_kwargs['events_kw']['hits'] is None:
        raise ValueError('Must specify a path to a pulse series or photon'
                         ' series using --hits.')

    return split_kwargs


def scan_llh(dom_tables_kw, hypo_kw, events_kw, scan_kw):
    """Script "main" function"""
    t00 = time.time()

    scan_values = []
    for dim in HYPO_PARAMS_T._fields:
        val_str = ''.join(scan_kw.pop(dim))
        val_str = val_str.lower().replace('pi', format(np.pi, '.17e'))
        scan_values.append(hrlist2list(val_str))

    dom_tables = init_obj.setup_dom_tables(**dom_tables_kw)
    hypo_handler = init_obj.setup_discrete_hypo(**hypo_kw)
    events_generator = init_obj.get_events(**events_kw)

    # Pop 'outdir' from `scan_kw` since we don't want to store this info in
    # the metadata dict.
    outdir = expand(scan_kw.pop('outdir'))
    mkdir(outdir)

    print('Scanning paramters')
    t0 = time.time()

    fast_llh = True

    if fast_llh:
        get_llh = dom_tables._get_llh
        dom_info = dom_tables.dom_info
        tables = dom_tables.tables
        table_norm = dom_tables.table_norm
        t_indep_tables = dom_tables.t_indep_tables
        t_indep_table_norm = dom_tables.t_indep_table_norm
        sd_idx_table_indexer = dom_tables.sd_idx_table_indexer
        metric_kw = {}
        def metric_wrapper(hypo, hits, hits_indexer, unhit_sd_indices,
                           time_window):
            sources = hypo_handler.get_sources(hypo)
            return get_llh(
                sources=sources,
                hits=hits,
                hits_indexer=hits_indexer,
                unhit_sd_indices=unhit_sd_indices,
                sd_idx_table_indexer=sd_idx_table_indexer,
                time_window=time_window,
                dom_info=dom_info,
                tables=tables,
                table_norm=table_norm,
                t_indep_tables=t_indep_tables,
                t_indep_table_norm=t_indep_table_norm
            )
    else:
        metric_kw = dict(dom_tables=dom_tables, tdi_table=None)
        get_llh = likelihood.get_llh
        def metric_wrapper(hypo, **metric_kw):
            sources = hypo_handler.get_sources(hypo)
            return get_llh(sources=sources, **metric_kw)

    n_points_total = 0
    metric_vals = []
    for _, event in events_generator:
        hits = event['hits']
        hits_indexer = event['hits_indexer']
        hits_summary = event['hits_summary']
        metric_kw['hits'] = hits
        metric_kw['hits_indexer'] = hits_indexer
        hit_sd_indices = hits_indexer['sd_idx']
        unhit_sd_indices = np.array(
            sorted(ALL_STRS_DOMS_SET.difference(hit_sd_indices)),
            dtype=np.uint32
        )
        metric_kw['unhit_sd_indices'] = unhit_sd_indices
        metric_kw['time_window'] = np.float32(
            hits_summary['time_window_stop'] - hits_summary['time_window_start']
        )

        t1 = time.time()
        metric_vals.append(scan(scan_values, metric_wrapper, metric_kw))
        dt = time.time() - t1

        n_points = metric_vals[-1].size
        n_points_total += n_points
        print('  ---> {:.3f} s, {:d} points ({:.3f} ms per LLH)'
              .format(dt, n_points, dt/n_points*1e3))
    dt = time.time() - t0

    info = OrderedDict([
        ('hypo_params', HYPO_PARAMS_T._fields),
        ('scan_values', scan_values),
        ('metric_name', 'llh'),
        ('metric_vals', metric_vals),
        ('scan_kw', sort_dict(scan_kw)),
        ('dom_tables_kw', sort_dict(dom_tables_kw)),
        ('hypo_kw', sort_dict(hypo_kw)),
        ('events_kw', sort_dict(events_kw)),
    ])

    outfpath = join(outdir, 'scan.pkl')
    print('Saving results in pickle file, path "{}"'.format(outfpath))
    pickle.dump(info, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    print('Total time to scan: {:.3f} s; {:.3f} ms avg per LLH'
          .format(time.time() - t00, dt/n_points_total*1e3))

    return metric_vals, info


if __name__ == '__main__':
    metric_vals, info = scan_llh(**parse_args()) # pylint: disable=invalid-name
