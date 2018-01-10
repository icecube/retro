#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Generate a json file summarizing a CLSim table
"""


from __future__ import absolute_import, division, print_function


__all__ = ['summarize_clsim_table', 'parse_args', 'main']

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
from glob import glob
from os.path import abspath, basename, dirname, isfile, join, splitext
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import CLSIM_TABLE_METANAME_PROTO, SPEED_OF_LIGHT_M_PER_NS
from retro import expand, interpret_clsim_table_fname, mkdir
from retro.table_readers import load_clsim_table


def summarize_clsim_table(table_fpath, table=None, save_summary=True,
                          outdir=None):
    """
    Parameters
    ----------
    table_fpath : string
        Path to table (or just the table's filename if `outdir` is specified)

    table : mapping, optional
        If the table has already been loaded, it can be passed here to avoid
        re-loading the table.

    save_summary : bool
        Whether to save the table summary to disk.

    outdir : string, optional
        If `save_summary` is True, write the summary to this directory. If
        `outdir` is not specified and `save_summary` is True, the summary will
        be written to the same directory that contains `table_fpath`.

    Returns
    -------
    table : dictionary
        See `load_clsim_table` for details of the data structure

    summary : dictionary

    """
    if save_summary:
        from pisa.utils.jsons import from_json, to_json

    table_fpath = expand(table_fpath)
    srcdir, clsim_fname = dirname(table_fpath), basename(table_fpath)
    invalid_fname = False
    try:
        fname_info = interpret_clsim_table_fname(clsim_fname)
    except ValueError:
        invalid_fname = True
        fname_info = {}

    if outdir is None:
        outdir = srcdir
    outdir = expand(outdir)
    mkdir(outdir)

    if invalid_fname:
        metapath = None
    else:
        metaname = (CLSIM_TABLE_METANAME_PROTO
                    .format(hash_val=fname_info['hash_val']))
        metapath = join(outdir, metaname)
    if metapath and isfile(metapath):
        meta = from_json(metapath)
    else:
        meta = dict()

    if table is None:
        table = load_clsim_table(table_fpath)

    summary = OrderedDict()
    for key in table.keys():
        if key in ['table']:
            continue
        summary[key] = table[key]
    if fname_info:
        for key in ['hash_val', 'string', 'depth_idx', 'seed']:
            summary[key] = fname_info[key]
    # TODO: Add hole ice info when added to tray_kw_to_hash
    if meta:
        summary['n_events'] = meta['tray_kw_to_hash']['NEvents']
        summary['ice_model'] = meta['tray_kw_to_hash']['IceModel']
        summary['tilt'] = not meta['tray_kw_to_hash']['DisableTilt']
        for key, val in meta.items():
            if key.endswith('_binning_kw'):
                summary[key] = val
    elif fname_info['fname_version'] == 1:
        summary['n_events'] = fname_info['n_events']
        summary['ice_model'] = 'spice_mie'
        summary['tilt'] = False
        summary['r_binning_kw'] = dict(min=0.0, max=400.0, n_bins=200, power=2)
        summary['costheta_binning_kw'] = dict(min=-1, max=1, n_bins=40)
        summary['t_binning_kw'] = dict(min=0.0, max=3000.0, n_bins=300)
        summary['costhetadir_binning_kw'] = dict(min=-1, max=1, n_bins=20)
        summary['deltaphidir_binning_kw'] = dict(min=0.0, max=np.pi, n_bins=20)

    # Save marginal distributions and info to file
    norm = (
        1
        / table['n_photons']
        / (SPEED_OF_LIGHT_M_PER_NS / table['phase_refractive_index']
           * np.mean(np.diff(table['t_bin_edges'])))
        #* table['angular_acceptance_fract']
        * (len(table['costheta_bin_edges']) - 1)
    )
    summary['norm'] = norm

    dim_names = ['r', 'costheta', 't', 'costhetadir', 'deltaphidir']
    n_dims = len(table['table_shape'])
    assert n_dims == len(dim_names)

    summary['dimensions'] = OrderedDict()
    for keep_axis, ax_name in zip(tuple(range(n_dims)), dim_names):
        remove_axes = list(range(n_dims))
        remove_axes.pop(keep_axis)
        remove_axes = tuple(remove_axes)
        axis = OrderedDict()
        axis['mean'] = norm * np.mean(table['table'], axis=remove_axes)
        axis['max'] = norm * np.max(table['table'], axis=remove_axes)
        summary['dimensions'][ax_name] = axis

    if save_summary:
        base_fname, _ = splitext(clsim_fname)
        outfpath = join(outdir, base_fname + '_summary.json')
        to_json(summary, outfpath)
        print('saved summary to "{}"'.format(outfpath))

    return table, summary


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory in which to save summary (if not specified, summary
        is saved to same directory as the table)'''
    )
    parser.add_argument(
        'table-fpaths', nargs='+',
        help='''Path(s) to CLSim table(s). Note that literal strings are
        glob-expanded.'''
    )
    return parser.parse_args()


def main():
    """Main function for calling summarize_clsim_table as a script"""
    t0 = time()
    args = parse_args()
    kwargs = vars(args)
    table_fpaths = []
    for fpath in kwargs.pop('table-fpaths'):
        table_fpaths.extend(glob(expand(fpath)))
    for fpath in table_fpaths:
        kwargs['table_fpath'] = fpath
        summarize_clsim_table(**kwargs)
    total_time = time() - t0
    avg_time = total_time / len(table_fpaths)
    print('Time to summarize table(s): {} s (average {} s/table)'
          .format(total_time, avg_time))


if __name__ == '__main__':
    main()
