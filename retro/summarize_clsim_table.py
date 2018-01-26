#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

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
from itertools import product
from os.path import abspath, basename, dirname, isfile, join, splitext
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
import retro
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
    table
        See `load_clsim_table` for details of the data structure

    summary : OrderedDict

    """
    t_start = time()
    if save_summary:
        from pisa.utils.jsons import from_json, to_json

    table_fpath = retro.expand(table_fpath)
    srcdir, clsim_fname = dirname(table_fpath), basename(table_fpath)
    invalid_fname = False
    try:
        fname_info = retro.interpret_clsim_table_fname(clsim_fname)
    except ValueError:
        invalid_fname = True
        fname_info = {}

    if outdir is None:
        outdir = srcdir
    outdir = retro.expand(outdir)
    retro.mkdir(outdir)

    if invalid_fname:
        metapath = None
    else:
        metaname = (retro.CLSIM_TABLE_METANAME_PROTO[-1]
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
        if key == 'table':
            continue
        summary[key] = table[key]
    if fname_info:
        for key in ('hash_val', 'string', 'depth_idx', 'seed'):
            summary[key] = fname_info[key]
    # TODO: Add hole ice info when added to tray_kw_to_hash
    if meta:
        summary['n_events'] = meta['tray_kw_to_hash']['NEvents']
        summary['ice_model'] = meta['tray_kw_to_hash']['IceModel']
        summary['tilt'] = not meta['tray_kw_to_hash']['DisableTilt']
        for key, val in meta.items():
            if key.endswith('_binning_kw'):
                summary[key] = val
    elif 'fname_version' in fname_info and fname_info['fname_version'] == 1:
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
        / (retro.SPEED_OF_LIGHT_M_PER_NS / table['phase_refractive_index']
           * np.mean(np.diff(table['t_bin_edges'])))
        #* table['angular_acceptance_fract']
        * (len(table['costheta_bin_edges']) - 1)
    )
    summary['norm'] = norm

    dim_names = ('r', 'costheta', 't', 'costhetadir', 'deltaphidir')
    n_dims = len(table['table_shape'])
    assert n_dims == len(dim_names)

    # Apply norm to underflow and overflow so magnitudes can be compared
    # relative to plotted marginal distributions
    for flow, idx in product(('underflow', 'overflow'), iter(range(n_dims))):
        summary[flow][idx] = summary[flow][idx] * norm

    retro.wstderr('Finding marginal distributions...\n')
    retro.wstderr('    masking off zeros in table...')
    t0 = time()
    nonzero_table = np.ma.masked_equal(table['table'], 0)
    retro.wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3, 3)))

    t0_marg = time()
    summary['dimensions'] = OrderedDict()
    for keep_axis, ax_name in zip(tuple(range(n_dims)), dim_names):
        remove_axes = list(range(n_dims))
        remove_axes.pop(keep_axis)
        remove_axes = tuple(remove_axes)
        axis = OrderedDict()

        retro.wstderr('    mean across non-{} axes...'.format(ax_name))
        t0 = time()
        axis['mean'] = norm * np.asarray(
            np.mean(table['table'], axis=remove_axes)
        )
        retro.wstderr(' ({} s)\n'.format(np.round(time() - t0, 3)))

        retro.wstderr('    median across non-{} axes...'.format(ax_name))
        t0 = time()
        axis['median'] = norm * np.asarray(
            np.ma.median(nonzero_table, axis=remove_axes)
        )
        retro.wstderr(' ({} s)\n'.format(np.round(time() - t0, 3)))

        retro.wstderr('    max across non-{} axes...'.format(ax_name))
        t0 = time()
        axis['max'] = norm * np.asarray(
            np.max(table['table'], axis=remove_axes)
        )
        retro.wstderr(' ({} s)\n'.format(np.round(time() - t0, 3)))
        summary['dimensions'][ax_name] = axis
    retro.wstderr(
        '  Total time to find marginal distributions: {} s\n'
        .format(np.round(time() - t0_marg, 3))
    )

    if save_summary:
        ext = None
        base_fname = clsim_fname
        while ext not in ('', '.fits'):
            base_fname, ext = splitext(base_fname)
            ext = ext.lower()
        outfpath = join(outdir, base_fname + '_summary.json.bz2')
        to_json(summary, outfpath)
        print('saved summary to "{}"'.format(outfpath))

    retro.wstderr('Time to summarize table: {} s\n'
                  .format(np.round(time() - t_start, 3)))

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
        table_fpaths.extend(glob(retro.expand(fpath)))
    for fpath in table_fpaths:
        kwargs['table_fpath'] = fpath
        summarize_clsim_table(**kwargs)
    total_time = time() - t0
    if len(table_fpaths) > 1:
        avg = np.round(total_time / len(table_fpaths), 3)
        retro.wstderr('Average time to summarize tables: {} s/table\n'.format(avg))


if __name__ == '__main__':
    main()
