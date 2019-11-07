#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Combine multiple Retro CLSim tables into a single table.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'SUM_KEYS',
    'NO_VALIDATE_KEYS',
    'NO_WRITE_KEYS',
    'combine_tables',
    'main',
]

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
from glob import glob
from os.path import abspath, basename, dirname, isfile, join, splitext
import sys
from time import time

import numpy as np
from six import string_types

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import COMPR_EXTENSIONS, expand, mkdir, nsort_key_func, wstderr
from retro.tables.clsim_tables import load_clsim_table_minimal


SUM_KEYS = ('n_photons', 'table', 'ckv_table', 't_indep_table', 't_indep_ckv_table')
"""Sum together values corresponding to these keys in all tables"""

NO_VALIDATE_KEYS = SUM_KEYS + (
    'ckv_template_map',
    'omkeys',
    'pca_reduced_table',
    'source_tables',
    'templates',
    'template_chi2s',
    'source_tables',
    'seed',
    'string',
    'dom',
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_azimuth',
    'dom_zenith',
    'z',
    'azimuth',
    'zenith',
)

NO_WRITE_KEYS = (
    'underflow',
    'overflow',
    'seed',
    'string',
    'dom',
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_azimuth',
    'dom_zenith',
    'z',
    'azimuth',
    'zenith',
    'ckv_template_map',
    'pca_reduced_table',
    'templates',
    'template_chi2s',
)


def combine_tables(table_fpaths, outdir=None, overwrite=False):
    """Combine multiple tables together into a single table.

    All tables specified must have the same binnings defined. Tables should
    also be produced using different random seeds (if all else besides
    n_photons is equal); if corresponding metadata files can be found in the
    same directories as the CLSim tables, this will be enforced prior to
    loading and combining the actual tables together.

    Parameters
    ----------
    table_fpaths : string or iterable thereof
        Each string is glob-expanded

    outdir : string, optional
        Directory to which to save the combined table; if not specified, the
        resulting table will be returned but not saved to disk.

    overwrite : bool
        Overwrite an existing table. If a table is found at the output path and
        `overwrite` is False, the function simply returns without raising an
        exception.

    Returns
    -------
    combined_table

    """
    t_start = time()

    # Get all input table filepaths, including glob expansion

    orig_table_fpaths = deepcopy(table_fpaths)
    if isinstance(table_fpaths, string_types):
        table_fpaths = [table_fpaths]
    table_fpaths_tmp = []
    for fpath in table_fpaths:
        table_fpaths_tmp.extend(glob(expand(fpath)))
    table_fpaths = sorted(table_fpaths_tmp, key=nsort_key_func)

    if not table_fpaths:
        raise ValueError(
            "Found no tables given `table_fpaths` = {}".format(orig_table_fpaths)
        )

    wstderr(
        'Found {} tables to combine:\n  {}\n'.format(
            len(table_fpaths), '\n  '.join(table_fpaths)
        )
    )

    # Create the output directory

    if outdir is not None:
        outdir = expand(outdir)
        mkdir(outdir)

    # Combine the tables

    combined_table = None
    table_keys = None
    for fpath in table_fpaths:
        table = load_clsim_table_minimal(fpath, mmap=True)

        base = basename(fpath)
        rootname, ext = splitext(base)
        if ext.lstrip('.') in COMPR_EXTENSIONS:
            base = rootname
        if 'source_tables' not in table:
            table['source_tables'] = np.array([base], dtype=np.string0)

        if combined_table is None:
            combined_table = table
            table_keys = set(table.keys())

            # Formulate output file paths and check if they exist (do on first
            # table to avoid finding out we are going to overwrite a file
            # before loading all the source tables)
            if outdir is not None:
                output_fpaths = OrderedDict(
                    (
                        (k, join(outdir, k + '.npy'))
                        for k in sorted(table_keys.difference(NO_WRITE_KEYS))
                    )
                )
                if not overwrite:
                    for fp in output_fpaths.values():
                        if isfile(fp):
                            raise IOError(
                                'File at {} already exists, NOT overwriting'.format(fp)
                            )
                wstderr(
                    'Output files will be written to:\n  {}\n'.format(
                        '\n  '.join(output_fpaths.values())
                    )
                )

            continue

        # Make sure keys are the same

        new_table_keys = set(table.keys())
        missing_keys = sorted(
            table_keys
            .difference(new_table_keys)
            .difference(NO_VALIDATE_KEYS)
        )
        additional_keys = sorted(
            new_table_keys
            .difference(table_keys)
            .difference(NO_VALIDATE_KEYS)
        )
        if missing_keys or additional_keys:
            raise ValueError(
                'Table is missing keys {} and/or has additional keys {}'.format(
                    missing_keys, additional_keys
                )
            )

        # Validate keys that should be equal

        for key in sorted(table_keys.difference(NO_VALIDATE_KEYS)):
            if not np.array_equal(table[key], combined_table[key]):
                raise ValueError('Unequal "{}" in file {}'.format(key, fpath))

        # Add values from keys that should be summed

        for key in SUM_KEYS:
            if key not in table:
                continue
            combined_table[key] += table[key]

        # Concatenate and sort new source table(s) in source_tables array

        combined_table['source_tables'] = np.sort(
            np.concatenate([combined_table['source_tables'], table['source_tables']])
        )

        # Make sure to clear table from memory since these can be quite large

        del table

    # Save the data to npy files on disk (in a sub-directory for all of this
    # table's files)
    if outdir is not None:
        wstderr('Writing files:\n')

        len_longest_fpath = np.max([len(p) for p in output_fpaths.values()])
        for key in sorted(table_keys.difference(NO_WRITE_KEYS)):
            fpath = output_fpaths[key]
            wstderr('  {} ...'.format(fpath.ljust(len_longest_fpath)))
            t0 = time()
            np.save(fpath, combined_table[key])
            wstderr(' ({:12.3f} s)\n'.format(time() - t0))

    wstderr(
        'Total time to combine tables: {} s\n'.format(np.round(time() - t_start, 3))
    )

    return combined_table


def main(description=__doc__):
    """Script interface to `combine_tables`, parsing command line args
    and passing to that function.

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--table-fpaths', nargs='+', required=True,
        help='''Path(s) to Retro CLSim tables. Note that literal strings are
        glob-expanded.'''
    )
    parser.add_argument(
        '--outdir', required=True,
        help='''Directory to which to save the combined table. Defaults to same
        directory as the first file path specified by --table-fpaths.'''
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='''Overwrite existing table key(s) if they exist in output directory.'''
    )
    args = parser.parse_args()
    combine_tables(**vars(args))


if __name__ == '__main__':
    main()
