#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Combine multiple Retro CLSim tables into a single table.
"""


from __future__ import absolute_import, division, print_function


__all__ = ['combine_clsim_tables', 'parse_args', 'main']

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
from glob import glob
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import expand, mkdir
from retro.table_readers import load_clsim_table


def combine_clsim_tables(table_fpaths, save=True, outdir=None):
    """Combine multiple CLSim-produced tables together into a single table.

    All tables specified must have the same binnings defined. Tables should
    also be produced using different random seeds; if corresponding metadata
    files can be found in the same directories as the CLSim tables, this will
    be enforced prior to loading and combining the actual tables together.

    Parameters
    ----------
    table_fpaths : string or iterable thereof
        Each string is glob-expanded

    save : bool
        Whether to save the result to disk.

    outdir : None or string
        Directory to which to save the combined table if `save` is True. If
        None is specified (the default), the table will be saved to the same
        directory as the first file path found in `fpath`.

    Returns
    -------
    combined_table

    """
    if isinstance(table_fpaths, basestring):
        table_fpaths = [table_fpaths]

    table_fpaths_tmp = []
    for fpath in table_fpaths:
        table_fpaths_tmp.extend(fp for fp in glob(expand(fpath)))
    table_fpaths = table_fpaths_tmp

    if outdir is None:
        outdir = dirname(table_fpaths[0])
    mkdir(outdir)

    combined_table = None
    for fpath in table_fpaths:
        table = load_clsim_table(expand(fpath))

        if combined_table is None:
            combined_table = table
            continue

        combined_table['table'] += table['table']
        combined_table['n_photons'] += table['n_photons']

    return combined_table


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--table-fpaths', nargs='+', required=True,
        help='''Path(s) to Retro CLSim tables. Note that literal strings are
        glob-expanded.'''
    )
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory to which to save the combined table. Defaults to same
        directory as the first file path specified by --table-fpaths.'''
    )
    return parser.parse_args()


def main():
    """Main function for calling combine_clsim_tables as a script"""
    args = parse_args()
    kwargs = vars(args)
    combine_clsim_tables(**kwargs)


if __name__ == '__main__':
    main()
