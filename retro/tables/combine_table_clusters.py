#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Combine multiple Retro tables as specified in a table-clustering file.
"""

from __future__ import absolute_import, division, print_function

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
from os.path import abspath, dirname, join
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import OMKEY_T
from retro.utils.misc import expand
from retro.tables.combine_clsim_tables import combine_clsim_tables


def combine_table_clusters(
    cluster_file,
    source_path_proto,
    basedir,
    cluster_idx,
    t_is_residual_time=None,
    overwrite=False,
):
    """Combine clustered tables together.

    The file should be Numpy .npy format and contain a structured Numpy array
    with field 'label'; subsequent fields are used to format the source table
    filename prototype (e.g. fields 'string' and 'dom' can be used if prototype
    is "table_{string}_{dom}.fits").

    Output table files are stored to a `cl{cluster_idx}` subdirectory within
    the specified directory.

    Parameters
    ----------
    cluster_file : str
    source_path_proto : str
    basedir : str
    cluster_idx : int
    t_is_residual_time : bool, optional
    overwrite : bool, optional

    """
    cluster_file = expand(cluster_file)
    outdir = join(expand(basedir), 'cl{}'.format(cluster_idx))

    clusters = np.load(cluster_file)
    labels = clusters['label']
    members = clusters[labels == cluster_idx]
    assert len(members) > 0

    omkeys = np.empty(len(members), dtype=OMKEY_T)
    omkeys[['string', 'dom']] = members[['string', 'dom']]

    table_fpaths = []
    names = members.dtype.names
    for member in members:
        pathspec = expand(source_path_proto.format(**dict(zip(names, member))))
        fpaths = glob(pathspec)
        if len(fpaths) == 0:
            raise ValueError('Cannot file(s) "{}"'.format(pathspec))
        table_fpaths.extend(fpaths)

    combine_clsim_tables(
        table_fpaths=table_fpaths,
        t_is_residual_time=t_is_residual_time,
        outdir=outdir,
        overwrite=overwrite,
    )
    np.save(join(outdir, 'omkeys.npy'), omkeys)


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--cluster-file', required=True,
        help='''Path to .npy file containgin cluster specification'''
    )
    parser.add_argument(
        '--cluster-idx', required=True, type=int,
        help='''Index of the cluster whose tables are to be combined'''
    )
    parser.add_argument(
        '--source-path-proto', required=True,
        help='''Prototype for finding source table files (or directories)'''
    )
    parser.add_argument(
        '--t-is-residual-time', action='store_true',
        help='''Whether time dimension represents residual time'''
    )
    parser.add_argument(
        '--basedir', required=True,
        help='''Parent directory for storing the table. The actual table files
        will be placed in $BASEDIR/cl$CLUSTER_IDX'''
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='''Overwrite a table that already exists at the destination
        path'''
    )
    return parser.parse_args()


if __name__ == '__main__':
    combine_table_clusters(**vars(parse_args()))
