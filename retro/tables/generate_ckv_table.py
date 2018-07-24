#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Convert raw Retro 5D table (which represent survival probabilities for light
traveling in a particular direction) to table for Cherenkov emitters with a
particular direction.

Output table will be in .npy-files-in-a-directory format for easy memory
mapping.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'generate_ckv_table',
    'parse_args'
]

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

from argparse import ArgumentParser
from os import remove
from os.path import abspath, dirname, isdir, isfile, join
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.tables.clsim_tables import load_clsim_table_minimal
from retro.utils.ckv import convolve_table
from retro.utils.misc import expand, mkdir


# TODO: allow different directional binning in output table
# TODO: write all keys of the table that are missing from the target directory


def generate_ckv_table(
    table,
    beta,
    oversample,
    num_cone_samples,
    t_is_residual_time=None,
    outdir=None,
    mmap_src=True,
    mmap_dst=False
):
    """
    Parameters
    ----------
    table : string or mapping
        If string, path to table file (or directory in the case of npy table).
        A mapping is assumed to be a table loaded as by
        `retro.table_readers.load_clsim_table_minimal`.

    beta : float in [0, 1]
        Beta factor, i.e. velocity of the charged particle divided by the speed
        of light in vacuum: `v/c`.

    oversample : int > 0
        Sample from each directional bin (costhetadir and deltaphidir) this
        many times. Increase to obtain a more accurate average over the range
        of directions that the resulting ckv-emitter-direction can take within
        the same output (directional) bin. Note that there is no unique
        information given by sampling (more than once) in the spatial
        dimensions, so these dimensions ignore `oversample`. Therefore,
        the computational cost is `oversample**2`.

    num_cone_samples : int > 0
        Number of samples around the circumference of the Cherenkov cone.

    t_is_residual_time : bool, optional
        Whether table time binning is time difference from direct-light (True)
        or time binning is absolute time from photon creation (False). If not
        specified, the value comes from the key "t_is_residual_time" in the
        loaded table. It is an error if these values disagree, or if neither
        the argument `t_is_residual_time` is specified or "t_is_residual_time"
        is not present in the loaded table.

    outdir : string or None
        If a string, use this directory to place the .npy file containing the
        ckv table. If `outdir` is None and `table` is a .npy-file-directory,
        this directory is used for `outdir`. If `outdir` is None and `table` is
        the path to a .fits file, `outdir` is the same name but with the .fits
        extension stripped. If `outdir` is None and `table` is a mapping, a
        ValueError is raised.
        npy-file-directory will be placed.

    mmap_src : bool, optional
        Whether to (attempt to) memory map the source `table` (if `table` is a
        string pointing to the file/directory). Default is `True`, as tables
        can easily exceed the memory capacity of a machine.

    mmap_dst : bool, optional
        Whether to memory map the destination `ckv_table`.

    """
    input_filename = None
    if isinstance(table, basestring):
        input_filename = expand(table)
        table = load_clsim_table_minimal(input_filename, mmap=mmap_src)

    if input_filename is None and outdir is None:
        raise ValueError('You must provide an `outdir` if `table` is a python'
                         ' object (i.e. not a file or directory path).')

    # Store original table to keep binning info, etc.
    full_table = table

    r_bin_edges = full_table['r_bin_edges']
    costheta_bin_edges = full_table['costheta_bin_edges']
    t_bin_edges = full_table['t_bin_edges']
    costhetadir_bin_edges = full_table['costhetadir_bin_edges']
    deltaphidir_bin_edges = full_table['deltaphidir_bin_edges']

    if 't_is_residual_time' in full_table:
        if t_is_residual_time is None:
            t_is_residual_time = full_table['t_is_residual_time']
        else:
            assert full_table['t_is_residual_time'] == t_is_residual_time
    assert t_is_residual_time is not None
    print('t_is_residual_time:', t_is_residual_time)

    # NOTE: we are making output binning same as input binning.

    n_phase = full_table['phase_refractive_index']
    n_group = full_table['group_refractive_index']
    cos_ckv = 1 / (n_phase * beta)
    if cos_ckv > 1:
        raise ValueError(
            'Particle moving at beta={} in medium with n_phase={} does not'
            ' produce Cherenkov light!'.format(beta, n_phase)
        )

    theta_ckv = np.arccos(cos_ckv)
    sin_ckv = np.sin(theta_ckv)

    # Extract just the "useful" part of the table, i.e., exclude under/overflow
    # bins.
    table = full_table['table'][(slice(1, -1),)*5]

    if outdir is None:
        if isdir(input_filename):
            outdir = input_filename
        elif isfile(input_filename):
            outdir = input_filename.rstrip('.fits')
            assert outdir != input_filename, str(input_filename)
    else:
        outdir = expand(outdir)
        if not isdir(outdir):
            mkdir(outdir)
    outdir = expand(outdir)
    ckv_table_fpath = join(outdir, 'ckv_table.npy')
    mkdir(outdir)

    if mmap_dst:
        # Allocate memory-mapped file
        ckv_table = np.lib.format.open_memmap(
            filename=ckv_table_fpath,
            mode='w+',
            dtype=np.float32,
            shape=table.shape
        )
    else:
        ckv_table = np.empty(shape=table.shape, dtype=np.float32)

    try:
        convolve_table(
            src=table,
            dst=ckv_table,
            cos_ckv=np.float32(cos_ckv),
            sin_ckv=np.float32(sin_ckv),
            r_bin_edges=r_bin_edges.astype(np.float32),
            ct_bin_edges=costheta_bin_edges.astype(np.float32),
            t_bin_edges=t_bin_edges.astype(np.float32),
            t_is_residual_time=t_is_residual_time,
            ctdir_bin_edges=costhetadir_bin_edges.astype(np.float32),
            dpdir_bin_edges=deltaphidir_bin_edges.astype(np.float32),
            num_cone_samples=num_cone_samples,
            oversample=oversample,
            n_group=n_group,
        )
    except:
        del ckv_table
        if mmap_dst:
            remove(ckv_table_fpath)
        raise

    if not mmap_dst:
        np.save(ckv_table_fpath, ckv_table)

    return ckv_table


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--table', required=True,
        help='''npy-table directories and/or .fits table files'''
    )
    parser.add_argument(
        '--t-is-residual-time', action='store_true',
        help='''Pass this flag if table's time binning is time difference from
        direct time.'''
    )
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='''Cherenkov emitter beta factor (v / c).'''
    )
    parser.add_argument(
        '--oversample', type=int, required=True,
        help='''Sample each output (costhetadir, deltaphidir) bin oversample^2
        times.'''
    )
    parser.add_argument(
        '--num-cone-samples', type=int, required=True,
        help='''Number of samples around the cone.'''
    )
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory in which to store the resulting table
        directory(ies).'''
    )
    return parser.parse_args()


if __name__ == '__main__':
    ckv_table = generate_ckv_table(**vars(parse_args())) # pylint: disable=invalid-name
