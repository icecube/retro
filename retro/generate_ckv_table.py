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
from os.path import abspath, dirname, expanduser, expandvars, isdir, isfile, join
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro
from retro.table_readers import load_clsim_table_minimal
from retro.ckv import convolve_table


__all__ = ['generate_ckv_table']


# TODO: allow different directional binning in output table

def generate_ckv_table(
        table, beta, oversample, num_cone_samples, outdir=None, mmap=True
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

    outdir : string or None
        If a string, use this directory to place the .npy file containing the
        ckv table. If `outdir` is None and `table` is a .npy-file-directory,
        this directory is used for `outdir`. If `outdir` is None and `table` is
        the path to a .fits file, `outdir` is the same name but with the .fits
        extension stripped. If `outdir` is None and `table` is a mapping, a
        ValueError is raised.
        npy-file-directory will be placed.

    mmap : bool, optional
        Whether to (attempt to) memory map the source `table` (if `table` is a
        string pointing to the file/directory). Default is `True`, as tables
        can easily exceed the memory capacity of a machine.

    """
    input_filename = None
    if isinstance(table, basestring):
        input_filename = expanduser(expandvars(table))
        table = load_clsim_table_minimal(input_filename, mmap=mmap)

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

    n_r_bins = len(r_bin_edges) - 1
    n_costheta_bins = len(costheta_bin_edges) - 1
    n_t_bins = len(t_bin_edges) - 1

    # NOTE: we are making output binning same as input binning.

    n_phase = table['phase_refractive_index']
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
    table = table['table'][(slice(1, -1),)*5]

    if outdir is None:
        if isdir(input_filename):
            outdir = input_filename
        elif isfile(input_filename):
            outdir = input_filename.rstrip('.fits')
            assert outdir != input_filename, str(input_filename)
    else:
        outdir = expanduser(expandvars(outdir))
        if not isdir(outdir):
            retro.mkdir(outdir)
    outdir = expanduser(expandvars(outdir))
    ckv_table_fpath = join(outdir, 'ckv_table.npy')
    retro.mkdir(outdir)

    # Allocate memory-mapped file
    ckv_table = np.lib.format.open_memmap(
        filename=ckv_table_fpath,
        mode='w+',
        dtype=np.float32,
        shape=table.shape
    )
    try:
        convolve_table(
            src=table,
            dst=np.float32(ckv_table),
            cos_ckv=np.float32(cos_ckv),
            sin_ckv=np.float32(sin_ckv),
            n_r=n_r_bins,
            n_ct=n_costheta_bins,
            n_t=n_t_bins,
            ctdir_bin_edges=costhetadir_bin_edges.astype(np.float32),
            dpdir_bin_edges=deltaphidir_bin_edges.astype(np.float32),
            num_cone_samples=num_cone_samples,
            oversample=oversample
        )
    except:
        del ckv_table
        remove(ckv_table_fpath)
        raise

    return ckv_table


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--table', required=True,
        help='''npy-table directories and/or .fits table files'''
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
