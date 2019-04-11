#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Convolve a TDI table that tabulates "regular" photons with a Cherenkov cone to
arrive at a Cherenkov TDI table.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'generate_ckv_tdi_table',
    'parse_args',
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
import pickle
import sys

import numpy as np
from six import string_types

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.ckv import convolve_table
from retro.utils.misc import expand, mkdir


# TODO: allow different directional binning in output table
# TODO: write all keys of the table that are missing from the target directory


def generate_ckv_tdi_table(
    tdi_table,
    beta,
    oversample,
    num_cone_samples,
    n_phase=None,
    outdir=None,
    mmap_src=True,
    mmap_dst=False,
):
    """
    Parameters
    ----------
    tdi_table : string or mapping
        If string, path to TDI table file (or directory containing a
        `tdi_table.npy' file).

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

    n_phase : float or None
        Required if `tdi_table` is an array; if `tdi_table` specifies a table
        location, then `n_phase` will be read from the `tdi_metadata.pkl`
        file.

    outdir : string or None
        If a string, use this directory to place the resulting
        `ckv_tdi_table.npy` file. This is optional if `tdi_table` specifies a
        file or directory (in which case the `outdir` will be inferred from
        this path).

    mmap_src : bool, optional
        Whether to (attempt to) memory map the source `tdi_table` (if `table`
        is a string pointing to the file/directory). Default is `True`, as
        tables can easily exceed the memory capacity of a machine.

    mmap_dst : bool, optional
        Whether to memory map the destination `ckv_tdi_table.npy` file.

    """
    input_filename = None
    input_dirname = None
    if isinstance(tdi_table, string_types):
        tdi_table = expand(tdi_table)
        if isdir(tdi_table):
            input_filename = join(tdi_table, 'tdi_table.npy')
        elif isfile(tdi_table):
            input_filename = tdi_table
        else:
            raise IOError(
                '`tdi_table` is not a directory or file: "{}"'
                .format(tdi_table)
            )
        input_dirname = dirname(input_filename)

    if input_filename is None and outdir is None:
        raise ValueError(
            'You must provide an `outdir` if `tdi_table` is a python object'
            ' (i.e., not a file or directory path).'
        )

    if input_filename is None and n_phase is None:
        raise ValueError(
            'You must provide `n_phase` if `tdi_table` is a python object'
            ' (i.e., not a file or directory path).'
        )

    if n_phase is None:
        meta = pickle.load(file(join(input_dirname, 'tdi_metadata.pkl'), 'rb'))
        n_phase = meta['n_phase']

    if outdir is None:
        outdir = input_dirname
    mkdir(outdir)

    if input_filename is not None:
        tdi_table = np.load(
            input_filename,
            mmap_mode='r' if mmap_src else None,
        )

    cos_ckv = 1 / (n_phase * beta)
    if cos_ckv > 1:
        raise ValueError(
            'Particle moving at beta={} in medium with n_phase={} does not'
            ' produce Cherenkov light!'.format(beta, n_phase)
        )

    ckv_tdi_table_fpath = join(outdir, 'ckv_tdi_table.npy')
    if isfile(ckv_tdi_table_fpath):
        print(
            'WARNING! Destination file exists "{}"'
            .format(ckv_tdi_table_fpath)
        )

    if mmap_dst:
        # Allocate memory-mapped file
        ckv_tdi_table = np.lib.format.open_memmap(
            filename=ckv_tdi_table_fpath,
            mode='w+',
            dtype=np.float32,
            shape=tdi_table.shape,
        )
    else:
        ckv_tdi_table = np.empty(shape=tdi_table.shape, dtype=np.float32)

    try:
        convolve_table(
            src=tdi_table,
            dst=ckv_tdi_table,
            cos_ckv=cos_ckv,
            num_cone_samples=num_cone_samples,
            oversample=oversample,
            costhetadir_min=-1,
            costhetadir_max=+1,
            phidir_min=-np.pi,
            phidir_max=+np.pi,
        )
    except:
        del ckv_tdi_table
        if mmap_dst:
            remove(ckv_tdi_table_fpath)
        raise

    if not mmap_dst:
        np.save(ckv_tdi_table_fpath, ckv_tdi_table)

    return ckv_tdi_table


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--tdi-table', required=True,
        help='''Path to TDI table or path to directory containing the file
        `tdi_table.npy`'''
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
        help='''Directory in which to store the resulting table; if not
        specified, output table will be stored alongside the input table'''
    )
    return parser.parse_args()


if __name__ == '__main__':
    ckv_tdi_table = generate_ckv_tdi_table(**vars(parse_args())) # pylint: disable=invalid-name
