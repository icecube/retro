# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for using a set of 5D (r, costheta, t, costhetadir, deltaphidir)
Cherenkov Retro tables (5D CLSim tables with directionality map convolved with
a Cherenkov cone)
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'CKV_TABLE_KEYS',
    'load_ckv_table'
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

from collections import OrderedDict
from os.path import abspath, basename, dirname, isfile, join
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DEBUG
from retro.utils.misc import expand, wstderr


CKV_TABLE_KEYS = [
    'n_photons', 'group_refractive_index', 'phase_refractive_index',
    'r_bin_edges', 'costheta_bin_edges', 't_bin_edges',
    'costhetadir_bin_edges', 'deltaphidir_bin_edges', 'ckv_table',
]


def load_ckv_table(fpath, mmap):
    """Load a Cherenkov table from disk.

    Parameters
    ----------
    fpath : string
        Path to directory containing the table's .npy files.

    mmap : bool
        Whether to memory map the table (if it's stored in a directory
        containing .npy files).

    Returns
    -------
    table : OrderedDict
        Items are
        - 'n_photons' :
        - 'group_refractive_index' :
        - 'phase_refractive_index' :
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'ckv_table' : np.ndarray
        - 't_indep_ckv_table' : np.ndarray (if available)

    """
    fpath = expand(fpath)
    table = OrderedDict()

    if DEBUG:
        wstderr('Loading ckv table from {} ...\n'.format(fpath))

    if isfile(fpath):
        assert basename(fpath) == 'ckv_table.npy'
        fpath = dirname(fpath)

    t0 = time()
    indir = fpath

    if mmap:
        mmap_mode = 'r'
    else:
        mmap_mode = None

    for key in CKV_TABLE_KEYS + ['t_indep_ckv_table']:
        fpath = join(indir, key + '.npy')
        if DEBUG:
            wstderr('    loading {} from "{}" ...'.format(key, fpath))

        if key in ['table', 'ckv_table']:
            this_mmap_mode = mmap_mode
        else:
            this_mmap_mode = None

        t1 = time()
        if isfile(fpath):
            table[key] = np.load(fpath, mmap_mode=this_mmap_mode)
        elif key != 't_indep_ckv_table':
            raise ValueError(
                'Could not find file "{}" for loading table key "{}"'
                .format(fpath, key)
            )

        if DEBUG:
            wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))

    if DEBUG:
        wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))

    return table
