# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for using a set of "raw" 5D (r, costheta, t, costhetadir, deltaphidir)
CLSim-produced Retro tables
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'MY_CLSIM_TABLE_KEYS',
    'CLSIM_TABLE_FNAME_PROTO',
    'CLSIM_TABLE_FNAME_RE',
    'CLSIM_TABLE_METANAME_PROTO',
    'CLSIM_TABLE_METANAME_RE',
    'interpret_clsim_table_fname',
    'load_clsim_table_minimal',
    'load_clsim_table'
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
from os.path import abspath, basename, dirname, isdir, isfile, join
import re
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DEBUG
from retro.tables.retro_5d_tables import TABLE_NORM_KEYS, get_table_norm
from retro.utils.misc import (
    expand, force_little_endian, get_decompressd_fobj, hrlist2list, wstderr
)


MY_CLSIM_TABLE_KEYS = [
    'table_shape',
    'n_photons',
    'group_refractive_index',
    'phase_refractive_index',
    'r_bin_edges',
    'costheta_bin_edges',
    't_bin_edges',
    'costhetadir_bin_edges',
    'deltaphidir_bin_edges',
    'table',
]

CLSIM_TABLE_FNAME_PROTO = [
    (
        'retro_nevts1000_{string}_DOM{depth_idx:d}.fits.*'
    ),
    (
        'clsim_table'
        '_set_{hash_val:s}'
        '_string_{string}'
        '_depth_{depth_idx:d}'
        '_seed_{seed}'
        '.fits'
    ),
    (
        'clsim_table'
        '_set_{hash_val:s}'
        '_string_{string}'
        '_dom_{dom:d}'
        '_seed_{seed:d}'
        '_n_{n_events:d}'
        '.fits'
    )
]
"""String templates for CLSim ("raw") retro tables. Note that `string` can
either be a specific string number OR either "ic" or "dc" indicating a generic
DOM of one of these two types located at the center of the detector, where z
location is averaged over all DOMs. `seed` can either be an integer or a
human-readable range (e.g. "0-9" for a table that combines toegether seeds, 0,
1, ..., 9)"""

CLSIM_TABLE_FNAME_RE = [
    re.compile(
        r'''
        retro
        _nevts(?P<n_events>[0-9]+)
        _(?P<string>[0-9a-z]+)
        _DOM(?P<depth_idx>[0-9]+)
        \.fits
        ''', re.IGNORECASE | re.VERBOSE
    ),
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _string_(?P<string>[0-9a-z]+)
        _depth_(?P<depth_idx>[0-9]+)
        _seed_(?P<seed>[0-9]+)
        \.fits
        ''', re.IGNORECASE | re.VERBOSE
    ),
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _string_(?P<string>[0-9a-z]+)
        _dom_(?P<dom>[0-9]+)
        _seed_(?P<seed>[0-9]+)
        _n_(?P<n_events>[0-9]+)
        \.fits
        ''', re.IGNORECASE | re.VERBOSE
    )
]

CLSIM_TABLE_METANAME_PROTO = [
    'clsim_table_set_{hash_val:s}_meta.json',
    (
        'clsim_table'
        '_set_{hash_val:s}'
        '_string_{string}'
        '_dom_{dom:d}'
        '_seed_{seed:d}'
        '_n_{n_events:d}'
        '_meta.json'
    )
]

CLSIM_TABLE_METANAME_RE = [
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _meta
        \.json
        ''', re.IGNORECASE | re.VERBOSE
    ),
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _string_(?P<string>[0-9a-z]+)
        _dom_(?P<dom>[0-9]+)
        _seed_(?P<seed>[0-9]+)
        _n_(?P<n_events>[0-9]+)
        _meta\.json
        ''', re.IGNORECASE | re.VERBOSE
    )
]


def interpret_clsim_table_fname(fname):
    """Get fields from fname and interpret these (e.g. by covnerting into
    appropriate Python types).

    The fields are parsed into the following types / values:
        - fname_version : int
        - hash_val : None or str
        - string : str (one of {'ic', 'dc'}) or int
        - depth_idx : int
        - seed : None, str (exactly '*'), int, or list of ints
        - n_events : None or int

    Parameters
    ----------
    fname : string

    Returns
    -------
    info : dict

    Raises
    ------
    ValueError
        If ``basename(fname)`` does not match the regexes
        ``CLSIM_TABLE_FNAME_RE``

    """
    fname = basename(fname)
    fname_version = None
    for fname_version in range(len(CLSIM_TABLE_FNAME_RE) - 1, -1, -1):
        match = CLSIM_TABLE_FNAME_RE[fname_version].match(fname)
        if match:
            break
    if not match:
        raise ValueError(
            'File basename "{}" does not match regex {} or any legacy regexes'
            .format(fname, CLSIM_TABLE_FNAME_RE[-1].pattern)
        )
    info = match.groupdict()
    info['fname_version'] = fname_version

    try:
        info['string'] = int(info['string'])
    except ValueError:
        assert isinstance(info['string'], basestring)
        assert info['string'].lower() in ['ic', 'dc']
        info['string'] = info['string'].lower()

    if fname_version == 1:
        info['seed'] = None
        info['hash_val'] = None
    elif fname_version == 2:
        info['n_events'] = None
        try:
            info['seed'] = int(info['seed'])
        except ValueError:
            if info['seed'] != '*':
                info['seed'] = hrlist2list(info['seed'])

    info['depth_idx'] = int(info['depth_idx'])

    ordered_keys = ['fname_version', 'hash_val', 'string', 'depth_idx', 'seed',
                    'n_events']
    ordered_info = OrderedDict()
    for key in ordered_keys:
        ordered_info[key] = info[key]

    return ordered_info


def load_clsim_table_minimal(fpath, step_length=None, mmap=False):
    """Load a CLSim table from disk (optionally compressed with zstd).

    Similar to the `load_clsim_table` function but the full table, including
    under/overflow bins, is kept and no normalization or further processing is
    performed on the table data besides populating the ouptput OrderedDict.

    Parameters
    ----------
    fpath : string
        Path to file to be loaded. If the file has extension 'zst', 'zstd', or
        'zstandard', the file will be decompressed using the `python-zstandard`
        Python library before passing to `pyfits` for interpreting.

    mmap : bool, optional
        Whether to memory map the table (if it's stored in a directory
        containing .npy files).

    Returns
    -------
    table : OrderedDict
        Items include
        - 'table_shape' : tuple of int
        - 'table' : np.ndarray
        - 't_indep_table' : np.ndarray (if available)
        - 'n_photons' :
        - 'phase_refractive_index' :
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :

    """
    table = OrderedDict()

    fpath = expand(fpath)

    if DEBUG:
        wstderr('Loading table from {} ...\n'.format(fpath))

    if isdir(fpath):
        t0 = time()
        indir = fpath
        if mmap:
            mmap_mode = 'r'
        else:
            mmap_mode = None
        for key in MY_CLSIM_TABLE_KEYS + ['t_indep_table', 't_is_residual_time']:
            fpath = join(indir, key + '.npy')
            if DEBUG:
                wstderr('    loading {} from "{}" ...'.format(key, fpath))
            t1 = time()
            if isfile(fpath):
                table[key] = np.load(fpath, mmap_mode=mmap_mode)
            elif key not in ['t_indep_table', 't_is_residual_time']:
                raise ValueError(
                    'Could not find file "{}" for loading table key "{}"'
                    .format(fpath, key)
                )
            if DEBUG:
                wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))
        if step_length is not None and 'step_length' in table:
            assert step_length == table['step_length']
        if DEBUG:
            wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))
        return table

    if not isfile(fpath):
        raise ValueError('Table does not exist at path "{}"'.format(fpath))

    import pyfits
    t0 = time()
    fobj = get_decompressd_fobj(fpath)
    pf_table = None
    try:
        pf_table = pyfits.open(fobj, mode='readonly', memmap=mmap)

        table['table_shape'] = pf_table[0].data.shape # pylint: disable=no-member
        table['n_photons'] = force_little_endian(
            pf_table[0].header['_i3_n_photons'] # pylint: disable=no-member
        )
        table['group_refractive_index'] = force_little_endian(
            pf_table[0].header['_i3_n_group'] # pylint: disable=no-member
        )
        table['phase_refractive_index'] = force_little_endian(
            pf_table[0].header['_i3_n_phase'] # pylint: disable=no-member
        )
        if step_length is not None:
            table['step_length'] = step_length

        n_dims = len(table['table_shape'])
        if n_dims == 5:
            # Space-time dimensions
            table['r_bin_edges'] = force_little_endian(
                pf_table[1].data # meters # pylint: disable=no-member
            )
            table['costheta_bin_edges'] = force_little_endian(
                pf_table[2].data # pylint: disable=no-member
            )
            table['t_bin_edges'] = force_little_endian(
                pf_table[3].data # nanoseconds # pylint: disable=no-member
            )

            # Photon directionality
            table['costhetadir_bin_edges'] = force_little_endian(
                pf_table[4].data # pylint: disable=no-member
            )
            table['deltaphidir_bin_edges'] = force_little_endian(
                pf_table[5].data # pylint: disable=no-member
            )

        else:
            raise NotImplementedError(
                '{}-dimensional table not handled'.format(n_dims)
            )

        table['table'] = force_little_endian(pf_table[0].data) # pylint: disable=no-member

        wstderr('    (load took {} s)\n'.format(np.round(time() - t0, 3)))

    except:
        wstderr('ERROR: Failed to load "{}"\n'.format(fpath))
        raise

    finally:
        del pf_table
        if hasattr(fobj, 'close'):
            fobj.close()
        del fobj

    return table


def load_clsim_table(fpath, step_length, angular_acceptance_fract,
                     quantum_efficiency):
    """Load a CLSim table from disk (optionally compressed with zstd).

    Parameters
    ----------
    fpath : string
        Path to file to be loaded. If the file has extension 'zst', 'zstd', or
        'zstandard', the file will be decompressed using the `python-zstandard`
        Python library before passing to `pyfits` for interpreting.

    Returns
    -------
    table : OrderedDict
        Items include
        - 'table_shape' : tuple of int
        - 'table' : np.ndarray
        - 't_indep_table' : np.ndarray
        - 'n_photons' :
        - 'group_refractive_index' :
        - 'phase_refractive_index' :

        If the table is 5D, items also include
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'table_norm'

    """
    table = OrderedDict()

    table = load_clsim_table_minimal(fpath=fpath, step_length=step_length)
    table['table_norm'] = get_table_norm(
        angular_acceptance_fract=angular_acceptance_fract,
        quantum_efficiency=quantum_efficiency,
        step_length=step_length,
        **{k: table[k] for k in TABLE_NORM_KEYS if k != 'step_length'}
    )
    table['t_indep_table_norm'] = quantum_efficiency * angular_acceptance_fract

    wstderr('Interpreting table...\n')
    t0 = time()
    n_dims = len(table['table_shape'])

    # Cut off first and last bin in each dimension (underflow and
    # overflow bins)
    slice_wo_overflow = (slice(1, -1),) * n_dims
    wstderr('    slicing to remove underflow/overflow bins...')
    t0 = time()
    table_wo_overflow = table['table'][slice_wo_overflow]
    wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    wstderr('    slicing and summarizing underflow and overflow...')
    t0 = time()
    underflow, overflow = [], []
    for n in range(n_dims):
        sl = tuple([slice(1, -1)]*n + [0] + [slice(1, -1)]*(n_dims - 1 - n))
        underflow.append(table['table'][sl].sum())

        sl = tuple([slice(1, -1)]*n + [-1] + [slice(1, -1)]*(n_dims - 1 - n))
        overflow.append(table['table'][sl].sum())
    wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    table['table'] = table_wo_overflow
    table['underflow'] = np.array(underflow)
    table['overflow'] = np.array(overflow)

    return table
