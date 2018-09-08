#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals

"""
Combine single-DOM time-independent Cartesian table tiles to create a TDI table.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'extract_meta_from_keys',
    'combine_tdi_tiles',
    'parse_args',
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
from glob import glob
import json
from os.path import abspath, dirname, isdir, isfile, join
import pickle
import sys

import numpy as np
import pyfits

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3info.angsens_model import load_angsens_model
from retro.i3info.extract_gcd import extract_gcd
from retro.utils.misc import expand, load_pickle, mkdir, wstderr, wstdout


def extract_meta_from_keys(keys, prefix):
    """Extract metadata contained in a key's name.

    Parameters
    ----------
    keys : list of strings
    prefix : string

    Returns
    -------
    meta : string

    """
    return next(k[len(prefix):] for k in keys if k.startswith(prefix))


def combine_tdi_tiles(
    source_dir,
    dest_dir,
    table_hash,
    gcd,
    bin_edges_file,
    tile_spec_file,
):
    """Combine individual time-independent tiles (one produced per DOM) into a single
    TDI table.

    Parameters
    ----------
    source_dir : str
    dest_dir : str
    bin_edges_file : str
    tile_spec_file : str

    """
    source_dir = expand(source_dir)
    dest_dir = expand(dest_dir)
    gcd = expand(gcd)
    bin_edges_file = expand(bin_edges_file)
    tile_spec_file = expand(tile_spec_file)
    mkdir(dest_dir)
    assert isdir(source_dir)
    assert isfile(bin_edges_file)
    assert isfile(tile_spec_file)

    gcd = extract_gcd(gcd)

    bin_edges = load_pickle(bin_edges_file)
    x_edges = bin_edges['x']
    y_edges = bin_edges['y']
    z_edges = bin_edges['z']
    ctdir_edges = bin_edges['costhetadir']
    phidir_edges = bin_edges['phidir']

    n_x = len(x_edges) - 1
    n_y = len(y_edges) - 1
    n_z = len(z_edges) - 1
    n_ctdir = len(ctdir_edges) - 1
    n_phidir = len(phidir_edges) - 1

    n_dir_bins = n_ctdir * n_phidir

    x_bw = (x_edges.max() - x_edges.min()) / n_x
    y_bw = (y_edges.max() - y_edges.min()) / n_y
    z_bw = (z_edges.max() - z_edges.min()) / n_z
    bin_vol = x_bw * y_bw * z_bw

    ctdir_min = ctdir_edges.min()
    ctdir_max = ctdir_edges.max()

    phidir_min = phidir_edges.min()
    phidir_max = phidir_edges.max()

    with file(tile_spec_file, 'r') as f:
        tile_specs = [l.strip() for l in f.readlines()]

    table = np.zeros(shape=(n_x, n_y, n_z, n_ctdir, n_phidir), dtype=np.float32)

    # Slice all table dimensions to exclude {under,over}flow bins
    central_slice = (slice(1, -1),)*5

    angsens_model = None
    ice_model = None
    disable_tilt = None
    disable_anisotropy = None
    n_phase = None
    n_group = None

    tiles_info = []

    for tile_spec in tile_specs:
        info = None
        try:
            fields = tile_spec.split()

            info = OrderedDict()

            info['tbl_idx'] = int(fields[0])
            info['string'] = int(fields[1])
            info['dom'] = int(fields[2])
            info['seed'] = int(fields[3])
            info['n_events'] = int(fields[4])

            info['x_min'] = float(fields[5])
            info['x_max'] = float(fields[6])
            info['n_x'] = int(fields[7])

            info['y_min'] = float(fields[8])
            info['y_max'] = float(fields[9])
            info['n_y'] = int(fields[10])

            info['z_min'] = float(fields[11])
            info['z_max'] = float(fields[12])
            info['n_z'] = int(fields[13])

            info['n_ctdir'] = int(fields[14])
            info['n_phidir'] = int(fields[15])

            tiles_info.append(info)

            tile_fpath = glob(join(
                source_dir,
                'clsim_table_set'
                '_{table_hash}'
                '_tile_{tbl_idx}'
                '_string_{string}'
                '_dom_{dom}'
                '_seed_{seed}'
                '_n_{n_events}'
                '.fits'.format(table_hash=table_hash, **info)
            ))[0]
            try:
                fits_table = pyfits.open(tile_fpath, mode='readonly', memmap=True)
            except:
                wstderr('Failed on tile_fpath "{}"'.format(tile_fpath))
                raise

            primary = fits_table[0]

            header = primary.header # pylint: disable=no-member
            keys = header.keys()

            this_gcd_i3_md5 = extract_meta_from_keys(keys, '_i3_gcd_i3_md5_')
            assert this_gcd_i3_md5 == gcd['source_gcd_i3_md5'], \
                    'this: {}, ref: {}'.format(this_gcd_i3_md5, gcd['source_gcd_i3_md5'])

            this_angsens_model = extract_meta_from_keys(keys, '_i3_angsens_')
            if angsens_model is None:
                angsens_model = this_angsens_model
                _, avg_angsens = load_angsens_model(angsens_model)
            else:
                assert this_angsens_model == angsens_model

            this_table_hash = extract_meta_from_keys(keys, '_i3_hash_')
            assert this_table_hash == table_hash

            this_ice_model = extract_meta_from_keys(keys, '_i3_ice_')
            if ice_model is None:
                ice_model = this_ice_model
            else:
                assert this_ice_model == ice_model

            this_disable_anisotropy = header['_i3_disable_anisotropy']
            if disable_anisotropy is None:
                disable_anisotropy = this_disable_anisotropy
            else:
                assert this_disable_anisotropy == disable_anisotropy

            this_disable_tilt = header['_i3_disable_tilt']
            if disable_tilt is None:
                disable_tilt = this_disable_tilt
            else:
                assert this_disable_tilt == disable_tilt

            this_n_phase = header['_i3_n_phase']
            if n_phase is None:
                n_phase = this_n_phase
            else:
                assert this_n_phase == n_phase

            this_n_group = header['_i3_n_group']
            if n_group is None:
                n_group = this_n_group
            else:
                assert this_n_group == n_group

            assert info['n_ctdir'] == n_ctdir
            assert info['n_phidir'] == n_phidir

            assert np.isclose(header['_i3_costhetadir_min'], ctdir_min)
            assert np.isclose(header['_i3_costhetadir_max'], ctdir_max)

            assert np.isclose(header['_i3_phidir_min'], phidir_min)
            assert np.isclose(header['_i3_phidir_max'], phidir_max)

            n_photons = header['_i3_n_photons']
            n_dir_bins = info['n_ctdir'] * info['n_phidir']

            this_x_bw = (info['x_max'] - info['x_min']) / info['n_x']
            this_y_bw = (info['y_max'] - info['y_min']) / info['n_y']
            this_z_bw = (info['z_max'] - info['z_min']) / info['n_z']

            assert this_x_bw == x_bw
            assert this_y_bw == y_bw
            assert this_z_bw == z_bw

            assert np.any(np.isclose(info['x_min'], x_edges))
            assert np.any(np.isclose(info['x_max'], x_edges))

            assert np.any(np.isclose(info['y_min'], y_edges))
            assert np.any(np.isclose(info['y_max'], y_edges))

            assert np.any(np.isclose(info['z_min'], z_edges))
            assert np.any(np.isclose(info['z_max'], z_edges))

            quantum_efficiency = 0.25 * gcd['rde'][info['string'] - 1, info['dom'] - 1]
            norm = n_dir_bins * quantum_efficiency * avg_angsens / (n_photons * bin_vol)
            if np.isnan(norm):
                print('\nTile {} norm is nan!'.format(info['tbl_idx']))
                print(
                    '    quantum_efficiency = {}, n_photons = {}'
                    .format(quantum_efficiency, n_photons)
                )
            elif norm == 0:
                print('\nTile {} norm is 0'.format(info['tbl_idx']))

            x_start = np.digitize(info['x_min'] + x_bw / 2, x_edges) - 1
            x_stop = np.digitize(info['x_max'] - x_bw / 2, x_edges)

            y_start = np.digitize(info['y_min'] + y_bw / 2, y_edges) - 1
            y_stop = np.digitize(info['y_max'] - y_bw / 2, y_edges)

            z_start = np.digitize(info['z_min'] + z_bw / 2, z_edges) - 1
            z_stop = np.digitize(info['z_max'] - z_bw / 2, z_edges)

            # NOTE: comparison excludes norm = 0 _and_ norm = NaN
            if norm > 0:
                assert not np.isnan(norm)
                table[x_start:x_stop, y_start:y_stop, z_start:z_stop, :, :] += (
                    norm * primary.data[central_slice] # pylint: disable=no-member
                )
        except:
            wstderr('Failed on tile_spec {}'.format(tile_spec))
            if info is not None:
                wstderr('Info:\n{}'.format(info))
            raise
        wstderr('.')

    wstderr('\n')

    metadata = OrderedDict()
    metadata['table_hash'] = table_hash
    metadata['disable_tilt'] = disable_tilt
    metadata['disable_anisotropy'] = disable_anisotropy
    metadata['gcd'] = gcd
    metadata['angsens_model'] = angsens_model
    metadata['ice_model'] = ice_model
    metadata['n_phase'] = n_phase
    metadata['n_group'] = n_group
    metadata['tiles_info'] = tiles_info

    outdir = join(
        dest_dir,
        'tdi_table_{}_tilt_{}_anisotropy_{}'.format(
            table_hash,
            'off' if disable_tilt else 'on',
            'off' if disable_anisotropy else 'on',
        )
    )
    mkdir(outdir)

    name = 'tdi_table.npy'
    outfpath = join(outdir, name)
    wstdout('saving table to "{}"\n'.format(outfpath))
    np.save(outfpath, table)

    #outfpath = join(outdir, 'tdi_bin_edges.json')
    #wstdout('saving bin edges to "{}"\n'.format(outfpath))
    #json.dump(
    #    bin_edges,
    #    file(outfpath, 'w'),
    #    sort_keys=False,
    #    indent=2,
    #)
    outfpath = join(outdir, 'tdi_bin_edges.pkl')
    wstdout('saving bin edges to "{}"\n'.format(outfpath))
    pickle.dump(
        bin_edges,
        file(outfpath, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL,
    )

    #outfpath = join(outdir, 'tdi_metadata.json')
    #wstdout('saving metadata to "{}"\n'.format(outfpath))
    #json.dump(
    #    metadata,
    #    file(outfpath, 'w'),
    #    sort_keys=False,
    #    indent=2,
    #)
    outfpath = join(outdir, 'tdi_metadata.pkl')
    wstdout('saving metadata to "{}"\n'.format(outfpath))
    pickle.dump(
        metadata,
        file(outfpath, 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def parse_args(description=__doc__):
    """Parse command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--source-dir', required=True,
    )
    parser.add_argument(
        '--dest-dir', required=True,
    )
    parser.add_argument(
        '--table-hash', required=True,
    )
    parser.add_argument(
        '--gcd', required=True,
    )
    parser.add_argument(
        '--bin-edges-file', required=True,
    )
    parser.add_argument(
        '--tile-spec-file', required=True,
    )
    return parser.parse_args()


if __name__ == '__main__':
    combine_tdi_tiles(**vars(parse_args()))
