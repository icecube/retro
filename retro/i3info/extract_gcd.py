#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wildcard-import

"""
Extract positional and calibration info for DOMs
and save the resulting dict in a pkl file for later use
"""

from __future__ import absolute_import, division, print_function

__all__ = ['N_STRINGS', 'N_DOMS', 'extract_gcd', 'parse_args']

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

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import bz2
from collections import OrderedDict
import gzip
import hashlib
import os
from os.path import abspath, basename, expanduser, expandvars, dirname, isfile, join, splitext
import pickle
from shutil import copyfile
from StringIO import StringIO
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DATA_DIR
from retro.utils.misc import mkdir


N_STRINGS = 86
N_DOMS = 60


def extract_gcd(gcd_file, outdir=None):
    """Extract info from a GCD in i3 format, optionally saving to a simple
    Python pickle file.

    Parameters
    ----------
    gcd_file : str
    outdir : str, optional
        If provided, the gcd info is saved to a .pkl file with same name as
        `gcd_file` just with extension replaced.

    Returns
    -------
    gcd_info : OrderedDict
        'source_gcd_name': basename of the `gcd_file` provided
        'source_gcd_md5': direct md5sum of `gcd_file` (possibly compressed)
        'source_gcd_i3_md5': md5sum of `gcd_file` after decompressing to .i3
        'geo': (86, 60, 3) array of DOM x, y, z coords in m rel to IceCube coord system
        'rde' : (86, 60) array with relative DOM efficiencies
        'noise' : (86, 60) array with noise rate, in Hz, for each DOM

    """
    gcd_file = expanduser(expandvars(gcd_file))
    src_gcd_dir = dirname(gcd_file)
    src_gcd_basename = basename(gcd_file)
    src_gcd_stripped = src_gcd_basename.rstrip('.bz2').rstrip('.gz').rstrip('.i3').rstrip('.pkl')

    outfname = src_gcd_stripped + '.pkl'
    data_dir_fpath = abspath(join(DATA_DIR, outfname))

    outfpath = None
    if outdir is not None:
        outdir = expanduser(expandvars(outdir))
        mkdir(outdir)
        outfpath = join(outdir, outfname)

        if isfile(data_dir_fpath) and data_dir_fpath != abspath(outfpath):
            copyfile(data_dir_fpath, outfpath)

    if isfile(data_dir_fpath):
        return pickle.load(open(data_dir_fpath, 'rb'))

    if outfpath is not None and isfile(outfpath):
        return pickle.load(open(outfpath, 'rb'))

    if src_gcd_dir:
        dirs = [src_gcd_dir]
    else:
        dirs = ['.']
        if 'I3_DATA' in os.environ:
            dirs.append(expanduser(expandvars('$I3_DATA/GCD')))

    compression = []
    parsed = False
    src_gcd_stripped = src_gcd_basename
    for _ in range(10):
        root, ext = splitext(src_gcd_stripped)
        if ext == '.gz':
            compression.append('gz')
            src_gcd_stripped = root
        elif src_gcd_stripped.endswith('.bz2'):
            compression.append('bz2')
            src_gcd_stripped = root
        elif src_gcd_stripped.endswith('.i3'):
            parsed = True
            src_gcd_stripped = root
            break
        elif src_gcd_stripped.endswith('.pkl'):
            for src_dir in dirs:
                fpath = join(src_dir, src_gcd_stripped)
                if isfile(fpath):
                    gcd_info = pickle.load(open(src_gcd_stripped, 'rb'))
                    if outdir is not None and outdir != src_gcd_dir:
                        copyfile(src_gcd_stripped, outfpath)
                    return gcd_info

    if not parsed:
        raise ValueError(
            'Could not parse compression suffixes for GCD file "{}"'
            .format(gcd_file)
        )

    decompressed = open(gcd_file, 'rb').read()
    source_gcd_md5 = hashlib.md5(decompressed).hexdigest()
    for comp_alg in compression:
        if comp_alg == 'gz':
            decompressed = gzip.GzipFile(fileobj=StringIO(decompressed)).read()
        elif comp_alg == 'bz2':
            decompressed = bz2.decompress(decompressed)
    decompressed_gcd_md5 = hashlib.md5(decompressed).hexdigest()

    from I3Tray import I3Units, OMKey # pylint: disable=import-error
    from icecube import dataclasses, dataio # pylint: disable=import-error, unused-variable

    gcd = dataio.I3File(gcd_file) # pylint: disable=no-member
    frame = gcd.pop_frame()

    # get detector geometry
    key = 'I3Geometry'
    while key not in frame.keys():
        frame = gcd.pop_frame()
    omgeo = frame[key].omgeo

    # get calibration
    key = 'I3Calibration'
    while key not in frame.keys():
        frame = gcd.pop_frame()
    dom_cal = frame[key].dom_cal

    # create output dict
    gcd_info = OrderedDict()
    gcd_info['source_gcd_name'] = src_gcd_basename
    gcd_info['source_gcd_md5'] = source_gcd_md5
    gcd_info['source_gcd_i3_md5'] = decompressed_gcd_md5
    gcd_info['geo'] = geo = np.zeros((N_STRINGS, N_DOMS, 3))
    gcd_info['noise'] = noise = np.zeros((N_STRINGS, N_DOMS))
    gcd_info['rde'] = rde = np.zeros((N_STRINGS, N_DOMS))

    for string_idx in range(N_STRINGS):
        for dom_idx in range(N_DOMS):
            omkey = OMKey(string_idx + 1, dom_idx + 1)
            geo[string_idx, dom_idx, 0] = omgeo.get(omkey).position.x
            geo[string_idx, dom_idx, 1] = omgeo.get(omkey).position.y
            geo[string_idx, dom_idx, 2] = omgeo.get(omkey).position.z
            try:
                noise[string_idx, dom_idx] = (
                    dom_cal[omkey].dom_noise_rate / I3Units.hertz
                )
            except KeyError:
                noise[string_idx, dom_idx] = 0.0

            try:
                rde[string_idx, dom_idx] = dom_cal[omkey].relative_dom_eff
            except KeyError:
                gcd_info['rde'][string_idx, dom_idx] = 0.

    #print(np.mean(gcd_info['rde'][:80]))
    #print(np.mean(gcd_info['rde'][79:]))

    if outfpath is not None:
        with open(outfpath, 'wb') as outfile:
            pickle.dump(gcd_info, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    return gcd_info


def parse_args(description=__doc__):
    """Parse command line args"""
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', metavar='GCD_FILE', dest='gcd_file', type=str,
        required=True,
        help='Input GCD file. See e.g. files in $I3_DATA/GCD directory.'
    )
    parser.add_argument(
        '--outdir', type=str, required=True,
        help='Directory into which to save the resulting .pkl file',
    )
    return parser.parse_args()


if __name__ == '__main__':
    extract_gcd(**vars(parse_args()))
