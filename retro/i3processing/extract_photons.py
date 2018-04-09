#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Read in an i3 file from CLsim and generate a pickle file containing the run
parameters (gcd file, number of sims, and event MC truth) and photon info (a
dict keyed by (string, dom) tuples containing time, wavelength, coszen, and
weight for each DOM).
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    extract_photons
    parse_args
'''.split()

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this i3_file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from argparse import ArgumentParser
from collections import OrderedDict
from os.path import expanduser, expandvars
import sys
import pickle
import re

import numpy as np

from icecube import dataclasses, simclasses, dataio # pylint: disable=import-error, unused-import


def extract_photons(i3file, outfile=None, photons_key='photons'):
    """Extract photon info (and metadata about the run) from an I3 file, and
    store the info to a simple pickle file.

    Parameters
    ----------
    i3file : string
        Path to the I3 file

    outfile : string, optiional
        If not provided, the output file name is the `i3file` but with
        ".i3(.bz2)" extension(s) stripped and replaced with "_photons.pkl";
        `outfile` will also be placed in the same directory as the `i3file`.

    photons_key : string
        Field name in the I3 frame that contains phton info.

    Returns
    -------
    photon_info

    """
    # dict to store the extracted info
    photon_info = OrderedDict([
        ('gcd', None),
        ('mc_truth', None),
        ('num_sims', 0),
        ('doms', OrderedDict())
    ])

    orig_fname = i3file
    fname = expanduser(expandvars(i3file))

    if outfile is None:
        outfile = re.sub(r'\.i3.*', '', fname) + '_photons.pkl'

    print('Extracting photons from "{}"\nand writing info to "{}"'
          .format(orig_fname, outfile))

    i3_file = dataio.I3File(fname, 'r')

    num_sims = 0
    mc_truth = None
    while i3_file.more():
        if photon_info['gcd'] is None:
            frame = i3_file.pop_frame()
        else:
            frame = i3_file.pop_daq()

        if frame.Stop == frame.TrayInfo:
            for config in frame.values():
                try:
                    streams = config.module_configs['streams']
                    gcd = streams['Prefix']
                except (KeyError, ValueError, AttributeError) as e:
                    print(e)
                    continue
                gcd_l = gcd.lower()
                if not ('gcd' in gcd_l or 'geocalibdetector' in gcd_l):
                    continue
                photon_info['gcd'] = gcd

        if photons_key not in frame:
            continue

        num_sims += 1
        if num_sims % 100 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

        if mc_truth is None:
            mct = frame['I3MCTree']
            primary = mct.get_primaries()[0]
            mc_truth = OrderedDict()
            mc_truth['pdg'] = int(primary.pdg_encoding)
            mc_truth['time'] = float(primary.time)
            mc_truth['x'] = float(primary.pos.x)
            mc_truth['y'] = float(primary.pos.y)
            mc_truth['z'] = float(primary.pos.z)
            mc_truth['zenith'] = float(primary.dir.zenith)
            mc_truth['azimuth'] = float(primary.dir.azimuth)
            mc_truth['energy'] = float(primary.energy)

        photon_series = frame['photons']

        for dom_key, pinfo in photon_series:
            if dom_key not in photon_info['doms']:
                photon_info['doms'][dom_key] = OrderedDict([
                    ('time', []),
                    ('wavelength', []),
                    ('coszen', []),
                ])

            for photon in pinfo:
                photon_info['doms'][dom_key]['time'].append(photon.time)
                photon_info['doms'][dom_key]['wavelength'].append(photon.wavelength)
                photon_info['doms'][dom_key]['coszen'].append(np.cos(photon.dir.zenith))

    photon_info['mc_truth'] = mc_truth
    photon_info['num_sims'] = num_sims

    if num_sims >= 100:
        sys.stdout.write('\n')

    # Convert I3OMKey object to integers so we can ditch the albatross known as
    # IceCube software
    photon_info['doms'] = {
        (int(k.string), int(k.om)): v for k, v in photon_info['doms'].items()
    }

    # Sort dict by (string, dom) such that output file is guaranteed to be
    # consistent
    sorted_data = OrderedDict()
    for dom in sorted(photon_info['doms'].keys()):
        data_dict = photon_info['doms'][dom]
        for k in data_dict.keys():
            data_dict[k] = np.array(data_dict[k], dtype=np.float32)
        sorted_data[dom] = data_dict
    photon_info['doms'] = sorted_data

    pickle.dump(photon_info,
                open(outfile, 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    return photon_info


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--photons-key', default='photons',
        help='''Field in the I3 frame that identifies photons. Default is
        "photons"'''
    )
    parser.add_argument(
        '--outfile', default=None,
        help='''Path to pickle file in which results are stored. If not
        specified, the name will be the input file with the ".i3(.bz2)"
        extension(s) replaced by "_photons.pkl" (and placed in same
        directory as the input file).'''
    )
    parser.add_argument(
        'i3file',
        help='Path to the I3 file from which you wish to extract photon info'
    )
    return parser.parse_args()


if __name__ == '__main__':
    photon_info = extract_photons(**vars(parse_args())) # pylint: disable=invalid-name
