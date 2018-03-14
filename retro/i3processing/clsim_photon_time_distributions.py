#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name, wrong-import-position

"""
Generate time histograms for each DOM from a photon raw data pickle file (as
extracted from a CLSim forward event simulation).
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    generate_histos
    parse_args
'''.split()

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
from collections import OrderedDict
import cPickle as pickle
from os.path import abspath, dirname, expanduser, expandvars, isfile
import re
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3info.extract_gcd import extract_gcd


def generate_histos(
        photons, hole_ice_model, t_max, num_bins, gcd=None, include_rde=True,
        include_noise=True, outfile=None
    ):
    """Generate time histograms from photons extracted from CLSim (repated)
    forward event simulations.

    Parameters
    ----------
    photons : string or mapping

    hole_ice_model : string
        Raw CLSim does not (currently) incorproate hole ice model; this is a
        modification to the angular acceptance of the phtons that CLSim
        returns, so must be specified (and applied) post-hoc (e.g., in this
        function).

    t_max : float
        Last edge in time binning (first edge is at 0), in units of ns.

    num_bins : int
        Number of time bins, which span from 0 to t_max.

    gcd : str or None, optional
        Path to GCD i3 or pkl file to get DOM coordinates, rde, and noise
        (where the latter two only have an effect if `include_rde` and/or
        `include_noise` are True). Regardless if this is specified, the code
        will attempt to automatically figure out the GCD file used to produce
        the table. If this succeeds and `gcd` is specified by the user, the
        user's value is checked against that found in the data. If the user
        does not specify `gcd`, the value found in the data is used. If neither
        `gcd` is provided nor one can be found in the data, an error is raised.

    include_rde : bool, optional
        Whether to use relative DOM efficiencies (RDE) to scale the results per
        DOM. RDE is included by default.

    include_noise : bool, optiional
        Whether to add the noise floor for each DOM to the results. Noise is
        included by default.

    outfile : str or None, optiional
        If a string is specified, save the histos to a pickle file by the name
        `outfile`. If not specified (or `None`), `histos` will not be written
        to a file.


    Returns
    -------
    histos : OrderedDict


    Raises
    ------
    ValueError
        If `gcd` is specified but does not match a GCD file found in the data

    ValueError
        If `gcd` is not specified and no GCD can be found in the data


    See also
    --------
    i3processing.sim
        Perform the repeated simulation to get photons at DOMs. Generates an i3
        file.

    i3processing.extract_photon_info
        Extract photon info (and pertinent metadata) from the i3 file produced
        from the above.

    retro_dom_pdfs
        Produce distributions corresponding to the histograms made here, but
        using Retro reco.

    """
    photons_file_name = None
    if isinstance(photons, basestring):
        photons_file_name = photons
        photons = pickle.load(open(photons_file_name, 'rb'))
    dom_info = photons['doms']

    bin_edges = np.linspace(0, t_max, num_bins + 1)
    bin_widths = np.diff(bin_edges)

    gcd_info = None
    if isinstance(gcd, basestring):
        exp_gcd = expanduser(expandvars(gcd))
        if exp_gcd.endswith('.pkl'):
            gcd_info = pickle.load(open(exp_gcd, 'rb'))
        elif '.i3' in exp_gcd:
            gcd_info = extract_gcd(exp_gcd)
        else:
            raise ValueError('No idea how to handle GCD file "{}"'.format(gcd))

    if photons['gcd']:
        try:
            gcd_from_data = expanduser(expandvars(photons['gcd']))
            if gcd_from_data.endswith('.pkl'):
                gcd_info_from_data = pickle.load(open(gcd_from_data, 'rb'))
            else:
                gcd_info_from_data = extract_gcd(gcd_from_data)
        except (AttributeError, KeyError, ValueError):
            raise
            #assert gcd_info is not None
        else:
            if gcd_info is None:
                gcd_info = gcd_info_from_data
            else:
                if gcd_info != gcd_info_from_data:
                    print('WARNING: Using different GCD from the one used'
                          ' during simulation!')

    if gcd_info is None:
        if photons_file_name is not None:
            photons_err = ' filename "{}"'.format(photons_file_name)
        raise ValueError(
            'No GCD info could be found from arg `gcd`={} or in `photons`'
            '{}'.format(gcd, photons_err)
        )

    rde = gcd_info['rde']
    noise_rate_hz = gcd_info['noise']
    mask = (rde == 0) | np.isnan(rde) | np.isinf(rde)
    operational_doms = ~mask
    rde = np.ma.masked_where(mask, rde)
    quantum_effieincy = rde

    histos = OrderedDict()
    keep_gcd_keys = ['source_gcd_name', 'source_gcd_md5', 'source_gcd_i3_md5']
    histos['gcd_info'] = OrderedDict([(k, gcd_info[k]) for k in keep_gcd_keys])
    histos['include_rde'] = include_rde
    histos['include_noise'] = include_noise
    histos['bin_edges'] = bin_edges
    histos['binning_spec'] = OrderedDict([
        ('domain', (0, t_max)),
        ('num_bins', num_bins),
        ('spacing', 'linear'),
        ('units', 'ns')
    ])

    # Note the first number in the file is a number approximately equal (but
    # greater than) the peak in the distribution, so is useless for us.
    possible_paths = [
        hole_ice_model,
        '$I3_SRC/ice-models/resources/models/angsens/' + hole_ice_model
    ]
    coeffs_loaded = False
    for path in possible_paths:
        path = expanduser(expandvars(path))
        if not isfile(path):
            continue
        try:
            poly_coeffs = np.loadtxt(path)[1:]
        except:
            pass
        else:
            coeffs_loaded = True
            break

    if not coeffs_loaded:
        raise ValueError('Could not load hole ice model at any of\n{}'
                         .format(possible_paths))

    # We want coszen = -1 to correspond to upgoing particles, but angular
    # sensitivity is given w.r.t. the DOM axis (which points "down" towards earth,
    # and therefore is rotated 180-deg). So rotate the coszen polynomial about cz=0
    # by negating the odd coefficients (coeffs are in ascending powers of "x".
    flipped_coeffs = np.empty_like(poly_coeffs)
    flipped_coeffs[0::2] = poly_coeffs[0::2]
    flipped_coeffs[1::2] = -poly_coeffs[1::2]
    angsens_poly = np.polynomial.Polynomial(flipped_coeffs, domain=(-1, 1))

    # Attach the weights to the data
    num_sims = photons['num_sims']
    for data_dict in photons['doms'].values():
        cz = data_dict['coszen']
        try:
            # Note that angular sensitivity will modify the total number of
            # photons detected, and the poly is normalized as such already, so no
            # normalization should be applied here.
            angsens_wt = angsens_poly(cz)
        except:
            print(np.min(cz), np.max(cz))
            raise

        data_dict['weight'] = angsens_wt / num_sims

        for k, array in data_dict.items():
            data_dict[k] = array.astype(np.float32)

    histos['results'] = results = OrderedDict()
    for (string, dom), data in dom_info.items():
        string_idx, dom_idx = string - 1, dom - 1
        if not operational_doms[string_idx, dom_idx]:
            continue

        hist, _ = np.histogram(
            data['time'],
            bins=bin_edges,
            weights=data['weight'],
            normed=False
        )
        if include_rde:
            hist *= quantum_effieincy[string_idx, dom_idx]
        if include_noise:
            hist += noise_rate_hz[string_idx, dom_idx] * bin_widths / 1e9
        results[(string, dom)] = hist

    if outfile is not None:
        outfile = expanduser(expandvars(outfile))
        print('Writing histos to\n"{}"'.format(outfile))
        pickle.dump(
            histos,
            open(outfile, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return histos, dom_info


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--photons', required=True,
        help='''Raw data pickle file containing photons'''
    )
    parser.add_argument(
        '--hole-ice-model', required=True,
        help='''Filepath to hole ice model to apply to the photons.'''
    )
    parser.add_argument(
        '--t-max', type=float, required=True,
        help='''Bin up to this maximum time'''
    )
    parser.add_argument(
        '--num-bins', type=int, required=True,
        help='''Number of bins to use for time histograms.'''
    )
    parser.add_argument(
        '--gcd', default=None,
        help='''GCD file used to obtaining relative DOM efficiencies
        (RDE) and noise (if --include-noise flag is specified). This is only
        necessary if one of those flags is set and if the GCD file cannot be
        determined from the input file.'''
    )
    parser.add_argument(
        '--include-rde', action='store_true',
        help='''Include relative DOM efficiency corrections (per DOM) to
        histograms (as obtained from GCD file).'''
    )
    parser.add_argument(
        '--include-noise', action='store_true',
        help='''Include noise offsets in histograms (as obtained from GCD
        file).'''
    )
    parser.add_argument(
        '--outfile', default=None,
        help='''Filepath for storing histograms. If not specified, a default
        name is derived from the --raw-data filename.'''
    )

    args = parser.parse_args()

    # Construct the output filename if none is provided
    if args.outfile is None:
        args.outfile = re.sub(r'_photons.pkl', '_photon_histos.pkl', args.photons)

    return args


if __name__ == '__main__':
    histos, dom_info = generate_histos(**vars(parse_args())) # pylint: disable=invalid-name
