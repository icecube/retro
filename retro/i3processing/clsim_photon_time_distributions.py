#!/usr/bin/env python

"""
Generate time histograms for each DOM from a photon raw data pickle file (as
extracted from a CLSim forward event simulation).
"""

from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import OrderedDict
import cPickle as pickle
from os.path import dirname, expanduser, expandvars
import sys

import numpy as np


RETRO_DIR = dirname(dirname(__file__))
if not RETRO_DIR:
    RETRO_DIR = '../..'


def generate_histos(
        raw_data, gcd, t_max, num_bins, include_rde=True, include_noise=True,
        outfile=None
    ):
    """
    Parameters
    ----------
    raw_data : string or mapping

    gcd : str
        Path to GCD i3 or pkl file to get DOM coordinates, rde, and noise
        (where the latter two only have an effect if `include_rde` and/or
        `include_noise` are True).

    t_max : float
        Last edge in time binning (first edge is at 0), in units of ns.

    num_bins : int
        Number of time bins, which span from 0 to t_max.

    include_rde : bool, optional
        RDE is included by default.

    include_noise : bool, optiional
        Noise is included by default.

    outfile : str, optiional
        If a string is specified, save the histos to a pickle file by the name
        `outfile`. If not specified (or `None`), `histos` will not be written
        to a file.

    Returns
    -------
    histos : OrderedDict

    """
    if isinstance(raw_data, basestring):
        raw_data = pickle.load(open(raw_data, 'rb'))
    dom_info = raw_data['doms']

    bins = np.linspace(0, t_max, num_bins + 1)
    bin_widths = np.diff(bins)

    if isinstance(gcd, basestring):
        gcd = expanduser(expandvars(gcd))
        if gcd.endswith('.pkl'):
            gcd_info = pickle.load(open(gcd, 'rb'))
        elif '.i3' in gcd:
            if RETRO_DIR not in sys.path:
                sys.path.append(RETRO_DIR)
            from retro.i3info.extract_gcd import extract_gcd
            gcd_info = extract_gcd(gcd)
        else:
            raise ValueError('No idea how to handle GCD file "{}"'.format(gcd))

    rde = gcd_info['rde']
    noise_rate_hz = gcd_info['noise']
    mask = (rde == 0) | np.isnan(rde) | np.isinf(rde)
    operational_doms = ~mask
    rde = np.ma.masked_where(mask, rde)
    quantum_effieincy = 0.25 * rde

    histos = OrderedDict()
    keep_gcd_keys = ['source_gcd_name', 'source_gcd_md5', 'source_gcd_i3_md5']
    histos['gcd_info'] = [gcd_info[k] for k in keep_gcd_keys]
    histos['include_rde'] = include_rde
    histos['include_noise'] = include_noise
    histos['binning'] = OrderedDict([
        ('t_min', 0),
        ('t_max', t_max),
        ('num_bins', num_bins),
        ('spacing', 'linear')
    ])
    histos['results'] = results = OrderedDict()
    for (string, dom), data in dom_info.items():
        string_idx, dom_idx = string - 1, dom - 1
        if not operational_doms[string_idx, dom_idx]:
            continue

        hist, _ = np.histogram(
            data['time'],
            bins=bins,
            weights=data['weight'],
            normed=False
        )
        hist *= quantum_effieincy[string_idx, dom_idx]
        if include_noise:
            hist += noise_rate_hz[string_idx, dom_idx] * bin_widths / 1e9
        results[(string, dom)] = hist

    if outfile is not None:
        outfile = expanduser(expandvars(outfile))
        pickle.dump(
            histos,
            open(outfile, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL
        )

    return histos


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--raw-data', required=True,
        help='''Raw data pickle file'''
    )
    parser.add_argument(
        '--gcd', required=True,
        help='''GCD pickle file used to obtaining relative DOM efficiencies
        (RDE) and noise (if --include-noise flag is specified).'''
    )
    parser.add_argument(
        '--outfile', required=True,
        help='''Filepath to which histogram data is stored.'''
    )
    parser.add_argument(
        '--num-bins', type=int, required=True,
        help='''Number of bins to use for time histograms.'''
    )
    parser.add_argument(
        '--t-max', type=float, required=True,
        help='''Bin up to this maximum time'''
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
    return parser.parse_args()


if __name__ == '__main__':
    _ = generate_histos(**vars(parse_args()))
