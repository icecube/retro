#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Extract MultiNest estimate from a processed event on disk.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['get_estimate', 'parse_args']

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
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.stats import estimate_from_llhp
from retro.utils.misc import expand


UNITS = {
    'x': 'm',
    'y': 'm',
    'z': 'm',
    'time': 'ns',
    'track_fraction': '',
    'track_energy': 'GeV',
    'cascade_energy': 'GeV',
    'energy': 'GeV',
    'track_zenith': 'rad',
    'track_azimuth': 'rad',
    'cascade_zenith': 'rad',
    'cascade_azimuth': 'rad',
}


def get_estimate(fpath, verbose=True):
    """Get estimate from llhp.npy file"""
    fpath = expand(fpath)
    llhp = np.load(fpath)
    estimate = estimate_from_llhp(llhp)
    if verbose:
        for dim, est  in estimate.items():
            mean, low, high = est['mean'], est['low'], est['high']
            print('{:s} : mean = {:9.3f} ; 95% interval = [{:9.3f}, {:9.3f}] {}'
                  .format(dim.rjust(20), mean, low, high, UNITS[dim]))
    return llhp, estimate


def parse_args(description=__doc__):
    """Parse command line options"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        'fpath'
    )
    return parser.parse_args()


if __name__ == '__main__':
    llhp, estimate = get_estimate(**vars(parse_args())) # pylint: disable=invalid-name
