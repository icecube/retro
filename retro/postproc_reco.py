#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Parse MultiNest *stats output file.
"""

from __future__ import absolute_import, print_function

__all__ = ['UNITS', 'parse', 'main']

from collections import OrderedDict
from os.path import abspath, dirname
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand
from retro.reco import CUBE_DIMS


UNITS = dict(
    x='m', y='m', z='m', t='ns', track_zenith='deg', track_azimuth='deg',
    track_energy='GeV', cascade_energy='GeV'
)


# TODO: handle multi-modal output


def parse(stats):
    """Parse the contents of a MultiNest *stats output file.

    Parameters
    ----------
    stats : iterable of str
        Result of open(file, 'r').readlines()

    Returns
    -------
    """
    found = False
    lineno = -1
    for lineno, line in enumerate(stats):
        if line.startswith('Dim No.'):
            found = True
            break
    if not found:
        raise ValueError('Could not find line with "Dim No." in file')

    stats = [s.strip() for s in stats[lineno + 1 : lineno + 1 + len(CUBE_DIMS)]]

    points = OrderedDict()
    errors = OrderedDict()

    for stat_line, dim in zip(stats, CUBE_DIMS):
        points[dim], errors[dim] = [float(x) for x in stat_line.split()[1:]]

    e_pt = points['energy']
    e_sd = errors['energy']
    tfrac_pt = points['track_fraction']
    tfrac_sd = errors['track_fraction']

    points['track_energy'] = e_pt * tfrac_pt
    points['cascade_energy'] = e_pt * (1 - tfrac_pt)

    errors['track_energy'] = e_sd * tfrac_pt
    errors['cascade_energy'] = e_sd * (1 - tfrac_pt)

    return points, errors


def main():
    """Load file specified on command line and print results of parsing it."""
    with open(expand(sys.argv[1]), 'r') as f:
        contents = f.readlines()

    try:
        points, errors = parse(contents)
    except ValueError:
        print('Failed to parse file "{}"'.format(sys.argv[1]))
        raise

    for dim in 't x y z track_zenith track_azimuth track_energy cascade_energy'.split():
        pt = points[dim]
        sd = errors[dim]
        if dim in ['track_zenith', 'track_azimuth']:
            pt = np.rad2deg(pt)
            sd = np.rad2deg(sd)

        print('{:14s} = {:8.3f} +/- {:5.1f} {}'.format(dim, pt, sd, UNITS[dim]))


if __name__ == '__main__':
    main()
