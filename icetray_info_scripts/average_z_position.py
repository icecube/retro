#!/usr/bin/env python
# pylint: disable=invalid-name, wildcard-import

"""
Print average Z position of each layer of IceCube and, separately, DeepCore
DOMs.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from I3Tray import * # pylint: disable=import-error
from I3Tray import OMKey # pylint: disable=import-error
from icecube import dataclasses # pylint: disable=import-error, unused-import
from icecube import dataio # pylint: disable=import-error


if __name__ == '__main__':
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', metavar='GCD_FILE', type=str,
        help='input GCD file',
        default='GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz'
    )
    args = parser.parse_args()

    geofile = dataio.I3File(args.file)
    g_frame = geofile.pop_frame()
    while 'I3Geometry' not in g_frame.keys():
        g_frame = geofile.pop_frame()
    geometry = g_frame['I3Geometry']
    omgeo = geometry.omgeo

    ic = np.full(shape=(86, 60, 3), fill_value=np.nan)
    for s in range(86):
        for o in range(60):
            ic[s, o, 0] = omgeo.get(OMKey(s+1, o+1)).position.x
            ic[s, o, 1] = omgeo.get(OMKey(s+1, o+1)).position.y
            ic[s, o, 2] = omgeo.get(OMKey(s+1, o+1)).position.z
    assert np.sum(np.isnan(ic)) == 0

    print('IceCube (non-DeepCore) DOMs:')
    ic_z = []
    for z in range(60):
        print('DOM depth index %i at z = %.2f +/- %.2f'
              % (z, np.mean(ic[0:78, z, 2]), np.std(ic[0:78, z, 2])))
        ic_z.append(np.mean(ic[0:78, z, 2]))

    print('DeepCore DOMs:')
    dc_z = []
    for z in range(60):
        print('DOM depth index %i at z = %.2f +/- %.2f'
              % (z, np.mean(ic[78:, z, 2]), np.std(ic[78:, z, 2])))
        dc_z.append(np.mean(ic[79:, z, 2]))

    print(ic_z)
    print(dc_z)
