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

    f = dataio.I3File(args.file)
    frame = f.pop_frame()
    key = 'I3Calibration'
    while key not in frame.keys():
        frame = f.pop_frame()
    print(frame[key].dom_cal[OMKey(86,42)].relative_dom_eff)
    print(frame[key].dom_cal[OMKey(86,42)].dom_noise_rate)

