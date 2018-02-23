#!/usr/bin/env python
# pylint: disable=invalid-name, wildcard-import

"""
Extract positional and calibration info for DOMs
and save the resulting dict in a pkl file for later use
"""
from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import numpy as np
import pickle

from I3Tray import * # pylint: disable=import-error
from I3Tray import OMKey # pylint: disable=import-error
from icecube import dataclasses # pylint: disable=import-error, unused-import
from icecube import dataio # pylint: disable=import-error
from I3Tray import I3Units


if __name__ == '__main__':
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--file', metavar='GCD_FILE', type=str,
        help='input GCD file',
        default=os.path.expandvars('$I3_DATA/GCD/GeoCalibDetectorStatus_IC86.2017.Run129700_V0.i3.gz')
    )
    args = parser.parse_args()

    f = dataio.I3File(args.file)
    frame = f.pop_frame()

    # get detector geometry
    key = 'I3Geometry'
    while key not in frame.keys():
        frame = f.pop_frame()
    omgeo = frame[key].omgeo

    # get calibration
    key = 'I3Calibration'
    while key not in frame.keys():
        frame = f.pop_frame()
    dom_cal = frame[key].dom_cal

    # create output dict
    data = {}
    data['geo'] = np.zeros((86,60,3))
    data['noise'] = np.zeros((86,60))
    data['rde'] = np.zeros((86,60))
    
    for s in xrange(86):
        for o in xrange(60):
            data['geo'][s,o,0] = omgeo.get(OMKey(s+1,o+1)).position.x
            data['geo'][s,o,1] = omgeo.get(OMKey(s+1,o+1)).position.y
            data['geo'][s,o,2] = omgeo.get(OMKey(s+1,o+1)).position.z
            try:
                data['noise'][s,o] = dom_cal[OMKey(s+1,o+1)].dom_noise_rate / I3Units.hertz
            except KeyError:
                data['noise'][s,o] = 0.
            try:
                data['rde'][s,o] = dom_cal[OMKey(s+1,o+1)].relative_dom_eff
            except KeyError:
                data['rde'][s,o] = 0.
    
    #for s in xrange(86):
    #    print(s, np.mean(data['rde'][s])) 

    #print(data['rde'][32]) 
    #print(data['rde'][44]) 
    print(np.mean(data['rde'][:80])) 
    print(np.mean(data['rde'][79:])) 

    with open('gcd_dict.pkl','wb') as of:
        pickle.dump(data, of)
