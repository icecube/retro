#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import

"""
Print average Z position of each layer of IceCube and, separately, DeepCore
DOMs.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import json
import os
from os.path import (abspath, dirname, expanduser, expandvars, isdir, isfile,
                     join)

import numpy as np

from I3Tray import OMKey # pylint: disable=import-error
from icecube import dataclasses # pylint: disable=import-error, unused-import
from icecube import dataio # pylint: disable=import-error

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import GEOM_FILE_PROTO, GEOM_META_PROTO
from retro import generate_geom_meta, get_file_md5


__all__ = ['N_STRINGS', 'N_OMS', 'extract_dom_coordinates',
           'parse_args', 'main']


N_STRINGS = 86
N_OMS = 60


def parse_args():
    """Parse command line arguments for `extract_dom_coordinates`"""
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--gcd', metavar='GCD_FILE', type=str,
        #default='$I3_DATA/GCD/GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz',
        default='$I3_DATA/GCD/GeoCalibDetectorStatus_2013.56429_V1.i3.gz',
        help='GCD file from which to extract DOM coordinates'
    )
    parser.add_argument(
        '--outdir', type=str, default='./',
        help='''Directory into which to save the .npy file containing the
        coordinates.'''
    )
    return parser.parse_args()


def extract_dom_coordinates(gcd, outdir):
    """Extract the DOM coordinates from a gcd file.

    Parameters
    ----------
    gcd : string
        Path to GCD file

    outdir : string
        Path to directory into which to store the resulting .npy file
        containing the coordinates array

    """
    gcd = expanduser(expandvars(gcd))
    outdir = expanduser(expandvars(outdir))

    gcd_md5 = get_file_md5(gcd)

    print('Extracting geometry from\n  "{}"'.format(abspath(gcd)))
    print('File MD5 sum is\n  {}'.format(gcd_md5))
    print('Will output geom file and metadata file to directory\n'
          '  "{}"'.format(abspath(outdir)))

    if not isfile(gcd):
        raise IOError('`gcd` file does not exist at "{}"'.format(gcd))

    if not isdir(outdir):
        os.makedirs(outdir)

    geofile = dataio.I3File(gcd)
    geometry = None
    while geofile.more():
        frame = geofile.pop_frame()
        if 'I3Geometry' in frame.keys():
            geometry = frame['I3Geometry']
            break
    if geometry is None:
        raise ValueError('Could not find geometry in file "{}"'.format(gcd))

    omgeo = geometry.omgeo

    geom = np.full(shape=(N_STRINGS, N_OMS, 3), fill_value=np.nan)
    for string in range(N_STRINGS):
        for om in range(N_OMS):
            geom[string, om, :] = (
                omgeo.get(OMKey(string+1, om+1)).position.x,
                omgeo.get(OMKey(string+1, om+1)).position.y,
                omgeo.get(OMKey(string+1, om+1)).position.z
            )

    assert np.sum(np.isnan(geom)) == 0

    geom_meta = generate_geom_meta(geom)
    geom_meta['sourcefile_path'] = gcd
    geom_meta['sourcefile_md5'] = gcd_md5

    outpath = join(outdir, GEOM_FILE_PROTO.format(**geom_meta))
    metapath = join(outdir, GEOM_META_PROTO.format(**geom_meta))

    json.dump(geom_meta, open(metapath, 'w'), indent=2)
    print('Saved metadata to\n  "{}"'.format(abspath(metapath)))
    np.save(outpath, geom)
    print('Saved geom to\n  "{}"'.format(abspath(outpath)))


def main():
    """Main function if being run as a script"""
    extract_dom_coordinates(**vars(parse_args()))


if __name__ == '__main__':
    main()
