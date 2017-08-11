from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from I3Tray import *
from icecube import dataclasses, dataio

if __name__ == '__main__':

    parser = ArgumentParser(description='''make 2d event pictures''')
    parser.add_argument( '-f', '--file', metavar='GCD_FILE', type=str, help='input GCD file', default='GeoCalibDetectorStatus_2013.56429_V1_Modified.i3.gz')
    args = parser.parse_args()

    geofile=dataio.I3File(args.file)
    g_frame = geofile.pop_frame()
    while 'I3Geometry' not in g_frame.keys():
        g_frame = geofile.pop_frame()
    geometry = g_frame["I3Geometry"]
    omgeo=geometry.omgeo
    IC = np.zeros((86,60,3))
    for s in xrange(86):
        for o in xrange(60):
            IC[s,o,0] = omgeo.get(OMKey(s+1,o+1)).position.x
            IC[s,o,1] = omgeo.get(OMKey(s+1,o+1)).position.y
            IC[s,o,2] = omgeo.get(OMKey(s+1,o+1)).position.z
    print 'IC'
    ic_z = []
    for z in xrange(60):
        print 'DOM %i at z = %.2f +/- %.2f'%(z, np.mean(IC[0:78,z,2]), np.std(IC[0:78,z,2]))
        ic_z.append(np.mean(IC[0:78,z,2]))
    print 'DC'
    dc_z = []
    for z in xrange(60):
        print 'DOM %i at z = %.2f +/- %.2f'%(z, np.mean(IC[78:,z,2]), np.std(IC[78:,z,2]))
        dc_z.append(np.mean(IC[79:,z,2]))

    print ic_z
    print dc_z
