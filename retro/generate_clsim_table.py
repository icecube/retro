#!/usr/bin/env python
# pylint: disable=invalid-name
"""
Tabulate the retro light flux in (theta, r, t, theta_dir, deltaphi_dir) bins.
"""

#===============================================================================
# TODO: put all options and config data together into a single hash that can
# uniquely identify a _set_ of tables (i.e. leave off DOM depth idx in the hash
# but include the entire set of geometry that is referenced).
#===============================================================================
# TODO: add angular sensitivity model to the values used to produce a hash


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
import os
import json
import sys

import numpy as np

#from icecube import *
from icecube import icetray # pylint: disable=import-error
from icecube.icetray import I3Units # pylint: disable=import-error
from icecube.clsim.tabulator import LinearAxis, PowerAxis, SphericalAxes # pylint: disable=import-error
from icecube.clsim.tablemaker.tabulator import TabulatePhotonsFromSource # pylint: disable=import-error
from I3Tray import I3Tray # pylint: disable=import-error

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retro import hash_obj


IC_AVG_Z = [
    501.58615386180389, 484.56641094501202, 467.53781871306592,
    450.52576974722058, 433.49294848319812, 416.48833328638324,
    399.45294854579828, 382.44884667029748, 365.4128210605719,
    348.40564121344153, 331.37281916691705, 314.36564088479065,
    297.3325641338642, 280.31782062237079, 263.28397310697113,
    246.27871821476862, 229.24294809194711, 212.23987227219803,
    195.20448733598758, 178.20051300831329, 161.16448681171124,
    144.15717980800531, 127.12435913085938, 110.11717947935446,
    93.085897103334091, 76.078589904002655, 59.045128015371468,
    42.029999953049881, 24.996410223153923, 7.9888461460001192,
    -9.0439743934533539, -26.049487190368847, -43.080769441066643,
    -60.087948872492866, -77.120897733248199, -94.128076993502106,
    -111.15923103919395, -128.16641000600961, -145.19935891567133,
    -162.20371852776944, -179.24769259721805, -196.25589713072165,
    -213.2888457469451, -230.29628186348157, -247.32910332312952,
    -264.33628121400488, -281.36910384740582, -298.34910231370191,
    -315.40756421211438, -332.38756502591644, -349.44602457682294,
    -366.45320559770636, -383.48474355844348, -400.49948746118793,
    -417.53371801131811, -434.51192259177185, -451.56307592147436,
    -468.54307634402545, -485.64474565554889, -502.7208975767478
]

DC_AVG_Z = [
    188.22000122070312, 178.20999799455916, 168.2000013078962,
    158.19000026157923, 148.17000034877233, 138.16000148228235,
    128.14999934605189, 118.14000047956195, 108.12571498325893,
    98.110001700265073, -159.19999912806921, -166.21000017438615,
    -173.2199990408761, -180.22999790736608, -187.23428562709265,
    -194.23999895368303, -201.25, -208.26000322614397,
    -215.26999991280692, -222.27999877929688, -229.29000200544084,
    -236.29428536551339, -243.2999986921038, -250.30999973842077,
    -257.31999860491072, -264.33000401088168, -271.34000069754467,
    -278.3499973842076, -285.35428728376115, -292.36000279017856,
    -299.36999947684154, -306.37999616350447, -313.39000156947543,
    -320.39999825613842, -327.40999494280135, -334.4142848423549,
    -341.42000034877231, -348.43000139508928, -355.44000244140625,
    -362.44999912806918, -369.46000017438615, -376.47000122070312,
    -383.47428676060269, -390.47999790736606, -397.49000331333707,
    -404.5, -411.51000104631697, -418.52000209263394,
    -425.52857317243303, -432.53428431919644, -439.53999982561385,
    -446.55000087193082, -453.55999755859375, -460.56999860491072,
    -467.58000401088168, -474.58857509068082, -481.59286063058033,
    -488.5999973842076, -495.57714407784596, -502.65428379603793
]


def parse_args(description=__doc__):
    """Parese command line options"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--subdet', required=True, choices=('ic', 'dc'),
        help='Calculate for IceCube z-pos (ic) or DeepCore z-pos (dc)'
    )
    parser.add_argument(
        '--depth-idx', type=int, required=True,
        help='''z-position/depth index (referenced to either IceCube or
        DeepCore average depths, according to which is passed to --subdet.'''
    )
    parser.add_argument(
        '--nevts', type=int, required=True,
        help='Number of events to simulate'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='''Random seed to use, in range of 32 bit uint: [0, 2**32-1]
        (default: 0)'''
    )
    parser.add_argument(
        '--outdir', default='./',
        help='Save table to this directory (default: "./")'
    )

    parser.add_argument(
        '--r-max', type=float, default=400,
        help='Radial binning maximum value, in meters (default: 400)'
    )
    parser.add_argument(
        '--r-power', type=float, default=2,
        help='Radial binning is regular in raidus to this power (default: 2)'
    )
    parser.add_argument(
        '--n-r-bins', type=int, default=200,
        help='Number of radial bins (default: 200)'
    )

    parser.add_argument(
        '--t-min', type=float, default=0,
        help='Time binning minimum value, in nanoseconds (default: 0)'
    )
    parser.add_argument(
        '--t-max', type=float, default=3000,
        help='Time binning maximum value, in nanoseconds (default: 3000)'
    )
    parser.add_argument(
        '--n-t-bins', type=int, default=300,
        help='Number of time bins (default: 300)'
    )

    parser.add_argument(
        '--n-costheta-bins', type=int, default=40,
        help='Number of costheta (cosine of zenith angle) bins (default: 40)'
    )

    parser.add_argument(
        '--n-costhetadir-bins', type=int, default=40,
        help='Number of costhetadir bins (default: 20)'
    )
    parser.add_argument(
        '--n-deltaphidir-bins', type=int, default=40,
        help='Number of deltaphidir bins (default: 20)'
    )

    return parser.parse_args()


def generate_clsim_table(subdet, depth_idx, nevts, seed, outdir,
                         r_max, r_power, n_r_bins,
                         n_costheta_bins,
                         t_min, t_max, n_t_bins,
                         n_costhetadir_bins,
                         n_deltaphidir_bins):
    outdir = os.path.expanduser(os.path.expandvars(outdir))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    n_bins = (n_r_bins * n_costheta_bins * n_t_bins * n_costhetadir_bins
              * n_deltaphidir_bins)

    if n_bins > 2**32:
        raise ValueError(
            'The flattened bin index in CLSim is represented by uint32 which'
            ' has max of 4294967296, but the binning specified comes to {}'
            ' bins.'
            .format(n_bins)
        )

    # Average Z coordinate (depth) for each layer of DOMs (see
    # `average_z_position.py`)
    # TODO: make these command-line arguments

    r_min = 0 # meters
    costheta_min, costheta_max = -1, 1
    costhetadir_min, costhetadir_max = -1, 1
    deltaphidir_min, deltaphidir_max = -np.pi, np.pi # rad

    r_binning_kw = dict(
        min=r_min, max=r_max, n_bins=n_r_bins, power=r_power
    )
    costheta_binning_kw = dict(
        min=costheta_min, max=costheta_max, n_bins=n_costheta_bins
    )
    t_binning_kw = dict(
        min=t_min, max=t_max, n_bins=n_t_bins
    )
    costhetadir_binning_kw = dict(
        min=costhetadir_min, max=costhetadir_max, n_bins=n_costhetadir_bins
    )
    deltaphidir_binning_kw = dict(
        min=deltaphidir_min, max=deltaphidir_max, n_bins=n_deltaphidir_bins
    )

    axes = SphericalAxes([
        # r: photon location, radius (m)
        PowerAxis(**r_binning_kw),
        # costheta: photon location, coszenith
        LinearAxis(**costheta_binning_kw),
        # t: photon location, time (ns)
        LinearAxis(**t_binning_kw),
        # costhetadir: photon direction, coszenith
        LinearAxis(**costhetadir_binning_kw),
        # deltaphidir: photon direction, (impact) azimuth angle (rad)
        LinearAxis(**deltaphidir_binning_kw)
    ])

    if subdet.lower() == 'ic':
        z_pos = IC_AVG_Z[depth_idx]
    elif subdet.lower() == 'dc':
        z_pos = DC_AVG_Z[depth_idx]

    print('Subdetector {}, depth index {} (z_avg = {} m)'
          .format(subdet, depth_idx, z_pos))

    # Parameters that will (or can be foreseen to) cause the tables to vary
    # depending on their values. These define what we will call a "set" of
    # tables.
    tray_kw_to_hash = dict(
        PhotonSource='retro',
        Zenith=180 * I3Units.degree, # orientation of source
        Azimuth=0 * I3Units.degree, # orientation of source
        NEvents=nevts,
        IceModel='spice_mie',
        DisableTilt=True,
        PhotonPrescale=1,
        Sensor='none',
    )

    hashable_params = dict(
        r_binning_kw=r_binning_kw,
        t_binning_kw=t_binning_kw,
        costheta_binning_kw=costheta_binning_kw,
        costhetadir_binning_kw=costhetadir_binning_kw,
        deltaphidir_binning_kw=deltaphidir_binning_kw,
        tray_kw_to_hash=tray_kw_to_hash
    )

    hash_val = hash_obj(hashable_params, fmt='hex')[:8]
    filename = ('clsim_table__set_{}__{}__depth_{}__seed_{}.fits'
                .format(hash_val, subdet, depth_idx, seed))
    filepath = os.path.join(outdir, filename)

    metaname = 'clsim_table__set_{}__params.json'.format(hash_val)
    metapath = os.path.join(outdir, metaname)
    if os.path.isfile(filepath):
        if overwrite:
            print('WARNING! Overwriting table metadata file at "{}"'
                  .format(metapath))
        else:
            raise ValueError(
                'Table metadata file already exists at "{}",'
                ' assuming table already generated or in process; not'
                ' overwriting.'.format(metapath)
            )
    json.dump(hashable_params, file(metapath, 'w'), sort_keys=True, indent=4)

    if os.path.isfile(filepath):
        if overwrite:
            print('WARNING! Overwriting table at "{}"'.format(filepath))
            os.remove(filepath)
        else:
            raise ValueError('Table already exists at "{}"! Not overwriting.'
                             .format(filepath))

    tray_kw_other = dict(
        # Note that hash includes the parameters used to construct the axes
        Axes=axes,

        # Parameters that indicate some "index" into the set defined above.
        # I.e., you will want to associate all seeds and all z positions
        # simulated together in the same set, but of course these parameters
        # will also change the tables produced.
        ZCoordinate=z_pos, # location of source
        Seed=seed,

        # Parameters that should have no bearing on the contents of the tables
        Energy=1 * I3Units.GeV,
        TabulateImpactAngle=True,
        Directions=None,
        Filename=filepath,
        FlasherWidth=127,
        FlasherBrightness=127,
        RecordErrors=False,
    )

    all_tray_kw = {}
    all_tray_kw.update(tray_kw_to_hash)
    all_tray_kw.update(tray_kw_other)

    icetray.logging.set_level_for_unit(
        'I3CLSimStepToTableConverter', 'TRACE'
    )
    icetray.logging.set_level_for_unit(
        'I3CLSimTabulatorModule', 'DEBUG'
    )
    icetray.logging.set_level_for_unit(
        'I3CLSimLightSourceToStepConverterGeant4', 'TRACE'
    )
    icetray.logging.set_level_for_unit(
        'I3CLSimLightSourceToStepConverterFlasher', 'TRACE'
    )

    tray = I3Tray()
    tray.AddSegment(TabulatePhotonsFromSource, 'generator', **all_tray_kw)
    tray.Execute()
    tray.Finish()


if __name__ == '__main__':
    generate_clsim_table(**vars(parse_args()))