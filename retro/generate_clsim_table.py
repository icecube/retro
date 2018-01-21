#!/usr/bin/env python
# pylint: disable=wrong-import-position, invalid-name

"""
Tabulate the retro light flux in (theta, r, t, theta_dir, deltaphi_dir) bins.
"""


# TODO: add angular sensitivity model to the values used to produce a hash
#       (currently "as.h2-50cm")
# TODO: command-line option to simply return the metadata for a config to e.g.
#       extract a hash value one would expect from the given params
# TODO: include detector geometry (probably original full detector geom...) in
#       hash value
# TODO: does x and y coordinate make a difference if we turn ice tilt on? if
#       so, we should be able to specify (str,om) coordinate for simulation


from __future__ import absolute_import, division, print_function


__all__ = ['generate_clsim_table_meta', 'generate_clsim_table', 'parse_args']

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller, Justin L. Lanfranchi

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
from os import access, environ, makedirs, pathsep, remove, X_OK
from os.path import abspath, dirname, isdir, isfile, join
import json
from numbers import Integral
from subprocess import check_call
import sys

import numpy as np

#from icecube import *
from icecube import icetray # pylint: disable=import-error
from icecube.icetray import I3Units # pylint: disable=import-error
from icecube.clsim.tabulator import LinearAxis, PowerAxis, SphericalAxes # pylint: disable=import-error
from icecube.clsim.tablemaker.tabulator import TabulatePhotonsFromSource # pylint: disable=import-error
from I3Tray import I3Tray # pylint: disable=import-error

if __name__ == '__main__' and __package__ is None:
    PARENT_DIR = dirname(dirname(abspath(__file__)))
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)
from retro import CLSIM_TABLE_FNAME_V2_PROTO, CLSIM_TABLE_METANAME_PROTO
from retro import expand, generate_geom_meta, hash_obj


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


def generate_clsim_table_meta(r_binning_kw, t_binning_kw, costheta_binning_kw,
                              costhetadir_binning_kw, deltaphidir_binning_kw,
                              tray_kw_to_hash):
    """
    Returns
    -------
    hash_val : string
        8 hex characters indicating hash value for the table

    metaname : string
        Filename for file that will contain the complete metadata used to
        define this set of tables

    """
    linear_binning_keys = sorted(['min', 'max', 'n_bins'])
    power_binning_keys = sorted(['min', 'max', 'power', 'n_bins'])
    for binning_kw in [t_binning_kw, costheta_binning_kw,
                       costhetadir_binning_kw, deltaphidir_binning_kw]:
        assert sorted(binning_kw.keys()) == linear_binning_keys
    assert sorted(r_binning_kw.keys()) == power_binning_keys

    tray_keys = sorted(['PhotonSource', 'Zenith', 'Azimuth', 'NEvents',
                        'IceModel', 'DisableTilt', 'PhotonPrescale', 'Sensor'])
    assert sorted(tray_kw_to_hash.keys()) == tray_keys

    hashable_params = dict(
        r_binning_kw=r_binning_kw,
        t_binning_kw=t_binning_kw,
        costheta_binning_kw=costheta_binning_kw,
        costhetadir_binning_kw=costhetadir_binning_kw,
        deltaphidir_binning_kw=deltaphidir_binning_kw,
        tray_kw_to_hash=tray_kw_to_hash
    )

    hash_val = hash_obj(hashable_params, fmt='hex')[:8]
    metaname = CLSIM_TABLE_METANAME_PROTO.format(hash_val=hash_val)

    return hash_val, metaname


# TODO: add parmeters for detector geometry, bulk ice model, hole ice model
# (i.e. this means angular sensitivity curve in its current implementation,
# though more advanced hole ice models could mean different things), and
# whether to use time difference from direct time
def generate_clsim_table(subdet, depth_idx, nevts, seed, tilt,
                         r_max, r_power, n_r_bins,
                         t_max, n_t_bins,
                         n_costheta_bins, n_costhetadir_bins,
                         n_deltaphidir_bins,
                         outdir, overwrite=False, compress=True):
    """Generate a CLSim table.

    Parameters
    ----------
    subdet : string, {'ic', 'dc'}

    depth_idx : int in [0, 59]

    nevts : int > 0
        Note that the number of photons is much larger than the number of
        events (related to the "brightness" of the defined source)

    seed : int in [0, 2**32)
        Seed for CLSim's random number generator

    tilt : bool
        Whether to enable ice layer tilt in simulation

    r_max : float > 0

    r_power : int > 0

    t_max : float > 0

    n_t_bins : int > 0

    n_costheta_bins : int > 0

    n_costhetadir_bins : int > 0

    n_deltaphidir_bins : int > 0

    outdir : string

    overwrite : bool, optional
        Whether to overwrite an existing table (default: False)

    compress : bool, optional
        Whether to pass the resulting table through zstandard compression
        (default: True)

    Raises
    ------
    ValueError
        If `compress` but `zstd` command-line utility cannot be found

    AssertionError, ValueError
        If illegal argument values are passed

    ValueError
        If `overwrite` is False and a table already exists at the target path

    Notes
    -----
    Binnings are as follows:
        * Radial binning is regular in the space of r**(1/r_power), with
          `n_r_bins` spanning from `r_min` to `r_max`.
        * Time binning is linearly spaced with `n_t_bins` spanning from `t_min`
          to `t_max`
        * Position zenith angle is binned regularly in the cosine of the zenith
          angle with `n_costhetadir_bins` spanning from `costheta_min` to
          `costheta_max`.
        * Position azimuth angle is _not_ binned
        * Photon directionality zenith angle is binned regularly in
          cosine-zenith space, with `n_costhetadir_bins` spanning from
          `costhetadir_min` to `costhetadir_max`
        * Photon directionality azimuth angle, since position azimuth angle is
          not binned, is translated into the absolute value of the azimuth
          angle relative to the azimuth position of the photon; this is called
          `deltaphidir`. There are `n_deltaphidir_bins` from `deltaphidir_min`
          to `deltaphidir_max`.

    The following are forced upon the above binning specifications (and
    remaining parameters are specified as arguments to the function)
        * t_min = 0
        * r_min = 0
        * costheta_min = -1
        * costheta_max = 1
        * costhetadir_min = -1
        * costhetadir_max = 1
        * deltaphidir_min = 0
        * deltaphidir_min = pi (rad)

    """
    assert isinstance(nevts, Integral) and nevts > 0
    assert isinstance(seed, Integral) and 0 <= seed < 2**32
    assert isinstance(r_power, Integral) and r_power > 0
    assert isinstance(n_r_bins, Integral) and n_r_bins > 0
    assert isinstance(n_t_bins, Integral) and n_t_bins > 0
    assert isinstance(n_costheta_bins, Integral) and n_costheta_bins > 0
    assert isinstance(n_costhetadir_bins, Integral) and n_costhetadir_bins > 0
    assert isinstance(n_deltaphidir_bins, Integral) and n_deltaphidir_bins > 0

    if compress and not any(access(join(path, 'zstd'), X_OK)
                            for path in environ['PATH'].split(pathsep)):
        raise ValueError('`zstd` command not found in path')

    outdir = expand(outdir)
    if not isdir(outdir):
        makedirs(outdir)

    # Note: + 2 accounts for under/overflow bins in each dimension
    n_bins = np.product([n_bins + 2 for n_bins in (n_r_bins,
                                                   n_costheta_bins,
                                                   n_t_bins,
                                                   n_costhetadir_bins,
                                                   n_deltaphidir_bins)])

    if n_bins > 2**32:
        raise ValueError(
            'The flattened bin index in CLSim is represented by uint32 which'
            ' has a max of 4 294 967 296, but the binning specified comes to'
            ' {} bins ({} times too many).'
            .format(n_bins, n_bins / 2**32)
        )

    # Average Z coordinate (depth) for each layer of DOMs (see
    # `average_z_position.py`)
    # TODO: make these command-line arguments

    t_min = 0 # ns
    r_min = 0 # meters
    costheta_min, costheta_max = -1.0, 1.0
    costhetadir_min, costhetadir_max = -1.0, 1.0
    deltaphidir_min, deltaphidir_max = 0.0, np.pi # rad

    r_binning_kw = dict(
        min=float(r_min),
        max=float(r_max),
        n_bins=int(n_r_bins),
        power=int(r_power)
    )
    costheta_binning_kw = dict(
        min=float(costheta_min),
        max=float(costheta_max),
        n_bins=int(n_costheta_bins)
    )
    t_binning_kw = dict(
        min=float(t_min),
        max=float(t_max),
        n_bins=int(n_t_bins)
    )
    costhetadir_binning_kw = dict(
        min=float(costhetadir_min),
        max=float(costhetadir_max),
        n_bins=int(n_costhetadir_bins)
    )
    deltaphidir_binning_kw = dict(
        min=float(deltaphidir_min),
        max=float(deltaphidir_max),
        n_bins=int(n_deltaphidir_bins)
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
    ]) # yapf: disable

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
        # Number of events will affect the tables, but n=999 and n=1000 will be
        # very similar (and not statistically independent if the seed is the
        # same). But a user is likely to want to test out same settings but
        # different statistics, so these sets need different hashes (unless we
        # want the user to also specify the nevts when identifying a set...)
        # Therefore, this is included in the hash to indicate a common set of
        # tables
        NEvents=nevts,
        IceModel='spice_mie',
        DisableTilt=not tilt,
        PhotonPrescale=1,
        Sensor='none'
    )

    hashable_params = dict(
        r_binning_kw=r_binning_kw,
        t_binning_kw=t_binning_kw,
        costheta_binning_kw=costheta_binning_kw,
        costhetadir_binning_kw=costhetadir_binning_kw,
        deltaphidir_binning_kw=deltaphidir_binning_kw,
        tray_kw_to_hash=tray_kw_to_hash
    )

    hash_val, metaname = generate_clsim_table_meta(**hashable_params)
    metapath = join(outdir, metaname)

    filename = CLSIM_TABLE_FNAME_V2_PROTO.format(hash_val=hash_val,
                                                 string=subdet,
                                                 depth_idx=depth_idx,
                                                 seed=seed)
    filepath = abspath(join(outdir, filename))

    #if isfile(metapath):
    #    if overwrite:
    #        print('WARNING! Overwriting table metadata file at "{}"'
    #              .format(metapath))
    #    else:
    #        raise ValueError(
    #            'Table metadata file already exists at "{}",'
    #            ' assuming table already generated or in process; not'
    #            ' overwriting.'.format(metapath)
    #        )
    json.dump(hashable_params, file(metapath, 'w'), sort_keys=True, indent=4)

    print('='*80)
    print('Metadata for the table set was written to\n  "{}"'.format(metapath))
    print('Table will be written to\n  "{}"'.format(filepath))
    print('='*80)

    exists_at = []
    for fpath in [filepath, filepath + '.zst']:
        if isfile(fpath):
            exists_at.append(fpath)

    if exists_at:
        names = ', '.join('"{}"'.format(fp) for fp in exists_at)
        if overwrite:
            print('WARNING! Deleting existing table(s) at ' + names)
            for fpath in exists_at:
                remove(fpath)
        else:
            raise ValueError('Table(s) already exist at {}; not'
                             ' overwriting.'.format(names))
    print('')

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

    if compress:
        print('Compressing table with zstandard via command line')
        print('  zstd -1 --rm "{}"'.format(filepath))
        check_call(['zstd', '-1', '--rm', filepath])
        print('done.')


def parse_args(description=__doc__):
    """Parese command line args.

    Returns
    -------
    args : Namespace

    """
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
        '--seed', type=int, required=True,
        help='Random seed to use, in range of 32 bit uint: [0, 2**32-1]'
    )

    parser.add_argument(
        '--tilt', action='store_true',
        help='Enable tilt in ice model'
    )

    parser.add_argument(
        '--r-max', type=float, required=True,
        help='Radial binning maximum value, in meters'
    )
    parser.add_argument(
        '--r-power', type=int, required=True,
        help='Radial binning is regular in raidus to this power'
    )
    parser.add_argument(
        '--n-r-bins', type=int, required=True,
        help='Number of radial bins'
    )

    parser.add_argument(
        '--t-max', type=float, required=True,
        help='Time binning maximum value, in nanoseconds'
    )
    parser.add_argument(
        '--n-t-bins', type=int, required=True,
        help='Number of time bins'
    )

    parser.add_argument(
        '--n-costheta-bins', type=int, required=True,
        help='Number of costheta (cosine of zenith angle) bins'
    )

    parser.add_argument(
        '--n-costhetadir-bins', type=int, required=True,
        help='Number of costhetadir bins'
    )
    parser.add_argument(
        '--n-deltaphidir-bins', type=int, required=True,
        help='''Number of deltaphidir bins (Note: span from 0 to pi; code
        assumes symmetry about 0)'''
    )

    parser.add_argument(
        '--outdir', default='./',
        help='Save table to this directory (default: "./")'
    )

    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite if the table already exists'
    )

    return parser.parse_args()


if __name__ == '__main__':
    generate_clsim_table(**vars(parse_args()))
