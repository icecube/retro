#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name, no-member, no-name-in-module
# pylint: disable=import-error

"""
Create a Retro table: Propagate light outwards from a DOM and tabulate the
photons. Uses CLSim (tabulator) to do the work of photon propagation.
"""

# TODO: command-line option to simply return the metadata for a config to e.g.
#       extract a hash value one would expect from the given params

from __future__ import absolute_import, division, print_function

__all__ = [
    'get_average_dom_z_coords',
    'generate_clsim_table',
    'parse_args',
]

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
from collections import OrderedDict
from os import (
    access, environ, getpid, pathsep, remove, X_OK
)
from os.path import (
    abspath, dirname, exists, expanduser, expandvars, isfile, join, split
)
import json
from numbers import Integral
import subprocess
import sys
import threading
import time

import numpy as np

from I3Tray import I3Tray
from icecube.clsim import (
    AutoSetGeant4Environment,
    GetDefaultParameterizationList,
    GetFlasherParameterizationList,
    #GetIceCubeDOMAcceptance,
    I3CLSimFunctionConstant,
    I3CLSimFlasherPulse,
    I3CLSimFlasherPulseSeries,
    I3CLSimLightSourceToStepConverterGeant4,
    I3CLSimLightSourceToStepConverterPPC,
    I3CLSimSpectrumTable,
)
from icecube.clsim.tabulator import (
    LinearAxis,
    PowerAxis,
    SphericalAxes,
)
from icecube.clsim.traysegments.common import (
    configureOpenCLDevices,
    parseIceModel,
)
from icecube import dataclasses
from icecube.dataclasses import (
    I3Direction,
    I3Particle,
    I3Position,
)
from icecube.icetray import I3Frame, I3Module, I3Units, logging, traysegment
from icecube.photospline.photonics import FITSTable
from icecube.phys_services import I3GSLRandomService

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3info.extract_gcd import extract_gcd
from retro.utils.misc import expand, hash_obj, mkdir
from retro.tables.clsim_tables import (
    CLSIM_TABLE_FNAME_PROTO,
    CLSIM_TABLE_METANAME_PROTO,
    CLSIM_TABLE_TILE_FNAME_PROTO,
    CLSIM_TABLE_TILE_METANAME_PROTO,
)


DOM_RADIUS = 0.16510*I3Units.m # 13" diameter
DOM_SURFACE_AREA = np.pi * DOM_RADIUS**2

BINNING_ORDER = {
    'spherical': [
        'r',
        'costheta',
        'phi',
        't',
        'costhetadir',
        'deltaphidir',
    ],
    'cartesian': [
        'x',
        'y',
        'z',
        't',
        'costhetadir',
        'phidir',
    ]
}


def get_average_dom_z_coords(geo):
    """Find average z coordinates for IceCube (non-DeepCore) and DeepCore
    "z-layers" of DOMs.

    A "z-layer" of DOMs is defined by all DOMs on all strings of a given string
    type with shared DOM (OM) indices.

    Parameters
    ----------
    geo : (n_strings, n_doms_per_string, 3) array
        (x, y, z) coordinate for string 1 (string index 0) DOM 1 (dom index 0) is found at geo[0, 0]

    Returns
    -------
    ic_avg_z : shape (n_doms_per_string) array
    dc_avg_z : shape (n_doms_per_string) array

    """
    ic_avg_z = geo[:78, :, 2].mean(axis=0)
    dc_avg_z = geo[78:, :, 2].mean(axis=0)
    return ic_avg_z, dc_avg_z


def make_retro_pulse(x, y, z, zenith, azimuth):
    """Retro pulses originate from a DOM with an (x, y, z) coordinate and
    (potentially) a zenith and azimuth orientation (though for now the latter
    are ignored).

    """
    pulse = I3CLSimFlasherPulse()
    pulse.type = I3CLSimFlasherPulse.FlasherPulseType.retro
    pulse.pos = I3Position(x, y, z)
    pulse.dir = I3Direction(zenith, azimuth)
    pulse.time = 0.0
    pulse.numberOfPhotonsNoBias = 10000.

    # Following values don't make a difference
    pulse.pulseWidth = 1.0 * I3Units.ns
    pulse.angularEmissionSigmaPolar = 360.0 * I3Units.deg
    pulse.angularEmissionSigmaAzimuthal = 360.0 * I3Units.deg

    return pulse


def unpin_threads(delay=60):
    """
    When AMD OpenCL fissions the CPU device, it pins each sub-device to a
    a physical core. Since we always use sub-device 0, this means that multiple
    instances of the tabulator on a single machine will compete for core 0.
    Reset thread affinity after *delay* seconds to prevent this from happening.
    """
    # pylint: disable=missing-docstring
    def which(program):
        def is_exe(fpath):
            return exists(fpath) and access(fpath, X_OK)

        def ext_candidates(fpath):
            yield fpath
            for ext in environ.get('PATHEXT', '').split(pathsep):
                yield fpath + ext

        fpath, _ = split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in environ['PATH'].split(pathsep):
                exe_file = join(path, program)
                for candidate in ext_candidates(exe_file):
                    if is_exe(candidate):
                        return candidate

    def taskset(pid, tt=None):
        # get/set the taskset affinity for pid
        # uses a binary number string for the core affinity
        l = [which('taskset'), '-p']
        if tt:
            l.append(hex(int(tt, 2))[2:])
        l.append(str(pid))
        p = subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = p.communicate()[0].split(':')[-1].strip()
        if not tt:
            return bin(int(output, 16))[2:]

    def resetTasksetThreads(main_pid):
        # reset thread taskset affinity
        time.sleep(delay)
        num_cpus = reduce(
            lambda b, a: b + int('processor' in a), open('/proc/cpuinfo').readlines(),
            0
        )
        tt = '1'*num_cpus
        #tt = taskset(main_pid)
        p = subprocess.Popen(
            [which('ps'), '-Lo', 'tid', '--no-headers', '%d'%main_pid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for tid in p.communicate()[0].split():
            tid = tid.strip()
            if tid:
                taskset(tid, tt)
    # only do this on linux
    try:
        open('/proc/cpuinfo')
    except IOError:
        return
    # and only if taskset exists
    if not which('taskset'):
        return
    threading.Thread(target=resetTasksetThreads, args=(getpid(),)).start()


@traysegment
def TabulateRetroSources(
    tray,
    name,
    source_gcd_i3_md5,
    binning_kw,
    axes,
    ice_model,
    angular_sensitivity,
    disable_tilt,
    disable_anisotropy,
    hash_val,
    dom_spec,
    dom_x,
    dom_y,
    dom_z,
    dom_zenith,
    dom_azimuth,
    seed,
    n_events,
    tablepath,
    tile=None,
    record_errors=False,
):
    dom_x = dom_x * I3Units.m
    dom_y = dom_y * I3Units.m
    dom_z = dom_z * I3Units.m
    dom_zenith = dom_zenith * I3Units.rad
    dom_azimuth = dom_azimuth * I3Units.rad

    tablepath = expanduser(expandvars(tablepath))

    random_service = I3GSLRandomService(seed)

    tray.AddModule(
        'I3InfiniteSource', name + 'streams',
        Stream=I3Frame.DAQ
    )

    tray.AddModule(
        'I3MCEventHeaderGenerator',
        name + 'gen_header',
        Year=2009,
        DAQTime=158100000000000000,
        RunNumber=1,
        EventID=1,
        IncrementEventID=True
    )

    flasher_pulse_series_name = 'I3FlasherPulseSeriesMap'

    def reference_source(x, y, z, zenith, azimuth, scale):
        source = I3Particle()
        source.pos = I3Position(x, y, z)
        source.dir = I3Direction(zenith, azimuth)
        source.time = 0.0
        # Following are not used (at least not yet)
        source.type = I3Particle.ParticleType.EMinus
        source.energy = 1.0*scale
        source.length = 0.0
        source.location_type = I3Particle.LocationType.InIce

        return source

    class MakeParticle(I3Module):
        def __init__(self, ctx):
            super(MakeParticle, self).__init__(ctx)
            self.AddOutBox('OutBox')
            self.AddParameter('source_function', '', lambda: None)
            self.AddParameter('n_events', '', 100)
            self.reference_source = None
            self.n_events = None
            self.emitted_events = None

        def Configure(self):
            self.reference_source = self.GetParameter('source_function')
            self.n_events = self.GetParameter('n_events')
            self.emitted_events = 0

        def DAQ(self, frame):
            pulseseries = I3CLSimFlasherPulseSeries()
            pulse = make_retro_pulse(
                x=dom_x,
                y=dom_y,
                z=dom_z,
                zenith=dom_zenith,
                azimuth=dom_azimuth,
            )
            pulseseries.append(pulse)
            frame[flasher_pulse_series_name] = pulseseries
            frame['ReferenceParticle'] = self.reference_source(
                x=dom_x,
                y=dom_y,
                z=dom_z,
                zenith=dom_zenith,
                azimuth=dom_azimuth,
                scale=1.0,
            )
            self.PushFrame(frame)
            self.emitted_events += 1
            if self.emitted_events >= self.n_events:
                self.RequestSuspension()

    tray.AddModule(
        MakeParticle,
        source_function=reference_source,
        n_events=n_events,
    )

    header = OrderedDict(FITSTable.empty_header)
    header['retro_dom_table'] = 0
    header['gcd_i3_md5_{:s}'.format(source_gcd_i3_md5)] = 0
    for n, axname in enumerate(binning_kw.keys()):
        header['ax_{:s}'.format(axname)] = n
        for key, val in binning_kw[axname].items():
            header['{}_{}'.format(axname, key)] = val
        if axname == 't':
            header['t_is_residual_time'] = 1
    header['ice_{:s}'.format(ice_model.replace('.', '_'))] = 0
    header['angsens_{:s}'.format(angular_sensitivity.replace('.', '_'))] = 0
    header['disable_tilt'] = disable_tilt
    header['disable_anisotropy'] = disable_anisotropy
    header['hash_{:s}'.format(hash_val)] = 0
    if tile is not None:
        header['tile'] = tile
    for key, value in dom_spec.items():
        header[key] = value
    header['dom_x'] = dom_x
    header['dom_y'] = dom_y
    header['dom_z'] = dom_z
    header['dom_zenith'] = dom_zenith
    header['dom_azimuth'] = dom_azimuth
    header['seed'] = seed
    header['n_events'] = n_events

    if hasattr(dataclasses, 'I3ModuleGeo'):
        tray.AddModule(
            'I3GeometryDecomposer',
            name + '_decomposeGeometry',
            If=lambda frame: 'I3OMGeoMap' not in frame
        )

    # at the moment the Geant4 paths need to be set, even if it isn't used
    # TODO: fix this
    if I3CLSimLightSourceToStepConverterGeant4.can_use_geant4:
        AutoSetGeant4Environment()

    ppc_converter = I3CLSimLightSourceToStepConverterPPC(photonsPerStep=200)
    # Is this even necessary?
    ppc_converter.SetUseCascadeExtension(False)
    particle_parameterizations = GetDefaultParameterizationList(
        ppc_converter,
        muonOnly=False,
    )

    # need a spectrum table in order to pass spectra to OpenCL
    spectrum_table = I3CLSimSpectrumTable()
    particle_parameterizations += GetFlasherParameterizationList(spectrum_table)

    logging.log_debug(
        'number of spectra (1x Cherenkov + Nx flasher): %d'
        % len(spectrum_table),
        unit='clsim',
    )

    opencl_devices = configureOpenCLDevices(
        UseGPUs=False,
        UseCPUs=True,
        OverrideApproximateNumberOfWorkItems=None,
        DoNotParallelize=True,
        UseOnlyDeviceNumber=None,
    )

    medium_properties = parseIceModel(
        expandvars('$I3_SRC/ice-models/resources/models/' + ice_model),
        disableTilt=disable_tilt,
        disableAnisotropy=disable_anisotropy,
    )

    tray.AddModule(
        'I3CLSimTabulatorModule',
        name + '_clsim',
        MCTreeName='', # doesn't apply since we use pulse series
        FlasherPulseSeriesName=flasher_pulse_series_name,
        RandomService=random_service,
        Area=DOM_SURFACE_AREA,
        WavelengthAcceptance=I3CLSimFunctionConstant(1.0), #GetIceCubeDOMAcceptance(domRadius=DOM_RADIUS),
        AngularAcceptance=I3CLSimFunctionConstant(1.0),
        MediumProperties=medium_properties,
        ParameterizationList=particle_parameterizations,
        SpectrumTable=spectrum_table,
        OpenCLDeviceList=opencl_devices,
        PhotonsPerBunch=200,
        EntriesPerPhoton=5000,
        Filename=tablepath,
        RecordErrors=record_errors,
        TableHeader=header,
        Axes=axes,
        SensorNormalize=False
    )

    unpin_threads()


# TODO: add to CLSim invocation parmeters for detector geometry, bulk ice model, hole
# ice model (i.e. this means angular sensitivity curve in its current implementation,
# though more advanced hole ice models could mean different things), and whether to use
# time difference from direct time
def generate_clsim_table(
    outdir,
    gcd,
    ice_model,
    angular_sensitivity,
    disable_tilt,
    disable_anisotropy,
    string,
    dom,
    n_events,
    seed,
    coordinate_system,
    binning,
    tableset_hash=None,
    tile=None,
    overwrite=False,
    compress=False,
):
    """Generate a CLSim table.

    See wiki.icecube.wisc.edu/index.php/Ice for information about ice models.

    Parameters
    ----------
    outdir : string

    gcd : string

    ice_model : str
        E.g. "spice_mie", "spice_lea", ...

    angular_sensitivity : str
        E.g. "h2-50cm", "9" (which is equivalent to "new25" because, like, duh)

    disable_tilt : bool
        Whether to force no layer tilt in simulation (if tilt is present in
        bulk ice model; otherwise, this has no effect)

    disable_anisotropy : bool
        Whether to force no bulk ice anisotropy (if anisotropy is present in
        bulk ice model; otherwise, this has no effect)

    string : int in [1, 86]

    dom : int in [1, 60]

    n_events : int > 0
        Note that the number of photons is much larger than the number of
        events (related to the "brightness" of the defined source).

    seed : int in [0, 2**32)
        Seed for CLSim's random number generator

    coordinate_system : string in {"spherical", "cartesian"}
        If spherical, base coordinate system is .. ::

            (r, theta, phi, t, costhetadir, (optionally abs)deltaphidir)

        If Cartesian, base coordinate system is .. ::

            (x, y, z, costhetadir, phidir)

        but if any of the coordinate axes are specified to have 0 bins, they
        will be omitted (but the overall order is maintained).

    binning : mapping
        If `coordinate_system` is "spherical", keys should be:
            "n_r_bins"
            "n_t_bins"
            "n_costheta_bins"
            "n_phi_bins"
            "n_costhetadir_bins"
            "n_deltaphidir_bins"
            "r_max"
            "r_power"
            "t_max"
            "t_power"
            "deltaphidir_power"
        If `coordinate_system` is "cartesian", keys should be:
            "n_x_bins"
            "n_y_bins"
            "n_z_bins"
            "n_costhetadir_bins"
            "n_phidir_bins"
            "x_min"
            "x_max"
            "y_min"
            "y_max"
            "z_min"
            "z_max"

    tableset_hash : str, optional
        Specify if the table is a tile used to generate a larger table

    tile : int >= 0, optional
        Specify if the table is a tile used to generate a larger table

    overwrite : bool, optional
        Whether to overwrite an existing table (default: False)

    compress : bool, optional
        Whether to pass the resulting table through zstandard compression
        (default: True)

    Raises
    ------
    ValueError
        If `compress` is True but `zstd` command-line utility cannot be found

    AssertionError, ValueError
        If illegal argument values are passed

    ValueError
        If `overwrite` is False and a table already exists at the target path

    Notes
    -----
    Binnings are as follows:
        * Radial binning is regular in the space of r**(1/r_power), with
          `n_r_bins` spanning from 0 to `r_max` meters.
        * Time binning is regular in the space of t**(1/t_power), with
          `n_t_bins` spanning from 0 to `t_max` nanoseconds.
        * Position zenith angle is binned regularly in the cosine of the zenith
          angle with `n_costhetadir_bins` spanning from -1 to +1.
        * Position azimuth angle is binned regularly, with `n_phi_bins`
          spanning from -pi to pi radians.
        * Photon directionality zenith angle (relative to IcedCube coordinate
          system) is binned regularly in cosine-zenith space, with
          `n_costhetadir_bins` spanning from `costhetadir_min` to
          `costhetadir_max`
        * Photon directionality azimuth angle; sometimes assumed to be
          symmetric about line from DOM to the center of the bin, so is binned
          as an absolute value, i.e., from 0 to pi radians. Otherwise, binned
          from -np.pi to +np.pi

    The following are forced upon the above binning specifications (and
    remaining parameters are specified as arguments to the function)
        * t_min = 0 (ns)
        * r_min = 0 (m)
        * costheta_min = -1
        * costheta_max = 1
        * phi_min = -pi (rad)
        * phi_max = pi (rad)
        * costhetadir_min = -1
        * costhetadir_max = 1
        * deltaphidir_min = 0 (rad)
        * deltaphidir_min = pi (rad)

    """
    assert isinstance(n_events, Integral) and n_events > 0
    assert isinstance(seed, Integral) and 0 <= seed < 2**32
    assert (
        (tableset_hash is not None and tile is not None)
        or (tableset_hash is None and tile is None)
    )

    n_bins_per_dim = []
    for key, val in binning.items():
        if not key.startswith('n_'):
            continue
        assert isinstance(val, Integral), '{} not an integer'.format(key)
        assert val >= 0, '{} must be >= 0'.format(key)
        n_bins_per_dim.append(val)

    # Note: + 2 accounts for under & overflow bins in each dimension
    n_bins = np.product([n + 2 for n in n_bins_per_dim if n > 0])

    assert n_bins > 0

    #if n_bins > 2**32:
    #    raise ValueError(
    #        'The flattened bin index in CLSim is represented by uint32 which'
    #        ' has a max of 4 294 967 296, but the binning specified comes to'
    #        ' {} bins ({} times too many).'
    #        .format(n_bins, n_bins / 2**32)
    #    )

    ice_model = ice_model.strip()
    angular_sensitivity = angular_sensitivity.strip()
    # For now, hole ice model is hard-coded in our CLSim branch; see
    #   clsim/private/clsim/I3CLSimLightSourceToStepConverterFlasher.cxx
    # in the branch you're using to check that this is correct
    assert angular_sensitivity == 'flasher_p1_0.30_p2_-1'

    gcd_info = extract_gcd(gcd)

    if compress and not any(access(join(path, 'zstd'), X_OK)
                            for path in environ['PATH'].split(pathsep)):
        raise ValueError('`zstd` command not found in path')

    outdir = expand(outdir)
    mkdir(outdir)

    axes = OrderedDict()
    binning_kw = OrderedDict()

    # Note that the actual binning in CLSim is performed using float32, so we
    # first "truncate" all values to that precision. However, the `LinearAxis`
    # function requires Python floats (which are 64 bits), so we have to
    # convert all values to to `float` when passing as kwargs to `LinearAxis`
    # (and presumably the values will be re-truncated to float32 within the
    # CLsim code somewhere). Hopefully following this procedure, the values
    # actually used within CLSim are what we want...? CLSim is stupid.
    ftype = np.float32

    if coordinate_system == 'spherical':
        binning['t_min'] = ftype(0) # ns
        binning['r_min'] = ftype(0) # meters
        costheta_min = ftype(-1.0)
        costheta_max = ftype(1.0)
        # See
        #   clsim/resources/kernels/spherical_coordinates.c.cl
        # in the branch you're using to check that the following are correct
        phi_min = ftype(3.0543261766433716e-01)
        phi_max = ftype(6.5886182785034180e+00)
        binning['costhetadir_min'] = ftype(-1.0)
        binning['costhetadir_max'] = ftype(1.0)
        binning['deltaphidir_min'] = ftype(-3.1808626651763916e+00)
        binning['deltaphidir_max'] = ftype(3.1023228168487549e+00)

        if binning['n_r_bins'] > 0:
            assert isinstance(binning['r_power'], Integral) and binning['r_power'] > 0
            r_binning_kw = OrderedDict([
                ('min', float(binning['r_min'])),
                ('max', float(binning['r_max'])),
                ('n_bins', int(binning['n_r_bins'])),
            ])
            if binning['r_power'] == 1:
                axes['r'] = LinearAxis(**r_binning_kw)
            else:
                r_binning_kw['power'] = int(binning['r_power'])
                axes['r'] = PowerAxis(**r_binning_kw)
            binning_kw['r'] = r_binning_kw

        if binning['n_costheta_bins'] > 0:
            costheta_binning_kw = OrderedDict([
                ('min', float(costheta_min)),
                ('max', float(costheta_max)),
                ('n_bins', int(binning['n_costheta_bins'])),
            ])
            axes['costheta'] = LinearAxis(**costheta_binning_kw)
            binning_kw['costheta'] = costheta_binning_kw

        if binning['n_phi_bins'] > 0:
            phi_binning_kw = OrderedDict([
                ('min', float(phi_min)),
                ('max', float(phi_max)),
                ('n_bins', int(binning['n_phi_bins'])),
            ])
            axes['phi'] = LinearAxis(**phi_binning_kw)
            binning_kw['phi'] = phi_binning_kw

        if binning['n_t_bins'] > 0:
            assert isinstance(binning['t_power'], Integral) and binning['t_power'] > 0
            t_binning_kw = OrderedDict([
                ('min', float(binning['t_min'])),
                ('max', float(binning['t_max'])),
                ('n_bins', int(binning['n_t_bins'])),
            ])
            if binning['t_power'] == 1:
                axes['t'] = LinearAxis(**t_binning_kw)
            else:
                t_binning_kw['power'] = int(binning['t_power'])
                axes['t'] = PowerAxis(**t_binning_kw)
            binning_kw['t'] = t_binning_kw

        if binning['n_costhetadir_bins'] > 0:
            costhetadir_binning_kw = OrderedDict([
                ('min', float(binning['costhetadir_min'])),
                ('max', float(binning['costhetadir_max'])),
                ('n_bins', int(binning['n_costhetadir_bins'])),
            ])
            axes['costhetadir'] = LinearAxis(**costhetadir_binning_kw)
            binning_kw['costhetadir'] = costhetadir_binning_kw

        if binning['n_deltaphidir_bins'] > 0:
            assert (
                isinstance(binning['deltaphidir_power'], Integral)
                and binning['deltaphidir_power'] > 0
            )
            deltaphidir_binning_kw = OrderedDict([
                ('min', float(binning['deltaphidir_min'])),
                ('max', float(binning['deltaphidir_max'])),
                ('n_bins', int(binning['n_deltaphidir_bins'])),
            ])
            if binning['deltaphidir_power'] == 1:
                axes['deltaphidir'] = LinearAxis(**deltaphidir_binning_kw)
            else:
                deltaphidir_binning_kw['power'] = int(binning['deltaphidir_power'])
                axes['deltaphidir'] = PowerAxis(**deltaphidir_binning_kw)
            binning_kw['deltaphidir'] = deltaphidir_binning_kw

    elif coordinate_system == 'cartesian':
        binning['t_min'] = ftype(0) # ns
        binning['costhetadir_min'], binning['costhetadir_max'] = ftype(-1.0), ftype(1.0)
        binning['phidir_min'], binning['phidir_max'] = ftype(-np.pi), ftype(np.pi) # rad

        if binning['n_x_bins'] > 0:
            x_binning_kw = OrderedDict([
                ('min', float(binning['x_min'])),
                ('max', float(binning['x_max'])),
                ('n_bins', int(binning['n_x_bins'])),
            ])
            axes['x'] = LinearAxis(**x_binning_kw)
            binning_kw['x'] = x_binning_kw

        if binning['n_y_bins'] > 0:
            y_binning_kw = OrderedDict([
                ('min', float(binning['y_min'])),
                ('max', float(binning['y_max'])),
                ('n_bins', int(binning['n_y_bins'])),
            ])
            axes['y'] = LinearAxis(**y_binning_kw)
            binning_kw['y'] = y_binning_kw

        if binning['n_z_bins'] > 0:
            z_binning_kw = OrderedDict([
                ('min', float(binning['z_min'])),
                ('max', float(binning['z_max'])),
                ('n_bins', int(binning['n_z_bins'])),
            ])
            axes['z'] = LinearAxis(**z_binning_kw)
            binning_kw['z'] = z_binning_kw

        if binning['n_t_bins'] > 0:
            assert isinstance(binning['t_power'], Integral) and binning['t_power'] > 0
            t_binning_kw = OrderedDict([
                ('min', float(binning['t_min'])),
                ('max', float(binning['t_max'])),
                ('n_bins', int(binning['n_t_bins'])),
            ])
            if binning['t_power'] == 1:
                axes['t'] = LinearAxis(**t_binning_kw)
            else:
                t_binning_kw['power'] = int(binning['t_power'])
                axes['t'] = PowerAxis(**t_binning_kw)
            binning_kw['t'] = t_binning_kw

        if binning['n_costhetadir_bins'] > 0:
            costhetadir_binning_kw = OrderedDict([
                ('min', float(binning['costhetadir_min'])),
                ('max', float(binning['costhetadir_max'])),
                ('n_bins', int(binning['n_costhetadir_bins'])),
            ])
            axes['costhetadir'] = LinearAxis(**costhetadir_binning_kw)
            binning_kw['costhetadir'] = costhetadir_binning_kw

        if binning['n_phidir_bins'] > 0:
            phidir_binning_kw = OrderedDict([
                ('min', float(binning['phidir_min'])),
                ('max', float(binning['phidir_max'])),
                ('n_bins', int(binning['n_phidir_bins'])),
            ])
            axes['phidir'] = LinearAxis(**phidir_binning_kw)
            binning_kw['phidir'] = phidir_binning_kw

    binning_order = BINNING_ORDER[coordinate_system]

    missing_dims = set(axes.keys()).difference(binning_order)
    if missing_dims:
        raise ValueError(
            '`binning_order` specified is {} but is missing dimension(s) {}'
            .format(binning_order, missing_dims)
        )

    axes_ = OrderedDict()
    binning_kw_ = OrderedDict()
    for dim in binning_order:
        if dim in axes:
            axes_[dim] = axes[dim]
            binning_kw_[dim] = binning_kw[dim]
    axes = axes_
    binning_kw = binning_kw_

    # NOTE: use SphericalAxes even if we're actually binning Cartesian since we
    # don't care how it handles e.g. volumes, and Cartesian isn't implemented
    # in CLSim yet
    axes = SphericalAxes(axes.values())

    # Construct metadata initially with items that will be hashed
    metadata = OrderedDict([
        ('source_gcd_i3_md5', gcd_info['source_gcd_i3_md5']),
        ('coordinate_system', coordinate_system),
        ('binning_kw', binning_kw),
        ('ice_model', ice_model),
        ('angular_sensitivity', angular_sensitivity),
        ('disable_tilt', disable_tilt),
        ('disable_anisotropy', disable_anisotropy)
    ])
    # TODO: this is hard-coded in our branch of CLSim; make parameter & fix here!
    if 't' in binning:
        metadata['t_is_residual_time'] = True

    if tableset_hash is None:
        hash_val = hash_obj(metadata, fmt='hex')[:8]
        print('derived hash:', hash_val)
    else:
        hash_val = tableset_hash
        print('tableset_hash:', hash_val)
    metadata['hash_val'] = hash_val
    if tile is not None:
        metadata['tile'] = tile

    dom_spec = OrderedDict([('string', string), ('dom', dom)])

    if 'depth_idx' in dom_spec and ('subdet' in dom_spec or 'string' in dom_spec):
        if 'subdet' in dom_spec:
            dom_spec['string'] = dom_spec.pop('subdet')

        string = dom_spec['string']
        depth_idx = dom_spec['depth_idx']

        if isinstance(string, str):
            subdet = dom_spec['subdet'].lower()
            dom_x, dom_y = 0, 0

            ic_avg_z, dc_avg_z = get_average_dom_z_coords(gcd_info['geo'])
            if string == 'ic':
                dom_z = ic_avg_z[depth_idx]
            elif string == 'dc':
                dom_z = dc_avg_z[depth_idx]
            else:
                raise ValueError('Unrecognized subdetector {}'.format(subdet))
        else:
            dom_x, dom_y, dom_z = gcd_info['geo'][string - 1, depth_idx]

        metadata['string'] = string
        metadata['depth_idx'] = depth_idx

        if tile is not None:
            raise ValueError(
                'Cannot produce tiled tables using "depth_idx"-style table groupings;'
                ' use "string"/"dom"-style tables instead.'
            )

        clsim_table_fname_proto = CLSIM_TABLE_FNAME_PROTO[1]
        clsim_table_metaname_proto = CLSIM_TABLE_METANAME_PROTO[0]

        print('Subdetector {}, depth index {} (z_avg = {} m)'
              .format(subdet, depth_idx, dom_z))

    elif 'string' in dom_spec and 'dom' in dom_spec:
        string = dom_spec['string']
        dom = dom_spec['dom']
        dom_x, dom_y, dom_z = gcd_info['geo'][string - 1, dom - 1]

        metadata['string'] = string
        metadata['dom'] = dom

        if tile is None:
            clsim_table_fname_proto = CLSIM_TABLE_FNAME_PROTO[2]
            clsim_table_metaname_proto = CLSIM_TABLE_METANAME_PROTO[1]
        else:
            clsim_table_fname_proto = CLSIM_TABLE_TILE_FNAME_PROTO[-1]
            clsim_table_metaname_proto = CLSIM_TABLE_TILE_METANAME_PROTO[-1]

        print('GCD = "{}"\nString {}, dom {}: (x, y, z) = ({}, {}, {}) m'
              .format(gcd, string, dom, dom_x, dom_y, dom_z))

    else:
        raise ValueError('Cannot understand `dom_spec` {}'.format(dom_spec))

    # Until someone figures out DOM tilt and ice column / bubble column / cable
    # orientations for sure, we'll just set DOM orientation to zenith=pi,
    # azimuth=0.
    dom_zenith = np.pi
    dom_azimuth = 0.0

    # Now add other metadata items that are useful but not used for hashing
    metadata['dom_x'] = dom_x
    metadata['dom_y'] = dom_y
    metadata['dom_z'] = dom_z
    metadata['dom_zenith'] = dom_zenith
    metadata['dom_azimuth'] = dom_azimuth
    metadata['seed'] = seed
    metadata['n_events'] = n_events

    metapath = join(outdir, clsim_table_metaname_proto.format(**metadata))
    tablepath = join(outdir, clsim_table_fname_proto.format(**metadata))

    # Save metadata as a JSON file (so it's human-readable by any tool, not
    # just Python--in contrast to e.g. pickle files)
    json.dump(metadata, file(metapath, 'w'), sort_keys=False, indent=4)

    print('='*80)
    print('Metadata for the table set was written to\n  "{}"'.format(metapath))
    print('Table will be written to\n  "{}"'.format(tablepath))
    print('='*80)

    exists_at = []
    for fpath in [tablepath, tablepath + '.zst']:
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

    tray = I3Tray()
    tray.AddSegment(
        TabulateRetroSources,
        'TabulateRetroSources',
        source_gcd_i3_md5=gcd_info['source_gcd_i3_md5'],
        binning_kw=binning_kw,
        axes=axes,
        ice_model=ice_model,
        angular_sensitivity=angular_sensitivity,
        disable_tilt=disable_tilt,
        disable_anisotropy=disable_anisotropy,
        hash_val=hash_val,
        dom_spec=dom_spec,
        dom_x=dom_x,
        dom_y=dom_y,
        dom_z=dom_z,
        dom_zenith=dom_zenith,
        dom_azimuth=dom_azimuth,
        seed=seed,
        n_events=n_events,
        tablepath=tablepath,
        tile=tile,
        record_errors=False,
    )

    logging.set_level_for_unit('I3CLSimStepToTableConverter', 'TRACE')
    logging.set_level_for_unit('I3CLSimTabulatorModule', 'DEBUG')
    logging.set_level_for_unit('I3CLSimLightSourceToStepConverterGeant4', 'TRACE')
    logging.set_level_for_unit('I3CLSimLightSourceToStepConverterFlasher', 'TRACE')

    tray.Execute()
    tray.Finish()

    if compress:
        print('Compressing table with zstandard via command line')
        print('  zstd -1 --rm "{}"'.format(tablepath))
        subprocess.check_call(['zstd', '-1', '--rm', tablepath])
        print('done.')


def parse_args(description=__doc__):
    """Parese command line args.

    Returns
    -------
    args : Namespace

    """
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--outdir', required=True,
        help='Save table to this directory (default: "./")'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite if the table already exists'
    )
    parser.add_argument(
        '--compress', action='store_true',
        help='Compress the table with zstd when complete'
    )

    parser.add_argument(
        '--gcd', required=True
    )
    parser.add_argument(
        '--ice-model', required=True
    )
    parser.add_argument(
        '--angular-sensitivity', required=True
    )

    parser.add_argument(
        '--disable-tilt', action='store_true',
        help='Force no tilt, even if ice model contains tilt'
    )
    parser.add_argument(
        '--disable-anisotropy', action='store_true',
        help='Force no anisotropy, even if ice model contains anisotropy'
    )

    parser.add_argument(
        '--string', type=int, required=True,
        help='String number in [1, 86]'
    )
    parser.add_argument(
        '--dom', type=int, required=True,
        help='''DOM number on string, in [1, 60]'''
    )
    parser.add_argument(
        '--n-events', type=int, required=True,
        help='Number of events to simulate'
    )
    parser.add_argument(
        '--seed', type=int, required=True,
        help='Random seed to use, in range of 32 bit uint: [0, 2**32-1]'
    )

    subparsers = parser.add_subparsers(
        dest='coordinate_system',
        help='''Choose the coordinate system for binning: "spherical" or
        "cartesian"'''
    )

    # -- Spherical (phi optional) + time + directionality binning -- #

    sph_parser = subparsers.add_parser(
        'spherical',
        help='Use spherical binning about the DOM',
    )

    sph_parser.add_argument(
        '--n-r-bins', type=int, required=True,
        help='Number of radial bins'
    )
    sph_parser.add_argument(
        '--n-costheta-bins', type=int, required=True,
        help='Number of costheta (cosine of position zenith angle) bins'
    )
    sph_parser.add_argument(
        '--n-phi-bins', type=int, required=True,
        help='Number of phi (position azimuth) bins'
    )
    sph_parser.add_argument(
        '--n-t-bins', type=int, required=True,
        help='Number of time bins (relative to direct time)'
    )
    sph_parser.add_argument(
        '--n-costhetadir-bins', type=int, required=True,
        help='Number of costhetadir bins'
    )
    sph_parser.add_argument(
        '--n-deltaphidir-bins', type=int, required=True,
        help='''Number of deltaphidir bins (Note: span from 0 to pi; code
        assumes symmetry about 0)'''
    )

    sph_parser.add_argument(
        '--r-max', type=float, required=False,
        help='Radial binning maximum value, in meters'
    )
    sph_parser.add_argument(
        '--r-power', type=int, required=False,
        help='Radial binning is regular in raidus to this power'
    )

    sph_parser.add_argument(
        '--deltaphidir-power', type=int, required=False,
        help='deltaphidir binning is regular in deltaphidir to this power'
    )

    sph_parser.add_argument(
        '--t-max', type=float, required=False,
        help='Time binning maximum value, in nanoseconds'
    )
    sph_parser.add_argument(
        '--t-power', type=int, required=False,
        help='Time binning is regular in time to this power'
    )

    # -- Cartesian + (optional time) + directionality binning -- #

    cart_parser = subparsers.add_parser(
        'cartesian',
        help='Use Cartesian binning in IceCube coord system',
    )

    cart_parser.add_argument(
        '--tableset-hash', required=False,
        help='''Hash for a larger table(set) of which this is one tile (i.e.,
        if --tile is provided)'''
    )
    cart_parser.add_argument(
        '--tile', type=int, required=False,
        help='Tile number; provide if this is a tile in a larger table'
    )

    cart_parser.add_argument(
        '--n-x-bins', type=int, required=True,
        help='Number of x bins'
    )
    cart_parser.add_argument(
        '--n-y-bins', type=int, required=True,
        help='Number of y bins'
    )
    cart_parser.add_argument(
        '--n-z-bins', type=int, required=True,
        help='Number of z bins'
    )
    cart_parser.add_argument(
        '--n-t-bins', type=int, required=True,
        help='Number of time bins (relative to direct time)'
    )
    cart_parser.add_argument(
        '--n-costhetadir-bins', type=int, required=True,
        help='Number of costhetadir bins'
    )
    cart_parser.add_argument(
        '--n-phidir-bins', type=int, required=True,
        help='''Number of phidir bins (Note: span from -pi to pi)'''
    )

    # -- Binning limits -- #

    cart_parser.add_argument(
        '--x-min', type=float, required=False,
        help='x binning minimum value, IceCube coordinate system, in meters'
    )
    cart_parser.add_argument(
        '--x-max', type=float, required=False,
        help='x binning maximum value, IceCube coordinate system, in meters'
    )

    cart_parser.add_argument(
        '--y-min', type=float, required=False,
        help='y binning minimum value, IceCube coordinate system, in meters'
    )
    cart_parser.add_argument(
        '--y-max', type=float, required=False,
        help='y binning maximum value, IceCube coordinate system, in meters'
    )

    cart_parser.add_argument(
        '--z-min', type=float, required=False,
        help='z binning minimum value, IceCube coordinate system, in meters'
    )
    cart_parser.add_argument(
        '--z-max', type=float, required=False,
        help='z binning maximum value, IceCube coordinate system, in meters'
    )

    cart_parser.add_argument(
        '--t-max', type=float, required=False,
        help='Time binning maximum value, in nanoseconds'
    )
    cart_parser.add_argument(
        '--t-power', type=int, required=False,
        help='Time binning is regular in time to this power'
    )

    all_kw = vars(parser.parse_args())

    general_kw = OrderedDict()
    for key in (
        'outdir',
        'overwrite',
        'compress',
        'gcd',
        'ice_model',
        'angular_sensitivity',
        'disable_tilt',
        'disable_anisotropy',
        'string',
        'dom',
        'n_events',
        'seed',
        'coordinate_system',
        'tableset_hash',
        'tile',
    ):
        if key in all_kw:
            general_kw[key] = all_kw.pop(key)
    binning = all_kw

    return general_kw, binning


if __name__ == '__main__':
    _general_kw, _binning = parse_args()
    generate_clsim_table(binning=_binning, **_general_kw)
