#!/usr/bin/env python
#!/cvmfs/i3.opensciencegrid.org/py2-v3/icetray-start
# -*- coding: utf-8 -*-
# pylint: disable=unused-import, invalid-name, import-error, attribute-defined-outside-init

"""
Use CLSim to simulate a particle; for now, just charged leptons (muons,
electrons, taus, and their antiparticles)
"""

# TODO: *ONLY* do the phton sim here, make hits in another step to keep this as
#       fast as possible
# TODO: get Geant4 propagation to work (or at least test if it works now...)

from __future__ import absolute_import, division, print_function

__all__ = [
    'MAX_RUN_NUM',
    'simulate_particle',
    'parse_args'
]

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from os.path import expandvars

import numpy as np


MAX_RUN_NUM = int(1e9)


def simulate_particle(
        outfile, run_num, num_events, gcd, particle_type, energy, x, y, z,
        coszen, azimuth, ice_model, no_tilt, hole_ice_model, use_geant4,
        dom_oversize=1, photon_history=False, max_parallel_events=int(1e5),
        device=None, use_cpu=False, double_precision=False,
        single_buffer=False, verbosity=0
    ):
    from icecube.icetray import I3Frame, I3Logger, I3LogLevel, I3Module, I3Units

    from I3Tray import I3Tray
    from icecube import clsim
    from icecube.dataclasses import (
        I3Geometry, I3Calibration, I3DetectorStatus, I3OMGeo, I3Orientation,
        I3DOMCalibration, I3DOMStatus, I3Particle, I3Position, I3Direction,
        I3MCTree
    )

    # Import I3 modules to add to the icetray (even if only added by string below)
    from icecube import phys_services
    from icecube import sim_services

    class GenerateEvent(I3Module):
        def __init__(self, context):
            I3Module.__init__(self, context)
            self.AddParameter('I3RandomService', 'the service', None)
            self.AddParameter('Type', '', I3Particle.ParticleType.EMinus)
            self.AddParameter('Energy', '', 10. * I3Units.TeV)
            self.AddParameter('NEvents', '', 1)
            self.AddParameter('XCoord', '', 0.)
            self.AddParameter('YCoord', '', 0.)
            self.AddParameter('ZCoord', '', 0.)
            self.AddParameter('Zenith', '', -1.)
            self.AddParameter('Azimuth', '', 0.)
            self.AddOutBox('OutBox')

        def Configure(self):
            self.rs = self.GetParameter('I3RandomService')
            self.particleType = self.GetParameter('Type')
            self.energy = self.GetParameter('Energy')
            self.nEvents = self.GetParameter('NEvents')
            self.xCoord = self.GetParameter('XCoord')
            self.yCoord = self.GetParameter('YCoord')
            self.zCoord = self.GetParameter('ZCoord')
            self.zenith = self.GetParameter('Zenith')
            self.azimuth = self.GetParameter('Azimuth')
            self.eventCounter = 0

        def DAQ(self, frame):
            daughter = I3Particle()
            daughter.type = self.particleType
            daughter.energy = self.energy
            daughter.pos = I3Position(self.xCoord, self.yCoord, self.zCoord)
            daughter.dir = I3Direction(self.zenith, self.azimuth)
            daughter.time = 0.
            daughter.location_type = I3Particle.LocationType.InIce

            primary = I3Particle()
            primary.type = I3Particle.ParticleType.NuMu
            primary.energy = self.energy
            primary.pos = I3Position(self.xCoord, self.yCoord, self.zCoord)
            primary.dir = I3Direction(0., 0., -1.)
            primary.time = 0.
            primary.location_type = I3Particle.LocationType.Anywhere

            mctree = I3MCTree()
            mctree.add_primary(primary)
            mctree.append_child(primary, daughter)

            frame['I3MCTree'] = mctree

            self.PushFrame(frame)

            self.eventCounter += 1
            if self.eventCounter == self.nEvents:
                self.RequestSuspension()


    if verbosity == 0:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_WARN)
    elif verbosity == 1:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_INFO)
    elif verbosity == 2:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_DEBUG)
    elif verbosity == 3:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_TRACE)
    else:
        raise ValueError('Unhandled verbosity level: %s' % verbosity)

    assert run_num > 0 and run_num <= MAX_RUN_NUM

    tray = I3Tray()

    # Random number generator
    randomService = phys_services.I3SPRNGRandomService(
        seed=123456,
        nstreams=MAX_RUN_NUM,
        streamnum=run_num
    )

    # Use a real GCD file for a real-world test
    tray.AddModule(
        'I3InfiniteSource',
        'streams',
        Prefix=expandvars(gcd),
        Stream=I3Frame.DAQ
    )

    tray.AddModule(
        'I3MCEventHeaderGenerator',
        'gen_header',
        Year=2009,
        DAQTime=158100000000000000,
        RunNumber=run_num,
        EventID=1,
        IncrementEventID=True
    )

    tray.AddModule(
        GenerateEvent,
        'GenerateEvent',
        Type=eval('I3Particle.ParticleType.' + particle_type),
        I3RandomService=randomService,
        NEvents=num_events,
        Energy=energy * I3Units.GeV,
        XCoord=x * I3Units.m,
        YCoord=y * I3Units.m,
        ZCoord=z * I3Units.m,
        Zenith=np.arccos(coszen) * I3Units.rad,
        Azimuth=azimuth * I3Units.rad,
    )

    # TODO: does PROPOSAL also propagate MuPlus and TauPlus? the tray segment
    # only adds propagators for MuMinus and TauMinus, so for now just use
    # these.

    if not use_geant4:
        # Use PROPOSAL mu/tau propagator (and ? for other things?)
        from icecube.simprod.segments import PropagateMuons
        # Random service for muon propagation
        randomServiceForPropagators = phys_services.I3SPRNGRandomService(
            seed=123456,
            nstreams=MAX_RUN_NUM * 2,
            streamnum=MAX_RUN_NUM + run_num
        )
        tray.AddSegment(
            PropagateMuons,
            'PROPOSAL_propagator',
            RandomService=randomServiceForPropagators,
            CylinderRadius=800 * I3Units.m,
            CylinderLength=1600 * I3Units.m,
            SaveState=True,
            InputMCTreeName='I3MCTree',
            OutputMCTreeName='I3MCTree',
            #bremsstrahlung = ,
            #photonuclear_family= ,
            #photonuclear= ,
            #nuclear_shadowing= ,
        )

    # Version of
    tray.AddSegment(
        clsim.I3CLSimMakeHits,
        'I3CLSimMakeHits',
        UseCPUs=use_cpu,
        UseGPUs=not use_cpu,
        UseOnlyDeviceNumber=device,
        MCTreeName='I3MCTree',
        OutputMCTreeName=None,
        FlasherInfoVectName=None,
        FlasherPulseSeriesName=None,
        MMCTrackListName='MMCTrackList',
        MCPESeriesName='MCPESeriesMap',
        PhotonSeriesName='photons',
        ParallelEvents=max_parallel_events,
        #TotalEnergyToProcess=0.,
        RandomService=randomService,
        IceModelLocation=ice_model,
        DisableTilt=no_tilt,
        UnWeightedPhotons=False,
        UseGeant4=use_geant4,
        #CrossoverEnergyEM=None,
        #CrossoverEnergyHadron=None,
        UseCascadeExtension=True,
        StopDetectedPhotons=True,
        PhotonHistoryEntries=photon_history,
        DoNotParallelize=max_parallel_events <= 1,
        DOMOversizeFactor=dom_oversize,
        UnshadowedFraction=0.9,
        HoleIceParameterization=hole_ice_model,
        ExtraArgumentsToI3CLSimModule=dict(
            enableDoubleBuffering=not single_buffer,
            IgnoreNonIceCubeOMNumbers=False,
            #GenerateCherenkovPhotonsWithoutDispersion = False,
            #WavelengthGenerationBias  = ?,
            #IgnoreMuons               = False,
            #DOMPancakeFactor          = 1.0,
            #Geant4PhysicsListName     = ?
            #Geant4MaxBetaChangePerStep= ?
            #Geant4MaxNumPhotonsPerStep= ?
            doublePrecision=double_precision,
            #FixedNumberOfAbsorptionLengths = np.nan,
            #LimitWorkgroupSize        = 0,
        ),
        If=lambda f: True
    )

    tray.AddModule(
        'I3Writer', 'write', CompressionLevel=9, filename=outfile
    )

    tray.AddModule('TrashCan', 'the can')

    tray.Execute()
    tray.Finish()

    del tray


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    #==========================================================================
    # Script parameters
    #==========================================================================

    parser.add_argument(
        '--outfile', type=str, required=True,
        help='''Name of the file to generate (excluding suffix, which will be
        ".i3.bz2"'''
    )
    parser.add_argument(
        '--run-num', type=int, required=True,
        help='''The run number for this simulation; unique run numbers get
        unique random numbers, so use different run numbers for different
        simulations! (1 <= run num < 1e9)'''
    )
    parser.add_argument(
        '--num-events', type=int, required=True,
        help='The number of events per run'
    )

    #==========================================================================
    # Detector parameters
    #==========================================================================

    parser.add_argument(
        '--gcd', required=True,
        help='Read in GCD file'
    )
    parser.add_argument(
        '--dom-oversize', type=float, default=1.0,
        help='DOM oversize factor'
    )

    #==========================================================================
    # Event parameters
    #==========================================================================

    # Particle params
    parser.add_argument(
        '--particle-type', required=True,
        choices=(
            'EMinus', 'EPlus', 'MuMinus', 'MuPlus', 'TauMinus', 'TauPlus'
        ),
        help='Particle type to propagate'
    )
    parser.add_argument(
        '-e', '--energy',
        type=float,
        help='Particle energy (GeV)'
    )
    parser.add_argument(
        '-x', type=float, required=True,
        help='Particle start x-coord (meters, in IceCube coordinates)'
    )
    parser.add_argument(
        '-y', type=float, required=True,
        help='Particle start y-coord (meters, in IceCube coordinates)'
    )
    parser.add_argument(
        '-z', type=float, required=True,
        help='Particle start z-coord (meters, in IceCube coordinates)'
    )
    parser.add_argument(
        '--coszen', type=float, required=True,
        help='''Particle cos-zenith angle (dir *from which* it came, in IceCube
        coordinates)'''
    )
    parser.add_argument(
        '--azimuth', type=float, required=True,
        help='''Particle azimuth angle (rad; dir *from which* it came, in
        IceCube coordinates)'''
    )

    #==========================================================================
    # Ice parameters
    #==========================================================================

    parser.add_argument(
        '--ice-model', required=True,
        help='''A clsim ice model file/directory (ice models *will* affect
        performance metrics, always compare using the same model!)'''
    )
    parser.add_argument(
        '--no-tilt', action='store_true',
        help='Do NOT use ice layer tilt.'
    )
    parser.add_argument(
        '--hole-ice-model', required=True,
        help='Specify a hole ice parameterization.'
    )

    #==========================================================================
    # Underlying software parameters (i.e., CLSim)
    #==========================================================================

    parser.add_argument(
        '--use-geant4', action='store_true',
        help='Use Geant4'
    )
    parser.add_argument(
        '--max-parallel-events',
        type=int, default=int(1e5),
        help='''maximum number of events(==frames) that will be processed in
        parallel; set to 1 or fewer to disable parallelism.'''
    )
    parser.add_argument(
        '--single-buffer', action='store_true',
        help='Use singlue buffer (i.e., turn off double buffering).'
    )
    parser.add_argument(
        '--use-cpu', action='store_true',
        help='Simulate using CPU instead of GPU'
    )
    parser.add_argument(
        '--double-precision', action='store_true',
        help='Compute using double precision'
    )
    parser.add_argument(
        '--device', type=int, default=None,
        help='(GPU) device number; only used if --use-cpu is NOT specified'
    )
    parser.add_argument(
        '--photon-history', action='store_true',
        help='Store photon history'
    )
    parser.add_argument(
        '-v', action='count', default=0, dest='verbosity',
        help='''Logging verbosity; repeat v for increased verbosity. Levels are
        Default: warn, -v: info, -vv: debug, and -vvv: trace. Note that debug
        and trace are unavailable if the IceCube software was built in release
        mode. See
        http://software.icecube.wisc.edu/documentation/projects/icetray/logging.html
        for more info.'''
    )

    args = parser.parse_args()

    if args.device is not None:
        print(' ')
        print(
            ' ** DEVICE selected using the --device command line'
            ' option. Only do this if you know what you are doing!'
        )
        print(
            ' ** You should be using the CUDA_VISIBLE_DEVICES and/or'
            ' GPU_DEVICE_ORDINAL environment variables instead.'
        )

    if not (args.outfile.endswith('.i3.bz2') or args.outfile.endswith('.i3')):
        args.outfile += '.i3.bz2'

    if args.outfile.endswith('.i3'):
        args.outfile += '.bz2'

    return args


if __name__ == '__main__':
    simulate_particle(**vars(parse_args()))
