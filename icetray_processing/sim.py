#!/usr/bin/env python
#!/cvmfs/i3.opensciencegrid.org/py2-v2/icetray-start


"""
Simulate a bare muon for testing
"""


from __future__ import print_function

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from os.path import expandvars

import numpy as np

from I3Tray import I3Tray
from icecube import clsim
from icecube.dataclasses import I3Geometry, I3Calibration, I3DetectorStatus, \
        I3OMGeo, I3Orientation, I3DOMCalibration, I3DOMStatus, I3Particle, \
        I3Position, I3Direction, I3MCTree
from icecube.icetray import I3Module, I3Logger, I3LogLevel, I3Frame, I3Units,\
        OMKey
# Import I3 modules to add to the icetray (even if only added by string below)
from icecube import phys_services
from icecube import sim_services


class generateEvent(I3Module):
    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddParameter("I3RandomService", "the service", None)
        self.AddParameter("Type", "", I3Particle.ParticleType.EMinus)
        self.AddParameter("Energy", "", 10.*I3Units.TeV)
        self.AddParameter("NEvents", "", 1)
        self.AddParameter("XCoord", "", 0.)
        self.AddParameter("YCoord", "", 0.)
        self.AddParameter("ZCoord", "", 0.)
        self.AddParameter("Zenith", "", -1.)
        self.AddParameter("Azimuth", "", 0.)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.rs = self.GetParameter("I3RandomService")
        self.particleType = self.GetParameter("Type")
        self.energy = self.GetParameter("Energy")
        self.nEvents = self.GetParameter("NEvents")
        self.xCoord = self.GetParameter("XCoord")
        self.yCoord = self.GetParameter("YCoord")
        self.zCoord = self.GetParameter("ZCoord")
        self.zenith = self.GetParameter("Zenith")
        self.azimuth = self.GetParameter("Azimuth")
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

        frame["I3MCTree"] = mctree

        self.PushFrame(frame)

        self.eventCounter += 1
        if self.eventCounter==self.nEvents:
            self.RequestSuspension()


def run_test(args):
    if args.verbosity == 0:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_WARN)
    elif args.verbosity == 1:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_INFO)
    elif args.verbosity == 2:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_DEBUG)
    elif args.verbosity == 3:
        I3Logger.global_logger.set_level(I3LogLevel.LOG_TRACE)
    else:
        raise ValueError("Unhandled verbosity level: %s", args.verbosity)

    MAX_RUN_NUM = int(1e9)
    assert args.run_num > 0 and args.run_num <= MAX_RUN_NUM

    tray = I3Tray()

    tray.AddService(
        "I3XMLSummaryServiceFactory", "summary",
        OutputFileName=args.xml_file
    )

    # Random number generator
    randomService = phys_services.I3SPRNGRandomService(
        seed=args.seed,
        nstreams=MAX_RUN_NUM,
        streamnum=args.run_num
    )

    # Use a real GCD file for a real-world test
    tray.AddModule(
        "I3InfiniteSource", "streams",
        Prefix=expandvars(args.GCDFILE),
        Stream=I3Frame.DAQ
    )

    tray.AddModule(
        "I3MCEventHeaderGenerator", "gen_header",
        Year=2009,
        DAQTime=158100000000000000,
        RunNumber=args.run_num,
        EventID=1,
        IncrementEventID=True
    )

    tray.AddModule(
        generateEvent, "generateEvent",
        Type=eval("I3Particle.ParticleType." + args.particle_type),
        I3RandomService=randomService,
        NEvents=args.num_events,
        Energy=args.energy*I3Units.GeV,
        XCoord=args.x*I3Units.m,
        YCoord=args.y*I3Units.m,
        ZCoord=args.z*I3Units.m,
        Zenith=np.arccos(args.coszen)*I3Units.rad,
        Azimuth=args.azimuth*I3Units.rad,
    )

    # TODO: does PROPOSAL also propagate MuPlus and TauPlus? the tray segment
    # only adds propagators for MuMinus and TauMinus, so for now just use
    # these.

    if not args.use_geant4:
        # Use PROPOSAL mu/tau propagator (and ? for other things?)
        from icecube.simprod.segments import PropagateMuons
        # Random service for muon propagation
        randomServiceForPropagators = phys_services.I3SPRNGRandomService(
            seed=args.seed,
            nstreams=MAX_RUN_NUM*2,
            streamnum=MAX_RUN_NUM + args.run_num
        )
        tray.AddSegment(
            PropagateMuons, "PROPOSAL_propagator",
            RandomService    = randomServiceForPropagators,
            CylinderRadius   = 800*I3Units.m,
            CylinderLength   = 1600*I3Units.m,
            SaveState        = True,
            InputMCTreeName  = "I3MCTree",
            OutputMCTreeName = "I3MCTree",
            #bremsstrahlung = ,
            #photonuclear_family= ,
            #photonuclear= ,
            #nuclear_shadowing= ,
        )

    tray.AddSegment(
        clsim.I3CLSimMakeHits, "makeCLSimHits",
        UseCPUs                     = args.use_cpu,
        UseGPUs                     = not args.use_cpu,
        UseOnlyDeviceNumber         = args.device,
        MCTreeName                  = "I3MCTree",
        MMCTrackListName            = None,
        MCPESeriesName              = "MCPESeriesMap",
        PhotonSeriesName            = "photons",
        ParallelEvents              = args.max_parallel_events,
        #TotalEnergyToProcess        = 1e12,
        RandomService               = randomService,
        IceModelLocation            = args.ice_model,
        DisableTilt                 = args.no_tilt,
        UnWeightedPhotons           = False,
        StopDetectedPhotons         = True,
        PhotonHistoryEntries        = args.photon_history,
        UseGeant4                   = args.use_geant4,
        CrossoverEnergyEM           = 0,
        CrossoverEnergyHadron       = 0,
        DoNotParallelize            = args.no_parallel,
        DOMOversizeFactor           = args.dom_oversize,
        UnshadowedFraction          = 1.0,
        UseHoleIceParameterization  = args.no_hole_ice,

        ExtraArgumentsToI3CLSimModule = dict(
            enableDoubleBuffering     = not args.single_buffer,
            IgnoreNonIceCubeOMNumbers = False,
            #GenerateCherenkovPhotonsWithoutDispersion = False,
            #WavelengthGenerationBias  = ?,
            #IgnoreMuons               = False,
            #DOMPancakeFactor          = 1.0,
            #Geant4PhysicsListName     = ?
            #Geant4MaxBetaChangePerStep= ?
            #Geant4MaxNumPhotonsPerStep= ?
            doublePrecision           = args.double_precision,
            #FixedNumberOfAbsorptionLengths = np.nan,
            #LimitWorkgroupSize        = 0,
        )
    )

    tray.AddModule(
        "I3Writer", "write",
        CompressionLevel=9,
        filename=args.i3_file
    )

    tray.AddModule("TrashCan", "the can")

    tray.Execute()
    tray.Finish()

    del tray


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Benchmark CLSim performance.",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    # Particle parameters
    parser.add_argument(
        "--particle-type",
        choices=('EMinus', 'EPlus', 'MuMinus', 'MuPlus', 'TauMinus', 'TauPlus'),
        type=str,
        default='MuMinus',
        help='Particle type to propagate'
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=20.0,
        help="Particle energy (GeV)"
    )
    parser.add_argument(
        "-x",
        type=float,
        default=0.0,
        help="Particle start x-coord (meters, in IceCube coordinates)"
    )
    parser.add_argument(
        "-y",
        type=float,
        default=0.0,
        help="Particle start y-coord (meters, in IceCube coordinates)"
    )
    parser.add_argument(
        "-z",
        type=float,
        default=-400.0,
        help="Particle start z-coord (meters, in IceCube coordinates)")
    parser.add_argument(
        "--coszen",
        type=float,
        default=-1.0,
        help="""Particle cos-zenith angle (dir *from which* it came, in IceCube
        coordinates)"""
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=0.0,
        help="""Particle azimuth angle (rad; dir *from which* it came, in IceCube
        coordinates)"""
    )
    # Ice parameters
    parser.add_argument(
        "--ice-model",
        type=str,
        default=expandvars("$I3_SRC/clsim/resources/ice/spice_lea"),
        help="""A clsim ice model file/directory (ice models *will* affect
        performance metrics, always compare using the same model!)"""
    )
    parser.add_argument(
        "--no-tilt",
        action="store_true",
        help="Do NOT use ice layer tilt."
    )
    parser.add_argument(
        "--no-hole-ice",
        action="store_true",
        help="Do NOT use hole ice parameterization."
    )
    parser.add_argument(
        "--use-geant4",
        action="store_true",
        help="Use Geant4"
    )
    parser.add_argument(
        "--dom-oversize",
        type=float,
        default=1.0,
        help="DOM oversize factor"
    )
    parser.add_argument(
        "--outbase",
        type=str,
        default="benchmark",
        help="""Base file name for output files; ".xml" and ".i3.bz2"
        extensions are appended to `OUTBASE` to generate the benchmark summary
        and data file names, respectively."""
    )
    parser.add_argument("-g", "--gcdfile", default=os.getenv('GCDfile'),
		  dest="GCDFILE", help="Read in GCD file")
    parser.add_argument(
        "--num-events",
        type=int,
        default=1,
        help="The number of events per run"
    )
    parser.add_argument(
        "--max-parallel-events",
        type=int,
        default=100,
        help="""maximum number of events(==frames) that will be processed in
        parallel"""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Initial seed for the random number generator"
    )
    parser.add_argument(
        "--run-num",
        type=int,
        default=1,
        help="The run number for this simulation"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Do NOT parallelize"
    )
    parser.add_argument(
        "--single-buffer",
        action="store_true",
        help="Use singlue buffer (i.e., turn off double buffering)."
    )
    parser.add_argument(
        "--use-cpu",
        action="store_true",
        help="Simulate using CPU instead of GPU"
    )
    parser.add_argument(
        "--double-precision",
        action="store_true",
        help="Compute using double precision"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="(GPU) device number"
    )
    parser.add_argument(
        "--photon-history",
        action="store_true",
        help="Store photon history"
    )
    parser.add_argument(
        "-v",
        action="count",
        dest="verbosity",
        default=0,
        help="""Logging verbosity; repeat v for increased verbosity. Levels are
        Default: warn, -v: info, -vv: debug, and -vvv: trace. Note that debug
        and trace are unavailable if the IceCube software was built in release
        mode. See
        http://software.icecube.wisc.edu/documentation/projects/icetray/logging.html
        for more info."""
    )

    args = parser.parse_args()

    if args.device is not None:
        print(" ")
        print(" ** DEVICE selected using the --device command line"
              " option. Only do this if you know what you are doing!")
        print(" ** You should be using the CUDA_VISIBLE_DEVICES and/or"
              " GPU_DEVICE_ORDINAL environment variables instead.")

    #fname = '%s_E=%s_x=%s_y=%s_z=%s_coszen=%s_azimuth=%s'%(args.particle_type,args.energy,args.x,args.y,args.z,args.coszen,args.azimuth)
    #args.xml_file = args.outbase + fname + ".xml"
    #args.i3_file = args.outbase + fname + ".i3.bz2"
    args.xml_file = args.outbase + ".xml"
    args.i3_file = args.outbase + ".i3.bz2"

    run_test(args)

    # Print benchmark info out if companion module is available
    try:
        from extract_info import extract_info
    except ImportError:
        print("Could not import extract_info companion script for printing"
              " results.")
    else:
        extract_info(args.xml_file)
