#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name, import-error

from __future__ import absolute_import, division, print_function

from optparse import OptionParser
from os.path import abspath, dirname
import os
import sys

usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option(
    "-o",
    "--outfile",
    default="./test_output.i3",
    dest="OUTFILE",
    help="Write output to OUTFILE (.i3{.gz} format)"
)
parser.add_option(
    "-i",
    "--infile",
    default="./test_input.i3",
    dest="INFILE",
    help="Read input from INFILE (.i3{.gz} format)"
)
parser.add_option(
    "-r",
    "--runnumber",
    type="int",
    default=1,
    dest="RUNNUMBER",
    help=
    "The run number for this simulation, is used as seed for random generator"
)
parser.add_option(
    "-f",
    "--filenr",
    type="int",
    default=1,
    dest="FILENR",
    help="File number, stream of I3SPRNGRandomService"
)
parser.add_option(
    "-g",
    "--gcdfile",
    default=os.getenv('GCDfile'),
    dest="GCDFILE",
    help="Read in GCD file"
)
parser.add_option(
    "-e",
    "--efficiency",
    type="float",
    default=1.,
    dest="EFFICIENCY",
    help="DOM Efficiency ... the same as UnshadowedFraction"
)
parser.add_option(
    "-n",
    "--noise",
    default="vuvuzela",
    dest="NOISE",
    help="Noise model (vuvuzela/poisson)"
)
parser.add_option(
    "--holeice",
    default=50,
    dest="HOLEICE",
    help="Pick the hole ice parameterization"
)
parser.add_option(
    "-s",
    "--scalehad",
    type="float",
    default=1.,
    dest="SCALEHAD",
    help="Scale light from hadrons"
)

(options, args) = parser.parse_args()
if len(args) != 0:
    crap = "Got undefined options:"
    for a in args:
        crap += a
        crap += " "
    parser.error(crap)

from I3Tray import *

from icecube import icetray, dataclasses, dataio, simclasses # pylint: disable=unused-import
from icecube import ( # pylint: disable=unused-import
    phys_services, sim_services, DOMLauncher, DomTools, clsim,
    trigger_sim
)

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3processing.RemoveLatePhotons import RemoveLatePhotons


def BasicHitFilter(frame):
    hits = 0
    if frame.Has("MCPESeriesMap_new"):
        hits = len(frame.Get("MCPESeriesMap_new"))
    return hits > 0


def BasicDOMFilter(frame):
    if frame.Has("InIceRawData"):
        return len(frame['InIceRawData']) > 0
    return False


tray = I3Tray()

print('Using RUNNR: ', options.RUNNUMBER)
try:
    holeice = int(options.HOLEICE)
except:
    print("Selecting hole ice with string: ")
    holeice = options.HOLEICE
print(holeice, type(holeice))

# Random service
tray.AddService("I3SPRNGRandomServiceFactory", "sprngrandom")(
    ("Seed", options.RUNNUMBER),
    ("StreamNum", options.FILENR),
    ("NStreams", 50129),
    ("instatefile", ""),
    ("outstatefile", ""),
)

# Now fire up the random number generator with that seed
randomService = phys_services.I3SPRNGRandomService(
    seed=options.RUNNUMBER, nstreams=50129, streamnum=options.FILENR
)

### START ###

tray.AddModule('I3Reader', 'reader', FilenameList=[options.INFILE])

####
## Remove photons from neutron decay and other processes that take too long (unimportant)
####

tray.AddModule(
    RemoveLatePhotons,
    "RemovePhotons",
    InputPhotonSeries="photons",
    TimeLimit=1E5
) #nanoseconds

####
## Make hits from photons (change efficiency here already!)
####

#tray.AddModule("I3GeometryDecomposer", "I3ModuleGeoMap")
#from ReduceHadronicLightyield import HadLightyield

#print("Scaling hadrons with: ", options.SCALEHAD)
#tray.AddModule(HadLightyield , "scalecascade",
#                Lightyield = options.SCALEHAD)

tray.AddSegment(
    clsim.I3CLSimMakeHitsFromPhotons,
    "makeHitsFromPhotons",
    MCTreeName="I3MCTree_clsim",
    PhotonSeriesName="photons",
    MCPESeriesName="MCPESeriesMap_new",
    RandomService=randomService,
    DOMOversizeFactor=1.,
    UnshadowedFraction=options.EFFICIENCY,
    UseHoleIceParameterization=holeice
)

#from icecube.BadDomList import bad_dom_list_static
#txtfile = os.path.expandvars('$I3_SRC') + '/BadDomList/resources/scripts/bad_data_producing_doms_list.txt'
#BadDoms = bad_dom_list_static.IC86_bad_data_producing_dom_list(118175, txtfile)
#tray.AddModule(BasicHitFilter, 'FilterNullMCPE', Streams = [icetray.I3Frame.DAQ, icetray.I3Frame.Physics])
#print(BadDoms)
mcpe_to_pmt = "MCPESeriesMap_new"
if options.NOISE == 'poisson':
    load("libnoise-generator")
    tray.AddModule(
        "I3NoiseGeneratorModule",
        "noiseic",
        InputPESeriesMapName=mcpe_to_pmt,
        OutputPESeriesMapName=mcpe_to_pmt + "_withNoise",
        InIce=True,
        IceTop=False,
        EndWindow=10. * I3Units.microsecond,
        StartWindow=10. * I3Units.microsecond,
        IndividualRates=True,
        #DOMstoExclude = BadDoms
    )
    mcpeout = mcpe_to_pmt + '_withNoise'
elif options.NOISE == 'vuvuzela':
    from icecube import vuvuzela
    tray.AddSegment(
        vuvuzela.AddNoise,
        'VuvuzelaNoise',
        InputName=mcpe_to_pmt,
        OutputName=mcpe_to_pmt + "_withNoise",
        #ExcludeList = BadDoms,
        StartTime=-11 * I3Units.microsecond,
        EndTime=11 * I3Units.microsecond,
        DisableLowDTCutoff=True
    )
    mcpeout = mcpe_to_pmt + '_withNoise'
elif options.NOISE == 'none':
    print('\n*******WARNING: Noiseless simulation!!********\n')
    mcpeout = mcpe_to_pmt

else:
    print('Pick a valid noise model!')
    exit()

tray.AddModule(
    "PMTResponseSimulator",
    "rosencrantz",
    Input=mcpeout,
    Output=mcpeout + "_weighted",
    MergeHits=True,
)

tray.AddModule(
    "DOMLauncher",
    "guildenstern",
    Input=mcpeout + "_weighted",
    Output="InIceRawData_unclean",
    UseTabulatedPT=True,
)

tray.AddModule("I3DOMLaunchCleaning", "launchcleaning")(
    ("InIceInput", "InIceRawData_unclean"),
    ("InIceOutput", "InIceRawData"),
    ("FirstLaunchCleaning", False),
    #("CleanedKeys",BadDoms)
)

# Dropping frames without InIceRawData
tray.AddModule(
    BasicDOMFilter,
    'FilterNullInIce',
    Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics]
)
###### triggering
tray.AddModule(
    'Delete',
    'delete_triggerHierarchy',
    Keys=['I3TriggerHierarchy', 'TimeShift']
)

#time_shift_args = { 'I3MCTreeNames': [],
#                    'I3MCPMTResponseMapNames': [],
#                    'I3MCHitSeriesMapNames' : [] }
time_shift_args = {'SkipKeys': []}

#gcd_file = dataio.I3File(options.GCDFILE)
tray.AddSegment(
    trigger_sim.TriggerSim,
    'trig',
    gcd_file=dataio.I3File(options.GCDFILE),
    time_shift_args=time_shift_args,
    run_id=options.RUNNUMBER
)
# Not skipping these keys for now (check what gets dropped in the L2)
skipkeys = [
    "MCPMTResponseMap",
    "MCTimeIncEventID",
    "clsim_stats",
    "InIceRawData_unclean",
]

tray.AddModule(
    "I3Writer",
    "writer",
    #SkipKeys=skipkeys, All of these get thrown out by the L2 anyways ... keep them?
    Filename=options.OUTFILE,
    Streams=[icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
)

tray.AddModule("TrashCan", "adios")

tray.Execute()
tray.Finish()
