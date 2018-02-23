#!/usr/bin/env python                                                                                
 
import argparse
import glob
import os.path
import sys
import icecube
from datetime import datetime

from optparse import OptionParser
from os.path import expandvars
import os, sys, random



usage = "usage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-o", "--outfile",default="./test_output.i3",
                  dest="OUTFILE", help="Write output to OUTFILE (.i3{.gz} format)")
parser.add_option("-i", "--infile",default="./test_input.i3",
                  dest="INFILE", help="Read input from INFILE (.i3{.gz} format)")
parser.add_option("-g", "--gcdfile", default=os.getenv('GCDfile'),
		  dest="GCDFILE", help="Read in GCD file")
(options,args) = parser.parse_args()

from I3Tray import *
from icecube import icetray, dataclasses, hdfwriter, dataio, STTools
from icecube import linefit, lilliput, cramer_rao
from icecube.icetray import I3Units
from icecube.filterscripts.offlineL2 import Globals
from icecube.filterscripts.offlineL2.Globals import  (deepcore_wg,
                                                     muon_wg, wimp_wg, cascade_wg,
                                                     fss_wg, fss_wg_finiteReco, ehe_wg, ehe_wg_Qstream)
from icecube.filterscripts.offlineL2.Rehydration import Rehydration, Dehydration
from icecube.filterscripts.offlineL2.level2_Reconstruction_Muon import OfflineMuonReco
from icecube.phys_services.which_split import which_split


from icecube.STTools.seededRT.configuration_services import I3DOMLinkSeededRTConfigurationService
          
### START ###


#hit_series = 'InIceRawData'
hit_series = 'SplitUncleanedInIcePulses'


tray = I3Tray()

tray.AddModule('I3Reader', 'reader',
            FilenameList = [options.GCDFILE, options.INFILE]
            )


def Data_check(frame):        
    if frame.Has("SRTInIcePulses_200_750_1"):
        return False
    if frame.Has("SRTInIcePulses_90_700_1"):  
      return False
    if frame.Has("SRTInIcePulses_90_700_1_COG"):
        return False
    return True

tray.AddModule(Data_check,"data check")

seededRTConfig1 = I3DOMLinkSeededRTConfigurationService(
    ic_ic_RTRadius              = 200.0*I3Units.m,
    ic_ic_RTTime                = 750.0*I3Units.ns,
    treat_string_36_as_deepcore = False,
    useDustlayerCorrection      = False,
    allowSelfCoincidence        = True
    )

tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'North_seededrt_1',
               InputHitSeriesMapName  = hit_series,
               OutputHitSeriesMapName = 'SRTInIcePulses_200_750_1',
               STConfigService        = seededRTConfig1,
               SeedProcedure          = 'HLCCoreHits',
               NHitsThreshold         = 2,
               MaxNIterations         = 1,
               Streams                = [icetray.I3Frame.Physics],
               If = which_split(split_name='InIceSplit') & (lambda f: (
                deepcore_wg(f) or wimp_wg(f)    or
                muon_wg(f)     or cascade_wg(f) or
                ehe_wg(f)      or fss_wg(f) ))) 


seededRTConfig2 = I3DOMLinkSeededRTConfigurationService(
    ic_ic_RTRadius              = 90.0*I3Units.m,
    ic_ic_RTTime                = 700.0*I3Units.ns,
    treat_string_36_as_deepcore = False,
    useDustlayerCorrection      = False,
    allowSelfCoincidence        = True
    )


tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'North_seededrt_2',
               InputHitSeriesMapName  = hit_series,
               OutputHitSeriesMapName = 'SRTInIcePulses_90_700_1',
               STConfigService        = seededRTConfig2,
               SeedProcedure          = 'HLCCoreHits',
               NHitsThreshold         = 2,
               MaxNIterations         = 1,
               Streams                = [icetray.I3Frame.Physics],   
               If = which_split(split_name='InIceSplit') & (lambda f: (
            deepcore_wg(f) or wimp_wg(f)    or
            muon_wg(f)     or cascade_wg(f) or
            ehe_wg(f)      or fss_wg(f) ))) 


seededRTConfig3 = I3DOMLinkSeededRTConfigurationService(
    ic_ic_RTRadius              = 90.0*I3Units.m,
    ic_ic_RTTime                = 700.0*I3Units.ns,
    treat_string_36_as_deepcore = False,
    useDustlayerCorrection      = False,
    allowSelfCoincidence        = True
    )

tray.AddModule('I3SeededRTCleaning_RecoPulseMask_Module', 'North_seededrt_3',
               InputHitSeriesMapName  = hit_series,
               OutputHitSeriesMapName = 'SRTInIcePulses_90_700_1_COG',
               STConfigService        = seededRTConfig3,
               SeedProcedure          = 'HLCCOGSTHits',
               NHitsThreshold         = 2,
               MaxNIterations         = 1,
               Streams                = [icetray.I3Frame.Physics],
               If = which_split(split_name='InIceSplit') & (lambda f: (
            deepcore_wg(f) or wimp_wg(f)    or
            muon_wg(f)     or cascade_wg(f) or
            ehe_wg(f)      or fss_wg(f) ))) 
    

def Remove1(frame):
    if frame.Has("SRTInIcePulses_200_750_1"):
        
        min0 = 100000
        hits = dataclasses.I3RecoPulseSeriesMap.from_frame(frame,'SRTInIcePulses')
        for item in hits:
            for vals in item.data():
                #            print vals.time
                if  vals.time < min0:
                    min1 = vals.time
                    item0 = item
                
        if frame.Has("SRTInIcePulses_rm1"):
            return False
                
        frame['STRInIcePulses_rm1'] = dataclasses.I3RecoPulseSeriesMapMask(frame, 'SRTInIcePulses', lambda omkey, index, pulse: omkey != item.key() )
                
        return True
    else: 
        return False
            
#    hits2 = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SRTInIcePulses_rm1')
#    print 'AA', hits2.keys() 
#    print 'BB', hits.keys()


#tray.AddModule(Remove1, "RM")

tray.AddModule('I3Writer', 'writer', Filename=options.OUTFILE, Streams=[icetray.I3Frame.DAQ,icetray.I3Frame.Physics], DropOrphanStreams=[icetray.I3Frame.DAQ])                                    
tray.AddModule('TrashCan', 'thecan')                                                              
tray.Execute()                                                                                    
tray.Finish()     
