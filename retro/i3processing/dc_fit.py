#!/usr/bin/env python
#Does the SPEFit on all hits, only DC hits (DC) and only Ice Cube hits (NoDC)

import argparse
import glob
import os.path
import sys
import icecube

from I3Tray import *
from icecube import icetray, dataclasses, hdfwriter, dataio
from icecube import linefit, lilliput, cramer_rao
from icecube.icetray import I3Units
from icecube.filterscripts.offlineL2 import Globals
from icecube.filterscripts.offlineL2.Globals import (which_split, deepcore_wg,
                                                     muon_wg, wimp_wg, cascade_wg,
                                                     fss_wg, fss_wg_finiteReco, ehe_wg, ehe_wg_Qstream)
from icecube.filterscripts.offlineL2.Rehydration import Rehydration, Dehydration
from icecube.filterscripts.offlineL2.level2_Reconstruction_Muon import OfflineMuonReco
icetray.load("SeededRTCleaning", False)


@icetray.traysegment
def SPE(tray, name, Pulses = '', If = lambda f: True, suffix = '',
        LineFit = 'LineFit',
        SPEFitSingle = 'SPEFitSingle',
        SPEFit = 'SPEFit2',
        SPEFitCramerRao = 'SPEFitCramerRao',
        N_iter = 2,
        ):
    
    
    #tray.AddSegment( improvedLinefit.simple, LineFit+suffix, inputResponse = Pulses, fitName = LineFit+suffix, If = If )

    tray.AddSegment( lilliput.I3SinglePandelFitter, SPEFitSingle+suffix, pulses = Pulses, seeds = [LineFit+suffix], If = If )

    if N_iter > 1:
        tray.AddSegment( lilliput.I3IterativePandelFitter, SPEFit+suffix, pulses = Pulses, n_iterations = N_iter, seeds = [ SPEFitSingle+suffix ], If = If )

    #use only first hits.  Makes sense for an SPE likelihood                                            
    tray.AddModule('CramerRao', name + '_' + SPEFitCramerRao + suffix,
                   InputResponse = Pulses,
                   InputTrack = SPEFit+suffix,
                   OutputResult = SPEFitCramerRao+suffix,
                   AllHits = False, # ! doesn't make sense to use all hit for SPE pdf                  
                   DoubleOutput = False, # Default                                                     
                   z_dependent_scatter = True, # Default                                               
                   If = lambda f: ((muon_wg(f) or fss_wg(f)) and which_split(f, split_name='InIceSplit')),
                   )

def DCPulses(omkey, index, pulse):
    if ((omkey.om >10) and (omkey.string >78)) or ((omkey.om > 40) and ((omkey.string == 26) or (omkey.string == 27) or (omkey.string == 45) or (omkey.string == 46) or (omkey.string == 35) or (omkey.string == 36) or (omkey.string == 37))):
        return True
    else:
        return False


def NoDCPulses(omkey, index, pulse):
    if ((omkey.om >10) and (omkey.string >78)) or ((omkey.om > 40) and ((omkey.string == 26) or (omkey.string == 27) or (omkey.string == 45) or (omkey.string == 46) or (omkey.string == 35) or (omkey.string == 36) or (omkey.string == 37))):
        return False
    else:
        return True


def ApplyMask(frame):
    tray = I3Tray()

    if frame.Has('SRTInIcePulses'):
        #print "yes SRTInIcePulses"
        #pulsemap0 = icecube.dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SRTInIcePulses')
        #for i,j in pulsemap0:
            #print(i,j)

        my_mask = icecube.dataclasses.I3RecoPulseSeriesMapMask(frame,'SRTInIcePulses', DCPulses)
        frame.Put('DCMask',my_mask)
        
        my_mask2 = icecube.dataclasses.I3RecoPulseSeriesMapMask(frame,'SRTInIcePulses', NoDCPulses)
        frame.Put('NoDCMask',my_mask2)
        
        #off_pulses = frame['DCMask']
        # print frame.keys()
    else:
        print "No SRTInIcePulses"

def SelectDC(frame):
    pulsemap = icecube.dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'DCMask')
    if (len(pulsemap) == 0):
        return False
    else:
        return True

def SelectNoDC(frame):
    pulsemap = icecube.dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'NoDCMask')
    if (len(pulsemap) == 0):
        return False
    else:
        return True



#for i,j in pulsemap:
            #print(i,j)

      #     print frame.keys()
    

def make_parser():
    """Make the argument parser"""
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-s","--simulation", action="store_true",
        default=False, dest="mc", help="Mark as simulation (MC)")
    parser.add_option("-i", "--input", action="store",
        type="string", default="", dest="infile",
        help="Input i3 file(s)  (use comma separated list for multiple files)")
    parser.add_option("-g", "--gcd", action="store",
        type="string", default="", dest="gcdfile",
        help="GCD file for input i3 file")
    parser.add_option("-o", "--output", action="store",
        type="string", default="", dest="outfile",
        help="Output i3 file")
    parser.add_option("-n", "--num", action="store",
        type="int", default=-1, dest="num",
        help="Number of frames to process")
    parser.add_option("--dstfile", action="store",
        type="string", default=None, dest="dstfile",
        help="DST root file (should be .root)")
    return parser


##############################
#  ICETRAY PROCESSING BELOW  #
##############################
def main(options, stats={}):
# build tray
    tray = I3Tray()
    
    if isinstance(options['infile'],list):
        infiles = [options['gcdfile']]
        infiles.extend(options['infile'])
    else:
        infiles = [options['gcdfile'], options['infile']]
    print 'infiles: ',infiles
    
    # test access to input and output files                                                                                                                               
    for f in infiles:
        if not os.access(f,os.R_OK):
            raise Exception('Cannot read from %s'%f)
        def test_write(f):
            if f:
                try:
                    open(f,'w')
                except IOError:
                    raise Exception('Cannot write to %s'%f)
                finally:
                    os.remove(f)
                    test_write(options['outfile'])
                    

# read input files                                                                                                                                                    
    tray.AddModule( "I3Reader", "Reader")(
        ("Filenamelist", infiles)
        )

    tray.AddSegment(Rehydration, 'rehydrator',
                    dstfile=options['dstfile'],
                    mc=options['mc'],
                    doNotQify=options['mc'],
                    )

#If=which_split(split_name='InIceSplit') & (lambda f: ehe_wg(f))
# If=lambda f: ( which_split(split_name='InIceSplit') and (ehe_wg(f)) )
    
# relic of redoing pole fits. That got taken out.                                                  
# but need to keep doing SRT cleaning for all the filters                                          
    tray.AddModule("I3SeededRTHitMaskingModule",  'North_seededrt',
                   MaxIterations = 3,
                   Seeds = 'HLCcore',
                   InputResponse = 'SplitInIcePulses',
                   OutputResponse = 'SRTInIcePulses',
                   If = lambda f: ( which_split(f, split_name='InIceSplit') and
                                    (deepcore_wg(f) or wimp_wg(f) or
                                     muon_wg(f) or cascade_wg(f) or
                                     ehe_wg(f) or fss_wg(f)) )
                   )

    tray.AddSegment(SPE, name+'SPE', 
                    If = lambda f: ((muon_wg(f) or cascade_wg(f) or wimp_wg(f) or fss_wg(f)) and which_split(f, split_name='InIceSplit')), #Includes every wg events listed in process.py when OfflineMuonReco is called
                    Pulses = "SRTInIcePulses",
                    suffix = "",
                    LineFit = 'LineFit',
                    SPEFitSingle = 'SPEFitSingle',
                    SPEFit = 'SPEFit2',
                    SPEFitCramerRao = 'SPEFit2CramerRao',
                    N_iter = 2)
    
    tray.AddModule(ApplyMask,"name")
    
    tray.AddSegment(SPE, name+'SPE',
                    Pulses = "DCMask",
                    If = lambda f: ((muon_wg(f) or cascade_wg(f) or wimp_wg(f) or fss_wg(f)) and which_split(f, split_name='InIceSplit') and SelectDC(f)),     
                    suffix = "",
                    LineFit = 'LineFitDC',
                    SPEFitSingle = 'SPEFitSingleDC',
                    SPEFit = 'SPEFit2DC',
                    SPEFitCramerRao = 'SPEFit2CramerRaoDC',
                    N_iter = 2)
    
    tray.AddSegment(SPE, name+'SPE',
                    Pulses = "NoDCMask",
                    If = lambda f: ((muon_wg(f) or cascade_wg(f) or wimp_wg(f) or fss_wg(f)) and which_split(f, split_name='InIceSplit') and SelectNoDC(f)),     
                    suffix = "",
                    LineFit = 'LineFitNoDC',
                    SPEFitSingle = 'SPEFitSingleNoDC',
                    SPEFit = 'SPEFit2NoDC',
                    SPEFitCramerRao = 'SPEFit2CramerRaoNoDC',
                    N_iter = 2)
    
   
# write i3 files for further processing
    tray.AddModule( "I3Writer", "EventWriter" ) (
        ( "Filename", options['outfile']),
        ( "Streams", [icetray.I3Frame.DAQ, icetray.I3Frame.Physics]),
        ( "DropOrphanStreams", [icetray.I3Frame.DAQ]),
        )
    
    tray.AddModule("TrashCan", "Bye")
 # make it go                                                                                                                                                          
    if options['num'] >= 0:
        tray.Execute(options['num'])
    else:
        tray.Execute()

    tray.Finish()

    # print more CPU usage info. than speicifed by default                                                                                                                
    tray.PrintUsage(fraction=1.0)
    for entry in tray.Usage():
        stats[entry.key()] = entry.data().usertime

    # clean up forcefully in case we're running this in a loop                                                                                                           
    del tray

if __name__ == '__main__':
    # run as script from the command line                                                                                                                                 
    # get parsed args                                                                                                                                                     
    parser = make_parser()
    (options,args) = parser.parse_args()
    opts = {}
    # convert to dictionary                                                                                                                                               
    for name in parser.defaults:
        value = getattr(options,name)
        if name == 'infile' and ',' in value:
            value = value.split(',') # split into multiple inputs                                                                                                         
        opts[name] = value

    # call main function                                                                                                                                                  
    main(opts)


    
