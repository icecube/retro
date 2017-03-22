#!/usr/bin/env python
#
# Copyright (c) 2012, 2015
# Jakob van Santen <jvansanten@icecube.wisc.edu>
# and the IceCube Collaboration <http://www.icecube.wisc.edu>
# 
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# 
# 
# $Id: photomc.py 136148 2015-08-12 17:46:32Z jvansanten $
# 
# @file photomc.py
# @version $LastChangedRevision: 136148 $
# @date $Date: 2015-08-12 13:46:32 -0400 (Wed, 12 Aug 2015) $
# @author Jakob van Santen

"""
Tabulate the photon flux from a light source in South Pole ice.
"""

from optparse import OptionParser
from icecube.icetray import I3Units
from os import path, unlink

usage = "usage: %prog [options] outputfile"
parser = OptionParser(usage, description=__doc__)

parser.add_option("--seed", dest="seed", type="int", default=None,
    help="Seed for random number generators; harvested from /dev/random if unspecified.")
parser.add_option("--record-errors", dest="errors", action="store_true", default=False,
    help="Record both weights and squares of weights (useful for error bars)")
parser.add_option("--step", dest="steplength", type="float", default=1,
    help="Sampling step length in meters [%default]")
parser.add_option("--overwrite", dest="overwrite", action="store_true", default=False,
    help="Overwrite output file if it already exists")
    
opts, args = parser.parse_args()

if len(args) != 1:
	parser.error("You must specify an output file!")
outfile = args[0]
if path.exists(outfile):
	if opts.overwrite:
		unlink(outfile)
	else:
		parser.error("Output file exists! Pass --overwrite to overwrite it.")

from icecube import icetray
from icecube.clsim.tablemaker.tabulator import TabulatePhotonsFromSource, generate_seed

outfile = args[0]
if opts.seed is None:
	opts.seed = generate_seed()

from I3Tray import I3Tray

tray = I3Tray()

icetray.logging.set_level_for_unit('I3CLSimStepToTableConverter', 'TRACE')
icetray.logging.set_level_for_unit('I3CLSimTabulatorModule', 'DEBUG')
icetray.logging.set_level_for_unit('I3CLSimLightSourceToStepConverterGeant4', 'TRACE')
icetray.logging.set_level_for_unit('I3CLSimLightSourceToStepConverterFlasher', 'TRACE')


# axes?
axes = None

#PhotonSource = ('cascade', 'flasher', 'infinite-muon', 'muon-segment')
# prescale : Only propagate 1/PRESCALE of photons. This is useful for controlling
# how many photons are simulated per source, e.g. for infinite muons where 
# multiple trajectories need to be sampled

#E, zen, z of lightsource (Energy (GeV), zenith angle and z position

tray.AddSegment(TabulatePhotonsFromSource,
                'generator',
                Seed=opts.seed,
                PhotonSource='retro',
                Zenith=180. * I3Units.degree,
                ZCoordinate=-200.,
                Energy=1., 
                NEvents=25000, 
                Filename=outfile,
                TabulateImpactAngle=True,
                PhotonPrescale=1,
                RecordErrors=opts.errors,
                FlasherWidth=127,
                FlasherBrightness=127,
                DisableTilt=True,
                IceModel='spice_mie',
                Axes=axes,
                Sensor='none')
    
tray.AddModule('TrashCan', 'MemoryHole')
tray.Execute()
tray.Finish()
