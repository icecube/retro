#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example IceTray script for performing reconstructions
"""

from __future__ import absolute_import, division, print_function

__author__ = "P. Eller"
__license__ = """Copyright 2017-2018 Justin L. Lanfranchi and Philipp Eller

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""


from argparse import ArgumentParser
from os.path import abspath, dirname, isdir, isfile, join
import sys

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)

from icecube import dataclasses, icetray, dataio
from I3Tray import *

from retro import __version__, init_obj
from retro.reco import METHODS, Reco

parser = ArgumentParser()

parser.add_argument(
            '--input-i3-file', type=str,
            required=True,
            nargs='+',
            help='''Input I3 file''',
        )

parser.add_argument(
            '--output-i3-file', type=str,
            required=True,
            help='''Output I3 file''',
        )

split_kwargs = init_obj.parse_args(dom_tables=True, tdi_tables=True, parser=parser)

other_kw = split_kwargs.pop("other_kw")

# instantiate Retro reco object
my_reco = Reco(**split_kwargs)

tray = I3Tray()

tray.AddModule('I3Reader', 'reader', FilenameList = other_kw['input_i3_file'])

tray.Add(my_reco, "retro", 
    methods='crs_prefit',
    reco_pulse_series_name='SRTTWOfflinePulsesDC',
    seeding_recos=["L5_SPEFit11", "LineFit_DC"],
    triggers=['I3TriggerHierarchy'],
    additional_keys=['L5_oscNext_bool'],
    filter='event["header"]["L5_oscNext_bool"]',
    point_estimator='median')

tray.AddModule("I3Writer", "writer",
    DropOrphanStreams = [icetray.I3Frame.DAQ],
    Streams = [icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation, icetray.I3Frame.DAQ, icetray.I3Frame.Physics],
    filename = other_kw['output_i3_file'], 
)

tray.AddModule('TrashCan', 'GoHomeYouReDrunk')
tray.Execute()
tray.Finish()

