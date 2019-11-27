#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Example IceTray script for performing reconstructions
"""

from __future__ import absolute_import, division, print_function

__author__ = "P. Eller"
__license__ = """Copyright 2017-2019 Justin L. Lanfranchi and Philipp Eller

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
from os.path import abspath, dirname
import sys

from icecube import dataclasses, icetray, dataio  # pylint: disable=unused-import
from I3Tray import I3Tray

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import __version__, init_obj
from retro.reco import Reco


def main():
    """Script to run Retro recos in icetray"""
    parser = ArgumentParser()

    parser.add_argument(
        "--input-i3-file", type=str,
        required=True,
        nargs="+",
        help="""Input I3 file""",
    )

    parser.add_argument(
        "--output-i3-file", type=str,
        required=True,
        help="""Output I3 file""",
    )

    split_kwargs = init_obj.parse_args(dom_tables=True, tdi_tables=True, parser=parser)

    other_kw = split_kwargs.pop("other_kw")

    # instantiate Retro reco object
    my_reco = Reco(**split_kwargs)

    tray = I3Tray()

    tray.AddModule(
        _type="I3Reader",
        _name="reader",
        FilenameList=other_kw["input_i3_file"],
    )

    tray.Add(
        _type=my_reco,
        _name="retro",
        methods="crs_prefit",
        reco_pulse_series_name="SRTTWOfflinePulsesDC",
        hit_charge_quant=0.05,
        min_hit_charge=0.25,
        seeding_recos=["L5_SPEFit11", "LineFit_DC"],
        triggers=["I3TriggerHierarchy"],
        additional_keys=["L5_oscNext_bool"],
        filter='event["header"]["L5_oscNext_bool"] and len(event["hits"]) >= 8',
        point_estimator="median",
    )

    tray.AddModule(
        _type="I3Writer",
        _name="writer",
        DropOrphanStreams=[icetray.I3Frame.DAQ],
        filename=other_kw["output_i3_file"],
    )

    tray.AddModule(_type="TrashCan", _name="GoHomeYouReDrunk")
    tray.Execute()
    tray.Finish()


if __name__ == "__main__":
    main()
