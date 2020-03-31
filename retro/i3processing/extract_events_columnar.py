#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order

"""
Extract information on events into columnar storage (npy arrays)
"""

from __future__ import absolute_import, division, print_function

__author__ = "P. Eller, J.L. Lanfranchi"
__license__ = """Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = []

from argparse import ArgumentParser
from collections import OrderedDict
try:
    from collections import Iterable, Sequence
except ImportError:
    from collections.abc import Iterable, Sequence
from copy import deepcopy
from os.path import abspath, basename, dirname, join
import pickle
import re
import sys
import traceback

import numpy as np

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import (
    I3POSITION_T,
    I3DIRECTION_T,
    I3PARTICLEID_T,
    I3PARTICLE_T,
    FLAT_PARTICLE_T,
    I3TIME_T,
    I3EVENTHEADER_T,
    OMKEY_T,
)
from retro.utils.misc import (
    dict2struct, expand, get_file_md5, mkdir, set_explicit_dtype
)





def _extract_event(pframe):
    from icecube import dataclasses, genie_icetray, icetray, recclasses, simclasses  # pylint: disable=unused-import


def _extract_events_from_single_file(pframe):
    from icecube.dataio import I3FrameSequence  # pylint: disable=no-name-in-module
    from icecube.icetray import I3Frame  # pylint: disable=no-name-in-module
    from retro.i3processing.extract_gcd import MD5_HEX_RE, extract_gcd_frames

