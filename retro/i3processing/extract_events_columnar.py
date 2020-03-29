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
    I3POSITION_T, I3DIRECTION_T, I3PARTICLEID_T, I3PARTICLE_T, FLAT_PARTICLE_T
)
from retro.utils.misc import (
    dict2struct, expand, get_file_md5, mkdir, set_explicit_dtype
)


def i3particleid_to_np(id):
    return np.array([(id.majorID, id.minorID)], dtype=I3PARTICLEID_T)


def i3position_to_np(pos):
    return np.array([(pos.x, pos.y, pos.z)], dtype=I3POSITION_T)


def i3direction_to_np(dir):
    return np.array([(dir.azimuth, dir.zenith)], dtype=I3DIRECTION_T)


def i3particle_to_np(particle):
    return np.array(
        [
            (
                i3particleid_to_np(particle.id),
                particle.pdg_encoding,
                particle.shape,
                i3position_to_np(particle.pos),
                i3direction_to_np(particle.dir),
                particle.time,
                particle.energy,
                particle.length,
                particle.speed,
                particle.fit_status,
                particle.location_type,
            )
        ],
        dtype=I3PARTICLE_T,
    )


def get_daughters(mctree, parent, parent_idx, level, max_level, np_particle_list):
    """
    Parameters
    ----------
    mctree : icecube.dataclasses.I3MCTree
    parent : icecube.dataclasses.I3Particle
    parent_idx : int
    level : int
    max_level : int
    np_particle_list : appendable sequence

    """
    if max_level >= 0 and level > max_level:
        return

    if parent:
        daughters = mctree.get_daughters(parent)
    else:
        daughters = mctree.get_primaries()

    if not daughters:
        return

    # Record index before we started appending
    idx0 = len(np_particle_list)

    # First append all daughters found
    for daughter in daughters:
        np_particle = i3particle_to_np(daughter)
        np_particle_list.append((level, parent_idx, np_particle))

    # Now recurse, appending any granddaughters (daughters to these daughters)
    # at the end
    for daughter_idx, daughter in enumerate(daughters, start=idx0):
        get_daughters(
            mctree=mctree,
            parent=daughter,
            parent_idx=daughter_idx,
            level=level + 1,
            max_level=max_level,
            np_particle_list=np_particle_list,
        )


def flatten_mctree(mctree, max_level=-1):
    """
    Parameters
    ----------
    mctree : dataclasses.I3MCTree
        Tree to flatten into a numpy array

    max_level : int, optional
        Recurse to but not beyond `max_level` depth within the tree. Primaries
        are level 0, secondaries level 1, tertiaries level 2, etc. Set to
        negative value to capture all levels.

    Returns
    -------
    flat_particles : shape (n_particles,) numpy.ndarray of dtype FLAT_PARTICLE_T
        All particles in the I3MCtree are recorded, up to but not exceeding
        `max_level` recursion depth in the tree.

    """
    np_particle_list = []
    get_daughters(
        mctree=mctree,
        parent=None,
        parent_idx=-1,
        level=0,
        max_level=max_level,
        np_particle_list=np_particle_list,
    )
    return np.array(tuple(np_particle_list), dtype=FLAT_PARTICLE_T)
