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





def attrs2np(obj, dtype, convert_to_ndarray=True):
    """Extract attributes of an object (and optionally, recursively, attributes
    of those attributes, etc.) into a numpy.ndarray based on the specification
    provided by `dtype`.

    Parameters
    ----------
    obj
    dtype : numpy.dtype
    convert_to_ndarray : bool, optional

    Returns
    -------
    vals : shape-(1,) numpy.ndarray of dtype `dtype`

    """
    vals = []
    if isinstance(dtype, np.dtype):
        descr = dtype.descr
    elif isinstance(dtype, Sequence):
        descr = dtype
    else:
        raise TypeError("{}".format(dtype))

    for name, sub_dtype in descr:
        val = getattr(obj, name)
        if isinstance(sub_dtype, (str, np.dtype)):
            vals.append(val)
        elif isinstance(sub_dtype, Sequence):
            vals.append(attrs2np(val, sub_dtype, convert_to_ndarray=False))
        else:
            raise TypeError("{}".format(sub_dtype))

    # Numpy converts tuples correctly; lists are interpreted differently
    vals = tuple(vals)

    if convert_to_ndarray:
        vals = np.array([vals], dtype=dtype)

    return vals


def getters2np(obj, dtype, fmt="{}"):
    """

    Examples
    --------
    To get all of the values of an I3PortiaEvent: .. ::

        getters2np(frame["PoleEHESummaryPulseInfo"], dtype=I3PORTIAEVENT_T, fmt="Get{}")

    """
    from icecube import icetray
    vals = []
    for n in dtype.names:
        attr_name = fmt.format(n)
        attr = getattr(obj, attr_name)
        val = attr()
        if isinstance(val, icetray.OMKey):
            val = attrs2np(val, dtype=OMKEY_T)
        vals.append(val)

    return np.array([tuple(vals)], dtype=dtype)


def mapscalar2np(mapping, dtype):
    """Convert a mapping (containing string keys and scalar-typed values) to a
    single-element Numpy array from the values of `mapping`, using keys
    defined by `dtype.names`.

    Use this function if you already know the `dtype` you want to use. Use
    `retro.utils.misc.dict2struct` directly if you do not know the dtype(s) of
    the mapping's values ahead of time.


    Parameters
    ----------
    mapping : mapping from strings to scalars

    dtype : numpy.dtype
        If scalar dtype, convert via `utils.dict2struct`. If structured dtype,
        convert keys specified by the struct field names and values are
        converted according to the corresponding type.


    Returns
    -------
    array : shape-(1,) numpy.ndarray of dtype `dtype`


    See Also
    --------
    dict2struct
        Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
        as you go (i.e., this is not known a priori)

    """
    if hasattr(dtype, "names"):  # structured dtype
        vals = tuple(mapping[name] for name in dtype.names)
    else:  # scalar dtype
        vals = tuple(mapping[key] for key in sorted(mapping.keys()))
    return np.array([vals], dtype=dtype)


def flatten_mctree(
    mctree,
    parent=None,
    parent_idx=-1,
    level=0,
    max_level=-1,
    flat_particles=deepcopy([]),
    convert_to_ndarray=True,
):
    """Flatten an I3MCTree into a sequence of particles with additional
    metadata "level" and "parent" for easily reconstructing / navigating the
    tree structure if need be.

    Parameters
    ----------
    mctree : icecube.dataclasses.I3MCTree
        Tree to flatten into a numpy array

    parent : icecube.dataclasses.I3Particle, optional

    parent_idx : int, optional

    level : int, optional

    max_level : int, optional
        Recurse to but not beyond `max_level` depth within the tree. Primaries
        are level 0, secondaries level 1, tertiaries level 2, etc. Set to
        negative value to capture all levels.

    flat_particles : appendable sequence

    convert_to_ndarray : bool, optional


    Returns
    -------
    flat_particles : list of tuples or ndarray of dtype `FLAT_PARTICLE_T`


    Examples
    --------
    This is a recursive function, with defaults defined for calling simply for
    the typical use case of flattening an entire I3MCTree and producing a
    numpy.ndarray with the results. .. ::

        flat_particles = flatten_mctree(frame["I3MCTree"])

    """
    if max_level < 0 or level <= max_level:
        if parent:
            daughters = mctree.get_daughters(parent)
        else:
            level = 0
            parent_idx = -1
            daughters = mctree.get_primaries()

        if daughters:
            # Record index before we started appending
            idx0 = len(flat_particles)

            # First append all daughters found
            for daughter in daughters:
                np_particle = attrs2np(daughter, I3PARTICLE_T)
                flat_particles.append((level, parent_idx, np_particle))

            # Now recurse, appending any granddaughters (daughters to these
            # daughters) at the end
            for daughter_idx, daughter in enumerate(daughters, start=idx0):
                flatten_mctree(
                    mctree=mctree,
                    parent=daughter,
                    parent_idx=daughter_idx,
                    level=level + 1,
                    max_level=max_level,
                    flat_particles=flat_particles,
                    convert_to_ndarray=False,
                )

    if convert_to_ndarray:
        flat_particles = np.array(flat_particles, dtype=FLAT_PARTICLE_T)

    return flat_particles


def _extract_event(pframe):
    from icecube import dataclasses, genie_icetray, icetray, recclasses, simclasses  # pylint: disable=unused-import


def _extract_events_from_single_file(pframe):
    from icecube.dataio import I3FrameSequence  # pylint: disable=no-name-in-module
    from icecube.icetray import I3Frame  # pylint: disable=no-name-in-module
    from retro.i3processing.extract_gcd import MD5_HEX_RE, extract_gcd_frames

