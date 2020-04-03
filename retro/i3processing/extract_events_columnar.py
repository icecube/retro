#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, wrong-import-order

"""
Extract information on events into columnar storage (npy arrays)
"""

from __future__ import absolute_import, division, print_function

__author__ = "J.L. Lanfranchi"
__license__ = """Copyright 2020 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__all__ = [
    "construct_arrays",
    "index_and_cat_scalar_arrays",
    "cat_vector_arrays",
    "ConvertI3ToNumpy",
]

from argparse import ArgumentParser
from collections import OrderedDict

try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence
from glob import glob
from os import listdir
from os.path import abspath, basename, dirname, isdir, isfile, join
import re
import sys

#import numba
import numpy as np
from six import string_types

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import retro_types as rt
from retro.utils.misc import expand, mkdir, nsort_key_func
from retro.i3processing.extract_common import (
    DATA_GCD_FNAME_RE,
    OSCNEXT_I3_FNAME_RE,
    dict2struct,
    find_gcds_in_dirs,
    maptype2np,
)


RUN_DIR_RE = re.compile(r"(?P<pfx>Run)?(?P<run>[0-9]+)", flags=re.IGNORECASE)
"""Matches MC run dirs, e.g. '140000' & data run dirs, e.g. 'Run00125177'"""


def construct_arrays(data, delete_while_filling=False):
    """Construct arrays to collect same-key scalars / vectors across frames

    Returns
    -------
    scalar_arrays : dict
    vector_arrays : dict

    """
    if isinstance(data, Mapping):
        data = [data]

    # Get type and size info

    scalar_dtypes = {}
    vector_dtypes = {}

    num_frames = len(data)
    for frame_d in data:
        # Must get all vector values for all frames to get both dtype and
        # total length, but only need to get a scalar value once to get its
        # dtype
        for key in set(frame_d.keys()).difference(scalar_dtypes.keys()):
            val = frame_d[key]
            # if val is None:
            #    continue
            dtype = val.dtype

            if np.isscalar(val):
                scalar_dtypes[key] = dtype
            else:
                if key not in vector_dtypes:
                    vector_dtypes[key] = [0, dtype]  # length, type
                vector_dtypes[key][0] += len(val)

    # Construct empty arrays

    scalar_arrays = {}
    vector_arrays = {}

    for key, dtype in scalar_dtypes.items():
        # Until we know we need one (i.e., when an event is missing this
        # `key`), the "valid" mask array is omitted
        scalar_arrays[key] = dict(data=np.empty(shape=num_frames, dtype=dtype))

    # `vector_arrays` contains "data" and "index" arrays.
    # `index` has the same number of entries as the scalar arrays,
    # and each entry points into the corresponding `data` array to
    # determine which vector data correspond to this scalar datum

    for key, (length, dtype) in vector_dtypes.items():
        vector_arrays[key] = dict(
            data=np.empty(shape=length, dtype=dtype),
            index=np.empty(shape=num_frames, dtype=rt.START_AND_LENGTH_T),
        )

    # Fill the arrays

    for frame_idx, frame_d in enumerate(data):
        for key, array_d in scalar_arrays.items():
            val = frame_d.get(key, None)
            if val is None:
                if "valid" not in array_d:
                    array_d["valid"] = np.ones(shape=num_frames, dtype=np.bool8)
                array_d["valid"][frame_idx] = False
            else:
                array_d["data"][frame_idx] = val
                if delete_while_filling:
                    del frame_d[key]

        for key, array_d in vector_arrays.items():
            index = array_d["index"]
            if frame_idx == 0:
                prev_start = 0
                prev_length = 0
            else:
                prev_start, prev_length = index[frame_idx - 1]

            start = int(prev_start + prev_length)

            val = frame_d.get(key, None)
            if val is None:
                index[frame_idx] = (start, 0)
            else:
                length = len(val)
                index[frame_idx] = (start, length)
                array_d["data"][start : start + length] = val
                if delete_while_filling:
                    del index, frame_d[key]

    return scalar_arrays, vector_arrays


def index_and_cat_scalar_arrays(category_array_map, category_dtype=None):
    """A given scalar array might or might not be present in each tup

    Parameters
    ----------
    category_array_map : OrderedDict
        Keys are the categories (e.g., run number or subrun number), values
        are the scalar array dicts (containing keys "data" and, optionally,
        "valid", and values are the actual Numpy arrays)

    category_dtype : numpy.dtype or None, optional
        If None, use numpy type inferred by the operation .. ::

            np.array(list(category_array_map.keys())).dtype

    Returns
    -------
    index : numba.typed.Dict
        Keys are of type `category_dtype`, values are of type
        `retro.retro_types.START_AND_LENGTH_T`.

    arrays : dict of dicts containing arrays

    """
    # Datatype of each data array (same key must have same dtype regardless
    # of which category)
    key_dtypes = OrderedDict()

    # All data arrays in one category have same length as one another;
    # record this length for each category
    category_array_lengths = OrderedDict()

    # Data arrays (and any valid arrays) will have this length
    total_length = 0

    # Record any keys that, for any category, already have a valid array
    # created, as these keys will require valid arrays to be created and
    # filled
    keys_with_valid_arrays = set()

    # Get and validate metadata about arrays

    for category, array_dicts in category_array_map.items():
        array_length = None
        for key, array_d in array_dicts.items():
            data = array_d["data"]
            valid = array_d.get("valid", None)

            if array_length is None:
                array_length = len(data)
            elif len(data) != array_length:
                raise ValueError(
                    "category={}, key={}, ref len={}, this len={}".format(
                        category, key, array_length, len(data)
                    )
                )

            if valid is not None:
                keys_with_valid_arrays.add(key)
                if len(valid) != array_length:
                    raise ValueError(
                        "category={}, key={}, ref len={}, this len={}".format(
                            category, key, array_length, len(data)
                        )
                    )

            dtype = data.dtype
            existing_dtype = key_dtypes.get(key, None)
            if existing_dtype is None:
                key_dtypes[key] = dtype
            elif dtype != existing_dtype:
                raise TypeError(
                    "category={}, key={}, dtype={}, existing_dtype={}".format(
                        category, key, dtype, existing_dtype
                    )
                )

        if array_length is None:
            array_length = 0

        print("category:", category, "array_length:", array_length, "total length", total_length)
        category_array_lengths[category] = array_length
        total_length += array_length
        print("category:", category, "total length", total_length)

    print(category_array_lengths)

    # Create the index; use numba.typed.Dict for direct use in Numba

    categories = np.array(list(category_array_map.keys()), dtype=category_dtype)
    category_dtype = categories.dtype
    #index = numba.typed.Dict.empty(
    #    key_type=numba.from_dtype(category_dtype),
    #    value_type=numba.from_dtype(rt.START_AND_LENGTH_T),
    #)
    index = OrderedDict()

    # Populate the index

    start = 0
    for category, array_length in zip(categories, category_array_lengths.values()):
        value = np.array([(start, array_length)], dtype=rt.START_AND_LENGTH_T)[0]
        index[category] = value
        # TODO: numba.typed.Dict fails:
        # print("value:", value, "ntd value:", index[category])
        start += array_length

    # Record keys that are missing in one or more categories

    all_keys = set(key_dtypes.keys())
    keys_with_missing_data = set()
    for category, array_dicts in category_array_map.items():
        keys_with_missing_data.update(all_keys.difference(array_dicts.keys()))

    # Create and populate `data` arrays and any necessary `valid` arrays

    keys_requiring_valid_array = set.union(
        keys_with_missing_data, keys_with_valid_arrays
    )

    arrays = OrderedDict()
    for key, dtype in key_dtypes.items():
        data = np.empty(shape=total_length, dtype=dtype)
        if key in keys_requiring_valid_array:
            valid = np.empty(shape=total_length, dtype=np.bool8)
        else:
            valid = None

        for category, array_dicts in category_array_map.items():
            start_and_length = index[category]
            start = start_and_length["start"]
            stop = start + start_and_length["length"]

            key_arrays = array_dicts.get(key, None)
            if key_arrays is None:
                valid[start:stop] = False
                continue

            data[start:stop] = key_arrays["data"]
            if "valid" in key_arrays:
                valid[start:stop] = key_arrays["valid"]

        arrays[key] = dict(data=data)
        if valid is not None:
            arrays[key]["valid"] = valid

    return index, arrays


def cat_vector_arrays(category_array_map):
    """Concatenate vector arrays

    Parameters
    ----------
    category_array_map : mapping

    Returns
    -------
    arrays

    """
    key_dtypes = OrderedDict()

    index_total_lengths = {}
    data_total_lengths = {}

    for category, array_dicts in category_array_map.items():
        for key, array_d in array_dicts.items():
            data = array_d["data"]
            scalar_index = array_d["scalar_index"]

            dtype = data.dtype
            existing_dtype = key_dtypes.get(key, None)
            if existing_dtype is None:
                key_dtypes[key] = dtype
                index_total_lengths[key] = 0
                data_total_lengths[key] = 0
            elif dtype != existing_dtype:
                raise TypeError(
                    "category={}, key={}, dtype={}, existing_dtype={}".format(
                        category, key, dtype, existing_dtype
                    )
                )

            index_total_lengths[key] += len(scalar_index)
            data_total_lengths[key] += len(data)

    # Concatenate arrays from all categories for each key

    arrays = OrderedDict()
    for key, dtype in key_dtypes.items():
        index = np.empty(shape=index_total_lengths[key], dtype=rt.START_AND_LENGTH_T)
        data = np.empty(shape=data_total_lengths[key], dtype=dtype)

        index_start = 0
        data_start = 0
        for category, array_dicts in category_array_map.items():
            array_d = array_dicts.get(key, None)
            if array_d is None:
                raise NotImplementedError("TODO")

            index_ = array_d["scalar_index"]
            data_ = array_d["data"]

            data_length = len(data_)
            data_stop = data_start + data_length
            data[data_start:data_stop] = data_

            index_length = len(index_)
            index_stop = index_start + index_length
            index[index_start:index_stop] = index_[:]

            if data_start != 0:
                index[index_start:index_stop]["start"] += data_start

            data_start = data_stop
            index_start = index_stop

        arrays[key] = dict(data=data, index=index)

    return arrays


class ConvertI3ToNumpy(object):
    """
    Convert icecube objects to Numpy typed objects
    """

    __slots__ = [
        "icetray",
        "dataio",
        "dataclasses",
        "i3_scalars",
        "custom_funcs",
        "getters",
        "mapping_str_simple_scalar",
        "mapping_str_structured_scalar",
        "mapping_str_attrs",
        "attrs",
        "unhandled_types",
        "frame",
        "failed_keys",
    ]

    def __init__(self):
        # pylint: disable=unused-variable, unused-import
        from icecube import icetray, dataio, dataclasses, recclasses, simclasses

        try:
            from icecube import millipede
        except ImportError:
            millipede = None

        try:
            from icecube import santa
        except ImportError:
            santa = None

        try:
            from icecube import genie_icetray
        except ImportError:
            genie_icetray = None

        try:
            from icecube import tpx
        except ImportError:
            tpx = None

        self.icetray = icetray
        self.dataio = dataio
        self.dataclasses = dataclasses

        self.i3_scalars = {
            icetray.I3Bool: np.bool8,
            icetray.I3Int: np.int32,
            dataclasses.I3Double: np.float64,
            dataclasses.I3String: np.string0,
        }

        self.custom_funcs = {
            dataclasses.I3MCTree: self.extract_flat_mctree,
            dataclasses.I3RecoPulseSeries: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMap: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapMask: self.extract_flat_pulse_series,
            dataclasses.I3RecoPulseSeriesMapUnion: self.extract_flat_pulse_series,
            dataclasses.I3SuperDSTTriggerSeries: self.extract_seq_of_same_type,
            dataclasses.I3TriggerHierarchy: self.extract_flat_trigger_hierarchy,
            dataclasses.I3VectorI3Particle: self.extract_singleton_seq_to_scalar,
            dataclasses.I3DOMCalibration: self.extract_i3domcalibration,
        }

        self.getters = {recclasses.I3PortiaEvent: (rt.I3PORTIAEVENT_T, "Get{}")}

        self.mapping_str_simple_scalar = {
            dataclasses.I3MapStringDouble: np.float64,
            dataclasses.I3MapStringInt: np.int32,
            dataclasses.I3MapStringBool: np.bool8,
        }

        self.mapping_str_structured_scalar = {}
        if genie_icetray:
            self.mapping_str_structured_scalar[
                genie_icetray.I3GENIEResultDict
            ] = rt.I3GENIERESULTDICT_SCALARS_T

        self.mapping_str_attrs = {dataclasses.I3FilterResultMap: rt.I3FILTERRESULT_T}

        self.attrs = {
            icetray.I3RUsage: rt.I3RUSAGE_T,
            icetray.OMKey: rt.OMKEY_T,
            dataclasses.TauParam: rt.TAUPARAM_T,
            dataclasses.LinearFit: rt.LINEARFIT_T,
            dataclasses.SPEChargeDistribution: rt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3Direction: rt.I3DIRECTION_T,
            dataclasses.I3EventHeader: rt.I3EVENTHEADER_T,
            dataclasses.I3FilterResult: rt.I3FILTERRESULT_T,
            dataclasses.I3Position: rt.I3POSITION_T,
            dataclasses.I3Particle: rt.I3PARTICLE_T,
            dataclasses.I3ParticleID: rt.I3PARTICLEID_T,
            dataclasses.I3VEMCalibration: rt.I3VEMCALIBRATION_T,
            dataclasses.SPEChargeDistribution: rt.SPECHARGEDISTRIBUTION_T,
            dataclasses.I3SuperDSTTrigger: rt.I3SUPERDSTTRIGGER_T,
            dataclasses.I3Time: rt.I3TIME_T,
            dataclasses.I3TimeWindow: rt.I3TIMEWINDOW_T,
            recclasses.I3DipoleFitParams: rt.I3DIPOLEFITPARAMS_T,
            recclasses.I3LineFitParams: rt.I3LINEFITPARAMS_T,
            recclasses.I3FillRatioInfo: rt.I3FILLRATIOINFO_T,
            recclasses.I3FiniteCuts: rt.I3FINITECUTS_T,
            recclasses.I3DirectHitsValues: rt.I3DIRECTHITSVALUES_T,
            recclasses.I3HitStatisticsValues: rt.I3HITSTATISTICSVALUES_T,
            recclasses.I3HitMultiplicityValues: rt.I3HITMULTIPLICITYVALUES_T,
            recclasses.I3TensorOfInertiaFitParams: rt.I3TENSOROFINERTIAFITPARAMS_T,
            recclasses.I3Veto: rt.I3VETO_T,
            recclasses.I3CLastFitParams: rt.I3CLASTFITPARAMS_T,
            recclasses.I3CscdLlhFitParams: rt.I3CSCDLLHFITPARAMS_T,
            recclasses.I3DST16: rt.I3DST16_T,
            recclasses.DSTPosition: rt.DSTPOSITION_T,
            recclasses.I3StartStopParams: rt.I3STARTSTOPPARAMS_T,
            recclasses.I3TrackCharacteristicsValues: rt.I3TRACKCHARACTERISTICSVALUES_T,
            recclasses.I3TimeCharacteristicsValues: rt.I3TIMECHARACTERISTICSVALUES_T,
            recclasses.CramerRaoParams: rt.CRAMERRAOPARAMS_T,
        }
        if millipede:
            self.attrs[
                millipede.gulliver.I3LogLikelihoodFitParams
            ] = rt.I3LOGLIKELIHOODFITPARAMS_T
        if santa:
            self.attrs[santa.I3SantaFitParams] = rt.I3SANTAFITPARAMS_T

        # Define types we know we don't handle; these will be expanded as new
        # types are encountered to avoid repeatedly failing on the same types

        self.unhandled_types = set(
            [
                # dataclasses.I3Geometry,
                # dataclasses.I3Calibration,
                # dataclasses.I3DetectorStatus,
                # dataclasses.I3DOMLaunchSeriesMap,
                # dataclasses.I3MapKeyVectorDouble,
                # dataclasses.I3RecoPulseSeriesMapApplySPECorrection,
                # dataclasses.I3SuperDST,
                # dataclasses.I3TimeWindowSeriesMap,
                # dataclasses.I3VectorDouble,
                # dataclasses.I3VectorOMKey,
                # dataclasses.I3VectorTankKey,
                # dataclasses.I3MapKeyDouble,
                # recclasses.I3DSTHeader16,
            ]
        )
        # if tpx:
        #    self.unhandled_types.add(tpx.I3TopPulseInfoSeriesMap)

        self.frame = None
        self.failed_keys = set()

    def extract_season(self, path, gcd_path=None, keys=None):
        """E.g., data/level7_v01.04/IC86.14"""
        path = expand(path)
        assert isdir(path), path

    def extract_run(self, path, gcd_path=None, keys=None):
        """E.g. .. ::

            data/level7_v01.04/IC86.14/Run00125177
            genie/level7_v01.04/140000

        Note that what can be considered "subruns" for both data and MC are
        represented as files in both, at least for this version of oscNext.

        """
        path = expand(path)
        assert isdir(path), path

        match = RUN_DIR_RE.match(basename(path))
        assert match, 'path not a run directory? "{}"'.format(basename(path))
        groupdict = match.groupdict()

        is_data = groupdict["pfx"] is not None
        is_mc = not is_data
        run_str = groupdict["run"]
        run_int = int(groupdict["run"].lstrip("0"))

        if is_mc:
            assert isinstance(gcd_path, string_types) and isfile(expand(gcd_path))
            gcd_path = expand(gcd_path)
        else:
            if gcd_path is None:
                gcd_path = path
            assert isinstance(gcd_path, string_types)
            gcd_path = expand(gcd_path)
            if not isfile(gcd_path):
                assert isdir(gcd_path)
                # TODO: use DATA_GCD_FNAME_RE
                gcd_path = glob(join(gcd_path, "*Run{}*GCD*.i3*".format(run_str)))
                assert len(gcd_path) == 1, gcd_path
                gcd_path = expand(gcd_path[0])

        subrun_filepaths = []
        for basepath in listdir(path):
            match = OSCNEXT_I3_FNAME_RE.match(basepath)
            if not match:
                continue
            groupdict = match.groupdict()
            assert int(groupdict["run"]) == run_int
            subrun_int = int(groupdict["subrun"])
            subrun_filepaths.append((subrun_int, join(path, basepath)))
        subrun_filepaths.sort()

        scalar_arrays = OrderedDict()
        vector_arrays = OrderedDict()

        for subrun, fpath in subrun_filepaths:
            scalar_arrays[subrun], vector_arrays[subrun] = self.extract_files(
                paths=[gcd_path, fpath], keys=keys
            )

        # Ensure sorting by subrun
        subrun_scalar_index, scalar_arrays = index_and_cat_scalar_arrays(scalar_arrays)
        vector_arrays = cat_vector_arrays(vector_arrays)

        return subrun_scalar_index, scalar_arrays, vector_arrays

    def extract_files(self, paths, keys=None):
        """Extract info from one or more i3 file(s)

        Parameters
        ----------
        paths : str or iterable thereof
        keys : str or iterable thereof, or None; optional

        Returns
        -------
        scalar_arrays
        vector_arrays

        """
        if isinstance(paths, str):
            paths = [paths]
        paths = [expand(path) for path in paths]
        i3file_iterator = self.dataio.I3FrameSequence()
        try:
            extracted_data = []
            for path in paths:
                i3file_iterator.add_file(path)
                while i3file_iterator.more():
                    frame = i3file_iterator.pop_frame()
                    if frame.Stop != self.icetray.I3Frame.Physics:
                        continue
                    data = self.extract_frame(frame=frame, keys=keys)
                    extracted_data.append(data)
                #i3file_iterator.close_last_file()
        finally:
            i3file_iterator.close()

        #return extracted_data
        return construct_arrays(extracted_data)

    def extract_frame(self, frame, keys=None):
        """Extract icetray frame objects to numpy typed objects

        Parameters
        ----------
        frame : icetray.I3Frame
        keys : str or iterable thereof, or None; optional

        """
        self.frame = frame

        auto_mode = False
        if keys is None:
            auto_mode = True
            keys = frame.keys()
        elif isinstance(keys, str):
            keys = [keys]
        keys = sorted(set(keys).difference(self.failed_keys))

        extracted_data = {}

        for key in keys:
            try:
                value = frame[key]
            except Exception:
                if auto_mode:
                    self.failed_keys.add(key)
                # else:
                #    extracted_data[key] = None
                continue

            try:
                np_value = self.extract_object(value)
            except Exception:
                print("failed on key {}".format(key))
                raise

            # if auto_mode and np_value is None:
            if np_value is None:
                continue

            extracted_data[key] = np_value

        return extracted_data

    def extract_object(self, obj, to_numpy=True):
        """Convert an object from a frame to a Numpy typed object.

        Note that e.g. extracting I3RecoPulseSeriesMap{Mask,Union} requires
        that `self.frame` be assigned the current frame to work.

        Parameters
        ----------
        obj : frame object
        to_numpy : bool, optional

        Returns
        -------
        np_obj : numpy-typed object or None

        """
        obj_t = type(obj)

        if obj_t in self.unhandled_types:
            return None

        dtype = self.i3_scalars.get(obj_t, None)
        if dtype:
            val = dtype(obj.value)
            if to_numpy:
                return val
            return val, dtype

        dtype_fmt = self.getters.get(obj_t, None)
        if dtype_fmt:
            return self.extract_getters(obj, *dtype_fmt, to_numpy=to_numpy)

        dtype = self.mapping_str_simple_scalar.get(obj_t, None)
        if dtype:
            return dict2struct(obj, set_explicit_dtype_func=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_structured_scalar.get(obj_t, None)
        if dtype:
            return maptype2np(obj, dtype=dtype, to_numpy=to_numpy)

        dtype = self.mapping_str_attrs.get(obj_t, None)
        if dtype:
            return self.extract_mapscalarattrs(obj, to_numpy=to_numpy)

        dtype = self.attrs.get(obj_t, None)
        if dtype:
            return self.extract_attrs(obj, dtype, to_numpy=to_numpy)

        func = self.custom_funcs.get(obj_t, None)
        if func:
            return func(obj, to_numpy=to_numpy)

        # New unhandled type found
        self.unhandled_types.add(obj_t)

        return None

    @staticmethod
    def extract_flat_trigger_hierarchy(obj, to_numpy=True):
        """Flatten a trigger hierarchy into a linear sequence of triggers,
        labeled such that the original hiercarchy can be recreated

        Parameters
        ----------
        obj : I3TriggerHierarchy
        to_numpy : bool, optional

        Returns
        -------
        flat_triggers : shape-(N-trigers,) numpy.ndarray of dtype FLAT_TRIGGER_T

        """
        iterattr = obj.items if hasattr(obj, "items") else obj.iteritems

        level_tups = []
        flat_triggers = []

        for level_tup, trigger in iterattr():
            level_tups.append(level_tup)
            level = len(level_tup) - 1
            if level == 0:
                parent_idx = -1
            else:
                parent_idx = level_tups.index(level_tup[:-1])
            # info_tup, _ = self.extract_attrs(trigger, TRIGGER_T, to_numpy=False)
            key = trigger.key
            flat_triggers.append(
                (
                    level,
                    parent_idx,
                    (
                        trigger.time,
                        trigger.length,
                        trigger.fired,
                        (key.source, key.type, key.subtype, key.config_id or 0),
                    ),
                )
            )

        if to_numpy:
            return np.array(flat_triggers, dtype=rt.FLAT_TRIGGER_T)

        return flat_triggers, rt.FLAT_TRIGGER_T

    def extract_flat_mctree(
        self,
        mctree,
        parent=None,
        parent_idx=-1,
        level=0,
        max_level=-1,
        flat_particles=None,
        to_numpy=True,
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

        flat_particles : appendable sequence or None, optional

        to_numpy : bool, optional


        Returns
        -------
        flat_particles : list of tuples or ndarray of dtype `FLAT_PARTICLE_T`


        Examples
        --------
        This is a recursive function, with defaults defined for calling simply for
        the typical use case of flattening an entire I3MCTree and producing a
        numpy.ndarray with the results. .. ::

            flat_particles = extract_flat_mctree(frame["I3MCTree"])

        """
        if flat_particles is None:
            flat_particles = []

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
                    info_tup, _ = self.extract_attrs(
                        daughter, rt.I3PARTICLE_T, to_numpy=False
                    )
                    flat_particles.append((level, parent_idx, info_tup))

                # Now recurse, appending any granddaughters (daughters to these
                # daughters) at the end
                for daughter_idx, daughter in enumerate(daughters, start=idx0):
                    self.extract_flat_mctree(
                        mctree=mctree,
                        parent=daughter,
                        parent_idx=daughter_idx,
                        level=level + 1,
                        max_level=max_level,
                        flat_particles=flat_particles,
                        to_numpy=False,
                    )

        if to_numpy:
            return np.array(flat_particles, dtype=rt.FLAT_PARTICLE_T)

        return flat_particles, rt.FLAT_PARTICLE_T

    def extract_flat_pulse_series(self, obj, frame=None, to_numpy=True):
        """Flatten a pulse series into a 1D array of ((<OMKEY_T>), <PULSE_T>)

        Parameters
        ----------
        obj : dataclasses.I3RecoPUlseSeries{,Map,MapMask,MapUnion}
        frame : iectray.I3Frame, required if obj is {...Mask, ...Union}
        to_numpy : bool, optional

        Returns
        -------
        flat_pulses : shape-(N-pulses) numpy.ndarray of dtype FLAT_PULSE_T

        """
        if isinstance(
            obj,
            (
                self.dataclasses.I3RecoPulseSeriesMapMask,
                self.dataclasses.I3RecoPulseSeriesMapUnion,
            ),
        ):
            if frame is None:
                frame = self.frame
            obj = obj.apply(frame)

        flat_pulses = []
        for omkey, pulses in obj.items():
            omkey = (omkey.string, omkey.om, omkey.pmt)
            for pulse in pulses:
                info_tup, _ = self.extract_attrs(
                    pulse, dtype=rt.PULSE_T, to_numpy=False
                )
                flat_pulses.append((omkey, info_tup))

        if to_numpy:
            return np.array(flat_pulses, dtype=rt.FLAT_PULSE_T)

        return flat_pulses, rt.FLAT_PULSE_T

    def extract_singleton_seq_to_scalar(self, seq, to_numpy=True):
        """Extract a sole object from a sequence and treat it as a scalar.
        E.g., I3VectorI3Particle that, by construction, contains just one
        particle


        Parameters
        ----------
        seq : sequence
        to_numpy : bool, optional


        Returns
        -------
        obj

        """
        assert len(seq) == 1
        return self.extract_object(seq[0], to_numpy=to_numpy)

    def extract_attrs(self, obj, dtype, to_numpy=True):
        """Extract attributes of an object (and optionally, recursively, attributes
        of those attributes, etc.) into a numpy.ndarray based on the specification
        provided by `dtype`.


        Parameters
        ----------
        obj
        dtype : numpy.dtype
        to_numpy : bool, optional


        Returns
        -------
        vals : tuple or shape-(1,) numpy.ndarray of dtype `dtype`

        """
        vals = []
        if isinstance(dtype, np.dtype):
            descr = dtype.descr
        elif isinstance(dtype, Sequence):
            descr = dtype
        else:
            raise TypeError("{}".format(dtype))

        for name, subdtype in descr:
            val = getattr(obj, name)
            if isinstance(subdtype, (str, np.dtype)):
                vals.append(val)
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    out = self.extract_attrs(val, subdtype, to_numpy=False)
                assert out is not None, "{}: {} {}".format(name, subdtype, val)
                info_tup, _ = out
                vals.append(info_tup)
            else:
                raise TypeError("{}".format(subdtype))

        # Numpy converts tuples correctly; lists are interpreted differently
        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_mapscalarattrs(self, mapping, subdtype=None, to_numpy=True):
        """Convert a mapping (containing string keys and scalar-typed values)
        to a single-element Numpy array from the values of `mapping`, using
        keys defined by `subdtype.names`.

        Use this function if you already know the `subdtype` you want to end up
        with. Use `retro.utils.misc.dict2struct` directly if you do not know
        the dtype(s) of the mapping's values ahead of time.


        Parameters
        ----------
        mapping : mapping from strings to scalars

        dtype : numpy.dtype
            If scalar dtype, convert via `utils.dict2struct`. If structured
            dtype, convert keys specified by the struct field names and values
            are converted according to the corresponding type.


        Returns
        -------
        array : shape-(1,) numpy.ndarray of dtype `dtype`


        See Also
        --------
        dict2struct
            Convert from a mapping to a numpy.ndarray, dynamically building `dtype`
            as you go (i.e., this is not known a priori)

        """
        keys = mapping.keys()
        if not isinstance(mapping, OrderedDict):
            keys.sort()

        out_vals = []
        out_dtype = []

        if subdtype is None:  # infer subdtype from values in mapping
            for key in keys:
                val = mapping[key]
                info_tup, subdtype = self.extract_object(val, to_numpy=False)
                out_vals.append(info_tup)
                out_dtype.append((key, subdtype))
        else:  # scalar subdtype
            for key in keys:
                out_vals.append(mapping[key])
                out_dtype.append((key, subdtype))

        out_vals = tuple(out_vals)

        if to_numpy:
            return np.array([out_vals], dtype=out_dtype)[0]

        return out_vals, out_dtype

    def extract_getters(self, obj, dtype, fmt="Get{}", to_numpy=True):
        """Convert an object whose data has to be extracted via methods that
        behave like getters (e.g., .`xyz = get_xyz()`).


        Parameters
        ----------
        obj
        dtype
        fmt : str
        to_numpy : bool, optional


        Examples
        --------
        To get all of the values of an I3PortiaEvent: .. ::

            extract_getters(frame["PoleEHESummaryPulseInfo"], dtype=rt.I3PORTIAEVENT_T, fmt="Get{}")

        """
        vals = []
        for name, subdtype in dtype.descr:
            getter_attr_name = fmt.format(name)
            getter_func = getattr(obj, getter_attr_name)
            val = getter_func()
            if not isinstance(subdtype, str) and isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "Failed to convert name {} val {} type {}".format(
                            name, val, type(val)
                        )
                    )
                val, _ = out
            # if isinstance(val, self.icetray.OMKey):
            #    val = self.extract_attrs(val, dtype=rt.OMKEY_T, to_numpy=False)
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=dtype)[0]

        return vals, dtype

    def extract_seq_of_same_type(self, seq, to_numpy=True):
        """Convert a sequence of objects, all of the same type, to a numpy array of
        that type.

        Parameters
        ----------
        seq : seq of N objects all of same type
        to_numpy : bool, optional

        Returns
        -------
        out_seq : list of N tuples or shape-(N,) numpy.ndarray of `dtype`

        """
        assert len(seq) > 0

        # Convert first object in sequence to get dtype
        val0 = seq[0]
        val0_tup, val0_dtype = self.extract_object(val0, to_numpy=False)
        data_tups = [val0_tup]

        # Convert any remaining objects
        for obj in seq[1:]:
            data_tups.append(self.extract_object(obj, to_numpy=False)[0])

        if to_numpy:
            return np.array(data_tups, dtype=val0_dtype)

        return data_tups, val0_dtype

    def extract_i3domcalibration(self, obj, to_numpy=True):
        """Extract the information from an I3DOMCalibration frame object"""
        vals = []
        for name, subdtype in rt.I3DOMCALIBRATION_T.descr:
            val = getattr(obj, name)
            if name == "dom_cal_version":
                if val == "unknown":
                    val = (-1, -1, -1)
                else:
                    val = tuple(int(x) for x in val.split("."))
            elif isinstance(subdtype, (str, np.dtype)):
                pass
            elif isinstance(subdtype, Sequence):
                out = self.extract_object(val, to_numpy=False)
                if out is None:
                    raise ValueError(
                        "{} {} {} {}".format(name, subdtype, val, type(val))
                    )
                val, _ = out
            else:
                raise TypeError(str(subdtype))
            vals.append(val)

        vals = tuple(vals)

        if to_numpy:
            return np.array([vals], dtype=rt.I3DOMCALIBRATION_T)[0]

        return vals, rt.I3DOMCALIBRATION_T


def main():
    """Main"""
    # pylint: disable=line-too-long
    #from processing.samples.oscNext.verification.general_mc_data_harvest_and_plot import ALL_OSCNEXT_VARIABLES
    mykeys = """L5_SPEFit11 LineFit_DC I3TriggerHierarchy SRTTWOfflinePulsesDC
    SRTTWOfflinePulsesDCTimeRange SplitInIcePulses SplitInIcePulsesTimeRange
    L5_oscNext_bool I3EventHeader I3TriggerHierarchy I3GenieResultDict
    I3MCTree""".split()
    all_keys = mykeys #sorted(set([k.split(".")[0] for k in ALL_OSCNEXT_VARIABLES.keys()] + mykeys))

    parser = ArgumentParser()
    parser.add_argument("--paths", nargs="+")
    datafiles = sorted(glob("/data/icecube/ana/LE/oscNext/pass2/data/level5_v01.04/IC86.14/Run00124566/oscNext_*.i3.zst"))
    gcd_path = "/data/icecube/gcd/Level2pass2_IC86.2014_data_Run00124566_0410_53_89_GCD.i3.zst"

    cnv = ConvertI3ToNumpy()
    return cnv.extract_files([gcd_path] + datafiles, keys=all_keys)


if __name__ == "__main__":
    OUT = main()
    #sa, va = main()
    #print(next(iter(sa.items())), next(iter(va.items())))
