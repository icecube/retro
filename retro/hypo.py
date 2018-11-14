# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Base class for hypotheses and associated helper functions
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "SrcHandling",
    "get_partial_match_expr",
    "deduce_sph_pairs",
    "aggregate_sources",
    "Hypo",
]

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017-2018 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from collections import Mapping, OrderedDict
from enum import IntEnum
from inspect import cleandoc
from os.path import abspath, dirname
import re
import sys

import numba

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit
from retro.const import EMPTY_SOURCES
from retro.utils.misc import check_kwarg_keys, get_arg_names


class SrcHandling(IntEnum):
    """Kinds of sources each hypothesis can generate"""
    none = 0
    pegleg = 1
    nonscaling = 2
    scaling = 3


def get_partial_match_expr(word, minchars):
    """Generate regex sub-expression for matching first `minchars` of a word or
    `minchars` + 1 of the word, or `minchars` + 2, ....

    Adapted from user sberry at https://stackoverflow.com/a/13405331

    Parameters
    ----------
    word : str
    minchars : int

    Returns
    -------
    expr : str

    """
    return '|'.join(word[:i] for i in range(len(word), minchars - 1, -1))


AZ_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("azimuthal", 2)))

ZEN_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("zenith", 3)))


def deduce_sph_pairs(param_names):
    """Attempt to deduce which param names represent (azimuth, zenith) pairs.

    Parameters
    ----------
    param_names : iterable of strings

    Returns
    -------
    sph_pairs : tuple of 2-tuples of strings

    Notes
    -----
    Works by looking for the prefix and suffix (if any) surrounding
      "az", "azi", ..., or "azimuthal"
    to match the prefix and suffix (if any) surrounding
      "zen", "zeni", ..., or "zenith"

    Examples
    --------
    >>> deduce_sph_pairs(("x", "azimuth", "zenith", "cascade_azimuth", "cascade_zenith"))
    (('azimuth', 'zenith'), ('cascade_azimuth', 'cascade_zenith'))

    """
    az_pfxs_and_sfxs = OrderedDict()
    zen_pfxs_and_sfxs = OrderedDict()
    for param_name in param_names:
        match = AZ_REGEX.match(param_name)
        if match:
            groups = match.groups()
            az_pfxs_and_sfxs[param_name] = (groups[0], groups[2])
            continue
        match = ZEN_REGEX.match(param_name)
        if match:
            groups = match.groups()
            zen_pfxs_and_sfxs[param_name] = (groups[0], groups[2])
            continue

    sph_pairs = []
    for az_param_name, az_pfx_and_sfx in az_pfxs_and_sfxs.items():
        for zen_param_name, zen_pfx_and_sfx in zen_pfxs_and_sfxs.items():
            if zen_pfx_and_sfx != az_pfx_and_sfx:
                continue
            sph_pairs.append((az_param_name, zen_param_name))

    return tuple(sph_pairs)


def aggregate_sources(hypos, **kwargs):
    """Aggregate sources from all hypotheses and create a single function to call into
    each hypothesis's pegleg function (if any exist).

    Parameters
    ----------
    **kwargs
        Passed into each hypo.get_sources method

    Returns
    -------
    sources : list of 1 or more ndarrays of dtype SRC_T
        Required to be at least 1 element for use by Numba

    source_kinds : list of SrcHandling of same length as `sources`
        Required to be at least 1 element for use by Numba

    num_pegleg_generators : int >= 0

    pegleg_generators : numba-callable generator
        Takes integer parameter `generator_num` and then acts as an iterator through
        that pegleg generator, which yields [sources], [source_kinds] on each iteration.
        A dummy generator is created if no hypotheses have pegleg generators (i.e.,
        `num_pegleg_generators == 0`) such that Numba gets consistent types

    """
    sources = []
    source_kinds = []
    pegleg_generators = []

    for hypo in hypos:
        for kind, val in hypo.get_sources(**kwargs).items():
            if kind in (SrcHandling.nonscaling, SrcHandling.scaling):
                sources.append(val)
                source_kinds.append(kind)
            elif kind is SrcHandling.pegleg:
                pegleg_generators.append(val)
            else:
                raise ValueError("Invalid sources kind")

    if len(sources) == 0:
        sources.append(EMPTY_SOURCES)
        source_kinds.append(SrcHandling.none)

    for func_num, func in enumerate(pegleg_generators):
        if not isinstance(func, numba.targets.registry.CPUDispatcher):
            try:
                func = numba.njit(cache=True, fastmath=True, nogil=True)(func)
            except:
                print("failed to numba-jit-compile func {}".format(func))
                raise
        pegleg_generators[func_num] = func

    num_pegleg_generators = len(pegleg_generators)

    conditional_lines = []
    lcls = locals()
    for func_num, func in enumerate(pegleg_generators):
        if func is None:
            continue

        # Define a variable (f0, f1, etc.) in local scope that is the callee
        func_name = "f{:d}".format(func_num)
        lcls[func_name] = func

        # Define the conditional
        conditional = "if" if len(conditional_lines) == 0 else "elif"
        conditional_lines.append(
            "    {} func_num == {}:".format(conditional, func_num)
        )

        # Define the execution statement
        conditional_lines.append(
            "        return {}(pegleg_step=pegleg_step)".format(func_name)
        )

    # Remove leading spaces uniformly in string
    py_pegleg_generators_str = cleandoc(
        """
        def py_pegleg_generators(generator_num):
        {body}
            raise ValueError("Invalid `generator_num`")
        """
    ).format(body="\n".join(conditional_lines))

    try:
        exec py_pegleg_generators_str in locals() # pylint: disable=exec-used
    except:
        print(py_pegleg_generators_str)
        raise

    pegleg_generators = numba_jit(**DFLT_NUMBA_JIT_KWARGS)(
        py_pegleg_generators # pylint: disable=undefined-variable
    )

    return sources, source_kinds, num_pegleg_generators, pegleg_generators


class Hypo(object):
    """
    Hypothesis base class; inheriting classes must override the `self._get_sources`
    attribute with an appropriate callable.

    Parameters
    ----------
    param_mapping : mapping or callable
        If mapping, keys are `external_param_names` and values are corresponding
        `internal_param_names`

        If callable, provide a function that takes named arguments discoverable via
        :meth:`retro.utils.misc.get_arg_names` and all of which are
        `external_param_names`; values passed to the function are the external
        parameters' values. The function must return a Mapping whose keys are
        `internal_param_names` and values are the corresonding internal parameter
        values. Note that a callable should also be defined to handle arbitrary
        additional **kwargs so parameters that the function doesn't care about will just
        pass through. An example callable usable as a `param_mapping` is::

            def mapper_func(cascade_energy, x, y, z, **kwargs):
                return dict(energy=cascade_energy, x=x, y=y, z=z)

    internal_param_names : string or iterable of strings
        names of paremeters used internally by :attr:`_get_sources` method (which
        is to be defined in subclasses)

    internal_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        pairs of (azimuth, zenith) parameters (in that order) from among
        `internal_param_names`; if None (default), pairs of spherical parameters are
        (attempted to be) deduced by their names (see :meth:`deduce_sph_pairs` for
        details).

    external_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        pairs of (azimuth, zenith) pairs of "external" parameter names (in that order);
        if None (default), pairs of spherical parameters are (attempted to be) deduced
        using their names (see :meth:`deduce_sph_pairs` for details).

    """
    def __init__(
        self,
        param_mapping,
        internal_param_names,
        internal_sph_pairs=None,
        external_sph_pairs=None,
    ):
        # -- Validation and translation of args -- #

        if not (isinstance(param_mapping, Mapping) or callable(param_mapping)):
            raise TypeError(
                "`param_mapping` must be mapping or callable; got {}".format(
                    type(param_mapping)
                )
            )

        if isinstance(internal_param_names, basestring):
            internal_param_names = (internal_param_names,)
        else:
            internal_param_names = tuple(internal_param_names)

        if callable(param_mapping):
            external_param_names = tuple(get_arg_names(param_mapping))
        else: # isinstance(param_mapping, Mapping):
            # Validate internal params specified in map match those in this object
            internal_specd = param_mapping.values()
            if not isinstance(param_mapping, OrderedDict):
                internal_specd = sorted(internal_specd)
            internal_specd_set = set(internal_specd)
            if internal_specd_set != set(internal_param_names):
                raise ValueError(
                    "`param_mapping` maps to internal names {} which don't match"
                    "`internal_param_names` {}"
                    .format(sorted(internal_specd), sorted(internal_param_names))
                )
            if len(internal_specd_set) != len(internal_specd):
                dupes = []
                for x in internal_specd:
                    if internal_specd.count(x) > 1 and x not in dupes:
                        dupes.append(x)
                raise ValueError(
                    "`param_mapping` must be a one-to-one mapping, but internal param"
                    " names (i.e., values in the `param_mapping`) {} are specified"
                    " multiple times".format(dupes)
                )

            if isinstance(param_mapping, OrderedDict):
                external_param_names = param_mapping.keys()
            else:
                # Extract external_param_names in same order as internal_param_names if
                # no order specified by user
                inv_map = {v: k for k, v in param_mapping.items()}
                external_param_names = tuple(inv_map[k] for k in internal_param_names)

            # Copy the actual dict/Mapping to a private variable
            _pmap = param_mapping

            # Redefine param_mapping as a callable for consistency of subsequent
            # code (just has to call `param_mapping` rather than handle either
            # dict _or_ callable every time)
            def _param_mapping(**kwargs):
                internal_kwargs = {
                    intnam: kwargs[extnam] for extnam, intnam in _pmap.items()
                }
                return internal_kwargs
            param_mapping = _param_mapping

        if internal_sph_pairs is None:
            internal_sph_pairs = deduce_sph_pairs(internal_param_names)
        elif len(internal_sph_pairs) == 2:
            # Detect singleton tuple of strings and convert to tuple-of-tuple
            is_singleton = True
            for p in internal_sph_pairs:
                if not isinstance(p, basestring):
                    is_singleton = False
            if is_singleton:
                internal_sph_pairs = (internal_sph_pairs,)

        if external_sph_pairs is None:
            external_sph_pairs = deduce_sph_pairs(external_param_names)
        elif len(external_sph_pairs) == 2:
            # Detect singleton tuple of strings and convert to tuple-of-tuple
            is_singleton = True
            for p in external_sph_pairs:
                if not isinstance(p, basestring):
                    is_singleton = False
            if is_singleton:
                external_sph_pairs = (external_sph_pairs,)

        # Validate and also generate `internal_pair_idxs`
        internal_sph_pair_idxs = []
        for pair in internal_sph_pairs:
            az_param_idx = None
            zen_param_idx = None
            for pname in pair:
                if pname not in internal_param_names:
                    raise ValueError(
                        'param "{}" not a valid internal parameter: {}'.format(
                            pname, internal_param_names
                        )
                    )
                if 'az' in pname:
                    az_param_idx = internal_param_names.index(pname)
                elif 'zen' in pname:
                    zen_param_idx = internal_param_names.index(pname)
                else:
                    raise ValueError(
                        'Spherical params must either have "az" or "zen" in them; "{}"'
                        'has neither'.format(pname)
                    )
            internal_sph_pair_idxs.append((az_param_idx, zen_param_idx))

        # Validate and also generate `external_pair_idxs`
        external_sph_pair_idxs = []
        for pair in external_sph_pairs:
            az_param_idx = None
            zen_param_idx = None
            for pname in pair:
                if pname not in external_param_names:
                    raise ValueError(
                        'param "{}" not a valid external parameter: {}'.format(
                            pname, external_param_names
                        )
                    )
                if 'az' in pname:
                    az_param_idx = external_param_names.index(pname)
                elif 'zen' in pname:
                    zen_param_idx = external_param_names.index(pname)
                else:
                    raise ValueError(
                        'Spherical params must either have "az" or "zen" in them; "{}"'
                        'has neither'.format(pname)
                    )
            external_sph_pair_idxs.append((az_param_idx, zen_param_idx))

        # -- Define class attributes to store the above values -- #

        self.param_mapping = param_mapping
        """callable : maps external param names/values to internal param names/values"""

        self.external_param_names = tuple(external_param_names)
        """tuple of str : all param names that must be provided externally"""

        self.external_sph_pairs = tuple(external_sph_pairs)
        """tuple of zero or more 2-tuples of strings : (az, zen) external param name pairs"""

        self.external_sph_pair_idxs = tuple(external_sph_pair_idxs)
        """tuple of zero or more 2-tuples of ints : (az, zen) external param index pairs"""

        self.internal_param_names = tuple(internal_param_names)
        """tuple of str : all param names that are used internally"""

        self.internal_sph_pairs = tuple(internal_sph_pairs)
        """tuple of zero or more 2-tuples of strings : (az, zen) internal param name pairs"""

        self.internal_sph_pair_idxs = tuple(internal_sph_pair_idxs)
        """tuple of zero or more 2-tuples of ints : (az, zen) internal param name pairs"""

        self.num_external_params = len(self.external_param_names)
        """int : number of parameters used by the kernel"""

        self.num_internal_params = len(self.internal_param_names)
        """int : number of parameters used by the kernel"""

        self.internal_param_values = OrderedDict([(n, None) for n in internal_param_names])
        """OrderedDict : {internal_param_name: val or None}"""

        self.external_param_values = OrderedDict([(n, None) for n in external_param_names])
        """OrderedDict : {external_param_name: val or None}"""

        self.sources = None

        self.source_kinds = None

        self.num_calls = 0
        """Number of calls to `get_sources`"""

        # Define dummy function
        def _get_sources(**kwargs): # pylint: disable=unused-argument
            """Must be replaced with your own callable"""
            raise NotImplementedError()

        self._get_sources = _get_sources
        """callable : inheriting classes must override with a callable that returns a
        dict"""

    def get_sources(self, **kwargs):
        """Get sources corresponding to hypothesis parameters when called via::

            get_sources(external_param_name0=value0, ...)

        where external param names and/or values are mapped to internal param names and
        values which are then passed to the `self._get_sources` callable.

        Parameters
        ----------
        **kwargs
            Keyword arguments keyed by (at least) external param names; extra kwargs are
            ignored (and not passed through to the internal `_get_sources` function).

        Returns
        -------
        sources : list of one or more ndarrays of dtype SRC_T
        kinds : list of SrcHandling enums

        """
        self.num_calls += 1
        for external_param_name in self.external_param_names:
            try:
                self.external_param_values[external_param_name] = kwargs[external_param_name]
            except KeyError:
                print('Missing param "{}" in passed args'.format(external_param_name))
                raise

        # Map external param names/values onto internal param names/values
        self.internal_param_values = self.param_mapping(**self.external_param_values)

        # Call internal function
        try:
            self.sources = self._get_sources(**self.internal_param_values)
        except (TypeError, KeyError):
            check_kwarg_keys(
                required_keys=self.internal_param_names,
                provided_kwargs=self.internal_param_values,
                meta_name="kwarg(s)",
                message_pfx="internal param names (kwargs to `_get_sources`):",
            )
            raise

        return self.sources
