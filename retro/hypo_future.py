# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Base class for hypotheses and associated helper functions
"""

from __future__ import absolute_import, division, print_function

__all__ = ["Hypo"]

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
from os.path import abspath, dirname
import sys

from six import PY2, string_types

if PY2:
    from funcsigs import signature
else:
    from inspect import signature

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import check_kwarg_keys, deduce_sph_pairs
from retro.utils.get_arg_names import get_arg_names


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

        if isinstance(internal_param_names, string_types):
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
                if not isinstance(p, string_types):
                    is_singleton = False
            if is_singleton:
                internal_sph_pairs = (internal_sph_pairs,)

        if external_sph_pairs is None:
            external_sph_pairs = deduce_sph_pairs(external_param_names)
        elif len(external_sph_pairs) == 2:
            # Detect singleton tuple of strings and convert to tuple-of-tuple
            is_singleton = True
            for p in external_sph_pairs:
                if not isinstance(p, string_types):
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

        # -- Record configuration -- #

        config = OrderedDict()
        config["class"] = type(self).__name__
        config["internal_param_names"] = internal_param_names
        config["external_param_names"] = external_param_names
        config["internal_sph_pairs"] = internal_sph_pairs
        config["external_sph_pairs"] = external_sph_pairs
        if callable(param_mapping):
            config["param_mapping"] = (
                param_mapping.__name__ + str(signature(param_mapping))
            )
        elif isinstance(param_mapping, Mapping):
            config["param_mapping"] = param_mapping
        else:
            raise ValueError("Cannot handle {}".format(type(param_mapping)))

        # -- Define dummy _get_sources function -- #

        def _get_sources(**kwargs): # pylint: disable=unused-argument
            """Must be replaced with your own callable"""
            raise NotImplementedError()

        # -- Define class attributes to store the above values -- #

        self.param_mapping = param_mapping
        """callable : maps external param names/values to internal param names/values"""

        self.external_params = OrderedDict([(n, None) for n in external_param_names])
        """OrderedDict : {external_param_name: val or None}"""

        self.external_param_names = tuple(external_param_names)
        """tuple of str : all param names that must be provided externally"""

        self.external_sph_pairs = tuple(external_sph_pairs)
        """tuple of zero or more 2-tuples of strings : (az, zen) external param name pairs"""

        self.external_sph_pair_idxs = tuple(external_sph_pair_idxs)
        """tuple of zero or more 2-tuples of ints : (az, zen) external param index pairs"""

        self.internal_param_names = (tuple(internal_param_names),)
        """tuple of str : all param names that are used internally"""

        self.internal_sph_pairs = tuple(internal_sph_pairs)
        """tuple of zero or more 2-tuples of strings : (az, zen) internal param name pairs"""

        self.internal_sph_pair_idxs = tuple(internal_sph_pair_idxs)
        """tuple of zero or more 2-tuples of ints : (az, zen) internal param name pairs"""

        self.num_external_params = len(self.external_param_names)
        """int : number of parameters used by the kernel"""

        self.num_internal_params = len(internal_param_names)
        """int : number of parameters used by the kernel"""

        self.internal_params = OrderedDict([(n, None) for n in internal_param_names])
        """OrderedDict : {internal_param_name: val or None}"""

        self.sources = None
        """tuple of ndarrays of dtype SRC_T : Arrays of sources"""

        self.sources_handling = None
        """tuple of retro.const.SrcHandling enums : How each corresponding sources array
        in `self.sources` is to be handled"""

        self.max_num_scalefactors = None
        """Maximum number of scalefactors hypo might use (the actual number of
        scalefactors used can change, e.g., as a function of pegleg step)"""

        self.max_scalefactors = None
        """tuple of scalars : Maximum values each scalefactor can take"""

        self.num_pegleg_generators = None
        """int >= 0 : Number of generators contained within `self.pegleg_generators`,
        i.e., it is valid to call .. ::

            self.pegleg_generators(i)

        where i is in range(self.num_pegleg_generators)"""

        self.pegleg_generators = None
        """numba callable : call via .. ::

            self.pegleg_generators(i)

        where i is in range(self.num_pegleg_generators)"""

        self.num_calls = 0
        """int > = 0 : Number of calls to `get_sources`"""

        self.config = config
        """OrderedDict : configuration settings critical for recording run state"""

        self._get_sources = _get_sources
        """callable : inheriting classes must replace"""

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
        sources_handling : list of SrcHandling enums
        num_pegleg_generators
        pegleg_generators

        """
        self.num_calls += 1
        for external_param_name in self.external_param_names:
            try:
                self.external_params[external_param_name] = kwargs[external_param_name]
            except KeyError:
                print(
                    'Missing param "{}" in passed kwargs {}'
                    .format(external_param_name, kwargs)
                )
                raise

        # Map external param names/values onto internal param names/values
        self.internal_params = self.param_mapping(**self.external_params)

        # Call internal function
        try:
            (
                self.sources,
                self.sources_handling,
                self.num_pegleg_generators,
                self.pegleg_generators,
            ) = self._get_sources(**self.internal_params)
        except (TypeError, KeyError):
            check_kwarg_keys(
                required_keys=self.internal_param_names,
                provided_kwargs=self.internal_params,
                meta_name="kwarg(s)",
                message_pfx="internal param names (kwargs to `_get_sources`):",
            )
            raise

        return (
            self.sources,
            self.sources_handling,
            self.num_pegleg_generators,
            self.pegleg_generators
        )

    def get_energy(self, pegleg_indices=None, scalefactors=None):
        """Get energy (in units of GeV) from current parameter values. Must
        implement/override in subclasses."""
        raise NotImplementedError("Must define `get_energy` in subclasses of Hypo")

    def get_derived_params(self, pegleg_indices=None, scalefactors=None):
        """Retrieve any derived params from component hypotheses.

        Parameters
        ----------
        pegleg_indices : optional
        scalefactors : optional

        Returns
        -------
        derived_params : OrderedDict

        """
        raise NotImplementedError()
