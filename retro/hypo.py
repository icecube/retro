# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Abstract/partially-concrete base class for hypotheses
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
from numbers import Number
from os.path import abspath, dirname
import re
import sys

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import get_arg_names



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
    (("azimuth", "zenith"), ("cascade_azimuth", "cascade_zenith"))

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


class Hypo(object):
    """
    Hypothesis base class; inheriting classes must override the `self._generate_sources`
    attribute with an appropriate callable.

    Parameters
    ----------
    param_mapping : mapping
        mapping from external parameter names to internally-used parameter names (see
        `Cascade.internal_param_names` for required internal param names)

    internal_param_names : string or iterable of strings

    internal_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        If None or not specified, pairs of spherical parameters are deduced by
        `deduce_sph_pairs`.

    external_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        If None or not specified, pairs of spherical parameters are deduced by
        `deduce_sph_pairs`.

    scaling_proto_energy : None or scalar > 0
        specify None to disable or specify scalar > 0 to treat the cascade as
        "scaling," i.e., a prototypical set of light sources are generated for the
        energy specified and modifying the energy from that merely scales the luminosity
        of each of those sources as opposed to generating an entirely new set of light
        sources; in this case, the topology of the cascade will not be as accurate but
        the speed of computing likelihoods increases

    """
    def __init__(
        self,
        param_mapping,
        internal_param_names,
        internal_sph_pairs=None,
        external_sph_pairs=None,
        scaling_proto_energy=None,
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
            if set(internal_specd) != set(internal_param_names):
                raise ValueError(
                    "`param_mapping` maps to internal names {} which don't match"
                    "`internal_param_names` {}"
                    .format(sorted(internal_specd), sorted(internal_param_names))
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
                    internal_name: kwargs[external_name]
                    for external_name, internal_name in _pmap.items()
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

        # Validate and also generate *pair_idxs
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

        # Validate and also generate *pair_idxs
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

        if not (
            scaling_proto_energy is None
            or isinstance(scaling_proto_energy, Number)
        ):
            raise TypeError("`scaling_proto_energy` must be None or a scalar")

        if isinstance(scaling_proto_energy, Number) and scaling_proto_energy <= 0:
            raise ValueError("If scalar, `scaling_proto_energy` must be > 0")

        # -- Define class attributes to store the above values -- #

        self.param_mapping = param_mapping
        """dict or callable : mapping from external to internal params"""

        self.external_param_names = external_param_names
        """tuple of str : all param names that must be provided externally"""

        self.external_sph_pairs = tuple(external_sph_pairs)
        """tuple of 2-tuples of strings : (az_param, zen_param) pairs"""

        self.external_sph_pair_idxs = tuple(external_sph_pair_idxs)
        """tuple of 2-tuples of ints : (az_param_idx, zen_param_idx) pairs"""

        self.internal_param_names = tuple(internal_param_names)
        """tuple of str : all param names that are used internally"""

        self.internal_sph_pairs = tuple(internal_sph_pairs)
        """tuple of zero or more 2-tuples of strings : (az, zen) interal param name pairs"""

        self.internal_sph_pair_idxs = tuple(internal_sph_pair_idxs)
        """tuple of 2-tuples of ints : (az_param_idx, zen_param_idx) pairs"""

        self.num_external_params = len(self.external_param_names)
        """int : number of parameters used by the kernel"""

        self.num_internal_params = len(self.internal_param_names)
        """int : number of parameters used by the kernel"""

        self.scaling_proto_energy = scaling_proto_energy
        """float or None : energy at which prototypical sources were generated"""

        self.is_scaling = scaling_proto_energy is not None
        """bool : whether the kernel's sources are to be treated as scaling sources"""

        # Define dummy function
        def generate_sources(**kwargs): # pylint: disable=unused-argument
            """Must be replaced with your own callable"""
            raise NotImplementedError()

        self._generate_sources = generate_sources
        """callable : inheriting classes must override with a callable that returns an
        ndarray of dtype SRC_T"""

    def generate_sources(self, **kwargs):
        """Generate sources from the hypothesis when called via .. ::

            generate_sources(external_param_name0=value0, ...)

        where external param names and/or values are mapped to internal param names and
        values which are then passed to the `self._generate_sources` callable.

        Parameters
        ----------
        **kwargs
            Keyword arguments keyed by external param names

        Returns
        -------
        sources : length-n_sources ndarray of dtype SRC_T

        """
        internal_kwargs = self.param_mapping(**kwargs)
        return self._generate_sources(**internal_kwargs)
