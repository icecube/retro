# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Cascade hypothesis class to generate photons expected from a cascade.
"""

from __future__ import absolute_import, division, print_function

__all__ = ["EM_CASCADE_PHOTONS_PER_GEV", "CascadeModel", "CascadeHypo"]

__author__ = "P. Eller, J.L. Lanfranchi"
__license__ = """Copyright 2017-2018 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from collections import Iterable, OrderedDict
import math
from numbers import Number
from os.path import abspath, dirname
import sys

import enum
import numpy as np
from scipy.stats import gamma, pareto

if __name__ == "__main__" and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import (
    EMPTY_SOURCES, SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI, SRC_CKV_BETA1, SrcHandling,
    dummy_pegleg_gens
)
from retro.hypo_future import Hypo
from retro.retro_types import SRC_T
from retro.utils.misc import check_kwarg_keys, validate_and_convert_enum


EM_CASCADE_PHOTONS_PER_GEV = 12818.970 #12805.3383311
"""Cascade photons per energy, in units of 1/GeV (see
``retro/i3info/track_and_cascade_photon_parameterizations.py``)"""


# TODO: a "pegleg"-like cascade which can add sources to an existing set to increase the
# energy of the cascade while maintaining more accurate topolgy with higher energy (as
# opposed to the scaling cascade, which has a fixed topology at one energy and simply
# scales luminosity of those sources to increase energy)

class CascadeModel(enum.IntEnum):
    """Cascade models defined"""

    spherical = 0
    """spherical point-radiator"""

    one_dim_v1 = 1
    """parameterization inspired by but not strictly following arXiv:1210.5140v2; set
    num_sources to 1 to produce a point-like Chrenkov cascade"""


class CascadeHypo(Hypo):
    """
    Kernel to produce light sources expected from a cascade hypothesis.

    Parameters
    ----------
    model : CascadeModel enum or convertible thereto
        see :class:`CascadeModel` for valid values

    param_mapping : dict-like (mapping)
        see docs for :class:`retro.hypo.Hypo` for details

    external_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        see docs for :class:`retro.hypo.Hypo` for details

    num_sources : int, optional
        specify an integer >= 1 to fix the number of sources, or specify None or an
        integer <= 0 to dynamically adjust the number of sources based on the energy of
        the cascade. See `retro/notebooks/energy_dependent_cascade_num_samples.ipynb`
        for the logic behind how `num_sources` is computed in this case

    scaling_proto_energy : None or scalar > 0
        specify None to disable or specify scalar > 0 to treat the cascade as
        "scaling," i.e., a prototypical set of light sources are generated for the
        energy specified and modifying the energy from that merely scales the luminosity
        of each of those sources as opposed to generating an entirely new set of light
        sources; in this case, the topology of the cascade will not be as accurate but
        the speed of computing likelihoods increases

    model_kwargs : mapping
        Additional keyword arguments required by chosen cascade `model`; if parameters
        in addition to those required by the model are passed, a ValueError will be
        raised
            * `spherical`, `one_dim_v1` take no additional parameters

    """
    def __init__(
        self,
        model,
        param_mapping,
        external_sph_pairs=None,
        num_sources=None,
        scaling_proto_energy=None,
        model_kwargs=None,
    ):
        # -- Validation and translation of args -- #

        # `model`

        model = validate_and_convert_enum(val=model, enum_type=CascadeModel)

        # `num_sources`

        if not (
            num_sources is None
            or (
                isinstance(num_sources, Number)
                and int(num_sources) == num_sources
            )
        ):
            raise TypeError("`num_sources` must be an integer or None")

        if num_sources is None or num_sources <= 0:
            num_sources = -1
        else:
            num_sources = int(num_sources)

        if model == CascadeModel.spherical:
            if num_sources < 1:
                num_sources = 1
            elif num_sources != 1:
                raise ValueError(
                    "spherical cascade only valid with `num_sources=1` or auto (< 0);"
                    " got `num_sources` = {}".format(num_sources)
                )
            internal_param_names = ("x", "y", "z", "time", "energy")
        else: # model == CascadeModel.one_dim_v1
            internal_param_names = ("x", "y", "z", "time", "energy", "azimuth", "zenith")

        if not (
            scaling_proto_energy is None
            or isinstance(scaling_proto_energy, Number)
        ):
            raise TypeError("`scaling_proto_energy` must be None or a scalar")

        # `scaling_proto_energy`

        if isinstance(scaling_proto_energy, Number) and scaling_proto_energy <= 0:
            raise ValueError("If scalar, `scaling_proto_energy` must be > 0")

        is_scaling = scaling_proto_energy is not None
        if is_scaling:
            internal_param_names = tuple(
                p for p in internal_param_names if p != "energy"
            )

        # `model_kwargs`

        if model_kwargs is None:
            model_kwargs = {}

        if model in (CascadeModel.spherical, CascadeModel.one_dim_v1):
            required_keys = ()
        else:
            raise NotImplementedError(
                "{} cascade model not implemented".format(model.name)
            )

        check_kwarg_keys(
            required_keys=required_keys,
            provided_kwargs=model_kwargs,
            meta_name="`model_kwargs`",
            message_pfx="{} cascade model:".format(model.name), # pylint: disable=no-member
        )

        # -- Initialize base class -- #

        super(CascadeHypo, self).__init__(
            param_mapping=param_mapping,
            internal_param_names=internal_param_names,
            external_sph_pairs=external_sph_pairs,
        )
        self.max_num_scalefactors = 1 if is_scaling else 0

        # -- Store attrs unique to a cascade -- #

        self.model = model
        self.num_sources = num_sources
        self.scaling_proto_energy = scaling_proto_energy
        self.is_scaling = is_scaling

        # -- Record configuration items unique to a muon hypo -- #

        self.config["model"] = model.name # pylint: disable=no-member
        self.config["num_sources"] = num_sources
        self.config["scaling_proto_energy"] = scaling_proto_energy

        # -- Create the self._get_sources attribute/callable -- #

        self._create_get_sources_func()

    def _create_get_sources_func(self):
        """Create the function that generates photon sources for a hypothesis.

        The created function is attached to this class as the private attribute
        `self._get_sources` and is intended to be called externally from
        `self.get_sources` (which does the appropriate translations from
        external names/values to internal names/values as used by _get_sources).

        """
        is_scaling = self.is_scaling
        if is_scaling:
            src_handling = SrcHandling.scaling
        else:
            src_handling = SrcHandling.nonscaling

        scaling_proto_energy = self.scaling_proto_energy

        if self.model == CascadeModel.spherical:

            def __get_sources(time, x, y, z, energy):
                """Point-like spherically-radiating cascade.

                Parameters
                ----------
                time, x, y, z, energy : scalars

                Returns
                -------
                sources
                sources_handling
                num_pegleg_generators
                pegleg_generators

                """
                if energy == 0:
                    return (EMPTY_SOURCES,), (SrcHandling.none,), 0, dummy_pegleg_gens

                sources = np.empty(shape=(1,), dtype=SRC_T)
                sources[0]["kind"] = SRC_OMNI
                sources[0]["time"] = time
                sources[0]["x"] = x
                sources[0]["y"] = y
                sources[0]["z"] = z
                sources[0]["photons"] = EM_CASCADE_PHOTONS_PER_GEV * energy

                return (sources,), (SrcHandling.none,), 0, dummy_pegleg_gens

            if is_scaling:
                def ___get_sources(time, x, y, z): # pylint: disable=missing-docstring
                    return __get_sources(
                        time=time,
                        x=x,
                        y=y,
                        z=z,
                        energy=scaling_proto_energy,
                    )
                ___get_sources.__doc__ = (
                    __get_sources.__doc__.replace(", energy", "")
                )
                _get_sources = ___get_sources
            else:
                _get_sources = __get_sources

        elif self.model == CascadeModel.one_dim_v1:
            # TODO: use quasi-random (low discrepancy) numbers instead of pseudo-random
            #       (e.g., Sobol sequence)

            # Create samples from angular zenith distribution.
            max_num_sources = int(1e5)
            if self.num_sources > max_num_sources:
                raise ValueError(
                    "Can only produce up to {} sources".format(max_num_sources)
                )

            # Parameterizations from arXiv:1210.5140v2
            zen_dist = pareto(b=1.91833423, loc=-22.82924369, scale=22.82924369)
            random_state = np.random.RandomState(0)
            precomputed_zen_samples = np.deg2rad(
                np.clip(
                    zen_dist.rvs(size=max_num_sources, random_state=random_state),
                    a_min=0,
                    a_max=180,
                )
            )

            # Create samples from angular azimuth distribution
            random_state = np.random.RandomState(2)
            precomputed_az_samples = random_state.uniform(
                low=0,
                high=2*np.pi,
                size=max_num_sources,
            )

            param_alpha = 2.01849
            param_beta = 1.45469

            min_cascade_energy = np.ceil(10**(-param_alpha / param_beta) * 100) / 100

            rad_len = 0.3975
            param_b = 0.63207
            rad_len_over_b = rad_len / param_b

            # numba closure doesn't have access to attributes of `self`, so extract
            # attributes we need as "regular" variables
            num_sources = self.num_sources

            def compute_actual_num_sources(num_sources, energy=None):
                """Compute actual number of sources to use given a specification
                `num_sources` for number of sources to use; take "limits" (minimum
                energy) into account and dynamically compute `actual_num_sources` if
                `num_sources` is < 0.

                Parameters
                ----------
                num_sources : int
                    if < 0, compute num_sources based on `energy`

                energy : scalar > 0, required only if `num_sources` < 0

                Returns
                -------
                actual_num_sources : int > 0

                """
                if num_sources < 0:
                    # Note that num_sources must be 1 for energy <= min_cascade_energy
                    # (param_a goes <= 0 at this value and below, causing an exception from
                    # gamma distribution)
                    if energy <= min_cascade_energy:
                        actual_num_sources = 1
                    else:
                        # See `retro/notebooks/energy_dependent_cascade_num_samples.ipynb`
                        actual_num_sources = int(np.round(
                            np.clip(
                                math.exp(0.77 * math.log(energy) + 2.3),
                                a_min=1,
                                a_max=None,
                            )
                        ))
                else:
                    actual_num_sources = num_sources
                return actual_num_sources

            def compute_longitudinal_samples(num_sources, energy):
                """Create longitudinal distribution of cascade's light sources.

                See arXiv:1210.5140v2

                Parameters
                ----------
                num_sources : int
                energy : scalar > 0

                Returns
                -------
                longitudinal_samples : length-`n_samples` ndarray of dtype float64

                """
                param_a = (
                    param_alpha
                    + (
                        param_beta
                        * math.log10(max(min_cascade_energy, energy))
                    )
                )
                # ~70% of exec time:
                longitudinal_dist = gamma(param_a, scale=rad_len_over_b)
                # ~10% of execution time:
                longitudinal_samples = longitudinal_dist.rvs(size=num_sources, random_state=1)
                return longitudinal_samples

            if is_scaling:
                actual_num_sources = compute_actual_num_sources(
                    num_sources=num_sources,
                    energy=scaling_proto_energy,
                )
                precomputed_long_samples = compute_longitudinal_samples(
                    num_sources=actual_num_sources,
                    energy=scaling_proto_energy,
                )
            else:
                # Define things just so they exist, even though we won't use them
                # (needed for Numba-compiled closures)
                actual_num_sources = num_sources
                precomputed_long_samples = np.empty(0)

            # TODO: speed up (~800 µs to generate sources at 10 GeV)...
            # * 70% of the time is spent instantiating the gamma dist
            # * 10% of the time is executing the `rvs` method of the gamma dist
            def __get_sources(time, x, y, z, energy, azimuth, zenith):
                """Cascade with both longitudinal and angular distributions (but no
                distribution off-axis). All emitters are located on the shower axis.

                Use as a hypo_kernel with the DiscreteHypo class.

                Note that the number of samples is proportional to the energy of the
                cascade.

                Parameters
                ----------
                time, x, y, z, energy, azimuth, zenith

                Returns
                -------
                sources
                source_handling
                num_pegleg_generators
                pegleg_generators

                """
                if energy == 0:
                    return (EMPTY_SOURCES,), (SrcHandling.none,), 0, dummy_pegleg_gens

                if is_scaling:
                    n_sources = actual_num_sources
                else:
                    n_sources = compute_actual_num_sources(
                        num_sources=num_sources,
                        energy=energy,
                    )

                opposite_zenith = np.pi - zenith
                opposite_azimuth = azimuth + np.pi

                sin_zen = math.sin(opposite_zenith)
                cos_zen = math.cos(opposite_zenith)
                sin_az = math.sin(opposite_azimuth)
                cos_az = math.cos(opposite_azimuth)
                dir_x = sin_zen * cos_az
                dir_y = sin_zen * sin_az
                dir_z = cos_zen

                if n_sources == 1:
                    sources = np.empty(shape=(1,), dtype=SRC_T)
                    sources[0]["kind"] = SRC_CKV_BETA1
                    sources[0]["time"] = time
                    sources[0]["x"] = x
                    sources[0]["y"] = y
                    sources[0]["z"] = z
                    sources[0]["photons"] = EM_CASCADE_PHOTONS_PER_GEV * energy

                    sources[0]["dir_costheta"] = cos_zen
                    sources[0]["dir_sintheta"] = sin_zen

                    sources[0]["dir_phi"] = opposite_azimuth
                    sources[0]["dir_cosphi"] = cos_az
                    sources[0]["dir_sinphi"] = sin_az

                    return (sources,), (src_handling,), 0, dummy_pegleg_gens

                # Create rotation matrix
                rot_mat = np.array(
                    [[cos_az * cos_zen, -sin_az, cos_az * sin_zen],
                     [sin_az * cos_zen, cos_zen, sin_az * sin_zen],
                     [-sin_zen, 0, cos_zen]]
                )

                if is_scaling:
                    longitudinal_samples = precomputed_long_samples
                else:
                    longitudinal_samples = compute_longitudinal_samples(
                        num_sources=n_sources,
                        energy=energy,
                    )

                # Grab precomputed samples from angular zenith distribution
                zen_samples = precomputed_zen_samples[:n_sources]

                # Grab precomputed samples from angular azimuth distribution
                az_samples = precomputed_az_samples[:n_sources]

                # Create angular vectors distribution
                sin_zen = np.sin(zen_samples)
                x_ang_dist = sin_zen * np.cos(az_samples)
                y_ang_dist = sin_zen * np.sin(az_samples)
                z_ang_dist = np.cos(zen_samples)
                ang_dist = np.concatenate(
                    (
                        x_ang_dist[np.newaxis, :],
                        y_ang_dist[np.newaxis, :],
                        z_ang_dist[np.newaxis, :]
                    ),
                    axis=0,
                )

                final_ang_dist = np.dot(rot_mat, ang_dist)
                final_phi_dist = np.arctan2(final_ang_dist[1], final_ang_dist[0])
                final_theta_dist = np.arccos(final_ang_dist[2])

                # Define photons per sample
                photons_per_sample = EM_CASCADE_PHOTONS_PER_GEV * energy / n_sources

                # Create photon array
                sources = np.empty(shape=n_sources, dtype=SRC_T)

                sources["kind"] = SRC_CKV_BETA1

                sources["time"] = time + longitudinal_samples / SPEED_OF_LIGHT_M_PER_NS
                sources["x"] = x + longitudinal_samples * dir_x
                sources["y"] = y + longitudinal_samples * dir_y
                sources["z"] = z + longitudinal_samples * dir_z

                sources["photons"] = photons_per_sample

                sources["dir_costheta"] = final_ang_dist[2]
                sources["dir_sintheta"] = np.sin(final_theta_dist)

                sources["dir_phi"] = final_phi_dist
                sources["dir_cosphi"] = np.cos(final_phi_dist)
                sources["dir_sinphi"] = np.sin(final_phi_dist)

                return (sources,), (src_handling,), 0, dummy_pegleg_gens

            if is_scaling:
                def ___get_sources(time, x, y, z, azimuth, zenith): # pylint: disable=missing-docstring
                    return __get_sources(
                        time=time,
                        x=x,
                        y=y,
                        z=z,
                        energy=scaling_proto_energy,
                        azimuth=azimuth,
                        zenith=zenith,
                    )
                ___get_sources.__doc__ = (
                    __get_sources.__doc__.replace(", energy", "")
                )
                _get_sources = ___get_sources
            else:
                _get_sources = __get_sources

        else:
            raise NotImplementedError(
                "{} cascade model is not implemented".format(self.model.name) # pylint: disable=no-member

            )

        self._get_sources = _get_sources

    def get_energy(self, pegleg_indices=None, scalefactors=None):
        """Get cascade energy.

        Parameters
        ----------
        pegleg_indices : must be None
        scalefactors : scalar or iterable of one scalar; required if is_scaling

        Returns
        -------
        energy
            Energy of cascade in GeV

        """
        assert pegleg_indices is None

        if isinstance(scalefactors, Iterable):
            scalefactors = tuple(scalefactors)[0]

        if self.is_scaling:
            assert scalefactors is not None
            return self.scaling_proto_energy * scalefactors

        return self.internal_params["energy"]

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
        derived_params = OrderedDict()

        # If scaling, energy is derived from scaling_proto_energy & scalefactor
        if self.is_scaling:
            derived_params["energy"] = self.get_energy(
                pegleg_indices=pegleg_indices,
                scalefactors=scalefactors,
            )

        return derived_params


def test_CascadeHypo():
    """Unit tests for CascadeHypo class"""
    dict_param_mapping = dict(
        x="x", y="y", z="z", time="time", cascade_energy="energy",
        cascade_azimuth="azimuth", cascade_zenith="zenith",
    )
    scaling_dict_param_mapping = {
        k: v for k, v in dict_param_mapping.items() if v != "energy"
    }
    sph_dict_param_mapping = {
        k: v for k, v in dict_param_mapping.items() if "zen" not in k and "az" not in k
    }
    sph_dict_scaling_param_mapping = {
        k: v for k, v in scaling_dict_param_mapping.items() if "zen" not in k and "az" not in k
    }

    def callable_param_mapping(
        x, y, z, time, cascade_energy, cascade_azimuth, cascade_zenith, **kwargs
    ): # pylint: disable=missing-docstring, unused-argument
        return dict(x=x, y=y, z=z, time=time, energy=cascade_energy,
                    azimuth=cascade_azimuth, zenith=cascade_zenith)

    def callable_scaling_param_mapping(
        x, y, z, time, cascade_azimuth, cascade_zenith, **kwargs
    ): # pylint: disable=missing-docstring, unused-argument
        return dict(x=x, y=y, z=z, time=time, azimuth=cascade_azimuth,
                    zenith=cascade_zenith)

    params = dict(x=0, y=0, z=0, time=0, cascade_energy=50, cascade_azimuth=np.pi/2,
                  cascade_zenith=np.pi/4)

    sph_params = {k: v for k, v in params.items() if "zen" not in k and "az" not in k}

    # dict for param mapping, enum model, dynamic num sources, not scaling
    cscd = CascadeHypo(
        param_mapping=dict_param_mapping,
        model=CascadeModel.one_dim_v1,
        num_sources=-1,
        scaling_proto_energy=None,
    )
    _, _, _, _ = cscd.get_sources(**params)

    # dict for param mapping, enum model, dynamic num sources, not scaling
    cscd = CascadeHypo(
        param_mapping=callable_param_mapping,
        model="one_dim_v1",
        num_sources=100,
        scaling_proto_energy=None,
    )

    _, _, _, _ = cscd.get_sources(**params)
    # callable for param mapping, str model, fixed num sources, scaling
    cscd = CascadeHypo(
        param_mapping=callable_scaling_param_mapping,
        model="one_dim_v1",
        num_sources=100,
        scaling_proto_energy=100,
    )
    _, _, _, _ = out = cscd.get_sources(**params)
    print(out[0][0][:10])
    print(out[1][0])

    # dict for param mapping, int model, auto num sources, scaling
    cscd = CascadeHypo(
        param_mapping=sph_dict_scaling_param_mapping,
        model=int(CascadeModel.spherical),
        num_sources=-1,
        scaling_proto_energy=100,
    )
    _, _, _, _ = cscd.get_sources(**sph_params)

    # dict for param mapping, int model, fixed num sources, not scaling
    cscd = CascadeHypo(
        param_mapping=sph_dict_param_mapping,
        model=CascadeModel.spherical,
        num_sources=1,
    )
    _, _, _, _ = cscd.get_sources(**sph_params)

    print("<< PASS : test_CascadeHypo >>")


if __name__ == "__main__":
    test_CascadeHypo()
