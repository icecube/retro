# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, invalid-name

"""
Muon hypothesis class to generate photons expected from a muon.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "ContinuousLossModel",
    "StochasticLossModel",
    "MuonHypo",
]

__author__ = 'P. Eller, J.L. Lanfranchi, K. Crust'
__license__ = '''Copyright 2017-2018 Philipp Eller, Justin L. Lanfranchi, Kevin Crust

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


import csv
import enum
import math
from os.path import abspath, dirname, join
import sys

import numpy as np
from scipy import interpolate

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import numba_jit, DFLT_NUMBA_JIT_KWARGS
from retro.const import (
    COS_CKV, SIN_CKV, THETA_CKV, SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV,
    TRACK_PHOTONS_PER_M, SRC_CKV_BETA1, EMPTY_SOURCES
)
from retro.hypo import Hypo
from retro.retro_types import SRC_T
from retro.utils.misc import check_kwarg_keys, validate_and_convert_enum


MMC_A_PARAM = 0.268
"""MMC paper, Table 4"""
MMC_B_PARAM = 0.00047
"""MMC paper, Table 4"""


def mmc_muon_energy_to_length(muon_energy):
    """Convert muon energy into a length, as parameterized in MMC paper [1]
    (i.e., averaging all energy loss processes and parameterization optimized
    for 20 GeV - 1 TeV).

    Parameters
    ----------
    length

    Returns
    -------
    muon_energy

    References
    ----------
    [1] arXiv:hep-ph/0407075, Table 4

    """
    return np.log(muon_energy / MMC_A_PARAM * MMC_B_PARAM + 1) / MMC_B_PARAM


def mmc_muon_length_to_energy(muon_length):
    """Convert muon length into an energy, as parameterized in MMC paper [1]
    (i.e., averaging all energy loss processes and parameterization optimized
    for 20 GeV - 1 TeV).

    Parameters
    ----------
    length

    Returns
    -------
    muon_energy

    References
    ----------
    [1] arXiv:hep-ph/0407075, Table 4

    """
    return (np.exp(muon_length*MMC_B_PARAM) - 1) * MMC_A_PARAM / MMC_B_PARAM


def test_mmc_muon_energy_to_length():
    """Unit tests for `mmc_muon_energy_to_length` function"""
    rand = np.random.RandomState(0)
    muon_energies = rand.uniform(low=1, high=1000, size=int(1e4))
    # round-trip test
    assert np.allclose(
        mmc_muon_length_to_energy(
            mmc_muon_energy_to_length(muon_energies)
        ),
        muon_energies
    )


def generate_table_interpolators():
    """Generate interpolators for tabulated muon energy loss model.

    Returns
    -------
    muon_energy_to_length : scipy.interpolate.UnivariateSpline
        Call with a muon energy to return its expected length

    muon_length_to_energy : scipy.interpolate.UnivariateSpline
        Call with a muon length to return its expected energy

    energy_bounds : tuple of 2 floats
        lower, upper limits of table

    """
    # Create spline (for table_energy_loss_muon)
    with open(join(RETRO_DIR, 'data', 'dedx_total_e.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile) # pylint: disable=invalid-name
        rows = list(reader)

    energies = np.array([float(x) for x in rows[0][1:]])

    max_energy = np.max(energies)
    min_energy = np.min(energies)

    # table fails roundtrip test below ~0.117 GeV, so set lower bound to 0.12 GeV
    energy_bounds = (max(min_energy, 0.12), max_energy)

    stopping_power = np.array([float(x) for x in rows[1][1:]])
    dxde = interpolate.UnivariateSpline(
        x=energies,
        y=1/stopping_power,
        s=0,
        k=3,
        ext=2, # ValueError if exxtrapolating
    )
    esamps = np.logspace(
        np.log10(min_energy),
        np.log10(max_energy),
        int(1e4),
    )
    dxde_samps = np.clip(dxde(esamps), a_min=0, a_max=np.inf)

    lengths = [0]
    for idx in range(len(esamps[1:])):
        lengths.append(np.trapz(y=dxde_samps[:idx+1], x=esamps[:idx+1]))
    lengths = np.clip(np.array(lengths), a_min=0, a_max=np.inf)

    muon_energy_to_length = interpolate.UnivariateSpline(
        x=esamps,
        y=lengths,
        k=1,
        s=0,
        ext=2, # ValueError if exxtrapolating
    )
    muon_length_to_energy = interpolate.UnivariateSpline(
        x=lengths[1:],
        y=esamps[1:],
        k=1,
        s=0,
        ext=2, # ValueError if exxtrapolating
    )

    return muon_energy_to_length, muon_length_to_energy, energy_bounds


def test_generate_table_interpolator():
    """Unit tests for `generate_table_interpolators` function"""
    muon_energy_to_length, muon_length_to_energy, (min_energy, max_energy) = (
        generate_table_interpolators()
    )
    muon_energies = np.logspace(
        np.log10(min_energy),
        np.log10(max_energy),
        int(1e6),
    )
    # round-trip test
    isclose = np.isclose(
        muon_length_to_energy(
            muon_energy_to_length(muon_energies)
        ),
        muon_energies
    )
    if not np.all(isclose):
        muon_energies = muon_energies[~isclose]
        conv_muen = muon_length_to_energy(
            muon_energy_to_length(muon_energies)
        )
        print(min_energy, max_energy, muon_energies, conv_muen)
    assert np.all(isclose)


class ContinuousLossModel(enum.IntEnum):
    """Muon continouous-loss models"""

    all_avg_const = 0
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterized as a constant energy loss per meter the muon travels"""

    all_avg_mmc = 1
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterized as in MMC paper (arXiv:hep-ph/0407075), Table 4"""

    all_avg_table = 2
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterization interpolated from a table"""


class StochasticLossModel(enum.IntEnum):
    """Muon stochastic-loss models"""

    none = 0
    """no (explicit) stochastic loss process is to be modeled; the energy lost
    to stochastics can still be accounted for in the choice of
    `continuous_loss_model` as an average over all loss processes, but the
    light produced will be less accurately modeled"""

    scaled_cascades = 1
    """create discrete scaling cascades to model stochastics, possibly with
    regularization a la millipede (N. Whitehorn thesis, AAT 3505520,
    ISBN: 9781267299499, section 4.4)"""

    discrete_cascades = 2
    """stochastic loss modeled as discrete cascades (of one of the kinds in
    :class:`retro.cascade_hypo.CascadeModel`"""


class MuonHypo(Hypo):
    """
    Kernel to produce light sources expected from a muon hypothesis.

    Split sources into those from continuous losses and, if specified, stochastic
    losses.

    Continuous losses are modeled as producing only invisible secondaries and the only
    visible light produced is Cherenkov radiation coming off of the muon. The
    specific loss model which translates the total of continuous energy losses
    into the length of the muon is chosen by `continuous_loss_model`. If no stochastic
    losses are modeled, then a loss model that averages all loss mechanisms is
    recommended.

    Stochastic losses produce what appear as cascades originating along the muon's path,
    but as of now, none are actually implemented in this class; hooks have been added,
    though, to hopefully make their implementation more straightforward.

    Parameters
    ----------
    continuous_loss_model : enum ContinuousLossModel, str, or int
        see :class:`ContinuousLossModel` for choices

    stochastic_loss_model : enum StochasticLossModel, str, int, or None
        see :class:`ContinuousLossModel` for choices; a value of `None` maps to
        ContinuousLossModel.none

    fixed_track_length : scalar, optional
        force generation of this many sources from the continous-loss model
        regardless of any specified energy (the hypothesis will not utilize
        an `energy` parameter), e.g. for pegleg-ing; if None or <= 0 is passed,
        the hypothesis number of sources depends on the energy specified

    param_mapping : Mapping or callable
        see :class:`retro.hypo.Hypo` for details

    external_sph_pairs : tuple, optional
        see :class:`retro.hypo.Hypo` for details

    continuous_loss_model_kwargs : mapping
        keyword arguments required by chosen `continuous_loss_model`;
        if parameters in addition to those required by the model are passed, a
        ValueError will be raised
            * `all_avg_const`, `all_avg_mmc`, and `all_avg_table` take a "time_step"
              parmeter, in units of nanoseconds

    stochastic_loss_model_kwargs : mapping
        keyword arguments required by chosen `stochastic_loss_model`,
        if any are required.

    """
    def __init__(
        self,
        continuous_loss_model,
        stochastic_loss_model,
        param_mapping,
        external_sph_pairs=None,
        fixed_track_length=None,
        continuous_loss_model_kwargs=None,
        stochastic_loss_model_kwargs=None,
    ):
        # -- Validation and translation of args -- #

        # `continuous_loss_model`

        continuous_loss_model = validate_and_convert_enum(
            val=continuous_loss_model,
            enum_type=ContinuousLossModel,
        )

        # `stochastic_loss_model`

        stochastic_loss_model = validate_and_convert_enum(
            val=stochastic_loss_model,
            enum_type=StochasticLossModel,
            none_evaluates_to=StochasticLossModel.none,
        )

        # `fixed_track_length`

        if fixed_track_length is None or fixed_track_length <= 0:
            fixed_track_length = 0

        # `continuous_loss_model_kwargs`

        if continuous_loss_model in (
            ContinuousLossModel.all_avg_const,
            ContinuousLossModel.all_avg_mmc,
            ContinuousLossModel.all_avg_table,
        ):
            required_keys = ("time_step",)
        else:
            raise NotImplementedError(
                "{} muon continuous loss model not implemented"
                .format(continuous_loss_model.name)
            )

        if continuous_loss_model_kwargs is None:
            continuous_loss_model_kwargs = {}
        check_kwarg_keys(
            required_keys=required_keys,
            provided_kwargs=continuous_loss_model_kwargs,
            meta_name="`continuous_loss_model_kwargs`",
            message_pfx=(
                "{} muon continuous loss model:"
                .format(continuous_loss_model.name)
            ),
        )

        # `stochastic_loss_model_kwargs`

        if stochastic_loss_model is StochasticLossModel.none:
            required_keys = ()
        elif stochastic_loss_model is StochasticLossModel.scaled_cascades:
            required_keys = ("cascade_model", "spacing",)
        else:
            raise NotImplementedError(
                "{} muon stochastic loss model not implemented"
                .format(continuous_loss_model.name)
            )

        if stochastic_loss_model_kwargs is None:
            stochastic_loss_model_kwargs = {}
        check_kwarg_keys(
            required_keys=required_keys,
            provided_kwargs=stochastic_loss_model_kwargs,
            meta_name="`stochastic_loss_model_kwargs`",
            message_pfx=(
                "{} muon stochastic loss model:"
                .format(stochastic_loss_model.name)
            ),
        )

        # -- Functons to find expected length from energy and vice versa -- #

        if continuous_loss_model is ContinuousLossModel.all_avg_const:
            self.muon_energy_to_length = lambda e: e / TRACK_M_PER_GEV
            self.muon_length_to_energy = lambda l: l * TRACK_M_PER_GEV
            self.muon_energy_bounds = (0, np.inf)
            self.muon_length_bounds = (0, np.inf)
        elif continuous_loss_model is ContinuousLossModel.all_avg_mmc:
            self.muon_energy_to_length = mmc_muon_energy_to_length
            self.muon_length_to_energy = mmc_muon_length_to_energy
            self.muon_energy_bounds = (0, np.inf)
            self.muon_length_bounds = (0, np.inf)
        elif continuous_loss_model is ContinuousLossModel.all_avg_table:
            self.muon_energy_to_length, self.muon_length_to_energy, bounds = (
                generate_table_interpolators()
            )
            self.muon_energy_bounds = tuple(bounds)
            self.muon_length_bounds = tuple(self.muon_energy_to_length(bounds))
        else:
            raise NotImplementedError()

        # -- Define `internal_param_names` -- #

        # All models have the following...
        internal_param_names = ("x", "y", "z", "time", "azimuth", "zenith")

        # If pegleg-ing (fixed_track_length > 0), energy/length are found
        # within the llh function, while this kernel simply produces _all_ possible
        # sources once
        if fixed_track_length == 0:
            if stochastic_loss_model is StochasticLossModel.none:
                internal_param_names += ("continuous_energy",)
            elif stochastic_loss_model is StochasticLossModel.scaled_cascades:
                raise NotImplementedError()

        # -- Initialize base class (retro.hypo.Hypo) -- #

        super(MuonHypo, self).__init__(
            param_mapping=param_mapping,
            internal_param_names=internal_param_names,
            external_sph_pairs=external_sph_pairs,
        )

        # -- Store attrs unique to a muon hypo -- #

        self.continuous_loss_model = continuous_loss_model
        self.stochastic_loss_model = stochastic_loss_model
        self.fixed_track_length = fixed_track_length
        self.continuous_loss_model_kwargs = continuous_loss_model_kwargs
        self.stochastic_loss_model_kwargs = stochastic_loss_model_kwargs

        # -- Create the _generate_sources function -- #

        self._create_source_generator_func()

    def _create_source_generator_func(self):
        muon_energy_to_length = self.muon_energy_to_length
        muon_length_to_energy = self.muon_length_to_energy
        min_energy, max_energy = self.muon_energy_bounds
        min_length = muon_energy_to_length(min_energy)
        time_step = self.continuous_loss_model_kwargs["time_step"]
        segment_length = time_step * SPEED_OF_LIGHT_M_PER_NS
        photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

        fixed_track_length = self.fixed_track_length
        if fixed_track_length > 0:
            if fixed_track_length < min_length:
                raise ValueError(
                    "(fixed_track_length = {} m) is less than (min_length = {} m)"
                    .format(fixed_track_length, min_length)
                )
            fixed_energy = muon_length_to_energy(fixed_track_length)
            assert min_energy <= fixed_energy <= max_energy
        else:
            fixed_track_length = 0
            fixed_energy = 0

        def _continuous_generator(x, y, z, time, azimuth, zenith, continuous_energy):
            if continuous_energy == 0:
                return [EMPTY_SOURCES], [False]

            if fixed_track_length > 0:
                length = fixed_track_length
            else:
                if continuous_energy < min_energy:
                    length = 0
                elif continuous_energy > max_energy:
                    raise ValueError("continuous_energy > max_energy")
                else:
                    length = muon_energy_to_length(continuous_energy)

            sampled_dt = np.arange(
                start=time_step*0.5,
                stop=length/SPEED_OF_LIGHT_M_PER_NS,
                step=time_step,
            )

            # At least one segment
            if len(sampled_dt) == 0:
                sampled_dt = np.array([length/2./SPEED_OF_LIGHT_M_PER_NS])

            # NOTE: add pi to make dir vector go in "math-standard" vector notation
            # (vector components point in direction of motion), as opposed to "IceCube"
            # vector notation (vector components point opposite to direction of
            # motion).
            opposite_zenith = np.pi - zenith
            opposite_azimuth = np.pi + azimuth

            dir_costheta = math.cos(opposite_zenith)
            dir_sintheta = math.sin(opposite_zenith)

            dir_cosphi = np.cos(opposite_azimuth)
            dir_sinphi = np.sin(opposite_azimuth)

            dir_x = dir_sintheta * dir_cosphi
            dir_y = dir_sintheta * dir_sinphi
            dir_z = dir_costheta

            sources = np.empty(shape=sampled_dt.shape, dtype=SRC_T)

            sources['kind'] = SRC_CKV_BETA1
            sources['time'] = time + sampled_dt
            sources['x'] = x + sampled_dt * (dir_x * SPEED_OF_LIGHT_M_PER_NS)
            sources['y'] = y + sampled_dt * (dir_y * SPEED_OF_LIGHT_M_PER_NS)
            sources['z'] = z + sampled_dt * (dir_z * SPEED_OF_LIGHT_M_PER_NS)
            sources['photons'] = photons_per_segment

            sources['dir_costheta'] = dir_costheta
            sources['dir_sintheta'] = dir_sintheta

            sources['dir_phi'] = opposite_azimuth
            sources['dir_cosphi'] = dir_cosphi
            sources['dir_sinphi'] = dir_sinphi

            sources['ckv_theta'] = THETA_CKV
            sources['ckv_costheta'] = COS_CKV
            sources['ckv_sintheta'] = SIN_CKV

            return sources

        # TODO: implement stochastic loss model(s)
        assert self.stochastic_loss_model is StochasticLossModel.none

        if fixed_track_length > 0:
            def _generate_sources(x, y, z, time, azimuth, zenith):
                return _continuous_generator(
                    x=x,
                    y=y,
                    z=z,
                    time=time,
                    azimuth=azimuth,
                    zenith=zenith,
                    continuous_energy=fixed_energy,
                )
            self._generate_sources = _generate_sources
        else:
            self._generate_sources = _continuous_generator

    def pegleg_eval(self, pegleg_idx):
        """Convert a pegleg index into track energy in GeV.

        Parameters
        ----------
        pegleg_idx : int

        Returns
        -------
        muon_energy : float

        """
        length = (
            pegleg_idx
            * self.continuous_loss_model_kwargs["time_step"]
            * SPEED_OF_LIGHT_M_PER_NS
        )
        return self.muon_length_to_energy(length)


if __name__ == "__main__":
    test_mmc_muon_energy_to_length()
    test_generate_table_interpolator()
