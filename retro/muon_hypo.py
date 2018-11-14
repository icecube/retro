# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name, invalid-name

"""
Muon hypothesis class to generate photons expected from a muon.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "TRACK_M_PER_GEV",
    "TRACK_PHOTONS_PER_M",
    "ConstABModel",
    "CONST_AB_MODEL_PARAMS",
    "ContinuousLossModel",
    "StochasticLossModel",
    "generate_const_a_b_converters",
    "test_generate_const_a_b_converters",
    "generate_gms_table_converters",
    "test_generate_gms_table_converters",
    "generate_min_energy_fit_converters",
    "test_generate_min_energy_fit_converters",
    "const_muon_energy_to_length",
    "const_muon_length_to_energy",
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

import enum
import math
from os.path import abspath, dirname, join
import sys

import numpy as np

RETRO_DIR = dirname(dirname(abspath(__file__)))
if __name__ == '__main__' and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DFLT_NUMBA_JIT_KWARGS, numba_jit
from retro.const import (
    ICE_DENSITY, MUON_REST_MASS, SPEED_OF_LIGHT_M_PER_NS, SRC_CKV_BETA1, EMPTY_SOURCES
)
from retro.hypo import Hypo, SrcHandling
from retro.retro_types import SRC_T
from retro.utils.misc import check_kwarg_keys, validate_and_convert_enum
from retro.utils.lerp import generate_lerp


# TODO: add ice density correction, bedrock, and possibly even air to muon model(s); see
#       `retro/notebooks/muon_length_vs_energy.ipynb`
# TODO: implement stochastic loss model(s)
# TODO: pegleg able to take logarithmic steps


TRACK_PHOTONS_PER_M = 2451.4544553
"""Track Cherenkov photons per length, in units of 1/m (see ``nphotons.py``)"""

TRACK_M_PER_GEV = 15 / 3.3
"""Constant model track length per energy, in units of m/GeV"""


class ConstABModel(enum.IntEnum):
    """Muon range <--> muon enegy simplifies if `a` (ionization loss) and `b` (stochastic
    loss) parameters are constant to ::

        range = log(1 + energy * b / a) / b

    Fits for `a` and `b` were done in the following:
        * MMC paper, table 3 [1]
        * MMC paper, table 4 [1]
        * PROPOSAL paper [2]
        * LEERA2 source code [3] (those are the values actually used here; rounded
          versions of the same were reported in the LEERA internal IceCube report [4])

    References
    ----------
    [1] arXiv:hep-ph/0407075
    [2] http://dx.doi.org/10.1016/j.cpc.2013.04.001
    [3] http://code.icecube.wisc.edu/svn/sandbox/terliuk/LEERA2/trunk_spl_tables/python/LEERA.py
    [4] https://internal-apps.icecube.wisc.edu/reports/data/icecube/2013/04/001/icecube_201304001_v2.pdf

    """
    proposal = 1
    leera2 = 2
    mmc3 = 3
    mmc4 = 4


CONST_AB_MODEL_PARAMS = {
    ConstABModel.proposal: dict(a=0.249, b=0.422e-3),
    ConstABModel.leera2: dict(a=0.225649, b=0.00046932),
    ConstABModel.mmc3: dict(a=0.259, b=0.363e-3),
    ConstABModel.mmc4: dict(a=0.268, b=0.47e-3),
}
"""`a` & `b` parameter values for various fits to constant-a&b model.
See :class:`ConstABModel` for details"""


class ContinuousLossModel(enum.IntEnum):
    """Muon continouous-loss models"""

    all_avg_const = 0
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterized as a constant energy loss per meter the muon travels"""

    all_avg_mmc3 = 1
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterized as in MMC paper (arXiv:hep-ph/0407075), Table 3"""

    all_avg_mmc4 = 2
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterized as in MMC paper (arXiv:hep-ph/0407075), Table 4"""

    all_avg_leera2 = 3

    all_avg_proposal = 4

    all_avg_gms_table = 5
    """energy loss averaged across all loss processes (continuous+stochastic),
    parameterization interpolated from a table"""

    min_energy_fit = 6
    """Assign the minimum energy to a given muon length seen in simulation; assumes
    energy will be added to this via some stochastic-loss model"""


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


def generate_const_a_b_converters(a=None, b=None, model=None):
    """Factory to generate functions to convert between muon length and energy, based on
    a simplified model of average energy losses. See :func:`ConstABModel` for details of
    the model.

    Either specify values for both `a` and `b` (and don't specfy `model`) or specify a
    `model` (and don't specify `a` or `b`).

    Parameters
    ----------
    a : float
        `a` parameter for a custom const-a / const-b muon energy loss model. Units are
        GeV/mwe. If `a` is specified, `b` must also be specified and `model` must _not_
        be specified.

    b : float
        `b` parameter for a custom const-a / const-b muon energy loss model. Units are
        1/mwe. If `b` is specified, `a` must also be specified and `model` must _not_
        be specified.

    model : ConstABModel enum or str or int convertible thereto
        See :class:`ConstABModel` for valid values

    Returns
    -------
    muon_energy_to_length : callable
    muon_length_to_energy : callable

    """
    if a is not None or b is not None:
        assert a is not None and b is not None
        assert model is None
        a = float(a)
        b = float(b)
        model_descr = "with a ~ {:.4e} & b ~ {:.4e}".format(a, b)

    if model is not None:
        assert a is None and b is None
        model_descr = "using a and b from {} model".format(model.name)

    if model is not None:
        model = validate_and_convert_enum(
            val=model,
            enum_type=ConstABModel,
        )
        params = CONST_AB_MODEL_PARAMS[model]
        a = params["a"]
        b = params["b"]

    a_over_b = a / b
    b_over_a = b / a

    # -- Define functions -- #

    def muon_energy_to_length(muon_energy): # pylint: disable=missing-docstring
        return np.log(1 + muon_energy * b_over_a) / b

    def muon_length_to_energy(muon_length): # pylint: disable=missing-docstring
        return (np.exp(muon_length * b) - 1) * a_over_b

    # -- Add docstring to functions -- #

    docstr = """Convert a muon {src} to expected {dst}.

        Model assumes `a` and `b` are constant, whereupon the range integral evaluates
        to ::

            range = log(1 + muon_energy * b/a) / b

        {model_descr}

        Parameters
        ----------
        muon_{src}

        Returns
        -------
        muon_{dst}

        """

    muon_energy_to_length.__doc__ = docstr.format(
        src="energy", dst="length", model_descr=model_descr,
    )
    muon_length_to_energy.__doc__ = docstr.format(
        src="length", dst="energy", model_descr=model_descr,
    )

    return muon_energy_to_length, muon_length_to_energy


def test_generate_const_a_b_converters():
    """Unit tests for `generate_const_a_b_converters` and the functions it produces."""
    required_param_names = set(["a", "b"])
    for model in list(ConstABModel):
        # parameters exist for all models
        assert model in CONST_AB_MODEL_PARAMS
        # names match exactly
        param_names = set(CONST_AB_MODEL_PARAMS[model].keys())
        assert param_names == required_param_names

    # round-trip tests
    muon_energies = np.logspace(start=-2, stop=4, num=int(1e4))
    for model in list(ConstABModel):
        muon_energy_to_length, muon_length_to_energy = generate_const_a_b_converters(
            model=model
        )
        assert np.allclose(
            muon_length_to_energy(muon_energy_to_length(muon_energies)),
            muon_energies
        )
    print("<< PASS : test_generate_const_a_b_converters >>")


def generate_gms_table_converters(losses="all"):
    """Generate converters for expected values of muon length <--> muon energy based on
    the tabulated muon energy loss model [1], spline-interpolated for smooth behavior
    within the range of tabulated energies / lengths.

    Note that "gms" in the name comes from the names of the authors of the table used.

    Parameters
    ----------
    losses : comma-separated str or iterable of strs
        Valid sub-values are {"all", "ionization", "brems", "photonucl", "pair_prod"}
        where if any in the list is specified to be "all" or if all of {"ionization",
        "brems", "photonucl", and "pair_prod"} are specified, this supercedes all
        other choices and the CSDA range values from the table are used..

    Returns
    -------
    muon_energy_to_length : callable
        Call with a muon energy to return its expected length

    muon_length_to_energy : callable
        Call with a muon length to return its expected energy

    energy_bounds : tuple of 2 floats
        (lower, upper) energy limits of table; below the lower limit, lengths are
        estimated to be 0 and above the upper limit, a ValueError is raised;
        corresponding behavior is enforced for lengths passed to `muon_length_to_energy`
        as well.

    References
    ----------
    [1] D. E. Groom, N. V. Mokhov, and S. I. Striganov, Atomic Data and Nuclear Data
        Tables, Vol. 78, No. 2, July 2001, p. 312. Table II-28.

    """
    if isinstance(losses, basestring):
        losses = tuple(x.strip().lower() for x in losses.split(","))

    VALID_MECHANISMS = ("ionization", "brems", "pair_prod", "photonucl", "all")
    for mechanism in losses:
        assert mechanism in VALID_MECHANISMS

    if "all" in losses or set(losses) == set(m for m in VALID_MECHANISMS if m != "all"):
        losses = ("all",)

    fpath = join(RETRO_DIR, "data", "muon_stopping_power_and_range_table_II-28.csv")
    table = np.loadtxt(fpath, delimiter=",")

    kinetic_energy = table[:, 0] # (GeV)
    total_energy = kinetic_energy + MUON_REST_MASS

    mev_per_gev = 1e-3
    cm_per_m = 1e2

    if "all" in losses:
        # Continuous-slowing-down-approximation (CSDA) range (cm * g / cm^3)
        csda_range = table[:, 7]
        mask = np.isfinite(csda_range)
        csda_range = csda_range[mask]
        ice_csda_range_m = csda_range / ICE_DENSITY / cm_per_m # (m)
        energy_bounds = (np.min(total_energy[mask]), np.max(total_energy[mask]))
        _, muon_energy_to_length = generate_lerp(
            x=total_energy[mask],
            y=ice_csda_range_m,
            low_behavior="constant",
            high_behavior="extrapolate",
            low_val=0,
        )
        _, muon_length_to_energy = generate_lerp(
            x=ice_csda_range_m,
            y=total_energy[mask],
            low_behavior="constant",
            high_behavior="extrapolate",
            low_val=0,
        )
    else:
        from scipy.interpolate import UnivariateSpline

        # All stopping powers given in (MeV / cm * cm^3 / g)
        stopping_power_by_mechanism = dict(
            ionization=table[:, 2],
            brems=table[:, 3],
            pair_prod=table[:, 4],
            photonucl=table[:, 5],
        )

        stopping_powers = []
        mask = np.zeros(shape=table.shape[0], dtype=bool)
        for mechanism in losses:
            addl_stopping_power = stopping_power_by_mechanism[mechanism]
            mask |= np.isfinite(addl_stopping_power)
            stopping_powers.append(addl_stopping_power)
        stopping_power = np.nansum(stopping_powers, axis=0)[mask]
        stopping_power *= cm_per_m * mev_per_gev * ICE_DENSITY

        valid_energies = total_energy[mask]
        energy_bounds = (valid_energies.min(), valid_energies.max())
        sample_energies = np.logspace(
            start=np.log10(valid_energies.min()),
            stop=np.log10(valid_energies.max()),
            num=1000,
        )
        spl = UnivariateSpline(x=valid_energies, y=1/stopping_power, s=0, k=3)
        ice_range = np.array(
            [spl.integral(valid_energies.min(), e) for e in sample_energies]
        )
        _, muon_energy_to_length = generate_lerp(
            x=sample_energies,
            y=ice_range,
            low_behavior="constant",
            high_behavior="extrapolate",
            low_val=0,
        )
        _, muon_length_to_energy = generate_lerp(
            x=ice_range,
            y=sample_energies,
            low_behavior="constant",
            high_behavior="extrapolate",
            low_val=0,
        )

    return muon_energy_to_length, muon_length_to_energy, energy_bounds


def test_generate_gms_table_converters():
    """Unit tests for `generate_table_converters` function"""
    e2l, l2e, (emin, emax) = generate_gms_table_converters()
    # round-trip test
    en = np.logspace(np.log10(emin), np.log10(emax), int(1e6))
    isclose = np.isclose(l2e(e2l(en)), en)
    if not np.all(isclose):
        en = en[~isclose]
        conv_muen = l2e(e2l(en))
        print(emin, emax, en, conv_muen)
        assert False
    print("<< PASS : test_generate_gms_table_converters >>")


def generate_min_energy_fit_converters(fit_data_path=None):
    """Generate muon energy <--> length conversion functions from fit data file.

    Parameters
    ----------
    fit_data_path : string, optional
        If not specified, default fit data file will be loaded.

    Returns
    -------
    muon_energy_to_length : callable
    muon_length_to_energy : callable
    energy_bounds : tuple of 2 floats

    """
    if fit_data_path is None:
        fit_data_path = join(RETRO_DIR, "data", "muon_min_energy_vs_len_fit.csv")

    en_len = np.loadtxt(fit_data_path, delimiter=", ")

    _, muon_energy_to_length = generate_lerp(
        x=en_len[:, 0],
        y=en_len[:, 1],
        low_behavior="constant",
        high_behavior="extrapolate",
        low_val=0,
    )

    _, muon_length_to_energy = generate_lerp(
        x=en_len[:, 1],
        y=en_len[:, 0],
        low_behavior="constant",
        high_behavior="extrapolate",
        low_val=0,
    )

    energy_bounds = np.min(en_len[:, 0]), np.max(en_len[:, 0])

    return muon_energy_to_length, muon_length_to_energy, energy_bounds


def test_generate_min_energy_fit_converters():
    """Unit tests for `generate_min_energy_fit_converters` function."""
    e2l, l2e, (emin, emax) = generate_min_energy_fit_converters()
    # round-trip test
    en = np.logspace(np.log10(emin), np.log10(emax), int(1e6))
    isclose = np.isclose(l2e(e2l(en)), en)
    if not np.all(isclose):
        en = en[~isclose]
        conv_muen = l2e(e2l(en))
        print(emin, emax, en, conv_muen)
        assert False
    print("<< PASS : test_generate_min_energy_fit_converters >>")


def const_muon_energy_to_length(muon_energy):
    """Convert muon energy into an expected length using a constant factor, as used
    elsewhere in low energy reconstructions.

    Note that I do not know the exact derivation of the constant.

    Parameters
    ----------
    muon_energy

    Returns
    -------
    muon_length

    """
    return muon_energy * TRACK_M_PER_GEV


def const_muon_length_to_energy(muon_length):
    """Convert muon length into an expected energy using a constant factor, as used
    elsewhere in low energy reconstructions.

    Note that I do not know the exact derivation of the constant.

    Parameters
    ----------
    muon_length

    Returns
    -------
    muon_energy

    """
    return muon_length / TRACK_M_PER_GEV


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

    pegleg_step_size : int, optional
        number of continuous-loss model sources to add each pegleg step; required if
        pegleg-ing (i.e., `fixed_track_length` > 0)

    param_mapping : Mapping or callable
        see :class:`retro.hypo.Hypo` for details

    external_sph_pairs : tuple, optional
        see :class:`retro.hypo.Hypo` for details

    continuous_loss_model_kwargs : mapping
        keyword arguments required by chosen `continuous_loss_model`;
        if parameters in addition to those required by the model are passed, a
        ValueError will be raised
            * `all_avg_const`, `all_avg_mmc*`, and `all_avg_gms_table` take a
              "time_step" parmeter, in units of nanoseconds

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
        pegleg_step_size=None,
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

        # `pegleg_step_size`

        if fixed_track_length == 0 and pegleg_step_size is not None:
            raise ValueError(
                "`pegleg_step_size` must be None if fixed_track_length <= 0"
            )
        if fixed_track_length > 0:
            assert float(pegleg_step_size) == int(pegleg_step_size)
            pegleg_step_size = int(pegleg_step_size)

        # `continuous_loss_model_kwargs`

        if continuous_loss_model in (
            ContinuousLossModel.all_avg_const,
            ContinuousLossModel.all_avg_mmc3,
            ContinuousLossModel.all_avg_mmc4,
            ContinuousLossModel.all_avg_proposal,
            ContinuousLossModel.all_avg_leera2,
            ContinuousLossModel.all_avg_gms_table,
            ContinuousLossModel.min_energy_fit,
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
                .format(continuous_loss_model.name) # pylint: disable=no-member
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
                .format(stochastic_loss_model.name) # pylint: disable=no-member
            ),
        )

        # -- Functons to find expected length from energy and vice versa -- #

        if continuous_loss_model is ContinuousLossModel.all_avg_const:
            self.muon_energy_to_length = const_muon_energy_to_length
            self.muon_length_to_energy = const_muon_length_to_energy
        elif continuous_loss_model in (
            ContinuousLossModel.all_avg_mmc3,
            ContinuousLossModel.all_avg_mmc4,
            ContinuousLossModel.all_avg_proposal,
            ContinuousLossModel.all_avg_leera2,
        ):
            model = continuous_loss_model.name[len('all_avg_'):]
            self.muon_energy_to_length, self.muon_length_to_energy = (
                generate_const_a_b_converters(model=model)
            )
        elif continuous_loss_model is ContinuousLossModel.all_avg_gms_table:
            self.muon_energy_to_length, self.muon_length_to_energy, _ = (
                generate_gms_table_converters()
            )
        elif continuous_loss_model is ContinuousLossModel.min_energy_fit:
            self.muon_energy_to_length, self.muon_length_to_energy, _ = (
                generate_min_energy_fit_converters()
            )
        else:
            raise NotImplementedError()

        # -- Define `internal_param_names` -- #

        # All models specify vertex and track direction
        internal_param_names = ("x", "y", "z", "time", "azimuth", "zenith")

        # If pegleg-ing (fixed_track_length > 0), energy/length are found
        # within the llh function and this kernel simply produces _all_ possible
        # sources the first time called
        if fixed_track_length == 0:
            internal_param_names += ("track_length",)

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
        self.pegleg_step_size = pegleg_step_size
        self.continuous_loss_model_kwargs = continuous_loss_model_kwargs
        self.stochastic_loss_model_kwargs = stochastic_loss_model_kwargs

        # -- Create the _get_sources function -- #

        self._create_get_sources_func()

    def _create_get_sources_func(self):
        time_step = self.continuous_loss_model_kwargs["time_step"]
        segment_length = time_step * SPEED_OF_LIGHT_M_PER_NS
        photons_per_segment = segment_length * TRACK_PHOTONS_PER_M

        fixed_track_length = self.fixed_track_length
        if fixed_track_length <= 0:
            fixed_track_length = 0

        def get_continuous_sources(x, y, z, time, azimuth, zenith, track_length): # pylint: disable=missing-docstring
            if track_length == 0:
                return [EMPTY_SOURCES], [SrcHandling.none]

            sampled_dt = np.arange(
                start=time_step*0.5,
                stop=track_length / SPEED_OF_LIGHT_M_PER_NS,
                step=time_step,
            )

            # At least one segment
            if len(sampled_dt) == 0:
                sampled_dt = np.array([track_length / 2. / SPEED_OF_LIGHT_M_PER_NS])

            # NOTE: add pi to make dir vector go in "math-standard" vector notation
            # (vector components point in direction of motion), as opposed to "IceCube"
            # vector notation (vector components point opposite to direction of
            # motion).
            opposite_zenith = np.pi - zenith
            opposite_azimuth = np.pi + azimuth

            dir_costheta = math.cos(opposite_zenith)
            dir_sintheta = math.sin(opposite_zenith)

            dir_cosphi = math.cos(opposite_azimuth)
            dir_sinphi = math.sin(opposite_azimuth)

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

            return [sources], [SrcHandling.nonscaling]

        assert self.stochastic_loss_model is StochasticLossModel.none

        if fixed_track_length > 0:
            pegleg_step_size = self.pegleg_step_size
            def _get_sources(x, y, z, time, azimuth, zenith):
                pegleg_sources = get_continuous_sources(
                    x=x,
                    y=y,
                    z=z,
                    time=time,
                    azimuth=azimuth,
                    zenith=zenith,
                    track_length=fixed_track_length,
                )
                num_pegleg_sources = len(pegleg_sources)

                @numba_jit(**DFLT_NUMBA_JIT_KWARGS)
                def pegleg_generator(): # pylint: disable=missing-docstring
                    for idx in range(0, num_pegleg_sources, pegleg_step_size):
                        yield (
                            [pegleg_sources[idx : idx + pegleg_step_size]],
                            [SrcHandling.nonscaling]
                        )

                return [pegleg_generator], [SrcHandling.pegleg]

        else:

            def _get_sources(x, y, z, time, azimuth, zenith, track_length):
                nonscaling_sources = get_continuous_sources(
                    x=x,
                    y=y,
                    z=z,
                    time=time,
                    azimuth=azimuth,
                    zenith=zenith,
                    track_length=track_length,
                )
                return [nonscaling_sources], [SrcHandling.nonscaling]

        self._get_sources = _get_sources

    def get_energy(self, pegleg_step=None, scalefactors=None): # pylint: disable=unused-argument
        """Retrieve the estimated energy of the last-produced muon.

        Parameters
        ----------
        pegleg_step : int

        scalefactors : scalar or iterable thereof, required if scaling sources present

        Returns
        -------
        estimated_muon_energy : float

        Raises
        ------
        ValueError
            * If fixed_track_length > 0 and no `pegleg_step` is specified
            * If no calls to get_sources have been made

        """
        if self.fixed_track_length <= 0:
            track_length = self.internal_param_values["track_length"]
        else:
            if pegleg_step is None:
                raise ValueError(
                    "Need to provide value for `pegleg_step` since kernel was"
                    " instantiated with a fixed track length"
                )
            track_length = (
                pegleg_step
                * self.continuous_loss_model_kwargs["time_step"]
                * SPEED_OF_LIGHT_M_PER_NS
            )

        continuous_energy = self.muon_length_to_energy(track_length)

        if self.stochastic_loss_model is StochasticLossModel.none:
            stochastic_energy = 0
        else:
            raise NotImplementedError("No handling of stochastic energy loss")

        return continuous_energy + stochastic_energy


if __name__ == "__main__":
    test_generate_const_a_b_converters()
    test_generate_gms_table_converters()
    test_generate_min_energy_fit_converters()
