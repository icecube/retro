# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Cascade hypothesis class to generate photons expected from a cascade.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['CascadeModel', 'CascadeHypo']

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

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
from numbers import Number
from os.path import abspath, dirname
import sys

import numpy as np
from scipy.stats import gamma, pareto

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import (
    COS_CKV, SIN_CKV, THETA_CKV, CASCADE_PHOTONS_PER_GEV, EMPTY_SOURCES,
    SPEED_OF_LIGHT_M_PER_NS, SRC_OMNI, SRC_CKV_BETA1
)
from retro.hypo import Hypo
from retro.retro_types import SRC_T


# TODO: a "pegleg"-like cascade which can simply add sources to an existing set to
# increase the energy of the cascade while maintaining more accurate topolgy (as opposed
# to the scaling cascade, which has a fixed topology at one energy and simply scales
# luminosity)

class CascadeModel(enum.IntEnum):
    """Cascade models defined"""

    spherical = 0
    """spherical point-radiator"""

    one_dim_v1 = 1
    """parameterization inspired by but not strictly following arXiv:1210.5140v2"""


class CascadeHypo(Hypo):
    """
    Kernel to produce light sources expected from a cascade hypothesis.

    Parameters
    ----------
    model : CascadeModel enum value or corresponding name string or int value
        see :class:`CascadeModel` for valid values

    param_mapping : dict-like (mapping)
        mapping from external parameter names to internally-used parameter names (see
        `Cascade.internal_param_names` for required internal param names)

    external_sph_pairs : tuple of 0 or more 2-tuples of strings, optional
        If None or not specified, pairs of spherical parameters are deduced by
        `deduce_sph_pairs`.

    num_sources : int, optional
        specify an integer >= 1 to fix the number of sources, or specify None or an
        integer <= 0 to dynamically adjust the number of sources based on the energy of
        the cascade. See `retro/notebooks/energy_dependent_cascade_num_samples.ipynb`
        for the reasoning behind how `num_sources` is computed in this case

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
        model,
        param_mapping,
        external_sph_pairs=None,
        num_sources=None,
        scaling_proto_energy=None,
    ):
        # -- validation and translation of args -- #

        # Translate int, str, or CascadeModel into CascadeModel for comparison via `is`
        if isinstance(model, basestring):
            model = getattr(CascadeModel, model)
        else:
            model = CascadeModel(model)

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

        if model is CascadeModel.spherical:
            if num_sources < 1:
                num_sources = 1
            else:
                raise ValueError("spherical cascade only valid with `num_sources=1`")

        if model is CascadeModel.spherical:
            internal_param_names = ("x", "y", "z", "time", "energy")
        else:
            internal_param_names = ("x", "y", "z", "time", "energy", "azimuth", "zenith")

        # -- Initialize base class -- #

        super(CascadeHypo, self).__init__(
            param_mapping=param_mapping,
            internal_param_names=internal_param_names,
            external_sph_pairs=external_sph_pairs,
            scaling_proto_energy=scaling_proto_energy,
        )

        # -- Store attrs unique to a cascade -- #

        self.model = model
        self.num_sources = num_sources
        self.scaling_proto_energy = scaling_proto_energy

        # -- Create the _generate_sources function -- #

        self._create_source_generator_func()

    def _create_source_generator_func(self):
        """Create the function that generates photon sources for a hypothesis.

        The created function is attached to this class as the private attribute
        `self._generate_sources` and is intended to be called externally from
        `self.generate_sources` (which does the appropriate translations from
        external names/values to internal names/values as used by _generate_sources).

        """
        if self.model is CascadeModel.spherical:

            def generate_sources(time, x, y, z, energy):
                """Point-like spherically-radiating cascade.

                Parameters
                ----------
                time, x, y, z, energy : scalars

                Returns
                -------
                sources : length-(0 or 1) ndarray of dtype SRC_T

                """
                if energy == 0:
                    return EMPTY_SOURCES

                sources = np.empty(shape=(1,), dtype=SRC_T)
                sources[0]['kind'] = SRC_OMNI
                sources[0]['time'] = time
                sources[0]['x'] = x
                sources[0]['y'] = y
                sources[0]['z'] = z
                sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * energy

                return sources

        elif self.model is CascadeModel.one_dim_v1:
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
            all_zen_samples = np.deg2rad(
                np.clip(
                    zen_dist.rvs(size=max_num_sources, random_state=random_state),
                    a_min=0,
                    a_max=180,
                )
            )

            # Create samples from angular azimuth distribution
            random_state = np.random.RandomState(2)
            all_azi_samples = random_state.uniform(
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

            # TODO: speed up (767 µs to generate sources at 10 GeV)
            def generate_sources(time, x, y, z, energy, azimuth, zenith):
                """Cascade with both longitudinal and angular distributions (but no
                distribution off-axis). All emitters are located on the shower axis.

                Use as a hypo_kernel with the DiscreteHypo class.

                Note that the nubmer of samples is proportional to the energy of the
                cascade.

                Parameters
                ----------
                time, x, y, z, energy, azimuth, zenith

                Returns
                -------
                sources

                """
                if energy == 0:
                    return EMPTY_SOURCES

                if num_sources < 0:
                    # Note that num_sources must be 1 for energy <= min_cascade_energy
                    # (param_a goes <= 0 at this value and below, causing an exception from
                    # gamma distribution)
                    if energy <= min_cascade_energy:
                        n_sources = 1
                    else:
                        # See `retro/notebooks/energy_dependent_cascade_num_samples.ipynb`
                        n_sources = int(np.round(
                            np.clip(
                                math.exp(0.77 * math.log(energy) + 2.3),
                                a_min=1,
                                a_max=None,
                            )
                        ))

                opposite_zenith = np.pi - zenith
                opposite_azimuth = azimuth + np.pi

                sin_zen = math.sin(opposite_zenith)
                cos_zen = math.cos(opposite_zenith)
                sin_azi = math.sin(opposite_azimuth)
                cos_azi = math.cos(opposite_azimuth)
                dir_x = sin_zen * cos_azi
                dir_y = sin_zen * sin_azi
                dir_z = cos_zen

                if n_sources == 1:
                    sources = np.empty(shape=(1,), dtype=SRC_T)
                    sources[0]['kind'] = SRC_CKV_BETA1
                    sources[0]['time'] = time
                    sources[0]['x'] = x
                    sources[0]['y'] = y
                    sources[0]['z'] = z
                    sources[0]['photons'] = CASCADE_PHOTONS_PER_GEV * energy

                    sources[0]['dir_costheta'] = cos_zen
                    sources[0]['dir_sintheta'] = sin_zen

                    sources[0]['dir_phi'] = opposite_azimuth
                    sources[0]['dir_cosphi'] = cos_azi
                    sources[0]['dir_sinphi'] = sin_azi

                    sources[0]['ckv_theta'] = THETA_CKV
                    sources[0]['ckv_costheta'] = COS_CKV
                    sources[0]['ckv_sintheta'] = SIN_CKV

                    return sources

                # Create rotation matrix
                rot_mat = np.array(
                    [[cos_azi * cos_zen, -sin_azi, cos_azi * sin_zen],
                     [sin_azi * cos_zen, cos_zen, sin_azi * sin_zen],
                     [-sin_zen, 0, cos_zen]]
                )

                # Create longitudinal distribution (from arXiv:1210.5140v2)
                param_a = (
                    param_alpha
                    + param_beta * math.log10(max(min_cascade_energy, energy))
                )

                long_dist = gamma(param_a, scale=rad_len_over_b)
                long_samples = long_dist.rvs(size=n_sources, random_state=1)

                # Grab samples from angular zenith distribution
                zen_samples = all_zen_samples[:n_sources]

                # Grab samples from angular azimuth distribution
                azi_samples = all_azi_samples[:n_sources]

                # Create angular vectors distribution
                sin_zen = np.sin(zen_samples)
                x_ang_dist = sin_zen * np.cos(azi_samples)
                y_ang_dist = sin_zen * np.sin(azi_samples)
                z_ang_dist = np.cos(zen_samples)
                ang_dist = np.concatenate(
                    (x_ang_dist[np.newaxis, :],
                     y_ang_dist[np.newaxis, :],
                     z_ang_dist[np.newaxis, :]),
                    axis=0
                )

                final_ang_dist = np.dot(rot_mat, ang_dist)
                final_phi_dist = np.arctan2(final_ang_dist[1], final_ang_dist[0])
                final_theta_dist = np.arccos(final_ang_dist[2])

                # Define photons per sample
                photons_per_sample = CASCADE_PHOTONS_PER_GEV * energy / n_sources

                # Create photon array
                sources = np.empty(shape=n_sources, dtype=SRC_T)

                sources['kind'] = SRC_CKV_BETA1

                sources['time'] = time + long_samples / SPEED_OF_LIGHT_M_PER_NS
                sources['x'] = x + long_samples * dir_x
                sources['y'] = y + long_samples * dir_y
                sources['z'] = z + long_samples * dir_z

                sources['photons'] = photons_per_sample

                sources['dir_costheta'] = final_ang_dist[2]
                sources['dir_sintheta'] = np.sin(final_theta_dist)

                sources['dir_phi'] = final_phi_dist
                sources['dir_cosphi'] = np.cos(final_phi_dist)
                sources['dir_sinphi'] = np.sin(final_phi_dist)

                sources['ckv_theta'] = THETA_CKV
                sources['ckv_costheta'] = COS_CKV
                sources['ckv_sintheta'] = SIN_CKV

                return sources

        else:
            raise NotImplementedError(
                '{} cascade model is not implemented'.format(self.model.name)
            )

        if self.is_scaling:
            # Define wrapper function that injects scaling_proto_energy
            def generate_sources_wrapper(**kwargs): # pylint: disable=missing-docstring
                kwargs['energy'] = self.scaling_proto_energy
                return generate_sources(**kwargs)
            generate_sources_wrapper.__doc__ = generate_sources.__doc__
            self._generate_sources = generate_sources_wrapper
        else:
            self._generate_sources = generate_sources
