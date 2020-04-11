#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

from __future__ import absolute_import, division, print_function

__all__ = ["MuonSecondariesLightOutput"]

__author__ = "E. Thyrum"
__license__ = """Copyright 2020 Emily Thyrum

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

from os.path import abspath, dirname, join
import sys

import numpy as np
from scipy import interpolate

RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
if __name__ == "__main__" and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import EM_CASCADE_PHOTONS_PER_GEV
from retro.utils.misc import expand


class MuonSecondariesLightOutput(object):
    """

    Parameters
    ----------
    energy_bins_file : str
        Path to energy bins .npy file

    histograms_file : str
        Path to histograms .npy file

    Notes
    -----
    Default files only apply to muons between 1 and 10000GeV.

    """

    def __init__(
        self,
        energy_bins_file=join(RETRO_DIR, "data", "muon_secondaries_light_output/energy_bins.npy"),
        histograms_file=join(RETRO_DIR, "data", "muon_secondaries_light_output/histograms.npy"),
    ):
        self.interpolators = []
        """linear interpolation function for each curve in histarray"""

        self.average_track_lengths = []
        """the "native" muon length for each curve in histarray"""

        # Load files for interpolating light output vs. track length for energy
        # ranges.
        self.energy_bins = np.load(expand(energy_bins_file))
        self.histarray = np.load(expand(histograms_file))

        for curve in self.histarray:
            curvex = []
            curveE = []
            for tup in curve:
                curvex.append(tup[0])
                curveE.append(tup[1])
            curvex = np.array(curvex)
            curveE = np.array(curveE)
            interpolator = interpolate.interp1d(
                curvex,
                curveE,
                kind="linear",
                bounds_error=False,
                fill_value=(curveE[0], curveE[-1]),
            )
            self.interpolators.append(interpolator)
            self.average_track_lengths.append(
                float(curve[-1][0] + (curve[-1][0] - curve[-2][0]) / 2)
            )

    def get_light_output(
        self,
        muon_starting_energy,
        total_track_length,
        segment_positions,
        segment_lengths,
    ):
        """
        Parameters
        ----------
        muon_starting_energy : scalar
        total_track_length : scalar
        segment_positions : numpy.ndarray, same shape as segment_positions
        segment_lengths : numpy.ndarray, same shape as segment_positions

        """
        # tells us the index of histarray to find the correct curve to use
        index = np.digitize(muon_starting_energy, self.energy_bins, right=False) - 1
        ourtracklen = self.average_track_lengths[index]

        # so that we don't alter the original segment_positions array
        segment_positions = np.copy(segment_positions)
        if total_track_length <= ourtracklen:
            segment_positions += ourtracklen - total_track_length
        else:
            segment_positions /= total_track_length / ourtracklen

        photons_per_segment = (
            self.interpolators[index](segment_positions)
            * segment_lengths
            * EM_CASCADE_PHOTONS_PER_GEV
        )

        return photons_per_segment


def test1(muon_starting_energy):
    import matplotlib as mpl
    try:
        mpl.use("agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    muo = MuonSecondariesLightOutput()

    spacing_array = []

    en1index = np.digitize(muon_starting_energy, muo.energy_bins, right=False) - 1

    for thing in muo.histarray:
        xarray = []
        for tup in thing:
            xarray.append(tup[0])
        spacing_array.append(xarray[-1] - xarray[-2])

    xarray = []
    photarray = []
    for tup in muo.histarray[en1index]:
        xarray.append(tup[0])
        photarray.append(tup[1] * spacing_array[en1index] * EM_CASCADE_PHOTONS_PER_GEV)
    fig, ax = plt.subplots()
    ax.plot(xarray, photarray, ls="none", marker="o")
    xarray = np.array(xarray)
    xnew = np.linspace(xarray[0], xarray[-1], 1000)
    spacenew = np.full_like(xnew, fill_value=xarray[1] - xarray[0])
    ax.plot(
        xnew,
        muo.get_light_output(
            muon_starting_energy, muo.average_track_lengths[en1index], xnew, spacenew
        ),
    )
    ax.set_title("Muon starting energy = {} GeV".format(muon_starting_energy))
    ax.set_xlabel("Position along muon track (m)")
    ax.set_ylabel("Photons/m produced by muon's secondary particles")
    fig.tight_layout()
    fig.savefig("muon_energy={}gev.png".format(muon_starting_energy))


def test2():
    for muon_starting_energy in np.logspace(np.log10(1), np.log10(1000), 10):
        test1(int(np.round(muon_starting_energy)))


def test3():
    import matplotlib as mpl
    try:
        mpl.use("agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt

    muo = MuonSecondariesLightOutput()

    for muon_starting_energy in np.logspace(np.log10(1), np.log10(1000), 10):
        muon_starting_energy = int(np.round(muon_starting_energy))


        index = np.digitize(muon_starting_energy, muo.energy_bins, right=False) - 1
        average_track_length = muo.average_track_lengths[index]
        total_track_lengths = [average_track_length * i/5 for i in [3, 4, 5, 6, 7]]

        fig, ax = plt.subplots()
        for total_track_length in total_track_lengths:
            segment_positions = np.linspace(0, total_track_length, 1000)
            segment_lengths = np.full_like(
                segment_positions,
                fill_value=5, #segment_positions[1] - segment_positions[0],
            )
            ax.plot(
                segment_positions,
                muo.get_light_output(
                    muon_starting_energy,
                    total_track_length,
                    segment_positions,
                    segment_lengths,
                ),
                label="{}".format(int(np.round(total_track_length))),
            )
            ax.set_title("Muon starting energy = {} GeV".format(muon_starting_energy))
            ax.set_xlabel("Position along muon track (m)")
            ax.set_ylabel("Normalized photons/m")
        ax.legend(loc="best", title="Track length (m)")
        fig.tight_layout()
        fig.savefig("paths_for_muon_energ={}gev.png".format(muon_starting_energy))


if __name__ == "__main__":
    test2()
    test3()
