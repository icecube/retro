"""
Hypo generic class
"""

# pylint: disable=wrong-import-position, too-many-instance-attributes


from __future__ import absolute_import, division, print_function

import math
import os
from os.path import abspath, dirname

import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (bin_edges_to_centers, BinningCoords, binspec_to_edges,
                   CASCADE_PHOTONS_PER_GEV, SPEED_OF_LIGHT_M_PER_NS,
                   HYPO_PARAMS_T, TimeSpaceCoord, TRACK_M_PER_GEV,
                   TRACK_PHOTONS_PER_M, TWO_PI)


__all__ = ['Hypo']


class Hypo(object):
    """
    Generic class for hypothesis for a given set of parameters. Subclass to
    create a specific implementation of a hypothesis.

    Parameters
    ----------
    params : HYPO_PARAMS_T
    track_e_scale : float
    cascade_e_scale : float

    """
    def __init__(self, params, origin=None, cascade_e_scale=1,
                 track_e_scale=1):
        # Convert types of passed values to those expected internally

        if origin is not None and not isinstance(origin, TimeSpaceCoord):
            origin = TimeSpaceCoord(*origin)
        if not isinstance(params, HYPO_PARAMS_T):
            params = HYPO_PARAMS_T(*params)

        # Store passed args as attrs

        self.params = params
        self.origin = origin
        self.cascade_e_scale = cascade_e_scale
        self.track_e_scale = track_e_scale

        # Pre-compute info about track

        self.track_length = params.track_energy * TRACK_M_PER_GEV

        sin_trck_zen = math.sin(self.params.track_zenith)
        self.track_dir_x = sin_trck_zen * math.cos(self.params.track_azimuth)
        self.track_dir_y = sin_trck_zen * math.sin(self.params.track_azimuth)
        self.track_dir_z = math.cos(self.params.track_zenith)

        # TODO: make this actual muon speed, which depends on energy
        self.track_speed_x = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_x
        self.track_speed_y = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_y
        self.track_speed_z = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_z

        self.track_photons_per_m = TRACK_PHOTONS_PER_M * self.track_e_scale
        self.track_photons = self.track_length * self.track_photons_per_m

        # Pre-compute info about cascade

        self.cascade_photons_per_gev = (
            CASCADE_PHOTONS_PER_GEV * self.cascade_e_scale
        )
        self.cascade_photons = (
            self.params.cascade_energy * self.cascade_photons_per_gev
        )

        # Total track + cascade

        self.tot_photons = self.cascade_photons + self.track_photons

        # Set default values for attrs computed by other methods/properties

        num_coords = len(BinningCoords._fields)
        self.bin_min, self.bin_max = [BinningCoords(*(np.nan,)*num_coords)]*2
        self.num_bins = BinningCoords(*(0,)*num_coords)

        self._bin_edges = None
        self._bin_centers = None
        self._bin_widths = None
        self._bin_num_factors = None

    def set_binning(self, start, stop, num_bins):
        """Define binnings of spherical coordinates assuming: linear binning in
        time, quadratic binning in radius, linear binning in cos(theta), and
        linear binning in phi.

        Parameters
        ----------
        start : BinningCoords namedtuple containing floats
            Lower-most bin edge in each dimension.

        stop : BinningCoords namedtuple containing floats
            Upper-most bin edge in each dimension.

        num_bins : BinningCoords namedtuple containing ints
            Number of bins in each dimension (note there will be
            ``num_bins + 1`` bin edges).

        """
        if not isinstance(start, BinningCoords):
            start = BinningCoords(*start)
        if not isinstance(stop, BinningCoords):
            stop = BinningCoords(*stop)
        if not isinstance(num_bins, BinningCoords):
            num_bins = BinningCoords(*num_bins)

        self.bin_min = start
        self.bin_max = stop
        self.num_bins = num_bins

        self._bin_edges = None
        self._bin_centers = None
        self._bin_widths = None
        self._bin_num_factors = None

    @property
    def bin_edges(self):
        """BinningCoords of floats : bin edges"""
        if self._bin_edges is None:
            self._bin_edges = binspec_to_edges(start=self.bin_min,
                                               stop=self.bin_max,
                                               num_bins=self.num_bins)
        return self._bin_edges

    @property
    def bin_centers(self):
        """BinningCoords of floats : bin centers"""
        if self._bin_centers is None:
            self._bin_centers = bin_edges_to_centers(self.bin_edges)
        return self._bin_centers

    @property
    def bin_widths(self):
        """BinningCoords of floats : bin widths"""
        if self._bin_widths is None:
            bin_edges = self.bin_edges
            self._bin_widths = BinningCoords(
                *(dim[1:] - dim[:-1] for dim in bin_edges)
            )
        return self._bin_widths

    @property
    def bin_num_factors(self):
        """BinningCoords of floats : factors used to ID which bin number a
        value falls in"""
        if self._bin_num_factors is None:
            self._bin_num_factors = BinningCoords(
                t=self.num_bins.t / (self.bin_max.t - self.bin_min.t),
                r=self.num_bins.r**2 / self.bin_max.r,
                theta=0.5 * self.num_bins.theta,
                phi=self.num_bins.phi / TWO_PI
            )
        return self._bin_num_factors

    def compute_matrices(self, *args, **kwargs):
        """Implement this method in subclasses of Hypo"""
        raise NotImplementedError()
