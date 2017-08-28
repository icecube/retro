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
from retro import HYPO_PARAMS_T, BinningCoords, TimeCartCoord
from retro import (CASCADE_PHOTONS_PER_GEV, SPEED_OF_LIGHT_M_PER_NS,
                   TRACK_M_PER_GEV, TRACK_PHOTONS_PER_M, PI, TWO_PI)
from retro import (bin_edges_to_centers, binspec_to_edges,
                   convert_to_namedtuple)


__all__ = ['Hypo']


class Hypo(object):
    """
    Generic class for hypothesis for a given set of parameters. Subclass to
    create a specific implementation of a hypothesis.

    Parameters
    ----------
    params : HYPO_PARAMS_T
    cascade_e_scale : float
    track_e_scale : float
    origin : None or BinningCoords namedtuple

    """
    def __init__(self, params, cascade_e_scale, track_e_scale, origin=None):
        # Convert types of passed values to those expected internally

        if origin is not None:
            origin = convert_to_namedtuple(origin, TimeCartCoord)
        params = convert_to_namedtuple(params, HYPO_PARAMS_T)

        # Store passed args as attrs

        self.params = params
        self.origin = origin
        self.cascade_e_scale = cascade_e_scale
        self.track_e_scale = track_e_scale

        # Pre-compute info about track

        self.track_length = params.track_energy * TRACK_M_PER_GEV
        # TODO: not simply linear; also, not correct speed here
        self.track_lifetime = self.track_length / SPEED_OF_LIGHT_M_PER_NS

        normal_zen = PI - self.params.track_zenith
        normal_az = -self.params.track_azimuth

        sin_zen = math.sin(normal_zen)
        self.track_dir_x = sin_zen * math.cos(normal_az)
        self.track_dir_y = sin_zen * math.sin(normal_az)
        self.track_dir_z = math.cos(normal_zen)

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
        bin_min = convert_to_namedtuple(start, BinningCoords)
        bin_max = convert_to_namedtuple(stop, BinningCoords)
        num_bins = convert_to_namedtuple(num_bins, BinningCoords)
        assert bin_min.t < bin_max.t
        assert bin_min.r == 0 < bin_max.r
        assert 0 <= bin_min.theta < bin_max.theta <= PI
        assert 0 <= bin_min.phi < bin_max.phi <= TWO_PI

        self.bin_min = bin_min
        self.bin_max = bin_max
        self.num_bins = num_bins

        #print('start:', start)
        #print('stop:', stop)
        #print('num_bins:', num_bins)

        self.num_bins = num_bins

        self._bin_edges = None
        self._bin_centers = None
        self._bin_widths = None
        self._bin_num_factors = None

    @property
    def bin_edges(self):
        """BinningCoords of floats : bin edges"""
        if self._bin_edges is None:
            self._bin_edges = binspec_to_edges(
                start=self.bin_min,
                stop=self.bin_max,
                num_bins=self.num_bins
            )
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
                r=self.num_bins.r / math.sqrt(self.bin_max.r - self.bin_min.r),
                theta=self.num_bins.theta / (math.cos(self.bin_min.theta)
                                             - math.cos(self.bin_max.theta)),
                phi=self.num_bins.phi / (self.bin_max.phi - self.bin_min.phi)
            )
            #print('bin_num_factors:', self._bin_num_factors)
        return self._bin_num_factors

    def set_origin(self, coord):
        """Implement this method in subclasses of Hypo"""
        raise NotImplementedError()

    def compute_matrices(self, *args, **kwargs):
        """Implement this method in subclasses of Hypo"""
        raise NotImplementedError()
