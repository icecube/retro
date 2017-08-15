# pylint: disable=wrong-import-position, line-too-long


from __future__ import absolute_import, division, print_function

import math
import os
from os.path import abspath, dirname
import time

import numba
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, CASCADE_PHOTONS_PER_GEV, FTYPE,
                   HYPO_PARAMS_T, PhotonInfo, SPEED_OF_LIGHT_M_PER_NS,
                   TimeSpaceCoord, TRACK_M_PER_GEV, TRACK_PHOTONS_PER_M,
                   TWO_PI, UITYPE)


NUMBA_UPDATE_ARRAYS = False

N_FTYPE = numba.float32 if FTYPE is np.float32 else numba.float64
N_ITYPE = numba.int64
N_UITYPE = numba.uint16 if UITYPE is np.uint16 else numba.uint64

# Define indices for accessing rows of `indices_array`
IDX_T_IX = BinningCoords._fields.index('t')
IDX_R_IX = BinningCoords._fields.index('r')
IDX_THETA_IX = BinningCoords._fields.index('theta')
IDX_PHI_IX = BinningCoords._fields.index('phi')

# Define indices for accessing rows of `values_array`
PHOT_CNT_IX = 0
PHOT_LEN_IX = 1
PHOT_THETA_IX = 2
PHOT_PHI_IX = 3


@numba.jit(nopython=True, fastmath=True, parallel=False, cache=True)
def update_arrays(indices_array, values_array,
                  number_of_increments, t_array_init, t, x, y, z, speed_x,
                  speed_y, speed_z, t_scaling_factor, r_scaling_factor,
                  theta_scaling_factor, track_azimuth_scaling_factor, segment_length,
                  cascade_photons, track_zenith, track_azimuth, phi_bin_width,
                  track_photons_per_m):
    for seg_idx in range(number_of_increments):
        var_t = t_array_init[seg_idx]
        relative_time = var_t - t
        var_x = x + speed_x * relative_time
        var_y = y + speed_y * relative_time
        var_z = z + speed_z * relative_time
        var_r = np.sqrt(var_x**2 + var_y**2 + var_z**2)
        var_theta = var_z / var_r
        var_phi = np.arctan2(var_y, var_x) % TWO_PI

        indices_array[IDX_T_IX, seg_idx] = var_t * t_scaling_factor
        indices_array[IDX_R_IX, seg_idx] = np.sqrt(var_r * r_scaling_factor)
        indices_array[IDX_THETA_IX, seg_idx] = (1 - var_theta) * theta_scaling_factor
        ind_phi = var_phi * track_azimuth_scaling_factor
        indices_array[IDX_PHI_IX, seg_idx] = ind_phi

        # Add track photons
        values_array[PHOT_CNT_IX, seg_idx] = segment_length * track_photons_per_m

        # Add track theta values
        values_array[PHOT_THETA_IX, seg_idx] = track_zenith

        # Add delta phi values
        values_array[PHOT_PHI_IX, seg_idx] = abs(track_azimuth - (ind_phi * phi_bin_width + (phi_bin_width / 2)))

    # TODO: should this be += to include both track and cascade photons at
    # 0? Or are all track photons accounted for at the "bin center" which
    # would be the first increment after 0?

    # Set cascade photons in 0th sample location
    values_array[PHOT_CNT_IX, 0] = cascade_photons


class SegmentedHypo(object):
    """
    Create hypo using individual segments and retrieve matrix that contains
    expected photons in each cell in spherical coordinate system with dom at
    origin. Binnnings and location of the DOM must be set.

    Parameters
    ----------
    params : HYPO_PARAMS_T
    track_e_scale : float
    cascade_e_scale : float
    time_increment
        If using constant time increments, length of time between photon
        dumps (ns)

    """
    def __init__(self, params, origin=None, cascade_e_scale=1, track_e_scale=1,
                 time_increment=1):
        if not isinstance(params, HYPO_PARAMS_T):
            params = HYPO_PARAMS_T(*params)
        self.t = params.t
        self.x = params.x
        self.y = params.y
        self.z = params.z
        self.track_zenith = params.track_zenith
        self.track_azimuth = params.track_azimuth
        self.track_energy = params.track_energy
        self.cascade_energy = params.cascade_energy

        # Directions
        sin_trck_zen = math.sin(self.track_zenith)
        self.track_dir_x = sin_trck_zen * math.cos(self.track_azimuth)
        self.track_dir_y = sin_trck_zen * math.sin(self.track_azimuth)
        self.track_dir_z = math.cos(self.track_azimuth)

        self.time_increment = time_increment
        self.segment_length = self.time_increment * SPEED_OF_LIGHT_M_PER_NS
        self.track_photons_per_m = TRACK_PHOTONS_PER_M * track_e_scale
        self.photons_per_segment = self.segment_length * self.track_photons_per_m
        self.cascade_photons_per_gev = CASCADE_PHOTONS_PER_GEV * cascade_e_scale

        self.speed_x = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_x
        self.speed_y = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_y
        self.speed_z = SPEED_OF_LIGHT_M_PER_NS * self.track_dir_z

        self.track_length = params.track_energy * TRACK_M_PER_GEV

        self.cascade_photons = params.cascade_energy * self.cascade_photons_per_gev
        self.track_photons = self.track_length * self.track_photons_per_m
        self.tot_photons = self.cascade_photons + self.track_photons

        # Default values
        self.number_of_increments = 0
        self.recreate_arrays = True
        self.origin = None
        self.photon_counts = None
        self.photon_avg_len = None
        self.photon_avg_phi = None
        self.photon_avg_theta = None

        if origin is not None:
            self.set_origin(coord=origin)

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
            Number of bins in each dimension (note there will be ``num_bins + 1``
            bin edges).

        """
        if not isinstance(num_bins, BinningCoords):
            num_bins = BinningCoords(*num_bins)
        if not isinstance(start, BinningCoords):
            start = BinningCoords(*start)
        if not isinstance(stop, BinningCoords):
            stop = BinningCoords(*stop)

        self.num_bins = num_bins
        self.bin_min = start
        self.bin_max = stop

        self.t_scaling_factor = self.num_bins.t / (self.bin_max.t - self.bin_min.t)
        self.r_scaling_factor = self.num_bins.r * self.num_bins.r / self.bin_max.r
        self.theta_scaling_factor = self.num_bins.theta / 2
        self.track_azimuth_scaling_factor = self.num_bins.phi / TWO_PI
        self.phi_bin_width = TWO_PI / self.num_bins.phi
        self.phi_half_bin_width = TWO_PI / self.num_bins.phi / 2

    #@profile
    def set_origin(self, coord):
        """Change the vertex to be relative to ``coord`` (e.g. a hit on DOM at
        the given position).

        Parameters
        ----------
        coord : TimeSpaceCoord or convertible thereto

        """
        if coord == self.origin:
            return

        if not isinstance(coord, TimeSpaceCoord):
            coord = TimeSpaceCoord(*coord)

        self.origin = coord

        self.t = self.t - self.origin.t
        self.x = self.x - self.origin.x
        self.y = self.y - self.origin.y
        self.z = self.z - self.origin.z

        orig_number_of_incr = self.number_of_increments

        # Create initial time array, using the midpoints of each time increment
        half_incr = self.time_increment / 2
        self.t_array_init = np.arange(
            self.t - half_incr,
            min(
                self.bin_max.t,
                self.track_length / SPEED_OF_LIGHT_M_PER_NS + self.t
            ) - half_incr,
            self.time_increment,
            dtype=FTYPE
        )
        self.t_array_init[0] = self.t

        # Set the number of time increments in the track
        self.number_of_increments = len(self.t_array_init)

        # Invalidate arrays if they changed shape
        if self.number_of_increments != orig_number_of_incr:
            self.recreate_arrays = True

    #@profile
    def compute_matrices(self, hit_dom_coord):
        """Use a single time array to simultaneously calculate all of the
        positions along the track, using information from __init__.

        """
        self.set_origin(coord=hit_dom_coord)

        if self.recreate_arrays:
            self.indices_array = np.empty(
                shape=(len(BinningCoords._fields), self.number_of_increments),
                dtype=UITYPE,
                order='C'
            )
            self.values_array = np.empty(
                (len(PhotonInfo._fields), self.number_of_increments),
                FTYPE
            )
            self.recreate_arrays = False

        if NUMBA_UPDATE_ARRAYS:
            update_arrays(self.indices_array,
                          self.values_array,
                          self.number_of_increments,
                          self.t_array_init,
                          self.t,
                          self.x,
                          self.y,
                          self.z,
                          self.speed_x,
                          self.speed_y,
                          self.speed_z,
                          self.t_scaling_factor,
                          self.r_scaling_factor,
                          self.theta_scaling_factor,
                          self.track_azimuth_scaling_factor,
                          self.segment_length,
                          self.cascade_photons,
                          self.track_zenith,
                          self.track_azimuth,
                          self.phi_bin_width,
                          self.track_photons_per_m)
            return

        relative_time = self.t_array_init - self.t
        var_x = self.x + self.speed_x * relative_time
        var_y = self.y + self.speed_y * relative_time
        var_z = self.z + self.speed_z * relative_time
        var_r = np.sqrt(np.square(var_x) + np.square(var_y) + np.square(var_z))
        var_theta = var_z / var_r
        var_phi = np.arctan2(var_y, var_x) % TWO_PI

        # Compute which bin index each segment is in

        # NOTE: indices_array is uint type, so float values are truncated,
        # should result in floor rounding
        self.indices_array[IDX_T_IX, :] = self.t_array_init * self.t_scaling_factor
        self.indices_array[IDX_R_IX, :] = np.sqrt(var_r * self.r_scaling_factor)
        self.indices_array[IDX_THETA_IX, :] = (1 - var_theta) * self.theta_scaling_factor
        self.indices_array[IDX_PHI_IX, :] = var_phi * self.track_azimuth_scaling_factor

        # Count segments in each bin
        t0 = time.time()
        self.segment_counts = {}
        for incr_idx in xrange(self.number_of_increments):
            bin_idx = BinningCoords(*self.indices_array[:, incr_idx])
            self.segment_counts[bin_idx] = (
                1 + self.segment_counts.get(bin_idx, 0)
            )

        # NOTE: The approx. projected length of a unit vector (onto track
        # dir) at Cherenkov angle for 1-100 GeV muon in ice with n ~ 1.78
        # (~55.8 deg) projected onto track's direction: cos(55.8 deg) ~ 0.562.

        # Stuff photon_info PhtonInfo namedtuples

        self.photon_info = {}
        for bin_idx, segment_count in self.segment_counts.items():
            self.photon_info[bin_idx] = PhotonInfo(
                count=segment_count * self.photons_per_segment,
                theta=self.track_zenith,
                phi=np.abs(
                    self.track_azimuth
                    - (bin_idx.phi*self.phi_bin_width + self.phi_half_bin_width)
                ),
                length=0.562
            )

        # TODO: should this be += to include both track and cascade photons at
        # 0? Or are all track photons accounted for at the "bin center" which
        # would be the first increment after 0?
        #zero_bin_idx = BinningCoords(t=0, r=0, theta=0, phi=0)
        zero_bin_idx = BinningCoords(*self.indices_array[:, 0])
        if zero_bin_idx in self.photon_info:
            orig_zero_bin_info = self.photon_info[zero_bin_idx]
            count = (self.cascade_photons + orig_zero_bin_info.count)

        print('zero bin before assignemnt:', orig_zero_bin_info)

        self.photon_info[zero_bin_idx] = PhotonInfo(
            count=count,
            theta=0,
            phi=0,
            length=0
        )
        t1 = time.time()
        print('time to fill dict:', t1 - t0)

        # Add track photons
        self.values_array[PHOT_CNT_IX, :] = self.segment_length * self.track_photons_per_m

        # TODO: should this be += to include both track and cascade photons at
        # 0? Or are all track photons accounted for at the "bin center" which
        # would be the first increment after 0?

        # Set cascade photons in 0th sample location
        self.values_array[PHOT_CNT_IX, 0] = self.cascade_photons

        # Add track theta values
        self.values_array[PHOT_THETA_IX, :] = self.track_zenith

        # Add delta phi values
        self.values_array[PHOT_PHI_IX, :] = np.abs(self.track_azimuth - (self.indices_array[IDX_PHI_IX, :] * self.phi_bin_width + (self.phi_bin_width / 2)))

        self.values_array[PHOT_LEN_IX, :] = 0.562
