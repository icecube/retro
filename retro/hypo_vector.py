# pylint: disable=print-statement, wrong-import-position, line-too-long


from __future__ import absolute_import, division

import math
import os
from os.path import abspath, dirname

import numba
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, CASCADE_PHOTONS_PER_GEV, FTYPE,
                   HYPO_PARAMS_T, SPEED_OF_LIGHT_M_PER_NS, TimeSpaceCoord,
                   TRACK_M_PER_GEV, TRACK_PHOTONS_PER_M, TWO_PI, UITYPE)


NUMBA_UPDATE_ARRAYS = False

N_FTYPE = numba.float32 if FTYPE is np.float32 else numba.float64
N_ITYPE = numba.int64
N_UITYPE = numba.uint16 if UITYPE is np.uint16 else numba.uint64


#CLASS_DTYPE_SPEC = [
#    ('t_v', N_FTYPE),
#    ('x_v', N_FTYPE),
#    ('y_v', N_FTYPE),
#    ('z_v', N_FTYPE),
#
#    ('time_increment', N_FTYPE),
#    ('segment_length', N_FTYPE),
#
#    ('speed_x', N_FTYPE),
#    ('speed_y', N_FTYPE),
#    ('speed_z', N_FTYPE),
#    ('track_length', N_FTYPE),
#    ('cascade_photons', N_FTYPE),
#    ('track_photons', N_FTYPE),
#    ('tot_photons', N_FTYPE),
#
#    ('t_scaling_factor', N_FTYPE),
#    ('r_scaling_factor', N_FTYPE),
#    ('theta_scaling_factor', N_FTYPE),
#    ('track_azimuth_scaling_factor', N_FTYPE),
#    ('phi_bin_width', N_FTYPE),
#
#    ('t', N_FTYPE),
#    ('x', N_FTYPE),
#    ('y', N_FTYPE),
#    ('z', N_FTYPE),
#
#    ('t_start', N_FTYPE),
#    ('t_stop', N_FTYPE),
#    ('number_of_increments', N_ITYPE),
#    ('recreate_arrays', numba.boolean),
#
#    ('variables_array', N_FTYPE[:, :]),
#    ('indices_array', N_UITYPE[:, :]),
#    ('values_array', N_FTYPE[:, :]),
#]

# Define indices for accessing rows of `variables_array`
T_VAR_IX = 0
X_VAR_IX = 1
Y_VAR_IX = 2
Z_VAR_IX = 3
R_VAR_IX = 4
TRCK_ZEN_VAR_IX = 5
PHI_VAR_IX = 6

# Define indices for accessing rows of `indices_array`
T_IDX_IX = 0
R_IDX_IX = 1
THETA_IDX_IX = 2
PHI_IDX_IX = 3

# Define indices for accessing rows of `values_array`
PHOT_VAL_IX = 0
TRCK_ZEN_VAL_IX = 1
DTRCK_AZ_VAL_IX = 2


#@numba.jit(nopython=True, fastmath=True, parallel=True, cache=True)
def update_arrays(indices_array, values_array,
                  number_of_increments, t_array_init, t, x, y, z, speed_x,
                  speed_y, speed_z, t_scaling_factor, r_scaling_factor,
                  theta_scaling_factor, track_azimuth_scaling_factor, segment_length,
                  cascade_photons, track_zenith, track_azimuth, phi_bin_width):
    for seg_idx in range(number_of_increments):
        var_t = t_array_init[seg_idx]
        relative_time = var_t - t
        var_x = x + speed_x * relative_time
        var_y = y + speed_y * relative_time
        var_z = z + speed_z * relative_time
        var_r = math.sqrt(var_x*var_x + var_y*var_y + var_z*var_z)
        var_theta = var_z / var_r
        var_phi = math.atan2(var_y, var_x) % TWO_PI

        #variables_array[T_VAR_IX, seg_idx] = var_t
        #variables_array[X_VAR_IX, seg_idx] = var_x
        #variables_array[Y_VAR_IX, seg_idx] = var_y
        #variables_array[Z_VAR_IX, seg_idx] = var_z
        #variables_array[R_VAR_IX, seg_idx] = var_r
        #variables_array[TRCK_ZEN_VAR_IX, seg_idx] = var_theta
        #variables_array[PHI_VAR_IX, seg_idx] = var_phi

        indices_array[T_IDX_IX, seg_idx] = var_t * t_scaling_factor
        indices_array[R_IDX_IX, seg_idx] = math.sqrt(var_r * r_scaling_factor)
        indices_array[THETA_IDX_IX, seg_idx] = (1 - var_theta) * theta_scaling_factor
        ind_phi = var_phi * track_azimuth_scaling_factor
        indices_array[PHI_IDX_IX, seg_idx] = ind_phi

        # Add track photons
        values_array[PHOT_VAL_IX, seg_idx] = segment_length * TRACK_PHOTONS_PER_M

        # TODO: should this be += to include both track and cascade photons at
        # 0? Or are all track photons accounted for at the "bin center" which
        # would be the first increment after 0?

        # Add track theta values
        values_array[TRCK_ZEN_VAL_IX, seg_idx] = track_zenith

        # Add delta phi values
        values_array[DTRCK_AZ_VAL_IX, seg_idx] = abs(track_azimuth - (ind_phi * phi_bin_width + (phi_bin_width / 2)))

    # Set cascade photons in 0th sample location
    values_array[PHOT_VAL_IX, 0] = cascade_photons


# TODO: jitclass makes this go insanely slow. What's the deal with that?
#@numba.jitclass(CLASS_DTYPE_SPEC)
class SegmentedHypo(object):
    """
    Create hypo using individual segments and retrieve matrix that contains
    expected photons in each cell in spherical coordinate system with dom at
    origin. Binnnings and location of the DOM must be set.

    Parameters
    ----------
    params : HYPO_PARAMS_T

    track_e_scale
        ?

    cascade_e_scale
        ?

    time_increment
        If using constant time increments, length of time between photon
        dumps (ns)

    """
    def __init__(self, params, cascade_e_scale=1, track_e_scale=1,
                 time_increment=1):
        if not isinstance(params, HYPO_PARAMS_T):
            params = HYPO_PARAMS_T(*params)
        self.params = params

        # Declare "constants"
        self.time_increment = time_increment
        self.segment_length = self.time_increment * SPEED_OF_LIGHT_M_PER_NS

        c_sin_zen = SPEED_OF_LIGHT_M_PER_NS * math.sin(self.params.track_zenith)
        self.speed_x = c_sin_zen * math.cos(self.params.track_azimuth)
        self.speed_y = c_sin_zen * math.sin(self.params.track_azimuth)
        self.speed_z = SPEED_OF_LIGHT_M_PER_NS * math.cos(self.params.track_zenith)

        self.track_length = params.track_energy * TRACK_M_PER_GEV

        self.cascade_photons = params.cascade_energy * CASCADE_PHOTONS_PER_GEV
        self.track_photons = self.track_length * TRACK_PHOTONS_PER_M
        self.tot_photons = self.cascade_photons + self.track_photons

        # Default values
        self.number_of_increments = 0
        self.recreate_arrays = True
        self.dom_coord = None

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

    #@profile
    def set_dom_location(self, coord):
        """Change the track vertex to be relative to a DOM at a given position.

        Parameters
        ----------
        coord : TimeSpaceCoord or convertible thereto

        """
        if coord == self.dom_coord:
            return

        if not isinstance(coord, TimeSpaceCoord):
            coord = TimeSpaceCoord(*coord)

        self.dom_coord = coord

        self.t = self.params.t - coord.t
        self.x = self.params.x - coord.x
        self.y = self.params.y - coord.y
        self.z = self.params.z - coord.z

        orig_number_of_incr = self.number_of_increments

        # Create initial time array, using the midpoints of each time increment
        half_incr = self.time_increment / 2
        self.t_array_init = np.arange(self.t - half_incr, min(self.bin_max.t, self.track_length / SPEED_OF_LIGHT_M_PER_NS + self.t) - half_incr, self.time_increment, FTYPE)
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
        self.set_dom_location(coord=hit_dom_coord)

        # Create array with variables
        if self.recreate_arrays:
            #self.variables_array = np.empty(
            #    (8, self.number_of_increments),
            #    FTYPE
            #)
            self.indices_array = np.empty(
                (4, self.number_of_increments),
                UITYPE
            )
            self.values_array = np.empty(
                (3, self.number_of_increments),
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
                          self.params.track_zenith,
                          self.params.track_azimuth,
                          self.phi_bin_width)
            return

        #self.variables_array[T_VAR_IX, :] = self.t_array_init
        #relative_time = self.variables_array[T_VAR_IX, :] - self.t
        #self.variables_array[X_VAR_IX, :] = self.x + self.speed_x * relative_time
        #self.variables_array[Y_VAR_IX, :] = self.y + self.speed_y * relative_time
        #self.variables_array[Z_VAR_IX, :] = self.z + self.speed_z * relative_time
        #self.variables_array[R_VAR_IX, :] = np.sqrt(np.square(self.variables_array[X_VAR_IX, :]) + np.square(self.variables_array[Y_VAR_IX, :]) + np.square(self.variables_array[Z_VAR_IX, :]))
        #self.variables_array[TRCK_ZEN_VAR_IX, :] = self.variables_array[Z_VAR_IX, :] / self.variables_array[R_VAR_IX, :]
        #self.variables_array[PHI_VAR_IX, :] = np.arctan2(self.variables_array[Y_VAR_IX, :], self.variables_array[X_VAR_IX, :]) % TWO_PI

        #self.indices_array[T_IDX_IX, :] = self.variables_array[T_VAR_IX, :] * self.t_scaling_factor
        #self.indices_array[R_IDX_IX, :] = np.sqrt(self.variables_array[R_VAR_IX, :] * self.r_scaling_factor)
        #self.indices_array[THETA_IDX_IX, :] = (1 - self.variables_array[TRCK_ZEN_VAR_IX, :]) * self.theta_scaling_factor
        #self.indices_array[PHI_IDX_IX, :] = self.variables_array[PHI_VAR_IX, :] * self.track_azimuth_scaling_factor

        relative_time = self.t_array_init - self.t
        var_x = self.x + self.speed_x * relative_time
        var_y = self.y + self.speed_y * relative_time
        var_z = self.z + self.speed_z * relative_time
        var_r = np.sqrt(np.square(var_x) + np.square(var_y) + np.square(var_z))
        var_theta = var_z / var_r
        var_phi = np.arctan2(var_y, var_x) % TWO_PI

        self.indices_array[T_IDX_IX, :] = relative_time * self.t_scaling_factor
        self.indices_array[R_IDX_IX, :] = np.sqrt(var_r * self.r_scaling_factor)
        self.indices_array[THETA_IDX_IX, :] = (1 - var_theta) * self.theta_scaling_factor
        self.indices_array[PHI_IDX_IX, :] = var_phi * self.track_azimuth_scaling_factor

        # Add track photons
        self.values_array[PHOT_VAL_IX, :] = self.segment_length * TRACK_PHOTONS_PER_M

        # TODO: should this be += to include both track and cascade photons at
        # 0? Or are all track photons accounted for at the "bin center" which
        # would be the first increment after 0?

        # Set cascade photons in 0th sample location
        self.values_array[PHOT_VAL_IX, 0] = self.cascade_photons

        # Add track theta values
        self.values_array[TRCK_ZEN_VAL_IX, :] = self.params.track_zenith

        # Add delta phi values
        self.values_array[DTRCK_AZ_VAL_IX, :] = np.abs(
            self.params.track_azimuth
            - (self.indices_array[PHI_IDX_IX, :] * self.phi_bin_width
               + (self.phi_bin_width / 2))
        )
