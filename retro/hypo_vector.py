# pylint: disable=print-statement, wrong-import-position, line-too-long


from __future__ import absolute_import, division

import math
import os
from os.path import abspath, dirname

import numba
import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import FTYPE, SPEED_OF_LIGHT_M_PER_NS


SPEED_OF_LIGHT = SPEED_OF_LIGHT_M_PER_NS * 1e9
TRACK_LENGTH_PER_GEV = 15 / 3.3
PHOTONS_PER_METER = 2451.4544553
CASCADE_PHOTONS_PER_GEV = 12805.3383311

UITYPE = np.uint64

N_FTYPE = numba.float32 if FTYPE is np.float32 else numba.float64
N_ITYPE = numba.int64
N_UITYPE = numba.uint16 if UITYPE is np.uint16 else numba.uint64


CLASS_DTYPE_SPEC = [
    ('t_v', N_FTYPE),
    ('x_v', N_FTYPE),
    ('y_v', N_FTYPE),
    ('z_v', N_FTYPE),
    ('theta_v', N_FTYPE),
    ('phi_v', N_FTYPE),
    ('track_energy', N_FTYPE),
    ('cascade_energy', N_FTYPE),

    ('time_increment', N_FTYPE),
    ('segment_length', N_FTYPE),

    ('speed_x', N_FTYPE),
    ('speed_y', N_FTYPE),
    ('speed_z', N_FTYPE),
    ('track_length', N_FTYPE),
    ('cascade_photons', N_FTYPE),
    ('track_photons', N_FTYPE),
    ('tot_photons', N_FTYPE),

    ('n_t_bins', N_ITYPE),
    ('n_r_bins', N_ITYPE),
    ('n_theta_bins', N_ITYPE),
    ('n_phi_bins', N_ITYPE),

    ('t_min', N_FTYPE),
    ('t_max', N_FTYPE),
    ('r_min', N_FTYPE),
    ('r_max', N_FTYPE),
    ('t_scaling_factor', N_FTYPE),
    ('r_scaling_factor', N_FTYPE),
    ('theta_scaling_factor', N_FTYPE),
    ('phi_scaling_factor', N_FTYPE),
    ('phi_bin_width', N_FTYPE),

    ('t', N_FTYPE),
    ('x', N_FTYPE),
    ('y', N_FTYPE),
    ('z', N_FTYPE),

    ('t_start', N_FTYPE),
    ('t_stop', N_FTYPE),
    ('number_of_increments', N_ITYPE),
    ('recreate_arrays', numba.boolean),

    ('variables_array', N_FTYPE[:, :]),
    ('indices_array', N_UITYPE[:, :]),
    ('values_array', N_FTYPE[:, :]),
]

# Define indices for accessing rows of `variables_array`
T_VAR_IX = 0
X_VAR_IX = 1
Y_VAR_IX = 2
Z_VAR_IX = 3
R_VAR_IX = 4
THETA_VAR_IX = 5
PHI_VAR_IX = 6

# Define indices for accessing rows of `indices_array`
T_IDX_IX = 0
R_IDX_IX = 1
THETA_IDX_IX = 2
PHI_IDX_IX = 3

# Define indices for accessing rows of `values_array`
PHOT_VAL_IX = 0
TRCK_THETA_VAL_IX = 1
DPHI_VAL_IX = 2


# TODO: jitclass makes this go insanely slow. What's the deal with that?
#@numba.jitclass(CLASS_DTYPE_SPEC)
class SegmentedHypo(object):
    """
    Create hypo using individual segments and retrieve matrix that contains
    expected photons in each cell in spherical coordinate system with dom at
    origin. Binnnings and location of the DOM must be set.

    Parameters
    ----------
    t_v
        time (ns)

    x_v, y_v, z_v
        vertex position (m)

    theta_v
        zenith (rad)

    phi_v
        azimuth (rad)

    track_energy
        track energy (GeV)

    cascade_energy
        cascade energy (GeV)

    time_increment
        If using constant time increments, length of time between photon
        dumps (ns)

    """
    def __init__(self, t_v, x_v, y_v, z_v, theta_v, phi_v, track_energy,
                 cascade_energy, time_increment=1):
        # Assign vertex
        self.t = t_v * 1e-9
        self.x = x_v
        self.y = y_v
        self.z = z_v
        self.theta_v = theta_v
        self.phi_v = phi_v

        # Declare constants
        self.time_increment = time_increment * 1e-9
        self.segment_length = self.time_increment * SPEED_OF_LIGHT

        # Calculate frequently used values
        sin_theta_v = math.sin(self.theta_v)
        cos_theta_v = math.cos(self.theta_v)
        sin_phi_v = math.sin(self.phi_v)
        cos_phi_v = math.cos(self.phi_v)
        self.speed_x = SPEED_OF_LIGHT * sin_theta_v * cos_phi_v
        self.speed_y = SPEED_OF_LIGHT * sin_theta_v * sin_phi_v
        self.speed_z = SPEED_OF_LIGHT * cos_theta_v

        # Cconvert track energy to length
        self.track_length = track_energy * TRACK_LENGTH_PER_GEV

        # Precalculate (nphotons.py) to avoid icetray
        self.cascade_photons = cascade_energy * CASCADE_PHOTONS_PER_GEV
        self.track_photons = self.track_length * PHOTONS_PER_METER
        self.tot_photons = self.cascade_photons + self.track_photons

        # Defaults
        self.number_of_increments = 0
        self.recreate_arrays = True

    def set_binning(self, n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_max,
                    r_max, t_min, r_min):
        """Define binnings of spherical coordinates assuming: linear binning in
        time, quadratic binning in radius, linear binning in cos(theta), and
        linear binning in phi.

        Parameters
        ----------
        n_t_bins, n_r_bins, n_theta_bins, n_phi_bins : int
            Number of bins (note this is _not_ bin edges) in each dimension.

        t_max : float
            max time (ns)

        r_max : float
            max radius (m)

        t_min : float
            min time (ns)

        r_min : float
            min radius (m)

        """
        self.n_t_bins = n_t_bins
        self.n_r_bins = n_r_bins
        self.n_theta_bins = n_theta_bins
        self.n_phi_bins = n_phi_bins
        self.t_min = t_min * 1e-9
        self.t_max = t_max * 1e-9
        self.r_min = r_min
        self.r_max = r_max
        self.t_scaling_factor = self.n_t_bins / (self.t_max - self.t_min)
        self.r_scaling_factor = self.n_r_bins * self.n_r_bins / self.r_max
        self.theta_scaling_factor = self.n_theta_bins / 2.
        self.phi_scaling_factor = self.n_phi_bins / np.pi / 2.
        self.phi_bin_width = 2. * np.pi / self.n_phi_bins

    def set_dom_location(self, t_dom=0, x_dom=0, y_dom=0, z_dom=0):
        """Change the track vertex to be relative to a DOM at a given position.

        Parameters
        ----------
        t_dom
            time (ns)

        x_dom, y_dom, z_dom
            position of the dom (m)

        """
        self.t = self.t - t_dom * 1e-9
        self.x = self.x - x_dom
        self.y = self.y - y_dom
        self.z = self.z - z_dom

        orig_number_of_incr = self.number_of_increments

        # Define bin edges
        incr_by_2 = self.time_increment / 2
        self.t_start = self.t - incr_by_2
        shifted_t_max = self.t_max - incr_by_2
        self.number_of_increments = (
            int(np.ceil((shifted_t_max - self.t_start) / self.time_increment))
        )
        self.t_stop = self.t_start + self.number_of_increments * self.time_increment

        #print 'relative start time: %.17e' % (self.t_start - self.t)
        #print 'relative stop  time: %.17e' % (self.t_stop - self.t)
        #print 'time increment     : %.17e' % ((self.t_stop - self.t_start) / self.number_of_increments)
        #print 'num increments     : %d' % self.number_of_increments

        # Invalidate arrays if they changed shape
        if self.number_of_increments != orig_number_of_incr:
            self.recreate_arrays = True

    #@profile
    def vector_photon_matrix(self):
        """Use a single time array to simultaneously calculate all of the
        positions along the track, using information from __init__.

        """
        if self.recreate_arrays:
            self.variables_array = np.empty((8, self.number_of_increments),
                                            FTYPE)

        self.variables_array[T_VAR_IX, :] = np.linspace(self.t_start,
                                                        self.t_stop,
                                                        self.number_of_increments)

        # Since we shifted by ``(time_increment / 2)``, reset first value to
        # avoid negative times
        self.variables_array[T_VAR_IX, 0] = self.t

        relative_time = self.variables_array[T_VAR_IX, :] - self.t

        self.variables_array[X_VAR_IX, :] = self.x + self.speed_x * relative_time
        self.variables_array[Y_VAR_IX, :] = self.y + self.speed_y * relative_time
        self.variables_array[Z_VAR_IX, :] = self.z + self.speed_z * relative_time
        self.variables_array[R_VAR_IX, :] = np.sqrt(np.square(self.variables_array[X_VAR_IX, :]) + np.square(self.variables_array[Y_VAR_IX, :]) + np.square(self.variables_array[Z_VAR_IX, :]))
        self.variables_array[THETA_VAR_IX, :] = self.variables_array[Z_VAR_IX, :] / self.variables_array[R_VAR_IX, :]
        self.variables_array[PHI_VAR_IX, :] = np.arctan2(self.variables_array[Y_VAR_IX, :], self.variables_array[X_VAR_IX, :]) % (2 * np.pi)

        # Create array with indices
        if self.recreate_arrays:
            self.indices_array = np.empty((4, self.number_of_increments), UITYPE)

        self.indices_array[T_IDX_IX, :] = self.variables_array[T_VAR_IX, :] * self.t_scaling_factor
        self.indices_array[R_IDX_IX, :] = np.sqrt(self.variables_array[R_VAR_IX, :] * self.r_scaling_factor)
        self.indices_array[THETA_IDX_IX, :] = (1 - self.variables_array[THETA_VAR_IX, :]) * self.theta_scaling_factor
        self.indices_array[PHI_IDX_IX, :] = self.variables_array[PHI_VAR_IX, :] * self.phi_scaling_factor

        # Create array to store values for each index
        if self.recreate_arrays:
            self.values_array = np.empty((3, self.number_of_increments), FTYPE)

        # Add track photons
        self.values_array[PHOT_VAL_IX, :] = self.segment_length * PHOTONS_PER_METER

        # TODO: should this be += to include both track and cascade photons at
        # 0? Or are all track photons accounted for at the "bin center" which
        # would be the first increment after 0?

        # Add cascade photons
        self.values_array[PHOT_VAL_IX, 0] = self.cascade_photons

        # Add track theta values
        self.values_array[TRCK_THETA_VAL_IX, :] = self.theta_v

        # Add delta phi values
        self.values_array[DPHI_VAL_IX, :] = np.abs(
            self.phi_v - (self.indices_array[PHI_IDX_IX, :]
                          * self.phi_bin_width + (self.phi_bin_width / 2))
        )

        self.recreate_arrays = False
