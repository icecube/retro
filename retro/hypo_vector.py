from __future__ import absolute_import, division

import math

import numpy as np


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

    theta
        zenith (rad)

    phi
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
                 cascade_energy, time_increment=1.):
        # Assign vertex
        self.t = t_v * 1e-9
        self.x = x_v
        self.y = y_v
        self.z = z_v
        self.theta_v = theta_v
        self.phi_v = phi_v

        # Declare constants
        self.time_increment = time_increment * 1e-9
        self.speed_of_light = 2.99e8
        self.segment_length = self.time_increment * self.speed_of_light

        # Calculate frequently used values
        self.sin_theta_v = math.sin(self.theta_v)
        self.cos_theta_v = math.cos(self.theta_v)
        self.sin_phi_v = math.sin(self.phi_v)
        self.cos_phi_v = math.cos(self.phi_v)
        self.speed_x = self.speed_of_light * self.sin_theta_v * self.cos_phi_v
        self.speed_y = self.speed_of_light * self.sin_theta_v * self.sin_phi_v
        self.speed_z = self.speed_of_light * self.cos_theta_v

        # Cconvert track energy to length
        self.track_length = 15. / 3.3 * track_energy

        # Precalculate (nphotons.py) to avoid icetray
        self.photons_per_meter = 2451.4544553
        self.cascade_photons = 12805.3383311 * cascade_energy
        self.track_photons = self.track_length * self.photons_per_meter
        self.tot_photons = self.cascade_photons + self.track_photons

    def set_binning(self, n_t_bins, n_r_bins, n_theta_bins, n_phi_bins, t_max,
                    r_max, t_min=0, r_min=0):
        """Define binnings of spherical coordinates assuming: linear binning in
        time, quadratic binning in radius, linear binning in cos(theta), and
        linear binning in phi.

        Parameters
        ----------
        t_min
            min time (ns)
        t_max
            max time (ns)
        r_min
            min radius (m)
        r_max
            max radius (m)

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

    def set_dom_location(self, t_dom=0., x_dom=0., y_dom=0., z_dom=0.):
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

    def vector_photon_matrix(self):
        """Use a single time array to simultaneously calculate all of the
        positions along the track, using information from __init__."""
        # Create initial time array, using the midpoints of each time increment
        self.t_array_init = np.arange(self.t, min(self.t_max, self.track_length / self.speed_of_light + self.t), self.time_increment, dtype=np.float32) - self.time_increment / 2
        self.t_array_init[0] = self.t

        # Set the number of time increments in the track
        self.number_of_increments = int(len(self.t_array_init))

        # Create array with variables
        self.variables_array = np.empty((8, self.number_of_increments), dtype=np.float32)
        self.t_array = self.variables_array[0, :]
        self.t_array[:] = self.t_array_init
        self.x_array = self.variables_array[1, :]
        self.x_array[:] = self.x + self.speed_x * (self.t_array - self.t)
        self.y_array = self.variables_array[2, :]
        self.y_array[:] = self.y + self.speed_y * (self.t_array - self.t)
        self.z_array = self.variables_array[3, :]
        self.z_array[:] = self.z + self.speed_z * (self.t_array - self.t)
        self.r_array = self.variables_array[4, :]
        self.r_array[:] = np.sqrt(np.square(self.x_array) + np.square(self.y_array) + np.square(self.z_array))
        self.cos_theta_array = self.variables_array[5, :]
        self.cos_theta_array[:] = self.z_array / self.r_array
        self.phi_array = self.variables_array[6, :]
        self.phi_array[:] = np.arctan2(self.y_array, self.x_array) % (2 * np.pi)

        # Create array with indices
        self.indices_array = np.empty((4, self.number_of_increments), dtype=np.uint16)
        self.t_index_array = self.indices_array[0, :]
        self.t_index_array[:] = self.t_array * self.t_scaling_factor
        self.r_index_array = self.indices_array[1, :]
        self.r_index_array[:] = np.sqrt(self.r_array * self.r_scaling_factor)
        self.theta_index_array = self.indices_array[2, :]
        self.theta_index_array[:] = (-self.cos_theta_array + 1.) * self.theta_scaling_factor
        self.phi_index_array = self.indices_array[3, :]
        self.phi_index_array[:] = self.phi_array * self.phi_scaling_factor

        # Create array to store values for each index
        self.values_array = np.empty((3, self.number_of_increments), dtype=np.float32)

        # Add track photons
        self.photon_array = self.values_array[0, :]
        self.photon_array[:] = self.segment_length * self.photons_per_meter

        # Add cascade photons
        self.photon_array[0] = self.cascade_photons

        # Add track theta values
        self.track_theta_array = self.values_array[1, :]
        self.track_theta_array[:] = self.theta_v

        # Add delta phi values
        self.delta_phi_array = self.values_array[2, :]
        self.delta_phi_array[:] = np.abs(self.phi_v - (self.phi_index_array * self.phi_bin_width + self.phi_bin_width / 2.))
