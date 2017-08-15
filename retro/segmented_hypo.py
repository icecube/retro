# pylint: disable=wrong-import-position

"""
SegmentedHypo class for segmented track hypo and cascade hypo, where each
segment is identical for the length of the track.
"""


from __future__ import absolute_import, division, print_function

import os
from os.path import abspath, dirname
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, FTYPE, PhotonInfo, SPEED_OF_LIGHT_M_PER_NS,
                   TimeSpaceCoord, TWO_PI, UITYPE)
from retro.hypo import Hypo


# Define indices for accessing rows of `indices_array`
IDX_T_IX = BinningCoords._fields.index('t')
IDX_R_IX = BinningCoords._fields.index('r')
IDX_THETA_IX = BinningCoords._fields.index('theta')
IDX_PHI_IX = BinningCoords._fields.index('phi')


class SegmentedHypo(Hypo):
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
        super(SegmentedHypo, self).__init__(params=params, origin=origin,
                                            cascade_e_scale=cascade_e_scale,
                                            track_e_scale=track_e_scale)
        self.time_increment = time_increment
        self.segment_length = self.time_increment * SPEED_OF_LIGHT_M_PER_NS
        self.photons_per_segment = self.segment_length * self.track_photons_per_m

        # Default values
        self.number_of_increments = 0
        self.allocate_arrays = True
        self.photon_info = None
        self.indices_array = None

        # Setup if origin was specified in init args
        if self.origin is not None:
            self.set_origin(coord=self.origin)

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

        self.t_rel = self.params.t - self.origin.t
        self.x_rel = self.params.x - self.origin.x
        self.y_rel = self.params.y - self.origin.y
        self.z_rel = self.params.z - self.origin.z

        orig_number_of_incr = self.number_of_increments

        # Create initial time array, using the midpoints of each time increment
        half_incr = self.time_increment / 2
        self.t_array_init = np.arange(
            self.t_rel - half_incr,
            min(
                self.bin_max.t,
                self.track_length / SPEED_OF_LIGHT_M_PER_NS + self.t_rel
            ) - half_incr,
            self.time_increment,
            dtype=FTYPE
        )
        self.t_array_init[0] = self.t_rel

        # Set the number of time increments in the track
        self.number_of_increments = len(self.t_array_init)

        # Invalidate arrays if they changed shape
        if self.number_of_increments != orig_number_of_incr:
            self.allocate_arrays = True

    # TODO: approximate timings (total ~1 ms)
    # 40% array indexing, unpacking
    # 24% dict access
    # 12% looping in python
    #  4% namedtuples vs. tuples

    #@profile
    def compute_matrices(self, hit_dom_coord):
        """Use a single time array to simultaneously calculate all of the
        positions along the track, using information from __init__.

        """
        self.set_origin(coord=hit_dom_coord)

        if self.allocate_arrays:
            self.indices_array = np.empty(
                shape=(len(BinningCoords._fields), self.number_of_increments),
                dtype=UITYPE,
                order='C'
            )
            self.allocate_arrays = False

        relative_time = self.t_array_init - self.t_rel
        var_x = self.x_rel + self.track_speed_x * relative_time
        var_y = self.y_rel + self.track_speed_y * relative_time
        var_z = self.z_rel + self.track_speed_z * relative_time
        var_r = np.sqrt(np.square(var_x) + np.square(var_y) + np.square(var_z))
        var_theta = var_z / var_r
        var_phi = np.arctan2(var_y, var_x) % TWO_PI

        # Compute which bin index each segment is in

        # NOTE: indices_array is uint type, so float values are truncated,
        # should result in floor rounding
        self.indices_array[IDX_T_IX, :] = (
            self.t_array_init * self.bin_num_factors.t
        )
        self.indices_array[IDX_R_IX, :] = (
            np.sqrt(var_r * self.bin_num_factors.r)
        )
        self.indices_array[IDX_THETA_IX, :] = (
            (1 - var_theta) * self.bin_num_factors.theta
        )
        self.indices_array[IDX_PHI_IX, :] = (
            var_phi * self.bin_num_factors.phi
        )

        # Count segments in each bin
        t0 = time.time()
        segment_counts = {}
        for incr_idx in range(self.number_of_increments):
            #bin_idx = BinningCoords(*self.indices_array[:, incr_idx])
            vals = self.indices_array[:, incr_idx]
            bin_idx = BinningCoords(*vals)
            previous_count = segment_counts.get(bin_idx, 0)
            segment_counts[bin_idx] = 1 + previous_count

        # NOTE: The approx. projected length of a unit vector (onto track
        # dir) at Cherenkov angle for 1-100 GeV muon in ice with n ~ 1.78
        # (~55.8 deg) projected onto track's direction: cos(55.8 deg) ~ 0.562.

        # Stuff photon_info PhtonInfo namedtuples

        self.photon_info = {}
        phi_bin_width = self.bin_widths.phi
        phi_half_bin_width = 0.5 * phi_bin_width
        for bin_idx, segment_count in segment_counts.iteritems():
            phi = abs(self.params.track_azimuth - (bin_idx.phi * phi_bin_width + phi_half_bin_width)) # pylint: disable=line-too-long
            count = segment_count * self.photons_per_segment
            p_info = PhotonInfo(count=count, theta=self.params.track_zenith, phi=phi, length=0.562) # pylint: disable=line-too-long
            #p_info = (count, self.params.track_zenith, phi, 0.562)
            self.photon_info[bin_idx] = p_info

        # TODO: should this be +=, average, or what to include both track and
        # cascade photons at 0? Should there be any track at 0? Or are all
        # track photons accounted for at the next "bin center" which would be
        # the first increment after 0?
        first_bin_idx = BinningCoords(*self.indices_array[:, 0])
        if False: #first_bin_idx in self.photon_info:
            orig_first_bin_info = self.photon_info[first_bin_idx]
            print('first bin, before assignemnt:', orig_first_bin_info)
            first_bin_photon_info = PhotonInfo(
                count=0.5 * (self.cascade_photons + orig_first_bin_info.count),
                theta=orig_first_bin_info.theta,
                phi=orig_first_bin_info.phi,
                # Weighted average of 0 for cascade & existing track
                length=(
                    (orig_first_bin_info.length * orig_first_bin_info.count)
                    / (self.cascade_photons + orig_first_bin_info.count)
                )
            )
        else:
            first_bin_photon_info = PhotonInfo(
                count=self.cascade_photons,
                theta=0,
                phi=0,
                length=0
            )

        self.photon_info[first_bin_idx] = first_bin_photon_info

        print('time to fill dict: %.3f ms' % ((time.time() - t0)*1000))
