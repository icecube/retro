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
from retro import FTYPE, UITYPE, BinningCoords, PhotonInfo, TimeSpaceCoord
from retro import SPEED_OF_LIGHT_M_PER_NS, TWO_PI
from retro import convert_to_namedtuple, spacetime_separation
from retro.hypo import Hypo


__all__ = ['IDX_T_IX', 'IDX_R_IX', 'IDX_THETA_IX', 'IDX_PHI_IX',
           'SegmentedHypo']


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
    def __init__(self, params, cascade_e_scale, track_e_scale, time_increment,
                 origin=None):
        super(SegmentedHypo, self).__init__(params=params, origin=origin,
                                            cascade_e_scale=cascade_e_scale,
                                            track_e_scale=track_e_scale)
        self.time_increment = time_increment
        self.segment_length = self.time_increment * SPEED_OF_LIGHT_M_PER_NS
        self.photons_per_segment = (
            self.segment_length * self.track_photons_per_m
        )

        # Default values
        self.num_segments = 0
        self.allocate_arrays = True
        self.photon_info = None
        self.indices_array = None

        # Setup if origin was specified in init args
        if self.origin is not None:
            self.set_origin(coord=self.origin)

    #@profile
    def set_origin(self, coord):
        """Change the vertex of the hypothesis to be relative to ``coord``
        (e.g. ``coord`` would be a hit on DOM at that time & position).

        Parameters
        ----------
        coord : TimeSpaceCoord or convertible thereto

        """
        if coord == self.origin:
            return

        coord = convert_to_namedtuple(coord, TimeSpaceCoord)

        #print('new origin being set:', coord)

        origin = coord

        t_start_rel = self.params.t - origin.t
        x_start_rel = self.params.x - origin.x
        y_start_rel = self.params.y - origin.y
        z_start_rel = self.params.z - origin.z

        st_sep = spacetime_separation(
            dt=-t_start_rel,
            dx=x_start_rel,
            dy=y_start_rel,
            dz=z_start_rel
        )
        #print('st_sep:', st_sep)

        if st_sep < 0:
            raise ValueError('Origin would violate causality for this'
                             ' hypothesis; refusing to set')

        self.origin = origin
        self.t_start_rel = t_start_rel
        self.x_start_rel = x_start_rel
        self.y_start_rel = y_start_rel
        self.z_start_rel = z_start_rel
        self.spacetime_separation = st_sep

        t_end_rel = self.t_start_rel + self.track_lifetime
        if t_end_rel > 0:
            t_end_rel = 0

        #print('params.t: %f, origin.t: %f, t_start_rel: %f'
        #      % (self.params.t, self.origin.t, self.t_start_rel))
        #print('self.bin_min.t:', self.bin_min.t)
        #print('self.bin_max.t:', self.bin_max.t)
        #print('self.origin.t:', self.origin.t)
        #print('self.track_lifetime:', self.track_lifetime)
        #print('self.t_start_rel:', self.t_start_rel)
        #print('t_end_rel:', t_end_rel)

        orig_num_segments = self.num_segments

        # Create initial time array, using the midpoints of each time increment
        # (or the endpoint if the track is too short to hit the midpoint of the
        # first bin)

        half_incr = 0.5 * self.time_increment
        track_end_half_segs = (t_end_rel - self.t_start_rel) // half_incr
        #print('track_end_half_segs:', track_end_half_segs)
        bin_max_half_segs = (self.bin_max.t - self.t_start_rel) // half_incr
        #print('bin_max_half_segs:', bin_max_half_segs)
        track_eff_end_half_segs = min(track_end_half_segs, bin_max_half_segs)
        #print('track_eff_end_half_segs:', track_eff_end_half_segs)
        self.num_segments = int(
            0.5 * (track_eff_end_half_segs if (track_eff_end_half_segs % 2 == 0)
                   else track_eff_end_half_segs - 1)
        )
        #print('num_segments:', self.num_segments)
        track_eff_endtime = (
            self.t_start_rel + self.num_segments * self.time_increment
        )
        #print('track_eff_endtime:', track_eff_endtime)
        self.segment_midpoint_times = np.linspace(
            self.t_start_rel + half_incr,
            track_eff_endtime - half_incr,
            self.num_segments,
            dtype=FTYPE
        )
        # The 0th element is not a midpoint at all, but the starting point of
        # the track and the point where we'll store cascade info
        self.segment_midpoint_times[0] = self.t_start_rel

        #print('segment_midpoint_times:', self.segment_midpoint_times)
        #print('segment_midpoint_times range: [%f, %f]'
        #      % (self.segment_midpoint_times.min(),
        #         self.segment_midpoint_times.max()))

        # Invalidate arrays if they changed shape
        if self.num_segments != orig_num_segments:
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
                shape=(len(BinningCoords._fields), self.num_segments),
                dtype=UITYPE,
                order='C'
            )
            self.allocate_arrays = False

        relative_time = self.segment_midpoint_times - self.t_start_rel
        var_x = self.x_start_rel + self.track_speed_x * relative_time
        var_y = self.y_start_rel + self.track_speed_y * relative_time
        var_z = self.z_start_rel + self.track_speed_z * relative_time
        var_r = np.sqrt(np.square(var_x) + np.square(var_y) + np.square(var_z))
        var_costheta = var_z / var_r
        var_phi = np.arctan2(var_y, var_x) % TWO_PI

        # Compute which bin index each segment is in

        # NOTE: indices_array is uint type, so float values are truncated,
        # should result in floor rounding
        self.indices_array[IDX_T_IX, :] = (
            (self.segment_midpoint_times - self.bin_min.t)
            * self.bin_num_factors.t
        )
        self.indices_array[IDX_R_IX, :] = (
            np.sqrt(var_r) * self.bin_num_factors.r
        )
        self.indices_array[IDX_THETA_IX, :] = (
            (1 - var_costheta) * self.bin_num_factors.theta
        )
        self.indices_array[IDX_PHI_IX, :] = (
            var_phi * self.bin_num_factors.phi
        )

        # Count segments in each bin
        t0 = time.time()
        segment_counts = {}
        for segment_idx in range(self.num_segments):
            indices = self.indices_array[:, segment_idx]
            bin_idx = BinningCoords(*indices)
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

        # Average the cascade info into the first bin
        bin_idx = BinningCoords(*self.indices_array[:, 0])
        old = self.photon_info.get(bin_idx, None)
        if old is None:
            combined = PhotonInfo(
                count=self.cascade_photons,
                theta=0,
                phi=0,
                length=0
            )
        else:
            combined = PhotonInfo(
                count=old.count + self.cascade_photons,
                theta=old.theta,
                phi=old.phi,
                length=0.5 * old.length
            )

        self.photon_info[bin_idx] = combined

        #print('time to fill dict: %.3f ms' % ((time.time() - t0)*1000))
