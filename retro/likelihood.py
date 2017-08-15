#!/usr/bin/env python

# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then tales of icecube hdf5 files with events (hits
series) in them as inputs and calculates lilkelihoods.

At the moment, these likelihoods can be single points or 1d or 2d scans.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from itertools import izip, product
import os
from os.path import abspath, dirname, isfile, join

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numba # pylint: disable=unused-import
import numpy as np
import pyfits
from pyswarm import pso
from scipy.special import gammaln

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import (BinningCoords, event_to_hypo_params, Events, expand,
                   FTYPE, HypoParams10D, HYPO_PARAMS_T, PhotonInfo, Pulses,
                   TimeSpaceCoord)
import hypo_fast # pylint: disable=unused-import
import hypo_vector # pylint: disable=unused-import


IC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_IC_DOM{dom:d}_r_cz_t_angles.fits'
)

DC_TABLE_FPATH_PROTO = (
    '{tables_dir:s}/retro_nevts1000_DC_DOM{dom:d}_r_cz_t_angles.fits'
)

DETECTOR_GEOM_FILE = join(dirname(abspath(__file__)), 'data', 'geo_array.npy')
DFLT_EVENTS_FPATH = (
    '/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600'
    '/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5'
)
N_PHI_BINS = 20
NOISE_CHARGE = 0.00000025
CASCADE_E_SCALE = 10 #2.
TRACK_E_SCALE = 10 #20.
NUM_SCAN_POINTS = 20
HYPO_T = hypo_vector.SegmentedHypo
CMAP = 'YlGnBu_r'

ABS_BOUNDS = HypoParams10D(
    t=(-1000, 1e6),
    x=(-1000, 1000),
    y=(-1000, 1000),
    z=(-1000, 1000),
    track_azimuth=(0, 2*np.pi),
    track_zenith=(-np.pi, np.pi),
    track_energy=(0, 1e3),
    cascade_azimuth=(0, 2*np.pi),
    cascade_zenith=(-np.pi, np.pi),
    cascade_energy=(0, 1e3)
)
"""Absolute bounds for scanning / minimizer to work within"""

REL_BOUNDS = HypoParams10D(
    t=(-300, 300),
    x=(-100, 100),
    y=(-100, 100),
    z=(-100, 100),
    track_azimuth=None,
    track_zenith=None,
    track_energy=None,
    cascade_azimuth=None,
    cascade_zenith=None,
    cascade_energy=None
)
"""Relative bounds defined about "truth" values for scanning / minimization"""

MIN_USE_RELATIVE_BOUNDS = True
"""Whether minimizer is to use relative bounds (where defined)"""

SCAN_USE_RELATIVE_BOUNDS = True
"""Whether scanning is to use relative bounds (where defined)"""

MIN_DIMS = [] #('t x y z track_zenith track_azimuth track_energy'.split())
"""Which dimensions to plug into minimizer (dims not fixed to truth)"""

SCAN_DIM_SETS = (
    't', 'x', #'y', 'z', 'track_zenith', 'track_azimuth', 'track_energy',
    #('t', 'x'), ('t', 'y'), ('t', 'z'), ('x', 'z'),
    #('track_zenith', 'track_azimuth'), ('track_zenith', 'z')
)
"""Which dimensions to scan. Tuples specify 2+ dimension scans"""


#@profile
def fill_photon_info(fpath, dom_depth_index, scale=1, photon_info=None):
    """Fill photon info namedtuple-of-dictionaries from FITS file.

    Parameters
    ----------
    fpath : string
        Path to FITS file corresponding to the passed ``dom_depth_index``.

    dom_depth_index : int
        Depth index (e.g. from 0 to 59)

    scale : float
        Scaling factor to apply to the photon counts from the table, e.g. for
        DOM efficiency.

    photon_info : None or PhotonInfo namedtuple of dicts
        If None, creates a new PhotonInfo namedtuple with empty dicts to fill.
        If one is provided, the existing component dictionaries are updated.

    Returns
    -------
    photon_info : PhotonInfo namedtuple of dicts
        Tuple fields are 'count', 'theta', 'phi', and 'length'. Each dict is
        keyed by `dom_depth_index` and values are the arrays loaded from the FITS file.

    bin_edges : BinningCoords namedtuple
        Each element of the tuple is an array of bin edges.

    """
    # pylint: disable=no-member
    if photon_info is None:
        photon_info = PhotonInfo(*([{}]*len(PhotonInfo._fields)))

    with pyfits.open(expand(fpath)) as table:
        if scale == 1:
            photon_info.count[dom_depth_index] = table[0].data
        else:
            photon_info.count[dom_depth_index] = table[0].data * scale

        photon_info.theta[dom_depth_index] = table[1].data
        photon_info.phi[dom_depth_index] = table[2].data
        photon_info.length[dom_depth_index] = table[3].data

        # Note that we invert (reverse and multiply by -1) time edges
        bin_edges = BinningCoords(t=-table[4].data[::-1], r=table[5].data,
                                  theta=table[6].data, phi=[])

    return photon_info, bin_edges


#@profile
def get_llh(hypo, event, detector_geometry, ic_photon_info, dc_photon_info,
            detailed_info_list=None):
    """Get log likelihood.

    Parameters
    ----------
    hypo : HYPO_T
    event : retro.Event namedtuple or convertible thereto
    detector_geometry : numpy.ndarray
    ic_photon_info : retro.PhotonInfo namedtuple or convertible thereto
    dc_photon_info : retro.PhotonInfo namedtuple or convertible thereto
    detailed_info_list : None or appendable sequence
        If a list is provided, it is appended with a dict containing detailed
        info from the calculation useful, e.g., for debugging. If ``None`` is
        passed, no detailed info is made available.

    Returns
    -------
    llh : float

    """
    llh = 0
    noise_counts = 0

    # Loop over pulses (aka hits)
    for string, om, pulse_time, pulse_charge in izip(*event.pulses):
        x, y, z = detector_geometry[string, om]

        hit_dom_coord = TimeSpaceCoord(t=pulse_time, x=x, y=y, z=z)

        # Get the photon expectations of the hypothesis in the DOM-hit
        # coordinates
        hypo.compute_matrices(hit_dom_coord=hit_dom_coord)

        # Get the retro table for the pulse

        # String indices 0-78 (numbers 1-79) are ordinary IceCube strings
        if 0 <= string <= 78:
            photon_counts_map = ic_photon_info.count[om]
            photon_avg_theta_map = ic_photon_info.theta[om]
            photon_avg_phi_map = ic_photon_info.phi[om]
            photon_avg_len_map = ic_photon_info.length[om]
        # String indices 79-85 (numbers 80-86) are ordinary DeepCore strings
        elif 79 <= string <= 85:
            photon_counts_map = dc_photon_info.count[om]
            photon_avg_theta_map = dc_photon_info.theta[om]
            photon_avg_phi_map = dc_photon_info.phi[om]
            photon_avg_len_map = dc_photon_info.length[om]
        else:
            raise ValueError('Unhandled string index %d (number %d)'
                             % (string, string + 1))

        # Get max llh between z_matrix and gamma_map
        # noise probability?
        expected_charge = 0
        for photon_idx, hypo_count in hypo.photon_counts:
            # These two agles need to be inverted because we're backpropagating
            # but want to match to forward-propagating photons
            hypo_theta = np.pi - hypo.photon_avg_theta[photon_idx]
            hypo_phi = np.pi - hypo.photon_avg_phi[photon_idx]
            #hypo_legth = hypo.photon_avg_length[photon_idx]

            # Get map
            map_count = photon_counts_map[photon_idx[0:3]]
            map_theta = photon_avg_theta_map[photon_idx[0:3]]
            map_phi = photon_avg_phi_map[photon_idx[0:3]]
            map_length = photon_avg_len_map[photon_idx[0:3]]

            # Assume source is totally directed at 0.73 rad (cherenkov angle)

            # Accept this fraction as isotropic light
            dir_fraction = map_length**2
            #print('map length = ', dir_fraction)
            #iso_fraction = 1. - dir_fraction

            # whats the cos(psi) between track direction and map?
            # accept this fraction of directional light
            # this is wrong i think...
            #proj_dir = np.arccos((np.cos(hypo_theta)*np.cos(map_theta) + np.sin(hypo_theta)*np.sin(map_theta)*np.cos(hypo_phi - map_phi)))

            proj_dir = (np.cos(hypo_theta) * np.cos(map_theta)
                        + (np.sin(hypo_theta) * np.sin(map_theta)
                           * np.cos(hypo_phi - map_phi)))

            #print(proj_dir)

            # How close to 0.754 is it? Use this to get a Gaussian weight
            delta = -proj_dir - 0.754
            accept_dir = np.exp(- delta**2 / 0.1) * dir_fraction
            #accept_iso = iso_fraction

            # Acceptance directed light
            total_charge = hypo_count * map_count
            #directional_q = hypo_legth * total_charge
            #isotropic_q = (1. - hypo_legth) * total_charge

            # Factor betwen isotropic and direction light
            #f_iso_dir = 10.

            #expected_charge += directional_q * (accept_iso/f_iso_dir + accept_dir) + isotropic_q * (accept_iso + accept_dir/f_iso_dir)
            #expected_charge += directional_q * accept_dir + isotropic_q * accept_iso
            expected_charge += total_charge * accept_dir

        if expected_charge < NOISE_CHARGE:
            noise_counts += 1

        expected_charge = max(NOISE_CHARGE, expected_charge)
        pulse_llh = -(pulse_charge * np.log(expected_charge)
                      - expected_charge
                      - gammaln(pulse_charge + 1))
        llh += pulse_llh

    if detailed_info_list is not None:
        detailed_info_list.append(dict(noise_counts=noise_counts))

    return llh

#@profile
def scan(llh_func, event, dims, scan_values, bin_edges, nominal_params=None,
         llh_func_kwargs=None):
    """Scan likelihoods for hypotheses changing one parameter dimension.

    Parameters
    ----------
    llh_func : callable
        Function used to compute a likelihood. Must take ``hypo`` and ``event``
        as first two arguments, where ``hypo`` is a HYPO_T object and ``event``
        is the argument passed here. Function must return just one value (the
        ``llh``)

    event : Event
        Event for which to get likelihoods

    dims : string or iterable thereof
        One of 't', 'x', 'y', 'z', 'azimuth', 'zenith', 'cascade_energy',
        or 'track_energy'.

    scan_values : iterable of floats, or iterable thereof
        Values to set for the dimension being scanned.

    nominal_params : None or HYPO_PARAMS_T namedtuple
        Nominal values for all param values. The value for the params being
        scanned are irrelevant, as this is replaced with each value from
        `scan_values`. Therefore this is optional if _all_ parameters are
        subject to the scan.

    llh_func_kwargs : mapping or None
        Keyword arguments to pass to `get_llh` function

    Returns
    -------
    all_llh : numpy.ndarray (len(scan_values[0]) x len(scan_values[1]) x ...)
        Likelihoods corresponding to each value in product(*scan_values).

    """
    if llh_func_kwargs is None:
        llh_func_kwargs = {}

    all_params = HYPO_PARAMS_T._fields

    # Need list of strings (dim names). If we just have a string, make it the
    # first element of a single-element tuple.
    if isinstance(dims, basestring):
        dims = (dims,)

    # Need iterable-of-iterables-of-floats. If we have just an iterable of
    # floats (e.g. for 1D scan), then make it the first element of a
    # single-element tuple.
    if np.isscalar(next(iter(scan_values))):
        scan_values = (scan_values,)

    if nominal_params is None:
        #assert len(dims) == len(all_params)
        nominal_params = HYPO_PARAMS_T(*([np.nan]*len(all_params)))

    # Make nominal into a list so we can mutate its values as we scan
    params = list(nominal_params)

    # Storage for unique vals in each dimension, from which to determine
    # overall dimensionality
    uniques = [set([])] * len(dims)

    # Get indices for each param that we'll be changing, in the order they will
    # be specified
    param_indices = []
    for dim in dims:
        param_indices.append(all_params.index(dim))

    all_llh = []
    for param_values in product(*scan_values):
        for idx, (pidx, pval) in enumerate(izip(param_indices, param_values)):
            params[pidx] = pval
            uniques[idx].add(pval)

        hypo = HYPO_T(params=params, cascade_e_scale=CASCADE_E_SCALE,
                      track_e_scale=TRACK_E_SCALE)
        hypo.set_binning(bin_edges)
        llh = llh_func(hypo, event, **llh_func_kwargs)
        all_llh.append(llh)

    all_llh = np.array(all_llh, dtype=FTYPE)
    all_llh.reshape(tuple(len(u) for u in uniques))

    return all_llh


#@profile
def main(events_fpath, tables_dir, start_index=None, stop_index=None):
    """Perform scans and minimization for events.

    Parameters
    ----------
    events_fpath : string
        Path to events HDF5 file

    tables_dir : string
        Path to directory containing the retro tables

    start_index : None or int
        Event index (as ordered in events file) to start on. Specify 0 or
        `None` to start with the first event. I.e., iterate over
        `range(start_index, stop_index)`.

    stop_index : None or int
        Event index (as ordered in events file) to stop before. I.e., iterate
        over `range(start_index, stop_index)`.

    """
    # pylint: disable=no-member
    events = Events(events_fpath)

    print('%d events found' % len(events))

    # --- load tables ---

    # Tables are not binned in phi, but we will do so for the hypo, therefore
    # need to apply norm
    norm = 1 / N_PHI_BINS

    # Correct for DOM wavelength acceptance, given cherenkov spectrum (source
    # photons); see tables/wavelegth.py
    #norm *= 0.0544243061857

    # Compensate for costheta bins (40) and wavelength accepet.
    #norm = 0.0544243061857 * 40.

    # Add in quantum efficiencies (?)
    dom_eff_ic = 0.25
    dom_eff_dc = 0.35

    # Define phi bin edges, as these are ignored in the tables (symmetry)
    phi_bin_edges = np.linspace(0, 2*np.pi, N_PHI_BINS + 1)

    ic_photon_info, dc_photon_info = None, None

    # Read in the actual tables
    ref_bin_edges = None
    for dom_depth_index in range(60):
        # IceCube (non-DeepCore) tables
        fpath = IC_TABLE_FPATH_PROTO.format(tables_dir=tables_dir,
                                            dom=dom_depth_index)
        if isfile(fpath):
            ic_photon_info, bin_edges = fill_photon_info(
                fpath=fpath,
                dom_depth_index=dom_depth_index,
                scale=norm * dom_eff_ic,
                photon_info=ic_photon_info
            )
            if ref_bin_edges is None:
                ref_bin_edges = bin_edges
            else:
                for test, ref in zip(bin_edges, ref_bin_edges):
                    assert np.array_equal(test, ref)
        else:
            print('No table for IC DOM depth index %i found at path "%s"'
                  % (dom_depth_index, fpath))

        # DeepCore tables
        fpath = DC_TABLE_FPATH_PROTO.format(tables_dir=tables_dir,
                                            dom=dom_depth_index)
        if isfile(fpath):
            dc_photon_info, bin_edges = fill_photon_info(
                fpath=fpath,
                dom_depth_index=dom_depth_index,
                scale=norm * dom_eff_dc,
                photon_info=dc_photon_info
            )
            if ref_bin_edges is None:
                ref_bin_edges = bin_edges
            else:
                for test, ref in zip(bin_edges, ref_bin_edges):
                    assert np.array_equal(test, ref)
        else:
            print('No table for IC DOM depth index %i found at path "%s"'
                  % (dom_depth_index, fpath))

    bin_edges = BinningCoords(t=bin_edges.t, r=bin_edges.r,
                              theta=bin_edges.theta,
                              phi=np.linspace(0, 2*np.pi))

    # Load detector geometry array
    detector_geometry = np.load(DETECTOR_GEOM_FILE)

    # Iterate through events
    for idx, event in enumerate(events[start_index:stop_index]):
        print('working on event #%i / event ID %d' % (idx, event.event))

        llh_func_kwargs = dict(detector_geometry=detector_geometry,
                               ic_photon_info=ic_photon_info,
                               dc_photon_info=dc_photon_info)

        truth_params = event_to_hypo_params(event)
        truth_hypo = HYPO_T(params=truth_params,
                            cascade_e_scale=CASCADE_E_SCALE,
                            track_e_scale=TRACK_E_SCALE)
        truth_hypo.set_binning(bin_edges)
        llh_truth = get_llh(hypo=truth_hypo, event=event, **llh_func_kwargs)
        print('llh at truth = %.2f' % llh_truth)

        min_results = None

        if MIN_DIMS:
            print('Will minimize following dimension(s): %s' % MIN_DIMS)

            variable_dims = []
            fixed_dims = []
            for dim in HYPO_PARAMS_T._fields:
                if dim in MIN_DIMS:
                    variable_dims.append(dim)
                else:
                    fixed_dims.append(dim)

            lower_bounds = []
            upper_bounds = []
            for dim in MIN_DIMS:
                if MIN_USE_RELATIVE_BOUNDS and getattr(REL_BOUNDS, dim) is not None:
                    nom_val = getattr(truth_params, dim)
                    lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                    upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                else:
                    lower, upper = getattr(ABS_BOUNDS, dim)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

            def get_llh_partial(args):
                pass

            xopt1, fopt1 = pso(get_llh_partial, lower_bounds, upper_bounds,
                               kwargs=llh_func_kwargs,
                               minstep=1e-5,
                               minfunc=1e-1,
                               debug=True)

        scan_results = None

        if SCAN_DIM_SETS:
            print('Will scan following sets of dimensions: %s' % str(SCAN_DIM_SETS))
            scan_results = OrderedDict()
            for dims in SCAN_DIM_SETS:
                print('Scanning dimension(s): %s...' % str(dims))
                if isinstance(dims, basestring):
                    dims = [dims]

                nominal_params = deepcopy(truth_params)
                scan_values = []
                for dim in dims:
                    if SCAN_USE_RELATIVE_BOUNDS and getattr(REL_BOUNDS, dim) is not None:
                        nom_val = getattr(nominal_params, dim)
                        lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                        upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                    else:
                        lower, upper = getattr(ABS_BOUNDS, dim)
                    scan_values.append(
                        np.linspace(lower, upper, NUM_SCAN_POINTS)
                    )

                llh = scan(llh_func=get_llh, event=event, dims=dims,
                           scan_values=scan_values, bin_edges=bin_edges,
                           nominal_params=nominal_params,
                           llh_func_kwargs=llh_func_kwargs)

                scan_results[tuple(dims)] = llh

                #z_vs = np.linspace(neutrino.z - 50, neutrino.z + 50, NUM_SCAN_POINTS)

                #llhs = []
                #noises = []
                #for z_v in z_vs:
                #    print('testing z = %.2f' % z_v)
                #    my_hypo = HYPO_T(neutrino.t, neutrino.x, neutrino.y, z_v,
                #                     theta=neutrino.theta, phi=neutrino.phi,
                #                     track_energy=track.energy,
                #                     cascade_energy=cascade.energy,
                #                     cascade_e_scale=CASCADE_E_SCALE,
                #                     track_e_scale=TRACK_E_SCALE)
                #    my_hypo.set_binning(t_bin_edges, r_bin_edges, theta_bin_edges,
                #                        phi_bin_edges)
                #    llh, noise = get_llh(hypo=my_hypo, t=t, x=x, y=y, z=z, q=q,
                #                         string=strings, om=oms,
                #                         ic_photon_info=ic_photon_info,
                #                         dc_photon_info=dc_photon_info)
                #    llhs.append(llh)
                #    noises.append(noise)

                #if not plot:
                #    return

                #plt.clf()
                #fig = plt.figure()
                #ax = fig.add_subplot(111)
                #ax.plot(z_vs, llhs)
                #ax.set_ylabel('llh')
                #ax.set_xlabel('Vertex z (m)')
                ## Truth
                #ax.axvline(neutrino.z, color='r')
                #ax.axvline(events.ml_recos[event_idx].vertex[3], color='g')
                #ax.axvline(events.spe_recos[event_idx].vertex[3], color='m')
                #ax.set_title('Event %i, E_cascade = %.2f GeV, E_track = %.2f GeV'
                #             % (evt, cascade.energy, track.energy))
                #plt.savefig('z_%s.png' % evt, dpi=150)


                #    if do_xz:
                #        x_points = 51
                #        y_points = 51
                #        x_vs = np.linspace(neutrino.x - 150, neutrino.x + 150, x_points)
                #        z_vs = np.linspace(neutrino.z - 100, neutrino.z + 100, y_points)
                #        llhs = []
                #        for z_v in z_vs:
                #            for x_v in x_vs:
                #                my_hypo = HYPO_T(neutrino.t, x_v, neutrino.y, z_v,
                #                               theta=neutrino.theta, phi=neutrino.phi,
                #                               track_energy=track.energy,
                #                               cascade_energy=cascade.energy,
                #                               cascade_e_scale=CASCADE_E_SCALE,
                #                               track_e_scale=TRACK_E_SCALE)
                #                my_hypo.set_binning(t_bin_edges, r_bin_edges,
                #                                    theta_bin_edges, phi_bin_edges)
                #                llh, noise = get_llh(hypo=my_hypo, t=t, x=x, y=y, z=z, q=q,
                #                                     string=strings, om=oms,
                #                                     ic_photon_info=ic_photon_info,
                #                                     dc_photon_info=dc_photon_info)
                #                print(' z = %.2f, x = %.2f : llh = %.2f' % (z_v, x_v, llh))
                #                llhs.append(llh)
                #        plt.clf()
                #        # [z, x]
                #        llhs = np.array(llhs)
                #        llhs = llhs.reshape(y_points, x_points)

                #        x_edges = np.linspace(x_vs[0] - np.diff(x_vs)[0]/2.,
                #                              x_vs[-1] + np.diff(x_vs)[0]/2.,
                #                              len(x_vs) + 1)
                #        z_edges = np.linspace(z_vs[0] - np.diff(z_vs)[0]/2.,
                #                              z_vs[-1] + np.diff(z_vs)[0]/2.,
                #                              len(z_vs) + 1)

                #        xx, yy = np.meshgrid(x_edges, z_edges)
                #        fig = plt.figure()
                #        ax = fig.add_subplot(111)
                #        ax.pcolormesh(xx, yy, llhs, cmap=CMAP)
                #        ax.set_ylabel('Vertex z (m)')
                #        ax.set_xlabel('Vertex x (m)')
                #        ax.set_xlim((x_edges[0], x_edges[-1]))
                #        ax.set_ylim((z_edges[0], z_edges[-1]))
                #        #truth
                #        ax.axvline(neutrino.x, color='r')
                #        ax.axhline(neutrino.z, color='r')
                #        ax.set_title('Event %i, E_cascade = %.2f GeV, E_track = %.2f GeV'
                #                     % (evt, cascade.energy, track.energy))
                #        plt.savefig('xz_%s.png' % evt, dpi=150)


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '-f', '--file', metavar='H5_FILE', type=str, required=True,
        help='''Events HDF5 file, containing (string, DOM, charge, time per
        event)''',
    )
    parser.add_argument(
        '--start-index', default=None, type=int,
        help='''Event index offset for event to start with (0-indexed)'''
    )
    parser.add_argument(
        '--stop-index', default=None, type=int,
        help='''Event index offset for event to start with (0-indexed)'''
    )
    parser.add_argument(
        '--tables-dir', metavar='DIR', type=str,
        default='/data/icecube/retro_tables/full1000',
        help='''Directory containing retro tables''',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(events_fpath=ARGS.file, start_index=ARGS.start_index,
         stop_index=ARGS.stop_index, tables_dir=ARGS.tables_dir)
