#!/usr/bin/env python

# pylint: disable=wrong-import-position

"""
Load retro tables into RAM, then tales of icecube hdf5 files with events (hits
series) in them as inputs and calculates lilkelihoods.

At the moment, these likelihoods can be single points or 1d or 2d scans.
"""


from __future__ import absolute_import, division, print_function

from argparse import ArgumentParser
from collections import OrderedDict, Sequence
from copy import deepcopy
import cPickle as pickle
from itertools import izip, product
import math
import os
from os.path import abspath, dirname, isdir, isfile, join
import time

import numba # pylint: disable=unused-import
import numpy as np
import pyfits
from pyswarm import pso

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))
from retro import DC_DOM_JITTER_NS, IC_DOM_JITTER_NS
from retro import (FTYPE, HYPO_PARAMS_T, BinningCoords, HypoParams10D,
                   PhotonInfo, TimeSpaceCoord)
from retro import event_to_hypo_params, expand, poisson_llh
from retro.events import Events
from analytic_hypo import AnalyticHypo # pylint: disable=unused-import
from segmented_hypo import SegmentedHypo, IDX_R_IX # pylint: disable=unused-import


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

# Completely arbitrary first guesses
EPS = 0.8
EPS_STAT = EPS
EPS_CLSIM_LENGTH_BINNING = 2*EPS
EPS_CLSIM_ANGLE_BINNING = EPS
NOISE_CHARGE = 0.00000025 * 100
CASCADE_E_SCALE = 1
TRACK_E_SCALE = 1
NUM_JITTER_SAMPLES = 5
JITTER_SIGMA = 5

N_PHI_BINS = 20
NUM_SCAN_POINTS = 100
HypoClass = SegmentedHypo
HYPOCLASS_KWARGS = dict(time_increment=1)
LLH_USE_AVGPHOT = False
LLH_USE_NOHIT = False

RESULTS_DIR = (
    'results_avgphot%d_nohit%d%s_cesc%d_tesc%d_noise%.1e_jitsamp%d%s'
    % (LLH_USE_AVGPHOT,
       LLH_USE_NOHIT,
       '_eps%.1f' % EPS_STAT if LLH_USE_AVGPHOT else '',
       CASCADE_E_SCALE,
       TRACK_E_SCALE,
       NOISE_CHARGE,
       NUM_JITTER_SAMPLES,
       '_jitsig%d' % JITTER_SIGMA if NUM_JITTER_SAMPLES > 1 else '')
)
if not isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, mode=0o2777)

ABS_BOUNDS = HypoParams10D(
    t=(-1000, 1e6),
    x=(-1000, 1000),
    y=(-1000, 1000),
    z=(-1000, 1000),
    track_azimuth=(0, 2*np.pi),
    track_zenith=(-np.pi, np.pi),
    track_energy=(0, 100),
    cascade_azimuth=(0, 2*np.pi),
    cascade_zenith=(-np.pi, np.pi),
    cascade_energy=(0, 100)
)
"""Absolute bounds for scanning / minimizer to work within"""

REL_BOUNDS = HypoParams10D(
    t=(-300, 300),
    #t=(-1000, 1000),
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
    't',
    'x',
    'y',
    'z',
    'track_zenith',
    'track_azimuth',
    'track_energy',
    'cascade_energy',
    #('t', 'x'),
    #('t', 'y'),
    #('t', 'z'),
    #('x', 'y'),
    #('x', 'z'),
    #('track_zenith', 'track_azimuth'),
    #('track_zenith', 'z'),
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
        keyed by `dom_depth_index` and values are the arrays loaded from the
        FITS file.

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
def get_neg_llh(hypo, event, detector_geometry, ic_photon_info, dc_photon_info,
                detailed_info_list=None):
    """Get log likelihood.

    Parameters
    ----------
    hypo : HypoClass
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
    neg_llh : float
        Negative of the log likelihood

    """
    t0 = time.time()

    neg_llh = 0
    noise_counts = 0

    eps_angle = EPS_STAT + EPS_CLSIM_ANGLE_BINNING
    eps_length = EPS_STAT + EPS_CLSIM_LENGTH_BINNING

    ## Initialize with _all_ DOMs: tuples of (string, depth)
    #n_strings, n_doms = detector_geometry.shape[:2]
    #no_hit_doms = [sd for sd in product(range(n_strings), range(n_doms))]
    #print(no_hit_doms[:10])
    #print(len(no_hit_doms))

    # Loop over pulses (aka hits)
    for string, om, pulse_time, pulse_charge in izip(*event.pulses):
        string_om = (string, om)
        #try:
        #    no_hit_doms.remove(string_om)
        #except ValueError:
        #    pass
        x, y, z = detector_geometry[string_om]

        # String indices 0-78 (numbers 1-79) are ordinary IceCube strings
        if 0 <= string <= 78:
            timing_jitter = IC_DOM_JITTER_NS
            # Get the retro table corresponding to the hit DOM
            retrosim_photon_counts = ic_photon_info.count[om]
            retrosim_photon_avg_theta = ic_photon_info.theta[om]
            retrosim_photon_avg_deltaphi = ic_photon_info.phi[om]
            retrosim_photon_avg_len = ic_photon_info.length[om]
        # String indices 79-85 (numbers 80-86) are DeepCore strings
        elif 79 <= string <= 85:
            timing_jitter = DC_DOM_JITTER_NS
            # Get the retro table corresponding to the hit DOM
            retrosim_photon_counts = dc_photon_info.count[om]
            retrosim_photon_avg_theta = dc_photon_info.theta[om]
            retrosim_photon_avg_deltaphi = dc_photon_info.phi[om]
            retrosim_photon_avg_len = dc_photon_info.length[om]
        else:
            raise ValueError('Unhandled string index %d (number %d)'
                             % (string, string + 1))

        # TODO: store each LLH computed for jitter times to info, so we can see
        # if this has any meaningful effect, if jitter times should be
        # modified, or if more points should be sampled from
        # [-timing_jitter to +timing_jitter].

        # TODO: Set jitter on DOM-by-DOM basis, not just DeepCore vs. IceCube
        #       (if it is apprciably different for different DOMs)
        # TODO: Jitter via e.g. smearing photons that go into retro tables

        best_pulse_neg_llh = np.inf
        if NUM_JITTER_SAMPLES == 0:
            jitter_dts = [0]
        else:
            jitter_dts = np.linspace(
                start=-JITTER_SIGMA*timing_jitter,
                stop=+JITTER_SIGMA*timing_jitter,
                num=NUM_JITTER_SAMPLES
            )

        for jitter_dt in jitter_dts:
            hit_dom_coord = TimeSpaceCoord(
                t=pulse_time + jitter_dt, x=x, y=y, z=z
            )

            # Get the photon expectations of the hypothesis in coordinates
            # relative to the hit DOM
            try:
                hypo.compute_matrices(hit_dom_coord=hit_dom_coord)
            except (IndexError, ValueError):
                pulse_neg_llh = -poisson_llh(expected=NOISE_CHARGE,
                                             observed=pulse_charge)
                if pulse_neg_llh < best_pulse_neg_llh:
                    best_pulse_neg_llh = pulse_neg_llh
                continue

            expected_charge = 0
            for bin_idx, hypo_photon_info in hypo.photon_info.iteritems():
                if bin_idx.t < 0:
                    continue

                hypo_count = hypo_photon_info.count

                # Get retro simulation table
                retro_idx = (bin_idx.t, bin_idx.r, bin_idx.theta)
                try:
                    retro_prob = retrosim_photon_counts[retro_idx]
                except:
                    continue
                hypo_cell_expected_charge = hypo_count * retro_prob

                if LLH_USE_AVGPHOT:
                    hypo_length = hypo_photon_info.length
                    retro_length = retrosim_photon_avg_len[retro_idx]
                    length_weight = (
                        (1 - abs(hypo_length - retro_length) + eps_length) / (1 + eps_length)
                    )
                    hypo_cell_expected_charge *= length_weight

                    ## These two agles need to be inverted because we're
                    ## backpropagating but want to match to forward-propagating
                    ## photons
                    #hypo_theta = np.pi - hypo_photon_info.theta
                    #hypo_phi = np.pi - hypo_photon_info.phi

                    retro_theta = retrosim_photon_avg_theta[retro_idx]
                    retro_phi = (
                        bin_idx.phi + retrosim_photon_avg_deltaphi[retro_idx]
                    )

                    # alpha is smallest angle between retro avg photon angle
                    # and hypo avg photon angle. Note that we invert the retro
                    # direction vector to account for it being a _reverse_
                    # simulation from DOM to cell, whereas hypo has photons
                    # start from cell and go outwards.
                    neg_sin_retro_theta = math.sin(retro_theta)
                    retro_x = neg_sin_retro_theta * math.cos(retro_phi)
                    retro_y = neg_sin_retro_theta * math.sin(retro_phi)
                    retro_z = math.cos(retro_theta)
                    cos_alpha = (retro_x*hypo.track_dir_x
                                 + retro_y*hypo.track_dir_y
                                 + retro_z*hypo.track_dir_z)

                    angle_weight = (
                        (0.5 + 0.5*cos_alpha + eps_angle + (1 - retro_length))
                        / (1 + eps_angle + (1 - retro_length))
                    )
                    hypo_cell_expected_charge *= angle_weight
                else:
                    angle_weight = 1

                # Charge at the DOM weighted by matching avg. photon direction
                # & length, multiplied by the probability of light getting from
                # the DOM to the hypo cell (and therefore vice versa) from
                # retro simulation
                expected_charge += hypo_cell_expected_charge

            if expected_charge < NOISE_CHARGE:
                noise_counts += 1
                # "Add" in noise (i.e.: expected charge must be at least as
                # large as noise level)
                expected_charge = NOISE_CHARGE

            # Poisson log likelihood (negative for minimizers)
            pulse_neg_llh = -poisson_llh(expected=expected_charge,
                                         observed=pulse_charge)
            #print('pulse_neg_llh:', pulse_neg_llh)
            if pulse_neg_llh < best_pulse_neg_llh:
                best_pulse_neg_llh = pulse_neg_llh

        neg_llh += best_pulse_neg_llh

    if detailed_info_list is not None:
        detailed_info_list.append(dict(noise_counts=noise_counts))

    #print('time to compute likelihood: %.5f ms' % ((time.time() - t0) * 1000))
    return neg_llh

#@profile
def scan(llh_func, event, dims, scan_values, bin_spec, nominal_params=None,
         llh_func_kwargs=None):
    """Scan likelihoods for hypotheses changing one parameter dimension.

    Parameters
    ----------
    llh_func : callable
        Function used to compute a likelihood. Must take ``hypo`` and ``event``
        as first two arguments, where ``hypo`` is a HypoClass object and
        ``event`` is the argument passed here. Function must return just one
        value (the ``llh``)

    event : Event
        Event for which to get likelihoods

    dims : string or iterable thereof
        One of 't', 'x', 'y', 'z', 'azimuth', 'zenith', 'cascade_energy',
        or 'track_energy'.

    scan_values : iterable of floats, or iterable thereof
        Values to set for the dimension being scanned.

    bin_spec

    nominal_params : None or HYPO_PARAMS_T namedtuple
        Nominal values for all param values. The value for the params being
        scanned are irrelevant, as this is replaced with each value from
        `scan_values`. Therefore this is optional if _all_ parameters are
        subject to the scan.

    llh_func_kwargs : mapping or None
        Keyword arguments to pass to `get_neg_llh` function

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

    scan_sequences = []
    shape = []
    for sv in scan_values:
        if not isinstance(sv, Sequence):
            sv = list(sv)
        scan_sequences.append(sv)
        shape.append(len(sv))

    if nominal_params is None:
        #assert len(dims) == len(all_params)
        nominal_params = HYPO_PARAMS_T(*([np.nan]*len(all_params)))

    # Make nominal into a list so we can mutate its values as we scan
    params = list(nominal_params)

    # Get indices for each param that we'll be changing, in the order they will
    # be specified
    param_indices = []
    for dim in dims:
        param_indices.append(all_params.index(dim))

    all_llh = []
    for param_values in product(*scan_sequences):
        for pidx, pval in izip(param_indices, param_values):
            params[pidx] = pval

        hypo = HypoClass(
            params=params,
            cascade_e_scale=CASCADE_E_SCALE,
            track_e_scale=TRACK_E_SCALE,
            **HYPOCLASS_KWARGS
        )
        hypo.set_binning(*bin_spec)
        neg_llh = llh_func(hypo, event, **llh_func_kwargs)
        all_llh.append(neg_llh)

    all_llh = np.array(all_llh, dtype=FTYPE)
    all_llh.reshape(shape)

    return all_llh


def main(events_fpath, tables_dir, geom_file, start_index=None,
         stop_index=None):
    """Perform scans and minimization for events.

    Parameters
    ----------
    events_fpath : string
        Path to events HDF5 file

    tables_dir : string
        Path to directory containing the retro tables

    geom_file : string
        File containing detector geometry

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

    # Load tables

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

    ic_photon_info, dc_photon_info = None, None

    # Read in the actual tables
    ref_bin_edges = None
    for dom_depth_index in range(60):
        # IceCube (non-DeepCore) DOM retro tables
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

        # DeepCore DOM retro tables
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

    # Take bin edges from those stored in retro tables but also add phi bins
    # manually since no phi dependence in retro tables
    bin_edges = BinningCoords(
        t=bin_edges.t,
        r=bin_edges.r,
        theta=bin_edges.theta,
        phi=np.linspace(0, 2*np.pi, N_PHI_BINS + 1)
    )

    # Generate bin_spec (used by hypotheses) based on the edges from tables
    bin_min = BinningCoords(*(np.min(d) for d in bin_edges))
    bin_max = BinningCoords(*(np.max(d) for d in bin_edges))
    num_bins = BinningCoords(*(len(d) - 1 for d in bin_edges))
    bin_spec = (bin_min, bin_max, num_bins)

    # Load detector geometry array
    detector_geometry = np.load(geom_file)

    # Iterate through events
    for idx, event in enumerate(events[start_index:stop_index]):
        print('Working on event #%i / event ID %d' % (idx, event.event))
        print('    neutrino:', event.neutrino)
        print('    track:', event.track)
        print('    cascade:', event.cascade)

        llh_func_kwargs = dict(detector_geometry=detector_geometry,
                               ic_photon_info=ic_photon_info,
                               dc_photon_info=dc_photon_info)

        truth_params = event_to_hypo_params(event)
        truth_hypo = HypoClass(
            params=truth_params,
            cascade_e_scale=CASCADE_E_SCALE,
            track_e_scale=TRACK_E_SCALE,
            **HYPOCLASS_KWARGS
        )
        truth_hypo.set_binning(*bin_spec)
        llh_truth = get_neg_llh(hypo=truth_hypo, event=event,
                                **llh_func_kwargs)
        print('llh at truth = %.2f' % llh_truth)

        if MIN_DIMS:
            print('Will minimize following dimension(s): %s' % MIN_DIMS)

            min_results = None

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
                if (MIN_USE_RELATIVE_BOUNDS
                        and getattr(REL_BOUNDS, dim) is not None):
                    nom_val = getattr(truth_params, dim)
                    lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                    upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                else:
                    lower, upper = getattr(ABS_BOUNDS, dim)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

            def get_neg_llh_partial(args):
                """Minimizer callable that only includes free params as args"""
                pass

            xopt1, fopt1 = pso(get_neg_llh_partial, lower_bounds, upper_bounds,
                               kwargs=llh_func_kwargs,
                               minstep=1e-5,
                               minfunc=1e-1,
                               debug=True)

        if SCAN_DIM_SETS:
            print('Will scan following sets of dimensions: %s'
                  % str(SCAN_DIM_SETS))
            for dims in SCAN_DIM_SETS:
                print('Scanning dimension(s): %s...' % str(dims))
                if isinstance(dims, basestring):
                    dims = [dims]

                nominal_params = deepcopy(truth_params)
                scan_values = []
                for dim in dims:
                    if (SCAN_USE_RELATIVE_BOUNDS
                            and getattr(REL_BOUNDS, dim) is not None):
                        nom_val = getattr(nominal_params, dim)
                        lower = nom_val + getattr(REL_BOUNDS, dim)[0]
                        upper = nom_val + getattr(REL_BOUNDS, dim)[1]
                    else:
                        lower, upper = getattr(ABS_BOUNDS, dim)
                    scan_values.append(
                        np.linspace(lower, upper, NUM_SCAN_POINTS)
                    )

                neg_llh = scan(
                    llh_func=get_neg_llh, event=event, dims=dims,
                    scan_values=scan_values, bin_spec=bin_spec,
                    nominal_params=nominal_params,
                    llh_func_kwargs=llh_func_kwargs
                )

                datadump = OrderedDict([
                    ('filename', event.filename),
                    ('event', event.event),
                    ('uid', event.uid),
                    ('dims', dims),
                    ('scan_values', scan_values),
                    ('neg_llh', neg_llh),
                    ('truth', tuple(getattr(truth_params, d) for d in dims)),
                    ('LLH_USE_AVGPHOT', LLH_USE_AVGPHOT),
                    #('LLH_USE_ANGLE', LLH_USE_ANGLE),
                    ('LLH_USE_NOHIT', LLH_USE_NOHIT)
                ])
                fname = ('scan_results_event_%d_uid_%d_dims_%s.pkl'
                         % (event.event, event.uid, '_'.join(dims)))
                fpath = join(RESULTS_DIR, fname)
                pickle.dump(
                    datadump,
                    file(fpath, 'wb'),
                    pickle.HIGHEST_PROTOCOL
                )
                print('saved scan to "%s"' % fpath)
            print('')


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
    parser.add_argument(
        '--geom-file', metavar='NPY_FILE', type=str,
        default=DETECTOR_GEOM_FILE,
        help='''NPY file containing DOM locations as (string, dom, x, y, z)
        entries'''
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(events_fpath=ARGS.file, geom_file=ARGS.geom_file,
         start_index=ARGS.start_index, stop_index=ARGS.stop_index,
         tables_dir=ARGS.tables_dir)
