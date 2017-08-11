#!/usr/bin/env python

# pylint: disable=print-statement, xrange-builtin, wrong-import-position

"""
Load retro tables into RAM, then tales of icecube hdf5 files with events (hits
series) in them as inputs and calculates lilkelihoods.

At the moment, these likelihoods can be single points or 1d or 2d scans.
"""


from __future__ import absolute_import, division

from argparse import ArgumentParser
from collections import namedtuple
from copy import deepcopy
from itertools import izip, product
from os.path import expanduser, expandvars, isfile

import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numba # pylint: disable=unused-import
import numpy as np
import pyfits
from pyswarm import pso
from scipy.special import gammaln

from hypothesis.hypo_fast import (FTYPE, HYPO_PARAMS_T, event_to_hypo_params,
                                  Hypo)
from particles import ParticleArray


IC_TABLE_FPATH_PROTO = (
    'tables/tables/full1000/retro_nevts1000_IC_DOM%i_r_cz_t_angles.fits'
)

DC_TABLE_FPATH_PROTO = (
    'tables/tables/full1000/retro_nevts1000_DC_DOM%i_r_cz_t_angles.fits'
)

PULSE_SERIES = 'SRTInIcePulses'
ML_RECO_NAME = 'IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'
SPE_RECO_NAME = 'SPEFit2'
DETECTOR_GEOM_FILE = 'likelihood/geo_array.npy'
DFLT_EVENTS_FPATH = (
    '/fastio/icecube/deepcore/data/MSU_sample/level5pt/numu/14600'
    '/icetray_hdf5/Level5pt_IC86.2013_genie_numu.014600.000000.hdf5'
)
N_PHI_BINS = 20
NOISE_CHARGE = 0.00000025
CASCADE_E_SCALE = 10 #2.
TRACK_E_SCALE = 10 #20.
NUM_SCAN_POINTS = 20
HYPO_T = Hypo
CMAP = 'YlGnBu_r'

ABS_BOUNDS = HYPO_PARAMS_T(
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

REL_BOUNDS = HYPO_PARAMS_T(
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

MIN_DIMS = [] #('t', 'x', 'y', 'z', 'track_zenith', 'track_azimuth',
#                'track_energy')
"""Which dimensions to plug into minimizer (dims not fixed to truth)"""

SCAN_DIM_SETS = (
    't', 'x', 'y', 'z', 'track_zenith', 'track_azimuth', 'track_energy',
    #('t', 'x'), ('t', 'y'), ('t', 'z'), ('x', 'z'),
    #('track_zenith', 'track_azimuth'), ('track_zenith', 'z')
)
"""Which dimensions to scan. Tuples specify 2+ dimension scans"""


Event = namedtuple(typename='Event', # pylint: disable=invalid-name
                   field_names=('event', 'pulses', 'interaction', 'neutrino',
                                'track', 'cascade', 'ml_reco', 'spe_reco'))

Pulses = namedtuple(typename='Pulses', # pylint: disable=invalid-name
                    field_names=('strings', 'oms', 'times', 'charges'))

PhotonInfo = namedtuple(typename='PhotonInfo', # pylint: disable=invalid-name
                        field_names=('count', 'theta', 'phi', 'length'))
"""Intended to contain dictionaries with DOM depth number as keys"""

BinEdges = namedtuple(typename='BinEdges', # pylint: disable=invalid-name
                      field_names=('t', 'r', 'theta', 'phi'))

def expand(p):
    """Expand path"""
    return expanduser(expandvars(p))


def fill_photon_info(fpath, dom, scale=1, photon_info=None):
    """Fill photon info namedtuple-of-dictionaries from FITS file.

    Parameters
    ----------
    fpath : string
    dom : int
        Depth index (e.g. from 0 to 59)
    norm : float
    dom_eff : float
    photon_info : None or PhotonInfo
        If None, creates a new PhotonInfo namedtuple with empty dicts to fill.
        If one is provided, the existing component dictionaries are updated.

    Returns
    -------
    photon_info : PhotonInfo namedtuple
    bin_edges : BinEdges namedtuple

    """
    # pylint: disable=no-member
    if photon_info is None:
        photon_info = PhotonInfo(*[{}]*len(PhotonInfo._fields))

    with pyfits.open(expand(fpath)) as table:
        photon_info.count[dom] = table[0].data * scale
        photon_info.theta[dom] = table[1].data
        photon_info.phi[dom] = table[2].data
        photon_info.length[dom] = table[3].data

        # Note that we invert (reverse and multiply by -1) time edges
        bin_edges = BinEdges(t=-table[4].data[::-1], r=table[5].data,
                             theta=table[6].data, phi=[])

    return photon_info, bin_edges


class Events(object):
    """Container for events extracted from an HDF5 file.

    Parameters
    ----------
    events_fpath : string
        Path to HDF5 file

    """
    def __init__(self, events_fpath):
        self.load(events_fpath)
        self.events = []
        self._num_events = 0
        self.pulses = None
        self.pulse_event_boundaries = None
        self.int_type = None

    def load(self, events_fpath):
        """Load events from file, populating `self`.

        Parameters
        ----------
        events_fpath : string
            Path to HDF5 file

        """
        with h5py.File(expand(events_fpath)) as h5:
            pulses = h5[PULSE_SERIES]
            pulse_events = pulses['Event']
            self.pulses = Pulses(
                strings=pulses['string'],
                oms=pulses['om'],
                times=pulses['times'],
                charges=pulses['charge'],
            )

            # Calculate the first index into the pulses array for each unique
            # event
            self.pulse_event_boundaries = [0]
            self.pulse_event_boundaries.extend(
                np.where(np.diff(pulse_events))[0]
            )

            self.interactions = h5['I3MCWeightDict']['InteractionType']

            nu = h5['trueNeutrino']
            self.neutrinos = ParticleArray(
                evt=nu['Event'],
                t=nu['time'],
                x=nu['x'],
                y=nu['y'],
                z=nu['z'],
                zen=nu['zenith'],
                az=nu['azimuth'],
                energy=nu['energy'],
                length=None,
                pdg=nu['type'],
                color='r',
                linestyle=':',
                label='Neutrino'
            )
            mu = h5['trueMuon']
            self.tracks = ParticleArray(
                evt=mu['Event'],
                t=mu['time'],
                x=mu['x'],
                y=mu['y'],
                z=mu['z'],
                zen=mu['zenith'],
                az=mu['azimuth'],
                energy=mu['energy'],
                length=mu['length'],
                forward=True,
                color='b',
                linestyle='-',
                label='track'
            )
            cascade = h5['trueCascade']
            self.cascades = ParticleArray(
                evt=cascade['Event'],
                t=cascade['time'],
                x=cascade['x'],
                y=cascade['y'],
                z=cascade['z'],
                zen=cascade['zenith'],
                az=cascade['azimuth'],
                energy=cascade['energy'],
                length=None,
                color='y',
                label='cascade'
            )
            reco = h5[ML_RECO_NAME]
            self.ml_recos = ParticleArray(
                evt=reco['Event'],
                t=reco['time'],
                x=reco['x'],
                y=reco['y'],
                z=reco['z'],
                zen=reco['zenith'],
                az=reco['azimuth'],
                energy=reco['energy'],
                length=None,
                color='g',
                label='Multinest'
            )
            reco = h5[SPE_RECO_NAME]
            self.spe_recos = ParticleArray(
                evt=reco['Event'],
                t=reco['time'],
                x=reco['x'],
                y=reco['y'],
                z=reco['z'],
                zen=reco['zenith'],
                az=reco['azimuth'],
                color='m',
                label='SPE'
            )
        self.events = self.neutrinos.evt
        self._num_events = len(self.events)

    def __len__(self):
        return self._num_events

    def __iter__(self):
        for idx in xrange(self._num_events):
            yield self[idx]

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Convert slice into (start, stop, step) tuple
            range_args = idx.indices(len(self))
            return [self[i] for i in xrange(*range_args)]

        neutrino = self.neutrinos[idx]
        event = neutrino.evt
        pulse_start_idx = self.pulse_event_boundaries[idx]
        if idx < self._num_events - 1:
            pulse_stop_idx = self.pulse_event_boundaries[idx + 1]
        else:
            pulse_stop_idx = None
        slc = slice(pulse_start_idx, pulse_stop_idx)
        event = Event(
            event=event,
            pulses=Pulses(
                strings=self.pulses.strings[slc],
                oms=self.pulses.oms[slc],
                times=self.pulses.times[slc],
                charges=self.pulses.charges[slc]
            ),
            interaction=self.interactions[idx],
            neutrino=self.neutrinos[idx],
            track=self.tracks[idx],
            cascade=self.cascades[idx],
            ml_reco=self.ml_recos[idx],
            spe_reco=self.spe_recos[idx]
        )
        return event


def get_llh(hypo, event, detector_geometry, ic_photon_info, dc_photon_info):
    """Get log likelihood.

    Parameters
    ----------
    hypo : HYPO_T
    event : Event namedtuple
    ic_photon_info : dict
    dc_photon_info : dict

    Returns
    -------
    llh : float

    """
    llh = 0
    n_noise = 0

    # Loop over pulses (aka hits)
    for string, om, pulse_time, pulse_charge in izip(event.pulses):
        x, y, z = detector_geometry[string, om]

        # Get the photon expectations of the hypothesis in the DOM-hit
        # coordinates
        n_phot, photon_theta, photon_phi, photon_length = hypo.get_matrices(
            Dt=pulse_time, Dx=x, Dy=y, Dz=z
        )

        # Get the retro table for the pulse
        retro_idx = om - 1
        if string < 79:
            # these are ordinary icecube strings
            n_phot_map = ic_photon_info.count[retro_idx]
            photon_theta_map = ic_photon_info.theta[retro_idx]
            photon_phi_map = ic_photon_info.phi[retro_idx]
            photon_length_map = ic_photon_info.length[retro_idx]
        else:
            # these are deepocre strings
            n_phot_map = dc_photon_info.count[retro_idx]
            photon_theta_map = dc_photon_info.theta[retro_idx]
            photon_phi_map = dc_photon_info.phi[retro_idx]
            photon_length_map = dc_photon_info.length[retro_idx]

        # Get max llh between z_matrix and gamma_map
        # noise probability?
        expected_charge = 0
        for photon_idx, hypo_count in n_phot:
            # These two agles need to be inverted, because we're
            # backpropagating but want to match to forward-propagating photons
            hypo_theta = np.pi - photon_theta[photon_idx]
            hypo_phi = np.pi - photon_phi[photon_idx]
            #hypo_legth = photon_length[photon_idx]

            # Get map
            map_count = n_phot_map[photon_idx[0:3]]
            map_theta = photon_theta_map[photon_idx[0:3]]
            map_phi = photon_phi_map[photon_idx[0:3]]
            map_length = photon_length_map[photon_idx[0:3]]

            # Assume source is totally directed at 0.73 rad (cherenkov angle)

            # Accept this fraction as isotropic light
            dir_fraction = map_length**2
            print 'map length = ', dir_fraction
            #iso_fraction = 1. - dir_fraction

            # whats the cos(psi) between track direction and map?
            # accept this fraction of directional light
            # this is wrong i think...
            #proj_dir = np.arccos((np.cos(hypo_theta)*np.cos(map_theta) + np.sin(hypo_theta)*np.sin(map_theta)*np.cos(hypo_phi - map_phi)))

            proj_dir = (np.cos(hypo_theta) * np.cos(map_theta)
                        + (np.sin(hypo_theta) * np.sin(map_theta)
                           * np.cos(hypo_phi - map_phi)))

            #print proj_dir

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
            n_noise += 1

        expected_charge = max(NOISE_CHARGE, expected_charge)
        pulse_llh = -(pulse_charge * np.log(expected_charge)
                      - expected_charge
                      - gammaln(pulse_charge + 1))
        llh += pulse_llh

    return llh


def scan(llh_func, event, dims, scan_values, nominal_params=None,
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

        hypo = HYPO_PARAMS_T(params=params, cascade_e_scale=CASCADE_E_SCALE,
                             track_e_scale=TRACK_E_SCALE)
        llh = llh_func(hypo, event, **llh_func_kwargs)
        all_llh.append(llh)

    all_llh = np.array(all_llh, dtype=FTYPE)
    all_llh.reshape(tuple(len(u) for u in uniques))

    return all_llh


def main(events_fpath, start_index=None, stop_index=None):
    """Perform scans and minimization for events.

    Parameters
    ----------
    events_fpath : string
        Path to events HDF5 file

    start_index : None or int
        Event index (as ordered in events file) to start on. Specify 0 or
        `None` to start with the first event. I.e., iterate over
        `range(start_index, stop_index)`.

    stop_index : None or int
        Event index (as ordered in events file) to stop before. I.e., iterate
        over `range(start_index, stop_index)`.

    """
    # pylint: disable=no-member
    #events_file_basename, _ = splitext(basename(events_fpath))
    events = Events(events_fpath)

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
    for dom in range(60):
        # IceCube (non-DeepCore) tables
        fpath = IC_TABLE_FPATH_PROTO % dom
        if isfile(fpath):
            ic_photon_info, bin_edges = fill_photon_info(
                fpath=fpath,
                dom=dom,
                scale=norm * dom_eff_ic,
                photon_info=ic_photon_info
            )
            if ref_bin_edges is None:
                ref_bin_edges = bin_edges
            else:
                for test, ref in zip(bin_edges, ref_bin_edges):
                    assert np.array_equal(test, ref)
        else:
            print 'No table for IC DOM depth index %i' % dom

        # DeepCore tables
        fpath = DC_TABLE_FPATH_PROTO % dom
        if isfile(fpath):
            dc_photon_info, bin_edges = fill_photon_info(
                fpath=fpath,
                dom=dom,
                scale=norm*dom_eff_dc,
                photon_info=dc_photon_info
            )
            if ref_bin_edges is None:
                ref_bin_edges = bin_edges
            else:
                for test, ref in zip(bin_edges, ref_bin_edges):
                    assert np.array_equal(test, ref)
        else:
            print 'No table for IC DOM depth index %i' % dom

    # --- load detector geometry array ---
    detector_geometry = np.load(DETECTOR_GEOM_FILE)

    # Iterate through events
    for idx, event in enumerate(events[start_index:stop_index]):
        print 'working on event #%i / event ID %d' % (idx, event.event)

        llh_func_kwargs = dict(event=event,
                               detector_geometry=detector_geometry,
                               ic_photon_info=ic_photon_info,
                               dc_photon_info=dc_photon_info)

        truth_params = event_to_hypo_params(event)
        llh_truth = get_llh(hypo=HYPO_PARAMS_T(truth_params), **llh_func_kwargs)

        if MIN_DIMS:
            print 'Will minimize following dimension(s): %s' % MIN_DIMS
            print 'llh at truth = %.2f' % llh_truth

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
                if MIN_USE_RELATIVE_BOUNDS and dim in REL_BOUNDS:
                    nom_val = getattr(truth_params, dim)
                    lower = nom_val + REL_BOUNDS[0]
                    upper = nom_val + REL_BOUNDS[1]
                else:
                    lower, upper = ABS_BOUNDS[dim]
                lower_bounds.append(lower)
                upper_bounds.append(upper)

            def get_llh_partial(args):
                pass

            xopt1, fopt1 = pso(get_llh_partial, lower_bounds, upper_bounds,
                               kwargs=llh_func_kwargs,
                               minstep=1e-5,
                               minfunc=1e-1,
                               debug=True)

            #print 'truth at t=%.2f, x=%.2f, y=%.2f, z=%.2f, theta=%.2f, phi=%.2f' % tuple(truth)
            #print 'with llh = %.2f' % llh_truth
            #print 'optimum at t=%.2f, x=%.2f, y=%.2f, z=%.2f, theta=%.2f, phi=%.2f' % tuple([p for p in xopt1])
            #print 'with llh = %.2f\n' % fopt1

        if SCAN_DIM_SETS:
            for dims in SCAN_DIM_SETS:
                if isinstance(dims, basestring):
                    dims = [dims]

                nominal_params = deepcopy(truth_params)
                scan_values = []
                for dim in dims:
                    if SCAN_USE_RELATIVE_BOUNDS and dim in REL_BOUNDS:
                        nom_val = getattr(nominal_params, dim)
                        lower = nom_val + REL_BOUNDS[0]
                        upper = nom_val + REL_BOUNDS[1]
                    else:
                        lower, upper = ABS_BOUNDS[dim]
                    scan_values.append(
                        np.linspace(lower, upper, NUM_SCAN_POINTS)
                    )

                llh = scan(llh_func=get_llh, event=event, dims=dims,
                           scan_values=scan_values,
                           nominal_params=nominal_params,
                           llh_func_kwargs=llh_func_kwargs)

                #z_vs = np.linspace(neutrino.z - 50, neutrino.z + 50, NUM_SCAN_POINTS)

                #llhs = []
                #noises = []
                #for z_v in z_vs:
                #    print 'testing z = %.2f' % z_v
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
                #                print ' z = %.2f, x = %.2f : llh = %.2f' % (z_v, x_v, llh)
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
        '-f', '--file', metavar='H5_FILE', type=str,
        default=DFLT_EVENTS_FPATH,
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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    main(events_fpath=ARGS.file, start_index=ARGS.start_index,
         stop_index=ARGS.stop_index)
