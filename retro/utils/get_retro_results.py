#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name


"""
Utilities for extracting and processing reconstruction information saved to disk
"""


from __future__ import absolute_import, division, print_function

__all__ = [
    'extract_from_leaf_dir',
    'augment_info',
    'get_retro_results',
    'parse_args',
]

from argparse import ArgumentParser
from collections import OrderedDict
from glob import glob
from operator import add
from os import walk
from os.path import abspath, basename, dirname, isdir, isfile, join, relpath
import pickle
import sys
import time

from dask.distributed import Client, LocalCluster, as_completed
import numpy as np
import pandas as pd

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.geom import rotate_points
from retro.utils.misc import expand, mkdir
from retro.utils.stats import estimate_from_llhp


KEEP_TRUTH_KEYS = (
    'pdg',
    'x',
    'y',
    'z',
    'time',
    'energy',
    'coszen',
    'azimuth',
    'unique_id',
    'highest_energy_daughter_pdg',
    'highest_energy_daughter_energy',
    'highest_energy_daughter_length',
    'highest_energy_daughter_coszen',
    'highest_energy_daughter_azimuth',
    'longest_daughter_pdg',
    'longest_daughter_energy',
    'longest_daughter_length',
    'longest_daughter_coszen',
    'longest_daughter_azimuth',
    #'cascade_pdg',
    #'cascade_energy',
    #'cascade_coszen',
    #'cascade_azimuth',
    'InteractionType',
    'LengthInVolume',
    'OneWeight',
    'TargetPDGCode',
    'TotalInteractionProbabilityWeight',
    'weight',
)

# apply to both "estimate_prefit" and "estimate"
KEEP_ATTRS = ('num_llh', 'max_llh', 'max_postproc_llh')
KEEP_RUN_INFO_KEYS = ('run_time',)

# only for "estimate"
KEEP_EST_FIT_META_KEYS = ('ins_logZ', 'ins_logZ_err', 'logZ', 'logZ_err')

# only for "estimate_prefit"
KEEP_EST_PRFT_FIT_META_KEYS = (
    'iterations',
    'num_failures',
    'num_mutation_successes',
    'num_simplex_successes',
    'stopping_flag',
)

PL_RECO_NAMES = (
    'Pegleg_Fit_MN',
    'Pegleg_Fit_SP_3Iter',
    'Pegleg_Fit_LB_3Iter',
    'pegleg',
)
RETRO_RECO_NAMES = ('retro_pft', 'retro')

SIMPLE_ERR_PARAMS = (
    'x',
    'y',
    'z',
    'rho',
    'position_phi',
    'time',
    'energy',
    'visible_energy',
    'coszen',
    'zenith',
    'azimuth',
    'track_energy',
    'track_coszen',
    'track_zenith',
    'track_azimuth',
    #'cascade_energy',
    #'cascade_coszen',
    #'cascade_zenith',
    #'cascade_azimuth',
)


def extract_from_leaf_dir(
    recodir,
    eventdir,
    flavdir,
    filenum,
    infos=None,
    recompute_estimate=False,
):
    """Given a leaf recodir and the corresponding leaf eventdir, extract information for
    the events contained within.

    Parameters
    ----------
    recodir : string
    eventdir : string
    flavdir : string
    filenum : string
    infos : mapping, optional
    recompute_estimate : bool, optional

    """
    if infos is None:
        infos = {}
    sys.stdout.write('{} '.format(filenum))

    tru_npy_f = join(eventdir, 'truth.npy')
    if isfile(tru_npy_f):
        truths = np.load(tru_npy_f)
    else:
        print('no truth info at path "{}"'.format(tru_npy_f))
        return infos

    pl_recos = OrderedDict()
    for pl_reco_name in PL_RECO_NAMES:
        fname = join(eventdir, 'recos', pl_reco_name + '.npy')
        if not isfile(fname):
            continue
        pl_recos[pl_reco_name] = np.load(fname)

    for est_fpath in glob(join(recodir, '*.crs_prefit_mn.estimate*.pkl')):
        with open(est_fpath, 'rb') as f:
            try:
                estimates = pickle.load(f)
            except EOFError:
                continue

        kinds = estimates['kind'].values
        params = estimates['param'].values
        mean_index = int(np.argwhere(kinds == 'mean'))
        lower_index = int(np.argwhere(kinds == 'lower_bound'))
        upper_index = int(np.argwhere(kinds == 'upper_bound'))

        if 'estimate_prefit' in est_fpath:
            is_prefit = True
            pfx = 'retro_pft_'
        else:
            is_prefit = False
            pfx = 'retro_'

        for event_idx, estimate in estimates.data_vars.items():
            event_idx = int(event_idx)
            info_key = (flavdir, filenum, event_idx)

            sfx = '_prefit' if is_prefit else ''
            if recompute_estimate:
                llhp_fpath = join(
                    recodir,
                    'evt{}.crs_prefit_mn.llhp{}.npy'.format(event_idx, sfx),
                )
                llhp = np.load(llhp_fpath)

                # TODO: record what method is & step so we can re-create estimates;
                # following only applies to "crs_prefit_mn"
                if is_prefit:
                    new_estimate = estimate_from_llhp(
                        llhp=llhp,
                        treat_dims_independently=False,
                        use_prob_weights=True,
                        priors_used=None,
                    )
                else:
                    new_estimate = estimate_from_llhp(
                        llhp=llhp,
                        treat_dims_independently=False,
                        use_prob_weights=True,
                        priors_used=estimate.attrs['priors_used'],
                    )
                est = new_estimate
            else:
                est = estimate

            est = est.__array__()

            try:
                if info_key not in infos:
                    info = infos[info_key] = dict(
                        flavdir=flavdir,
                        filenum=filenum,
                        event_idx=event_idx,
                    )

                    # -- Get truth into info dict -- #

                    truth = truths[event_idx]
                    for k in KEEP_TRUTH_KEYS:
                        #if k in truth:
                        info[k] = truth[k]

                    # -- Get Pegleg recos into info dict -- #

                    for pl_reco_name, pl_reco in pl_recos.items():
                        for param_name in pl_reco.dtype.names:
                            info['{}_{}'.format(pl_reco_name, param_name)] = (
                                pl_reco[event_idx][param_name]
                            )
                else:
                    info = infos[info_key]

                # -- Get Retro recos & metadata into info dict -- #

                attrs = estimate.attrs
                for attr in KEEP_ATTRS:
                    info[pfx + attr] = attrs[attr]

                run_info = attrs['run_info']
                for key in KEEP_RUN_INFO_KEYS:
                    info[pfx + key] = run_info[key]

                fit_meta = run_info['fit_meta']
                if is_prefit:
                    keep_fit_meta_keys = KEEP_EST_PRFT_FIT_META_KEYS
                else:
                    keep_fit_meta_keys = KEEP_EST_FIT_META_KEYS
                for key in keep_fit_meta_keys:
                    info[pfx + key] = fit_meta[key]

                for param_index, param in enumerate(params):
                    info[pfx + param] = est[mean_index, param_index]
                    lb = est[lower_index, param_index]
                    ub = est[upper_index, param_index]
                    width = ub - lb
                    if 'az' in param:
                        width = np.abs((width + np.pi) % (2*np.pi) - np.pi)
                    info[pfx + param + '_lower_bound'] = lb
                    info[pfx + param + '_upper_bound'] = ub
                    info[pfx + param + '_width'] = width

            except Exception as e:
                print(
                    'ERROR: event_idx {} in file "{}": {}'
                    .format(event_idx, est_fpath, e)
                )
                raise

    return infos.values()


def augment_info(info):
    """Add additional columns to DataFrame produced by get_retro_results (e.g.,
    reconstruction errors).

    Modifies `info` in-place.

    Parameters
    ----------
    info : pandas.DataFrame

    """
    # Copy "truth"-prefixed variables to have no prefix
    for param in SIMPLE_ERR_PARAMS:
        if 'truth_' + param in info and param not in info:
            info[param] = info['truth_' + param]

    # True track params
    if 'track_coszen' not in info and 'highest_energy_daughter_coszen' in info:
        info['track_coszen'] = info['highest_energy_daughter_coszen']
    if 'track_zenith' not in info and 'highest_energy_daughter_zenith' in info:
        info['track_zenith'] = info['highest_energy_daughter_zenith']
    if 'track_azimuth' not in info and 'highest_energy_daughter_azimuth' in info:
        info['track_azimuth'] = info['highest_energy_daughter_azimuth']
    if 'track_energy' not in info and 'highest_energy_daughter_energy' in info:
        info['track_energy'] = info['highest_energy_daughter_energy']

    # TODO: intelligently define track, cascade PDG
    if 'track_pdg' not in info and 'highest_energy_daughter_pdg' in info:
        info['track_pdg'] = info['highest_energy_daughter_pdg']

    # Define coszen or zenith depending on which is already defined
    for pfx in ('', 'track_'): #, 'cascade_'):
        cz_col = pfx + 'coszen'
        zen_col = pfx + 'zenith'
        if cz_col not in info:
            info[cz_col] = np.cos(info[zen_col])
        if zen_col not in info:
            info[zen_col] = np.arccos(info[cz_col])

    # -- Derived from the above -- #

    #info['visible_energy'] = info['cascade_energy'] + info['track_energy']
    #info['bjorken_y'] = info['cascade_energy'] / info['energy']
    info['rho'] = np.hypot(info['x'] - 50, info['y'] + 50)
    info['position_phi'] = np.arctan2(info['y'] + 50, info['x'] - 50)

    # -- Reco params -- #

    for reco in PL_RECO_NAMES + RETRO_RECO_NAMES:
        for pfx in ('', 'track_', 'cascade_'):
            zen_col = '{}_{}zenith'.format(reco, pfx)
            cz_col = '{}_{}coszen'.format(reco, pfx)
            if cz_col not in info and zen_col in info:
                info[cz_col] = np.cos(info[zen_col])
            if zen_col not in info and cz_col in info:
                info[zen_col] = np.arccos(info[cz_col])

        trck_en_col = '{}_track_energy'.format(reco)
        if trck_en_col not in info:
            continue
        reco_track_energy = info[trck_en_col]
        reco_cascade_energy = info['{}_cascade_energy'.format(reco)]
        info['{}_energy'.format(reco)] = reco_track_energy + 2 * reco_cascade_energy
        info['{}_visible_energy'.format(reco)] = reco_track_energy + reco_cascade_energy

        reco_x = info['{}_x'.format(reco)]
        reco_y = info['{}_y'.format(reco)]
        info['{}_rho'.format(reco)] = np.hypot(reco_x + 50, reco_y - 50)
        info['{}_position_phi'.format(reco)] = np.arctan2(reco_y + 50, reco_x - 50)

        # -- Errors -- #

        # "Simple" errors to compute
        for param in SIMPLE_ERR_PARAMS:
            reco_col = '{}_{}'.format(reco, param)
            if reco_col not in info or param not in info:
                continue
            err = info[reco_col] - info[param]
            if 'azimuth' in param:
                err = (err + np.pi) % (2*np.pi) - np.pi
            info['{}_{}_error'.format(reco, param)] = err

        # Angle error
        for pfx in ('', 'track_'): #, 'cascade_'):
            true_zen_col = '{}zenith'.format(pfx)
            true_az_col = '{}azimuth'.format(pfx)
            reco_zen_col = '{}_{}zenith'.format(reco, pfx)
            reco_az_col = '{}_{}azimuth'.format(reco, pfx)

            if not (
                true_zen_col in info
                and true_az_col in info
                and reco_zen_col in info
                and reco_az_col in info
            ):
                print('skipping "{}_{}angle_error"'.format(reco, pfx))
                continue

            true_zen = info[true_zen_col].values
            true_az = info[true_az_col].values
            reco_zen = info[reco_zen_col].values
            reco_az = info[reco_az_col].values

            q_theta = np.empty_like(true_zen)
            q_phi = np.empty_like(true_zen)

            rotate_points(
                p_theta=reco_zen,
                p_phi=reco_az,
                rot_theta=-true_zen,
                rot_phi=-true_az,
                q_theta=q_theta,
                q_phi=q_phi,
            )
            info['{}_{}angle_error'.format(reco, pfx)] = q_theta


def get_retro_results(
    outdir,
    recos_basedir,
    events_basedir,
    recompute_estimate=False,
    overwrite=False,
):
    """Extract all rectro reco results from a reco directory tree, merging with original
    event information from correspoding source events directory tree. Results are
    populated to a Pandas DataFrame, saved to disk, and this is returned to the user.

    Parameters
    ----------
    outdir : string
    recos_basedir : string
    events_basedir : string
    recompute_estimate : bool, optional
    overwrite : bool, optional

    """
    t0 = time.time()
    outdir = abspath(expand(outdir))
    if not isdir(outdir):
        mkdir(outdir)
    outfile_path = join(outdir, 'reconstructed_events.feather')
    if not overwrite and isfile(outfile_path):
        raise IOError('Output file path already exists at "{}"'.format(outfile_path))

    #if parallel:
    #    cluster = LocalCluster(threads_per_worker=1, diagnostics_port=None)
    #    client = Client(cluster)

    results = []
    try:
        # Walk directory hierarchy
        #futures = []
        for reco_dirpath, _, files in walk(recos_basedir, followlinks=True):
            is_leafdir = False
            for f in files:
                if f[-3:] == 'pkl' and f[:3] in ('slc', 'evt'):
                    is_leafdir = True
                    break
            if not is_leafdir:
                continue
            rel_dirpath = relpath(path=reco_dirpath, start=recos_basedir)
            if events_basedir is not None:
                event_dirpath = join(events_basedir, rel_dirpath)
                if not isdir(event_dirpath):
                    raise IOError('Event directory does not exist: "{}"'
                                  .format(event_dirpath))

            abs_reco_dirpath = abspath(reco_dirpath)
            filenum = basename(abs_reco_dirpath)
            flavdir = basename(dirname(abs_reco_dirpath))

            kwargs = dict(
                recodir=reco_dirpath,
                eventdir=event_dirpath,
                flavdir=flavdir,
                filenum=filenum,
                recompute_estimate=recompute_estimate,
            )
            #print(kwargs)
            #futures.append(client.submit(extract_from_leaf_dir, **kwargs))
            results.append(extract_from_leaf_dir(**kwargs))

        #results = [f.result() for f in as_completed(futures)]

    finally:
        pass
    #    cluster.close()
    #    client.close()
    #    del client
    #    del cluster

    # Convert to a single list containing all events
    #for r in results:
    #    print(type(r))
    #    print(r)
    all_events = reduce(add, results, [])

    # Convert to pandas DataFrame
    all_events = pd.DataFrame(all_events)

    # Save to disk
    all_events.to_feather(outfile_path)
    print('\nAll_events saved to "{}"\n'.format(outfile_path))

    nevents = len(all_events)
    dt = time.time() - t0
    print('\nTook {:.3f} s to extract {} events'.format(dt, nevents))

    return all_events


def parse_args():
    """Parse command line arguments.

    Returns
    -------
    kw : dict
        Arguments to be passed to
    """
    parser = ArgumentParser(
        description="""Extract reco and truth information and merge into a Pandas
        DataFrame. Results will be saved to "<outdir>/reconstructed_events.feather" in
        the feather file format (https://github.com/wesm/feather)."""
    )

    parser.add_argument(
        '--outdir', required=True
    )
    parser.add_argument(
        '--recos-basedir', required=True,
        help='Path to base directory containing Retro reconstruction information'
    )
    parser.add_argument(
        '--events-basedir', required=True,
        help='Path to base directory containing source events that were reconstructed'
    )
    parser.add_argument(
        '--recompute-estimate', action='store_true',
        help='''Recompute estimate from raw LLHP (if these are stored to disk; otherwise
        fail)'''
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite output file if it already exists',
    )

    kw = vars(parser.parse_args())

    return kw


if __name__ == '__main__':
    get_retro_results(**parse_args())
