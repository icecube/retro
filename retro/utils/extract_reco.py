#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Extract reco and reco error info
"""

from __future__ import absolute_import, division, print_function


__all__ = ["extract_reco", "summarize_reco"]


from collections import OrderedDict
from os import makedirs, walk
from os.path import abspath, basename, dirname, expanduser, expandvars, join, isdir, isfile, relpath
import pickle
import sys

import numpy as np
import pandas as pd

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import FitStatus
from retro.utils.stats import weighted_percentile
from retro.utils.weight_diff_tails import weight_diff_tails


def extract_reco(reco, recodir, eventsdir=None):
    """Extract truth and reco information.

    Parameters
    ----------
    reco : str
        Name of reconstruction to extract

    recodir : str
        Root directory to recurse into searching for recos

    eventsdir : str, optional
        If specified, directory structure must mirror that in `recodir`. If not
        specified, defaults to `recodir`.

    Returns
    -------
    events : numpy.array

    truth : numpy.array
        All events' truth values

    recos : numpy.array
        All events' reconstructed values (note if some but not all
        reconstructions were run on a file, the entire array is included from
        that file but you should only look at events with "fit_status" of
        `FitStatus.OK`)

    """
    if eventsdir is None:
        eventsdir = recodir
    recodir = abspath(expanduser(expandvars(recodir)))
    eventsdir = abspath(expanduser(expandvars(eventsdir)))

    # If the leaf reco/events/truth dir "recos" was specified, must go one up
    # to find events/truth
    if basename(recodir) == "recos":
        recodir = dirname(recodir)
    if basename(eventsdir) == "recos":
        eventsdir = dirname(eventsdir)

    all_events = []
    all_truths = []
    all_recos = []

    for events_dirpath, _, filenames in walk(eventsdir):
        if not "events.npy" in filenames:
            continue
        reco_filepath = join(
            recodir,
            relpath(events_dirpath, start=eventsdir),
            "recos",
            "{}.npy".format(reco),
        )
        if not isfile(reco_filepath):
            continue

        events = np.load(join(events_dirpath, "events.npy"))
        recos = np.load(reco_filepath)
        assert len(events) == len(recos)

        # Truth may or may not be present
        truth_filepath = join(events_dirpath, "truth.npy")
        if isfile(truth_filepath):
            truth = np.load(truth_filepath)
            all_truths.append(truth)

        all_events.append(events)
        all_recos.append(recos)

    assert len(all_truths) == 0 or len(all_truths) == len(all_events)
    if all_truths:
        truth = np.concatenate(all_truths)
    else:
        truth = None

    events = np.concatenate(all_events)
    recos = np.concatenate(all_recos)

    return events, truth, recos


def summarize_reco(
    truth,
    recos,
    point_estimator="median",
    outdir=None,
    verbosity=0,
):
    """
    Parameters
    ----------
    params : str or iterable thereof, optional
        If not specified, "standard" param names will be searched for in the
        reco

    point_estimator : str in {'mean', 'median', 'max'}, optional (default is 'median')
        This function only handles a point estimate of each parameter; use
        `point_estimator` to arrive at this from recos that end up with a
        posterior distribution.

    outdir : optional
        If specified, results will be written to this directory. If not
        specified, nothing is saved to disk.

    verbosity

    Returns
    -------
    summary : pandas.DataFrame

    """
    weights = truth["weight"]
    if "fit_status" in recos.dtype.names:
        mask = recos["fit_status"] == FitStatus.OK
        num_missing = np.count_nonzero(recos["fit_status"] == FitStatus.NotSet)
        num_invalid = len(mask) - np.count_nonzero(mask) - num_missing
        weights = weights[mask]
    else:
        mask = None

    summary = []
    for param in recos.dtype.names:
        try:
            pvals = recos[param]

            if pvals.dtype.names and point_estimator in pvals.dtype.names:
                pvals = pvals[point_estimator]

            if param.startswith('cascade'):
                if param == 'cascade_energy':
                    true_pname = 'total_cascade_em_equiv_energy'
                else:
                    true_pname = 'total_' + param
            else:
                true_pname = param

            if mask is None:
                mask = np.isfinite(pvals)
                num_missing = 0
                num_invalid = len(mask) - np.count_nonzero(mask) - num_missing
                weights = weights[mask]

            err = pvals[mask] - truth[true_pname][mask]
            if 'azimuth' in param:
                err = ((err + np.pi) % (2*np.pi)) - np.pi
            elif 'energy' in param:
                err /= truth[true_pname]

            if param == 'coszen':
                corrected_weights, _ = weight_diff_tails(
                    diff=err,
                    weights=weights,
                    inbin_lower=-1,
                    inbin_upper=+1,
                    range_lower=-1,
                    range_upper=+1,
                )
            else:
                corrected_weights = weights

            minval, q5, q25, median, q75, q95, maxval = weighted_percentile(
                a=err[mask],
                q=(0, 5, 25, 50, 75, 95, 100),
                weights=corrected_weights,
            )

            info = OrderedDict()
            info['reco'] = reco
            info['param'] = param
            info['n_invalid'] = len(mask) - np.count_nonzero(mask)
            info['err_mean'] = np.average(err[mask], weights=weights[:len(pvals)][mask])
            info['err_median'] = median
            info['err_min'] = minval
            info['err_max'] = maxval
            info['err_iq50'] = q75 - q25
            info['err_iq90'] = q95 - q5
            summary.append(info)
        except:
            sys.stderr.write('ERROR! -> "{}" reco, param "{}"\n'.format(reco, param))
            raise

    summary = pd.DataFrame(summary)
    summary = summary[info.keys()]
    summary.sort_values(by=info.keys(), inplace=True)

    for param in ['mean', 'median', 'min', 'max']:
        reco_perf['err_abs{}'.format(param)] = reco_perf['err_{}'.format(param)].abs()

    if outdir is not None:
        outdir = expanduser(expandvars(outdir))
        if not isdir(outdir):
            makedirs(outdir, mode=0o750)
            if verbosity > 0:
                sys.stderr.write('created dir "{}"\n'.format(outdir))

        outfpath = join(outdir, RECO_PERF_FNAME)
        reco_perf.to_pickle(outfpath)
        if verbosity > 0:
            sys.stderr.write('wrote reco performance summary to "{}"\n'.format(outfpath))

        outfpath = join(outdir, TRUTH_FNAME)
        #pickle.dump(truth, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        np.save(outfpath, truth)
        if verbosity > 0:
            sys.stderr.write('wrote all truth info to "{}"\n'.format(outfpath))

        outfpath = join(outdir, RECOS_FNAME)
        pickle.dump(reco_vals, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        if verbosity > 0:
            sys.stderr.write('wrote all reco info to "{}"\n'.format(outfpath))

    return reco_perf, reco_vals, truth
