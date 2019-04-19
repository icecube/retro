#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Formulate priors for Retro Reco from previously-run reconstructions.

Three steps:
    * Extract truth values and reconstructions of various parameters into a
      simplified format; summarize performance of each reco and store all this
      info to disk. The performance info can be used to determine which recos
      are the best candidates to use as priors for the next steps.
    * Load the truth & reco information from disk and fit the error
      distributions with various `scipy.stats.distributions`.
    * Summarize results, finding the distribution with lowest mean-squared
      error (or other metric) for each parameter for a given reconstruction
"""

from __future__ import absolute_import, division, print_function


__all__ = [
    'RECOS',
    'DISTRIBUTIONS',
    'PATH_PROTO',
    'TRUTH_FNAME',
    'RECOS_FNAME',
    'RECO_PERF_FNAME',
    'DISTS_HASH_DIR',
    'ABS_FLAVS',
    'TIMEOUT',
    'extract',
    'fit_distributions_to_data',
    'fit',
    'main',
]


from argparse import ArgumentParser
from collections import Iterable, OrderedDict
import hashlib
import multiprocessing as mp
from os import makedirs
from os.path import abspath, dirname, expanduser, expandvars, join, isdir, isfile
import pickle
import sys
import time

import numpy as np
import pandas as pd
from six import string_types
from scipy.stats import distributions as stats_distributions
from scipy.stats._continuous_distns import _distn_names  # pylint: disable=protected-access

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.retro_types import FitStatus
from retro.utils.stats import weighted_percentile, fit_cdf
from retro.utils.weight_diff_tails import weight_diff_tails


RECOS = tuple(sorted(
    "CascadeLast_DC DipoleFit_DC L4_ToIEval2 L4_ToIEval3 L4_iLineFit L5_SPEFit11"
    " LineFit LineFit_DC MM_DC_LineFitI_MM_DC_Pulses_1P_C05 MM_IC_LineFitI MPEFit"
    " MPEFitMuEX PoleMuonLinefit PoleMuonLlhFit SPEFit2 SPEFit2MuEX_FSS SPEFit2_DC"
    " SPEFitSingle SPEFitSingle_DC ToI_DC"
    .split()
))

PARAMS = ('x', 'y', 'z', 'time', 'coszen', 'azimuth')

DISTRIBUTIONS = tuple(sorted(_distn_names))

#PATH_PROTO = '~/oscNext/pass2/genie/level5/{abs_flav:d}9002/oscNext_genie_level5_pass2.{abs_flav:d}9002.{i:06d}'
PATH_PROTO = '/data/icecube/sim/ic86/i3/oscNext/pass2/genie/level5/{abs_flav:d}9002/oscNext_genie_level5_pass2.{abs_flav:d}9002.{i:06d}'

TRUTH_FNAME = 'truth.npy'

RECOS_FNAME = 'recos.pkl'

RECO_PERF_FNAME = 'reco_perf.pkl'

DISTS_HASH_DIR = "distributions_sha256"

ABS_FLAVS = (12, 14, 16)

TIMEOUT = None


def extract(
    path_proto=PATH_PROTO,
    recos=RECOS,
    params=PARAMS,
    outdir=None,
    verbosity=0,
):
    """Extract truth and reco information, and produce summary of reco errors,
    optionally saving all of this to disk.

    Parameters
    ----------
    path_proto : str
    recos : str or iterable thereof, optional
    params : str or iterable thereof, optional
    outdir : optional
        If specified, results will be written to this directory. If not
        specified, nothing is saved to disk.

    Returns
    -------
    reco_perf : pandas.DataFrame
        Summary of reconstructions' performance
    reco_vals : numpy.array
    truth : numpy.array

    """
    if isinstance(recos, string_types):
        recos = [recos]

    recos_by_flav = OrderedDict()
    truth_by_flav = OrderedDict()

    for abs_flav in ABS_FLAVS:
        if verbosity > 1:
            sys.stderr.write('{}\n'.format(abs_flav))
        flav_truth = []

        # TODO: use globbing rather than specifying path_proto as such (also
        # allows for missing intermediate numbers)

        i = -1
        while True:
            i += 1
            fpath = join(path_proto.format(abs_flav=abs_flav, i=i), 'truth.npy')

            try:
                vals = np.load(fpath)
            except IOError:
                break
            flav_truth.append(vals)
            if verbosity > 1:
                sys.stderr.write('{}\n'.format(fpath))
        if flav_truth:
            flav_truth = np.concatenate(flav_truth)
        truth_by_flav[abs_flav] = flav_truth

        flav_recos = OrderedDict()
        for reco in recos:
            reco_vals = []
            i = -1
            while True:
                i += 1
                fpath = join(
                    path_proto.format(abs_flav=abs_flav, i=i),
                    'recos',
                    '{reco:s}.npy'.format(reco=reco)
                )
                try:
                    vals = np.load(fpath)
                except IOError:
                    break
                reco_vals.append(vals)
                if verbosity > 1:
                    sys.stderr.write('{}\n'.format(fpath))
            if reco_vals:
                reco_vals = np.concatenate(reco_vals)
                flav_recos[reco] = reco_vals
            if len(reco_vals) != len(flav_truth):
                if verbosity > 0:
                    sys.stderr.write(
                        'abs_flav {} reco "{}" has len {} but truth has len {}\n'
                        .format(abs_flav, reco, len(reco_vals), len(flav_truth))
                    )
        recos_by_flav[abs_flav] = flav_recos

    reco_vals = OrderedDict()
    truth = []
    for abs_flav in ABS_FLAVS:
        truth.append(truth_by_flav[abs_flav])
        for reco, vals in recos_by_flav[abs_flav].items():
            if reco not in reco_vals:
                reco_vals[reco] = []
            reco_vals[reco].append(vals)

    reco_vals = OrderedDict(
        [(reco, np.concatenate(vals)) for reco, vals in reco_vals.items()]
    )
    truth = np.concatenate(truth)

    reco_perf = []
    for reco, vals in reco_vals.items():
        weights = truth['weight']
        for param in params:
            try:
                xform = None
                try:
                    pvals = vals[param]
                except (KeyError, ValueError):
                    if 'coszen' in param:
                        reco_pname = param.replace('coszen', 'zenith')
                        pvals = vals[reco_pname]
                        xform = np.cos
                    elif 'zenith' in param:
                        reco_pname = param.replace('zenith', 'coszen')
                        pvals = vals[reco_pname]
                        xform = np.arccos

                if pvals.dtype.names and 'median' in pvals.dtype.names:
                    pvals = pvals['median']

                if xform is not None:
                    pvals = xform(pvals)

                if param.startswith('cascade'):
                    if param == 'cascade_energy':
                        true_pname = 'total_cascade_em_equiv_energy'
                    else:
                        true_pname = 'total_' + param
                else:
                    true_pname = param

                err = pvals - truth[true_pname][:len(pvals)]
                if 'fit_status' in vals.dtype.names:
                    mask = np.isfinite(err) & (vals['fit_status'] == FitStatus.OK)
                else:
                    mask = np.isfinite(err)
                valid_err = err[mask]
                if len(valid_err) == 0:
                    raise ValueError("{} {}".format(reco, param))
                if param == 'coszen':
                    cw, _ = weight_diff_tails(
                        diff=err,
                        weights=weights[:len(pvals)][mask],
                        inbin_lower=-1,
                        inbin_upper=+1,
                        range_lower=-1,
                        range_upper=+1,
                    )
                else:
                    cw = weights[:len(pvals)][mask]
                minval, q5, q25, median, q75, q95, maxval = weighted_percentile(
                    a=err[mask],
                    q=(0, 5, 25, 50, 75, 95, 100),
                    weights=cw,
                )
                iq50 = q75 - q25
                iq90 = q95 - q5
                info = OrderedDict()
                info['reco'] = reco
                info['param'] = param
                info['n_invalid'] = len(err) - len(valid_err)
                info['err_mean'] = np.average(err[mask], weights=weights[:len(pvals)][mask])
                info['err_median'] = median
                info['err_min'] = minval
                info['err_max'] = maxval
                info['err_iq50'] = iq50
                info['err_iq90'] = iq90
            except:
                sys.stderr.write('ERROR! -> "{}" reco, param "{}"\n'.format(reco, param))
                raise
            reco_perf.append(info)
    info = reco_perf[0]
    reco_perf = pd.DataFrame(reco_perf)
    reco_perf = reco_perf[info.keys()]
    reco_perf.sort_values(by=info.keys(), inplace=True)

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


def fit_distributions_to_data(
    data,
    weights,
    distributions=DISTRIBUTIONS,
    outfile=None,
    n_procs=None,
    timeout=None,
    verbosity=0,
):
    """
    Parameters
    ----------
    data : array of same len as `weights`
    weights : array of same len as `data`
    distributions : str, scipy.stats.distributions, or iterable thereof; optional
    outfile : str, optional
    n_procs : int, optional

    Returns
    -------
    results_d : OrderedDict

    """
    if outfile is not None:
        assert outfile.endswith('.pkl')

    if distributions is None:
        distributions = DISTRIBUTIONS

    if isinstance(distributions, string_types) or not isinstance(distributions, Iterable):
        distributions = [distributions]

    tmp_distributions = []
    for distribution in distributions:
        if isinstance(distribution, string_types):
            distribution = getattr(stats_distributions, distribution)
        tmp_distributions.append(distribution)
    distributions = tmp_distributions

    if verbosity > 0:
        sys.stdout.write(
            'distributions to fit:\n{}\n'.format(' '.join(d.name for d in distributions))
        )

    # Get CDF values at each data point
    sortind = np.argsort(data)
    sorted_data = data[sortind]
    sorted_weights = weights[sortind]
    cumsum = np.cumsum(sorted_weights)
    cdf = cumsum / cumsum[-1]

    kwds = [
        dict(
            x=sorted_data,
            cdf=cdf,
            distribution=d,
            x_is_data=True,
            verbosity=verbosity,
        )
        for d in distributions
    ]

    if n_procs is None:
        n_procs = mp.cpu_count()
    n_procs = min(n_procs, len(distributions))

    pool = mp.Pool(processes=n_procs)
    workers = [(k['distribution'].name, pool.apply_async(fit_cdf, kwds=k)) for k in kwds]
    results = []
    for dist_name, worker in workers:
        try:
            result = worker.get(timeout=timeout)
        except mp.TimeoutError:
            sys.stderr.write(
                'ERROR: fitting "{}" distribution to data exceeded timeout; moving on\n'
                .format(dist_name)
            )
        else:
            results.append(result)

    if len(results) == 0:
        raise ValueError("No fits succeeded before timeout of {} s".format(timeout))
    results_d = OrderedDict()
    for distribution in distributions:
        for result in results:
            if result['name'] == distribution.name:
                results_d[distribution.name] = result
                results.remove(result)
                continue

    if outfile is not None:
        pickle.dump(
            results_d,
            open(outfile, 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return results_d


def fit(datadir, reco, params=PARAMS, distributions=DISTRIBUTIONS, timeout=TIMEOUT, verbosity=0):
    """
    Parameters
    ----------
    datadir : str
    reco : str
    params : str or iterable thereof
    distributions : str, scipy.stats.distributions, or iterable thereof
    timeout : float >= 0, optional

    """
    datadir = expanduser(expandvars(datadir))
    assert isdir(datadir)

    if isinstance(params, string_types):
        params = [params]

    if isinstance(distributions, string_types) or not isinstance(distributions, Iterable):
        distributions = [distributions]

    tmp_distributions = []
    for distribution in distributions:
        if isinstance(distribution, string_types):
            distribution = getattr(stats_distributions, distribution)
        tmp_distributions.append(distribution)
    distributions = tmp_distributions

    distribution_names = sorted(set([d.name for d in distributions]))

    params = sorted(params)
    distribution_names = sorted(distribution_names)

    dist_str = " ".join(distribution_names)
    hasher = hashlib.sha256()
    hasher.update(dist_str)
    dists_hash = hasher.hexdigest()[:10]
    hash_dirpath = join(datadir, DISTS_HASH_DIR)
    if not isdir(hash_dirpath):
        makedirs(hash_dirpath, 0o750)
    hash_fpath = join(hash_dirpath, dists_hash)
    if not isfile(hash_fpath):
        with open(hash_fpath, 'w') as f:
            f.write(dists_hash + "  " + dist_str + "\n")

    truth = pickle.load(open(join(datadir, TRUTH_FNAME), 'rb'))
    reco_vals = pickle.load(open(join(datadir, RECOS_FNAME), 'rb'))

    if verbosity > 0:
        sys.stdout.write(
            'fitting reco "{}"\nparams\n    {}\nwith distributions (sha256={})\n    {}\n'
            .format(reco, ",".join(params), dists_hash, ",".join(distribution_names))
        )

    fits = OrderedDict()
    for param in params:
        err = reco_vals[reco][param] - truth[param]
        mask = np.isfinite(err) & (reco_vals[reco]['fit_status'] == FitStatus.OK)
        valid_err = err[mask]
        valid_wts = truth['weight'][mask]
        if verbosity > 0:
            sys.stdout.write('='*80 + '\n')
            sys.stdout.write('Fitting reco "{}", param "{}"...\n'.format(reco, param))
        t0 = time.time()
        results_d = fit_distributions_to_data(
            data=valid_err,
            weights=valid_wts,
            distributions=distributions,
            outfile=None,
            n_procs=None,
            timeout=timeout,
            verbosity=verbosity,
        )
        fits[param] = results_d
        if verbosity > 0:
            sys.stdout.write(' ... fitting took {:.1f} s\n'.format(time.time() - t0))
            sys.stdout.write('='*80 + '\n\n')

    outfpath = join(
        datadir,
        (
            'fit_info__reco={}__params={}__dists_sha256={}.pkl'
            .format(reco, ','.join(params), dists_hash)
        )
    )
    pickle.dump(fits, open(outfpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if verbosity > 0:
        sys.stdout.write('Wrote results to "{}"\n'.format(outfpath))


def main(descr=__doc__):
    """command line interface to `extract` and `fit` functions"""
    parser = ArgumentParser(description=descr)
    subparsers = parser.add_subparsers(help='sub-command help')

    extract_parser = subparsers.add_parser(
        'extract',
        help='Extract recos, truth, and produce a summary of how each reco performs',
    )
    extract_parser.set_defaults(func=extract)
    extract_parser.add_argument('--outdir', required=True)
    extract_parser.add_argument('--path-proto', default=PATH_PROTO)
    extract_parser.add_argument('--recos', nargs='+', default=RECOS)
    extract_parser.add_argument('--params', nargs='+', default=PARAMS)

    fit_parser = subparsers.add_parser(
        'fit',
        help='Fit reco error with various distributions',
    )
    fit_parser.set_defaults(func=fit)
    fit_parser.add_argument('--datadir', required=True)
    fit_parser.add_argument('--reco', required=True)
    fit_parser.add_argument('--params', nargs='+', default=PARAMS)
    fit_parser.add_argument('--distributions', nargs='+', default=DISTRIBUTIONS)
    fit_parser.add_argument(
        '--timeout', type=float, default=TIMEOUT,
        help='''each distribution fit to each parameter is allowed up to this
        many seconds to perform its entire fitting process'''
    )

    kwargs = vars(parser.parse_args())
    func = kwargs.pop('func')
    func(**kwargs)


if __name__ == '__main__':
    main()
