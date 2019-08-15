# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Convenience functions for initializing major objects needed for Retro
likelihood processing (includes instantiating objects and loading the data
needed for them).
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'setup_dom_tables',
    'setup_discrete_hypo',
    'get_hits',
    'parse_args',
]

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

from argparse import ArgumentParser
from collections import Iterable, Mapping, OrderedDict
from copy import deepcopy
from operator import getitem
from os import listdir, walk
from os.path import abspath, dirname, isdir, isfile, join, splitext
import re
import sys
import time

import numpy as np
from six import string_types

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import const, load_pickle
from retro.hypo.discrete_hypo import DiscreteHypo
from retro.hypo import discrete_cascade_kernels as dck
from retro.hypo import discrete_muon_kernels as dmk
from retro.i3info.angsens_model import load_angsens_model
from retro.i3info.extract_gcd import extract_gcd
from retro.retro_types import (
    HIT_T, SD_INDEXER_T, HITS_SUMMARY_T, TriggerConfigID, TriggerTypeID, TriggerSourceID
)
from retro.tables.retro_5d_tables import (
    NORM_VERSIONS, TABLE_KINDS, Retro5DTables
)
from retro.utils.data_mc_agreement import quantize_min_q_filter
from retro.utils.misc import expand, nsort_key_func


I3_FNAME_INFO_RE = re.compile(
    r'''
    Level(?P<proc_level>.+) # processing level e.g. 5p or 5pt (???)
    _(?P<detector>[^.]+)    # detector, e.g. IC86
    \.(?P<year>\d+)         # year
    _(?P<generator>.+)      # generator, e.g. genie
    _(?P<flavor>.+)         # flavor, e.g. nue
    \.(?P<run>\d+)          # run, e.g. 012600
    \.(?P<filenum>\d+)      # file number, e.g. 000000
    ''', (re.VERBOSE | re.IGNORECASE)
)


def setup_dom_tables(
    dom_tables_kind,
    dom_tables_fname_proto,
    gcd,
    norm_version='binvol2.5',
    use_sd_indices=const.ALL_STRS_DOMS,
    num_phi_samples=None,
    ckv_sigma_deg=None,
    template_library=None,
    compute_t_indep_exp=True,
    no_noise=False,
    force_no_mmap=False,
):
    """Instantiate and load single-DOM tables.

    Parameters
    ----------
    dom_tables_kind : str
    dom_tables_fname_proto : str
    gcd : str
    norm_version : str, optional
    use_sd_indices : sequence, optional
    num_phi_samples : int, optional
    ckv_sigma_deg : float, optional
    template_library : str, optional
    compute_t_indep_exp : bool, optional
    no_noise : bool, optional
    force_no_mmap : bool, optional

    Returns
    -------
    dom_tables : Retro5DTables

    """
    print('Instantiating and loading DOM tables')
    t0 = time.time()

    dom_tables_fname_proto = expand(dom_tables_fname_proto)

    # TODO: set mmap based on memory?
    if force_no_mmap:
        mmap = False
    else:
        mmap = 'uncompr' in dom_tables_kind

    if dom_tables_kind in ['raw_templ_compr', 'ckv_templ_compr']:
        template_library = np.load(expand(template_library))
    else:
        template_library = None

    gcd = extract_gcd(gcd)

    if no_noise:
        gcd['noise'] = np.zeros_like(gcd['noise'])

    # Instantiate single-DOM tables class
    dom_tables = Retro5DTables(
        table_kind=dom_tables_kind,
        geom=gcd['geo'],
        rde=gcd['rde'],
        noise_rate_hz=gcd['noise'],
        compute_t_indep_exp=compute_t_indep_exp,
        norm_version=norm_version,
        num_phi_samples=num_phi_samples,
        ckv_sigma_deg=ckv_sigma_deg,
        template_library=template_library,
        use_sd_indices=use_sd_indices,
    )

    if '{subdet' in dom_tables_fname_proto:
        doms = const.ALL_DOMS
        for subdet in ['ic', 'dc']:
            if subdet == 'ic':
                strings = const.IC_STRS
            else:
                strings = const.DC_STRS

            for dom in doms:
                fpath = dom_tables_fname_proto.format(
                    subdet=subdet, dom=dom, depth_idx=dom - 1
                )

                shared_table_sd_indices = []
                for string in strings:
                    sd_idx = const.get_sd_idx(string=string, om=dom, pmt=0)
                    if sd_idx not in use_sd_indices:
                        continue
                    shared_table_sd_indices.append(sd_idx)

                if not shared_table_sd_indices:
                    continue

                dom_tables.load_table(
                    fpath=fpath,
                    sd_indices=shared_table_sd_indices,
                    mmap=mmap,
                )

    elif '{string}' in dom_tables_fname_proto:
        raise NotImplementedError('dom_tables_fname_proto with {string} not'
                                  ' implemented')

    elif '{string_idx}' in dom_tables_fname_proto:
        raise NotImplementedError('dom_tables_fname_proto with {string_idx}'
                                  ' not implemented')

    elif '{cluster_idx}' in dom_tables_fname_proto:
        cluster_idx = -1
        while True:
            cluster_idx += 1
            dpath = dom_tables_fname_proto.format(cluster_idx=cluster_idx)
            if not isdir(dpath):
                print('failed to find', dpath)
                break

            # TODO: make the omkeys field generic to all tables & place
            # loading & intersection ops within the `load_table` method.
            omkeys = np.load(join(dpath, 'omkeys.npy'))
            sd_indices = set(const.omkeys_to_sd_indices(omkeys))
            shared_table_sd_indices = sd_indices.intersection(use_sd_indices)

            dom_tables.load_table(
                fpath=dpath,
                sd_indices=shared_table_sd_indices,
                mmap=mmap,
            )

    else:
        stacked_tables_fpath = expand(join(
            dom_tables_fname_proto,
            'stacked_{}.npy'.format(dom_tables.table_name)
        ))
        stacked_tables_meta_fpath = expand(join(
            dom_tables_fname_proto,
            'stacked_{}_meta.pkl'.format(dom_tables.table_name)
        ))
        stacked_t_indep_tables_fpath = expand(join(
            dom_tables_fname_proto,
            'stacked_{}.npy'.format(dom_tables.t_indep_table_name)
        ))
        dom_tables.load_stacked_tables(
            stacked_tables_meta_fpath=stacked_tables_meta_fpath,
            stacked_tables_fpath=stacked_tables_fpath,
            stacked_t_indep_tables_fpath=stacked_t_indep_tables_fpath,
            mmap_t_indep=mmap,
        )

    for table in dom_tables.tables:
        assert np.all(np.isfinite(table['weight'])), 'table not finite!'
        assert np.all(table['weight'] >= 0), 'table is negative!'
        assert np.min(table['index']) >= 0, 'table has negative index'
        if dom_tables.template_library is not None:
            assert np.max(table['index']) < dom_tables.template_library.shape[0], \
                    'table too large index'
    if dom_tables.template_library is not None:
        assert np.all(np.isfinite(dom_tables.template_library)), 'templates not finite!'
        assert np.all(dom_tables.template_library >= 0), 'templates have negative values!'

    print('  -> {:.3f} s\n'.format(time.time() - t0))

    return dom_tables


def setup_tdi_tables(tdi=None, mmap=False):
    """Load and instantiate (Cherenkov) TDI tables.

    Parameters
    ----------
    tdi : sequence of strings, optional
        Path to TDI tables' `ckv_tdi_table.npy` files, or paths to
        directories containing those files; one entry per TDI table

    mmap : bool

    Returns
    -------
    tdi_tables : tuple of 0 or more numpy arrays
    tdi_metas : tuple of 0 or more OrderedDicts

    """
    if tdi is None:
        return (), ()

    mmap_mode = 'r' if mmap else None

    tdi_tables = []
    tdi_metas = []
    for tdi_ in tdi:
        if tdi_ is None:
            continue
        tdi_ = expand(tdi_)
        if isdir(tdi_):
            tdi_ = join(tdi_, 'ckv_tdi_table.npy')

        print('Loading and instantiating TDI table at "{}"'.format(tdi_))

        be = load_pickle(join(dirname(tdi_), 'tdi_bin_edges.pkl'))
        meta = load_pickle(join(dirname(tdi_), 'tdi_metadata.pkl'))
        meta['bin_edges'] = be
        tdi_table = np.load(tdi_, mmap_mode=mmap_mode)

        tdi_metas.append(meta)
        tdi_tables.append(tdi_table)

    return tuple(tdi_tables), tuple(tdi_metas)


def setup_discrete_hypo(cascade_kernel=None, track_kernel=None, track_time_step=None):
    """Convenience function for instantiating a discrete hypothesis with
    specified kernel(s).

    Parameters
    ----------
    cascade_kernel : string or None
        One of {"point", "point_ckv", or "one_dim"}
    track_kernel : string or None
    track_time_step : float or None

    Returns
    -------
    hypo_handler

    """
    generic_kernels = []
    generic_kernels_kwargs = []

    pegleg_kernel = None
    pegleg_kernel_kwargs = None

    scaling_kernel = None
    scaling_kernel_kwargs = None

    if cascade_kernel is not None:
        assert cascade_kernel in dck.CASCADE_KINDS, str(cascade_kernel)
        cascade_kernel_func = getattr(dck, cascade_kernel + '_cascade')
        cascade_kernel_kwargs = dict()
        if cascade_kernel.startswith('scaling'):
            if scaling_kernel is not None:
                raise ValueError('can only have one scaling kernel')
            scaling_kernel = cascade_kernel_func
            scaling_kernel_kwargs = cascade_kernel_kwargs
        else:
            generic_kernels.append(cascade_kernel_func)
            generic_kernels_kwargs.append(cascade_kernel_kwargs)

    if track_kernel is not None:
        assert track_kernel in dmk.MUON_KINDS, str(track_kernel)
        print('track_kernel:', track_kernel)
        track_kernel_func = getattr(dmk, track_kernel + '_muon')
        track_kernel_kwargs = dict(dt=track_time_step)
        if track_kernel.startswith('pegleg'):
            if pegleg_kernel is not None:
                raise ValueError('can only have one pegleg kernel')
            pegleg_kernel = track_kernel_func
            pegleg_kernel_kwargs = track_kernel_kwargs
        else:
            generic_kernels.append(track_kernel_func)
            generic_kernels_kwargs.append(dict(dt=track_time_step))

    hypo_handler = DiscreteHypo(
        generic_kernels=generic_kernels,
        generic_kernels_kwargs=generic_kernels_kwargs,
        pegleg_kernel=pegleg_kernel,
        pegleg_kernel_kwargs=pegleg_kernel_kwargs,
        scaling_kernel=scaling_kernel,
        scaling_kernel_kwargs=scaling_kernel_kwargs,
    )

    return hypo_handler


def get_events(
    events_root,
    gcd_dir,
    start=None,
    stop=None,
    step=None,
    agg_start=None,
    agg_stop=None,
    agg_step=None,
    truth=None,
    photons=None,
    pulses=None,
    recos=None,
    triggers=None,
    angsens_model=None,
    hits=None,
):
    """Iterate through a Retro events directory, getting events in a
    the form of a nested OrderedDict, with leaf nodes numpy structured arrays.

    It is attempted to iterate through the source intelligently to minimize
    disk access and in a thread-safe manner.

    Parameters
    ----------
    events_root : string or iterable thereof
        Path(s) to Retro events directory(ies) (each such directory corresponds
        to a single i3 file and contains an "events.npy" file).

    start, stop, step : optional
        Arguments passed to ``slice`` for only retrieving select events from
        the source.

    agg_start : None or int with 0 <= agg_start <= agg_stop, optional

    agg_stop : None or int with 0 <= agg_start <= agg_stop, optional

    agg_step : None or int >= 1, optional

    truth : bool, optional
        Whether to extract Monte Carlo truth information from the file (only
        applicable to simulation).

    photons : sequence of strings, optional
        Photon series names to extract. If any are specified, you must specify
        `angsens_model` for proper weighting of the photons. Default is to
        not extract any photon series.

    pulses : sequences of strings, optional
        Pulse series names to extract. Default is to not extract any pulse
        series.

    recos : sequence of strings, optional
        Reconstruction names to extract. Default is to not extract any
        reconstructions.

    triggers : sequence of strings
        Trigger hierarchy names to extract. Required if pulses are specified to
        be used as hits (such that the time window can be computed). Default is
        to not extract any trigger hierarchies.

    angsens_model : string, optional
        Required if `photons` specifies any photon series to extract, as this angular
        sensitivity model is applied to the photons to arrive at expectation for
        detected photons (though without taking into account any further details of the
        DOM's ability to detect photons)

    hits : string or sequence thereof, optional
        Path to photon or pulse series to extract as ``event["hits"]`` field.
        Default is to not populate "hits".

    Yields
    ------
    event : nested OrderedDict
        Attribute "meta" (i.e., accessed via event.meta) is itself an
        OrderedDict added to `event` to contain additional information:
        "events_root", "num_events", and "event_idx".

    """
    if isinstance(events_root, string_types):
        events_roots = [expand(events_root)]
    else:
        if not isinstance(events_root, Iterable):
            raise TypeError("`events_root` must be string or iterable thereof")
        events_roots = []
        for events_root_ in events_root:
            if not isinstance(events_root_, string_types):
                raise TypeError(
                    "Each value in an iterable `events_root` must be a string"
                )
            events_roots.append(expand(events_root_))

    slice_kw = dict(start=start, stop=stop, step=step)

    if agg_start is None:
        agg_start = 0
    else:
        agg_start_ = int(agg_start)
        assert agg_start_ == agg_start
        agg_start = agg_start_

    if agg_step is None:
        agg_step = 1
    else:
        agg_step_ = int(agg_step)
        assert agg_step_ == agg_step
        agg_step = agg_step_

    assert agg_start >= 0
    assert agg_step >= 1

    if agg_stop is not None:
        assert agg_stop > agg_start >= 0
        agg_stop_ = int(agg_stop)
        assert agg_stop_ == agg_stop
        agg_stop = agg_stop_

    if truth is not None and not isinstance(truth, bool):
        raise TypeError("`truth` is invalid type: {}".format(type(truth)))

    if photons is not None and not isinstance(photons, (string_types, Iterable)):
        raise TypeError("`photons` is invalid type: {}".format(type(photons)))

    if pulses is not None and not isinstance(pulses, (string_types, Iterable)):
        raise TypeError("`pulses` is invalid type: {}".format(type(pulses)))

    if recos is not None and not isinstance(recos, (string_types, Iterable)):
        raise TypeError("`recos` is invalid type: {}".format(type(recos)))

    if triggers is not None and not isinstance(triggers, (string_types, Iterable)):
        raise TypeError("`triggers` is invalid type: {}".format(type(triggers)))

    if hits is not None and not isinstance(hits, string_types):
        raise TypeError("`hits` is invalid type: {}".format(type(hits)))

    agg_event_idx = -1
    for events_root in events_roots:
        for dirpath, dirs, files in walk(events_root, followlinks=True):
            dirs.sort(key=nsort_key_func)

            if "events.npy" not in files:
                continue

            file_iterator_tree = OrderedDict()

            num_events, event_indices, headers = iterate_file(
                join(dirpath, 'events.npy'), **slice_kw
            )

            meta = OrderedDict(
                [
                    ("events_root", dirpath),
                    ("num_events", num_events),
                    ("event_idx", None),
                    ("agg_event_idx", None),
                ]
            )

            event_indices_iter = iter(event_indices)
            file_iterator_tree['header'] = iter(headers)

            # -- Translate args with defaults / find dynamically-specified things -- #

            if truth is None:
                truth_ = isfile(join(dirpath, 'truth.npy'))
            else:
                truth_ = truth

            if photons is None:
                dpath = join(dirpath, 'photons')
                if isdir(dpath):
                    photons_ = [splitext(d)[0] for d in listdir(dpath)]
                else:
                    photons_ = False
            elif isinstance(photons, string_types):
                photons_ = [photons]
            else:
                photons_ = photons

            if pulses is None:
                dpath = join(dirpath, 'pulses')
                if isdir(dpath):
                    pulses_ = [splitext(d)[0] for d in listdir(dpath) if 'TimeRange' not in d]
                else:
                    pulses_ = False
            elif isinstance(pulses, string_types):
                pulses_ = [pulses]
            else:
                pulses_ = list(pulses)

            if recos is None:
                dpath = join(dirpath, 'recos')
                if isdir(dpath):
                    # TODO: make check a regex including colons, etc. so we don't
                    # accidentally exclude a valid reco that starts with "slc"
                    recos_ = []
                    for fname in listdir(dpath):
                        if fname[:3] in ("slc", "evt"):
                            continue
                        fbase = splitext(fname)[0]
                        if fbase.endswith(".llhp"):
                            continue
                        recos_.append(fbase)
                else:
                    recos_ = False
            elif isinstance(recos, string_types):
                recos_ = [recos]
            else:
                recos_ = list(recos)

            if triggers is None:
                dpath = join(dirpath, 'triggers')
                if isdir(dpath):
                    triggers_ = [splitext(d)[0] for d in listdir(dpath)]
                else:
                    triggers_ = False
            elif isinstance(triggers, string_types):
                triggers_ = [triggers]
            else:
                triggers_ = list(triggers)

            # Note that `hits_` must be defined after `pulses_` and `photons_`
            # since `hits_` is one of these
            if hits is None:
                if pulses_ is not None and len(pulses_) == 1:
                    hits_ = ['pulses', pulses_[0]]
                elif photons_ is not None and len(photons_) == 1:
                    hits_ = ['photons', photons_[0]]
            elif isinstance(hits, string_types):
                hits_ = hits.split('/')
            else:
                raise TypeError("{}".format(type(hits)))

            # -- Populate the file iterator tree -- #

            if truth_:
                num_truths, _, truths = iterate_file(
                    fpath=join(dirpath, 'truth.npy'), **slice_kw
                )
                assert num_truths == num_events
                file_iterator_tree['truth'] = iter(truths)

            if photons_:
                photons_ = sorted(photons_)
                file_iterator_tree['photons'] = iterators = OrderedDict()
                for photon_series in photons_:
                    num_phs, _, photon_serieses = iterate_file(
                        fpath=join(dirpath, 'photons', photon_series + '.pkl'), **slice_kw
                    )
                    assert num_phs == num_events
                    iterators[photon_series] = iter(photon_serieses)

            if pulses_:
                file_iterator_tree['pulses'] = iterators = OrderedDict()
                for pulse_series in sorted(pulses_):
                    num_ps, _, pulse_serieses = iterate_file(
                        fpath=join(dirpath, 'pulses', pulse_series + '.pkl'), **slice_kw
                    )
                    assert num_ps == num_events
                    iterators[pulse_series] = iter(pulse_serieses)
                    #(
                    #    quantize_min_q_filter(ps, qmin=0.4, quantum=0.05)
                    #    for ps in iter(pulse_serieses)
                    #)

                    num_tr, _, time_ranges = iterate_file(
                        fpath=join(
                            dirpath,
                            'pulses',
                            pulse_series + 'TimeRange' + '.npy'
                        ),
                        **slice_kw
                    )
                    assert num_tr == num_events
                    iterators[pulse_series + 'TimeRange'] = iter(time_ranges)

            if recos_:
                file_iterator_tree['recos'] = iterators = OrderedDict()
                for reco in sorted(recos_):
                    num_recoses, _, recoses = iterate_file(
                        fpath=join(dirpath, 'recos', reco + '.npy'), **slice_kw
                    )
                    assert num_recoses == num_events
                    iterators[reco] = iter(recoses)

            if triggers_:
                file_iterator_tree['triggers'] = iterators = OrderedDict()
                for trigger_hier in sorted(triggers_):
                    num_th, _, trigger_hiers = iterate_file(
                        fpath=join(dirpath, 'triggers', trigger_hier + '.pkl'), **slice_kw
                    )
                    assert num_th == num_events
                    iterators[trigger_hier] = iter(trigger_hiers)

            if hits_ is not None and hits_[0] == 'photons':
                angsens_model, _ = load_angsens_model(angsens_model)
            else:
                angsens_model = None

            while True:
                try:
                    event = extract_next_event(file_iterator_tree)
                except StopIteration:
                    break

                if hits_ is not None:
                    hits_array, hits_indexer, hits_summary = get_hits(
                        event=event, path=hits_, angsens_model=angsens_model
                    )
                    event['hits'] = hits_array
                    event['hits_indexer'] = hits_indexer
                    event['hits_summary'] = hits_summary

                agg_event_idx += 1

                event.meta = deepcopy(meta)
                event.meta["event_idx"] = next(event_indices_iter)
                event.meta["agg_event_idx"] = agg_event_idx

                if agg_stop is not None and agg_event_idx >= agg_stop:
                    return

                if agg_event_idx < agg_start or (agg_event_idx - agg_start) % agg_step != 0:
                    continue

                yield event

            while True:
                try:
                    event = extract_next_event(file_iterator_tree)
                except StopIteration:
                    break

                if hits_ is not None:
                    hits_array, hits_indexer, hits_summary = get_hits(
                        event=event, path=hits_, angsens_model=angsens_model
                    )
                    event['hits'] = hits_array
                    event['hits_indexer'] = hits_indexer
                    event['hits_summary'] = hits_summary

                agg_event_idx += 1

                event.meta = deepcopy(meta)
                event.meta["event_idx"] = next(event_indices_iter)
                event.meta["agg_event_idx"] = agg_event_idx

                if agg_stop is not None and agg_event_idx >= agg_stop:
                    return

                if agg_event_idx < agg_start or (agg_event_idx - agg_start) % agg_step != 0:
                    continue

                yield event

            for key in list(file_iterator_tree.keys()):
                del file_iterator_tree[key]
            del file_iterator_tree


def iterate_file(fpath, start=None, stop=None, step=None, mmap_mode=None):
    """Iterate through the elements in a pickle (.pkl) or numpy (.npy) file. If
    a pickle file, structure must be a sequence of objects, one object per
    event. If a numpy file, it must be a one-dimensional structured array where
    each "entry" in the array contains the information from one event.

    Parameters
    ----------
    fpath : string
    start, stop, step : optional
        Arguments passed to `slice` for extracting select events from the
        file.
    mmap_mode : None or string in {"r", "r+", "w+", "c"}
        Only applicable if `fpath` is a numpy .npy file; see help for
        `numpy.memmap` for more information on each mode. Note that memory
        mapping a file is useful for not consuming too much memory and being
        able to simultaneously write to the same reco output file from multiple
        processes (presumably each process working on different events) from
        multiple processes BUT too many open file handles can result in an
        exception. Default is `None` (file is not memory mapped, instead entire
        file is read into memory).

    Yields
    ------
    info : OrderedDict
        Information extracted from the file for each event.

    """
    slicer = slice(start, stop, step)
    _, ext = splitext(fpath)
    if ext == '.pkl':
        events = load_pickle(fpath)
    elif ext == '.npy':
        try:
            events = np.load(fpath, mmap_mode=mmap_mode)
        except:
            sys.stderr.write('failed to load "{}"\n'.format(fpath))
            raise
    else:
        raise ValueError(fpath)

    num_events_in_file = len(events)
    indices = range(num_events_in_file)[slicer]
    sliced_events = events[slicer]

    return num_events_in_file, indices, sliced_events


def get_path(event, path):
    """Extract an item at `path` from an event which is usable as a nested
    Python mapping (i.e., using `getitem` for each level in `path`).

    Parameters
    ----------
    event : possibly-nested mapping
    path : iterable of strings

    Returns
    -------
    node
        Whatever lives at the specified `path` (could be a scalar, array,
        another mapping, etc.)

    """
    if isinstance(path, string_types):
        path = [path]
    node = event
    for subpath in path:
        try:
            node = getitem(node, subpath)
        except:
            if hasattr(event, "meta"):
                sys.stderr.write("event.meta: {}\n".format(event.meta))
            sys.stderr.write(
                "node = {} type = {}, subpath = {} type = {}\n".format(
                    node, type(node), subpath, type(subpath)
                )
            )
            raise
    return node


def get_hits(event, path, angsens_model=None):
    """From an event, take either pulses or photons (optionally applying
    weights to the latter for angular sensitivity) and create the three
    structured numpy arrays necessary for Retro to process the information as
    "hits".

    Parameters
    ----------
    event

    path

    angsens_model : str, numpy.polynomial.Polynomial, or None
        If specified and photons are extracted, weights for the photons will be
        applied according to the angular sensitivity model specified.
        Otherwise, each photon will carry a weight of one.

    Returns
    -------
    hits : shape (n_hits,) array of dtype HIT_T
    hits_indexer : shape (n_hit_doms,) array of dtype SD_INDEXER_T
    hits_summary : shape (1,) array of dtype HITS_SUMMARY_T

    """
    photons = path[0] == 'photons'

    series = get_path(event, path)

    if photons:
        time_window_start = 0.
        time_window_stop = 0.
        if angsens_model is not None:
            if isinstance(angsens_model, string_types):
                angsens_poly, _ = load_angsens_model(angsens_model)
            elif isinstance(angsens_model, np.polynomial.Polynomial):
                angsens_poly = angsens_model
            else:
                raise TypeError('`angsens_model` is {} but must be either'
                                ' string or np.polynomial.Polynomial'
                                .format(type(angsens_model)))

    else:
        trigger_hierarchy = event['triggers']['I3TriggerHierarchy']
        time_window_start = np.inf
        time_window_stop = -np.inf
        for trigger in trigger_hierarchy:
            source = trigger['source']

            # Do not expand the in-ice window based on GLOBAL triggers (of
            # any TriggerTypeID)
            if source == TriggerSourceID.GLOBAL:
                continue

            tr_type = trigger['type']
            config_id = trigger['config_id']
            tr_time = trigger['time']

            # TODO: rework to _only_ use TriggerConfigID?
            # Below values can be extracted by running
            # $I3_SRC/trigger-sim/resources/scripts/print_trigger_configuration.py -g GCDFILE
            trigger_handled = False
            if tr_type == TriggerTypeID.SIMPLE_MULTIPLICITY:
                if source == TriggerSourceID.IN_ICE:
                    if config_id == TriggerConfigID.SMT8_IN_ICE:
                        trigger_handled = True
                        left_dt = -4e3
                        right_dt = 5e3 + 6e3
                    elif config_id == TriggerConfigID.SMT3_DeepCore:
                        trigger_handled = True
                        left_dt = -4e3
                        right_dt = 2.5e3 + 6e3
            elif tr_type == TriggerTypeID.VOLUME:
                if source == TriggerSourceID.IN_ICE:
                    trigger_handled = True
                    left_dt = -4e3
                    right_dt = 1e3 + 6e3
            elif tr_type == TriggerTypeID.STRING:
                if source == TriggerSourceID.IN_ICE:
                    trigger_handled = True
                    left_dt = -4e3
                    right_dt = 1.5e3 + 6e3

            if not trigger_handled:
                raise NotImplementedError(
                    'Trigger TypeID {}, SourceID {}, config_id {} not'
                    ' implemented'
                    .format(TriggerTypeID(tr_type).name, # pylint: disable=no-member
                            TriggerSourceID(source).name, # pylint: disable=no-member
                            config_id)
                )

            time_window_start = min(time_window_start, tr_time + left_dt)
            time_window_stop = max(time_window_stop, tr_time + right_dt)

    hits = []
    hits_indexer = []
    offset = 0

    for (string, dom, pmt), p in series:
        sd_idx = const.get_sd_idx(string=string, om=dom, pmt=pmt)
        num = len(p)
        sd_hits = np.empty(shape=num, dtype=HIT_T)
        sd_hits['time'] = p['time']
        if not photons:
            sd_hits['charge'] = p['charge']
        elif angsens_model:
            sd_hits['charge'] = angsens_poly(p['coszen'])
        else:
            sd_hits['charge'] = 1

        hits.append(sd_hits)
        hits_indexer.append((sd_idx, offset, num))
        offset += num

    hits = np.concatenate(hits) #, dtype=HIT_T)
    if hits.dtype != HIT_T:
        raise TypeError('got dtype {}'.format(hits.dtype))

    hits_indexer = np.array(hits_indexer, dtype=SD_INDEXER_T)

    hit_times = hits['time']
    hit_charges = hits['charge']
    total_charge = np.sum(hit_charges)

    earliest_hit_time = hit_times.min()
    latest_hit_time = hit_times.max()
    average_hit_time = np.sum(hit_times * hit_charges) / total_charge

    num_hits = len(hits)
    num_doms_hit = len(hits_indexer)

    hits_summary = np.array(
        (
            earliest_hit_time,
            latest_hit_time,
            average_hit_time,
            total_charge,
            num_hits,
            num_doms_hit,
            time_window_start,
            time_window_stop,
        ),
        dtype=HITS_SUMMARY_T
    )

    return hits, hits_indexer, hits_summary


def extract_next_event(file_iterator_tree, event=None):
    """Recursively extract events from file iterators, where the structure of
    the iterator tree is reflected in the produced event.

    Parameters
    ----------
    file_iterator_tree : nested OrderedDict, leaf nodes are file iterators
    event : None or OrderedDict

    Returns
    -------
    event : OrderedDict

    """
    if event is None:
        event = OrderedDict()

    num_iterators = 0
    num_stopped_iterators = 0
    for key, val in file_iterator_tree.items():
        if isinstance(val, Mapping):
            event[key] = extract_next_event(val)
        else:
            num_iterators += 1
            try:
                event[key] = next(val)
            except StopIteration:
                num_stopped_iterators += 1

    if num_stopped_iterators > 0:
        if num_stopped_iterators != num_iterators:
            raise ValueError(
                "num_stopped_iterators = {} but num_iterators = {}".format(
                    num_stopped_iterators, num_iterators
                )
            )
        raise StopIteration

    return event


def parse_args(
    dom_tables=False,
    tdi_tables=False,
    hypo=False,
    events=False,
    description=None,
    parser=None,
):
    """Parse command line arguments.

    If `parser` is supplied, args are added to that; otherwise, a new parser is
    generated. Defaults to _not_ include any of the command-line arguments.

    Parameters
    ----------
    dom_tables : bool, optional
        Whether to include args for instantiating and loading single-DOM
        tables. Default is False.

    tdi_tables : bool, optional
        Whether to include args for instantiating and loading TDI tables.
        Default is False.

    hypo : bool
        Whether to include args for instantiating a DiscreteHypo and its hypo
        kernels. Default is False.

    events : bool
        Whether to include args for loading events. Default is False.

    description : string, optional

    parser : argparse.ArgumentParser, optional
        An existing parser onto which these arguments will be added.

    Returns
    -------
    split_kwargs : OrderedDict
        Optionally contains keys "dom_tables_kw", "hypo_kw", "events_kw",
        and/or "other_kw", where each is included only if there are keyword
        arguments for that grouping; values are dicts containing the keyword
        arguments and values as specified by the user on the command line (with
        some translation applied to convert arguments into a form usable
        directly by functions in Retro). "other_kw" only shows up if there are
        values that don't fall into one of the other categories.

    """
    if parser is None:
        parser = ArgumentParser(description=description)

    if dom_tables or events:
        parser.add_argument(
            '--angsens-model',
            choices='nominal  h1-100cm  h2-50cm  h3-30cm 9'.split(),
            help='''Angular sensitivity model; only necessary if loading tables
            or photon hits.'''
        )

    if dom_tables:
        group = parser.add_argument_group(
            title='Single-DOM tables arguments',
        )

        group.add_argument(
            '--dom-tables-kind',
            required=True,
            choices=TABLE_KINDS,
            help='''Kind of single-DOM table to use.'''
        )
        group.add_argument(
            '--dom-tables-fname-proto',
            required=True,
            help='''Must have one of the brace-enclosed fields "{string}" or
            "{subdet}", and must have one of "{dom}" or "{depth_idx}". E.g.:
            "my_tables_{subdet}_{depth_idx}"'''
        )
        group.add_argument(
            '--use-doms', required=True,
            choices='all dc dc_subdust'.split()
        )

        group.add_argument(
            '--gcd',
            required=True,
            help='''IceCube GCD file; can either specify an i3 file, or the
            extracted pkl file used in Retro.'''
        )
        group.add_argument(
            '--norm-version',
            required=False, default='binvol2.5',
            choices=NORM_VERSIONS,
            help='''Norm version.'''
        )
        group.add_argument(
            '--num-phi-samples', type=int, default=None,
        )
        group.add_argument(
            '--ckv-sigma-deg', type=float, default=None,
        )
        group.add_argument(
            '--template-library', default=None,
        )
        group.add_argument(
            '--no-noise', action='store_true',
            help='''Set noise rates to 0 in the GCD (e.g. for processing
            raw photons)'''
        )
        group.add_argument(
            '--no-t-indep', action='store_true',
            help='''Do NOT load t-indep tables (time-independent expectations
            would have to be handled by specifying a TDI table'''
        )
        group.add_argument(
            '--force-no-mmap', action='store_true',
            help='''Specify to NOT memory map the tables. If not specified, a
            sensible default is chosen for the type of tables being used.'''
        )

    if tdi_tables:
        group = parser.add_argument_group(
            title='TDI tables arguments',
        )
        group.add_argument(
            '--tdi',
            action='append',
            help='''Path to TDI table's `ckv_tdi_table.npy` file or path
            to directory containing that file; repeat --tdi to specify multiple
            TDI tables (making sure more finely-binned tables are specified
            BEFORE more coarsely-binned tables)'''
        )

    if hypo:
        group = parser.add_argument_group(
            title='Hypothesis handler and kernel parameters',
        )

        group.add_argument(
            '--cascade-kernel',
            required=True,
            choices=dck.CASCADE_KINDS,
        )
        group.add_argument(
            '--track-kernel',
            required=True,
            choices=dmk.MUON_KINDS,
        )
        group.add_argument(
            '--track-time-step', type=float,
            required=True,
        )

    if events:
        group = parser.add_argument_group(
            title='Events and fields within the events to extract',
        )

        group.add_argument(
            '--events-root', type=str,
            required=True,
            nargs='+',
            help='''Retro events directory(ies) (i.e., each must contain an
            "events.npy" file)''',
        )
        group.add_argument(
            '--start', type=int, default=None,
            help='''Process events defined by slicing *each file* by
            [start:stop:step]. Default is to start at 0.'''
        )
        group.add_argument(
            '--stop', type=int, default=None,
            help='''Process events defined by slicing *each file*
            [start:stop:step]. Default is None, i.e., take events until they
            run out.'''
        )
        group.add_argument(
            '--step', type=int, default=None,
            help='''Process events defined by slicing *each file* by
            [start:stop:step]. Default is None, i.e., take every event from
            `start` through (including)  `stop - 1`.'''
        )
        group.add_argument(
            '--agg-start', type=int, default=None,
            help='''Apply [agg-start:agg-stop:agg-step] slicing the aggregate
            of events returned after slicing each individual events file with
            [start:stop:step].'''
        )
        group.add_argument(
            '--agg-stop', type=int, default=None,
            help='''Apply [agg-start:agg-stop:agg-step] slicing the aggregate
            of events returned after slicing each individual events file with
            [start:stop:step].'''
        )
        group.add_argument(
            '--agg-step', type=int, default=None,
            help='''Apply [agg-start:agg-stop:agg-step] slicing the aggregate
            of events returned after slicing each individual events file with
            [start:stop:step].'''
        )
        group.add_argument(
            '--photons', nargs='+',
            help='''Photon series name(s).''',
        )
        group.add_argument(
            '--pulses', nargs='+',
            help='''Name of pulse series to extract. Repeat --pulses to specify
            multiple pulse series.''',
        )
        group.add_argument(
            '--truth', action='store_true',
            help='''Require extraction of Monte Carlo truth information (if not
            specified, truth will still be loaded if "truth.npy" file is
            found).''',
        )
        group.add_argument(
            '--recos', nargs='+',
            help='''Name of reconstruction to extract. Repeat --reco to extract
            multiple reconstructions.''',
        )
        group.add_argument(
            '--triggers', nargs='+',
            help='''Name(s) of reconstruction(s) to extract.''',
        )
        group.add_argument(
            '--hits', default=None,
            help='''Path to item to use as "hits", e.g.
            "pulses/OfflinePulses".''',
        )

    args = parser.parse_args()
    kwargs = vars(args)

    dom_tables_kw = {}
    tdi_tables_kw = {}
    hypo_kw = {}
    events_kw = {}
    other_kw = {}
    if dom_tables:
        code = setup_dom_tables.__code__
        dom_tables_kw = {k: None for k in code.co_varnames[:code.co_argcount]}
    if tdi_tables:
        code = setup_tdi_tables.__code__
        tdi_tables_kw = {k: None for k in code.co_varnames[:code.co_argcount]}
    if hypo:
        code = setup_discrete_hypo.__code__
        hypo_kw = {k: None for k in code.co_varnames[:code.co_argcount]}
    if events:
        code = get_events.__code__
        events_kw = {k: None for k in code.co_varnames[:code.co_argcount]}

    if dom_tables:
        use_doms = kwargs.pop('use_doms').strip().lower()
        if use_doms == 'all':
            use_sd_indices = const.ALL_STRS_DOMS
        elif use_doms == 'dc':
            use_sd_indices = const.DC_ALL_STRS_DOMS
        elif use_doms == 'dc_subdust':
            use_sd_indices = const.DC_ALL_SUBDUST_STRS_DOMS
        else:
            raise ValueError(use_doms)
        print('number of doms = {}'.format(len(use_sd_indices)))
        kwargs['use_sd_indices'] = use_sd_indices
        kwargs['compute_t_indep_exp'] = not kwargs.pop('no_t_indep')

    for key, val in kwargs.items():
        taken = False
        for kw in [dom_tables_kw, tdi_tables_kw, hypo_kw, events_kw]:
            if key not in kw:
                continue
            kw[key] = val
            taken = True
        if not taken:
            other_kw[key] = val

    split_kwargs = OrderedDict()
    if dom_tables:
        split_kwargs['dom_tables_kw'] = dom_tables_kw
    if tdi_tables:
        split_kwargs['tdi_tables_kw'] = tdi_tables_kw
    if hypo:
        split_kwargs['hypo_kw'] = hypo_kw
    if events:
        split_kwargs['events_kw'] = events_kw
    if other_kw:
        split_kwargs['other_kw'] = other_kw

    return split_kwargs
