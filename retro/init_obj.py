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
from collections import Mapping, OrderedDict
from operator import getitem
from os import listdir
from os.path import abspath, dirname, isdir, isfile, join, splitext
import re
import sys
import time

import numpy as np

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
from retro.utils.misc import expand


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
    step_length=1.0,
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
    step_length : float, optional
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
                    sd_idx = const.get_sd_idx(string=string, dom=dom)
                    if sd_idx not in use_sd_indices:
                        continue
                    shared_table_sd_indices.append(sd_idx)

                if not shared_table_sd_indices:
                    continue

                dom_tables.load_table(
                    fpath=fpath,
                    sd_indices=shared_table_sd_indices,
                    step_length=step_length,
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
                step_length=step_length,
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
    events_base,
    start=0,
    stop=None,
    step=None,
    truth=True,
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
    events_base : string
        Path to a Retro events directory (i.e., directory that corresponds to a
        single i3 file).

    start, stop, step : optional
        Arguments passed to ``slice`` for only retrieving select events from
        the source.

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

    angsens_model : string
        Required if `photons` specifies any photon series to extract, as this angular
        sensitivity model is applied to the photons to arrive at expectation for
        detected photons (though without taking into account any further details of the
        DOM's ability to detect photons)

    hits : string or sequence thereof, optional
        Path to photon or pulse series to extract as ``event["hits"]`` field.
        Default is to not populate "hits".

    Yields
    ------
    event_idx : int
        Index of the event relative to its position in the file. E.g. if you
        use ``start=5``, the first `event_idx` yielded will be 5.

    event : nested OrderedDict

    """
    if not (isdir(events_base) and isfile(join(events_base, 'events.npy'))):
        raise ValueError(
            '`events_base` does not appear to be a Retro event directory: "{}"'
            ' is either not a directory or does not contain an "events.npy"'
            ' file.'.format(events_base)
        )

    slice_kw = dict(start=start, stop=stop, step=step)
    file_iterator_tree = OrderedDict()
    file_iterator_tree['header'] = iterate_file(
        join(events_base, 'events.npy'), **slice_kw
    )

    if truth is None:
        truth = isfile(join(events_base, 'truth.npy'))

    if photons is None:
        dpath = join(events_base, 'photons')
        if isdir(dpath):
            photons = [splitext(d)[0] for d in listdir(dpath)]
        else:
            photons = False
    elif isinstance(photons, str):
        photons = [photons]

    if pulses is None:
        dpath = join(events_base, 'pulses')
        if isdir(dpath):
            pulses = [splitext(d)[0] for d in listdir(dpath) if 'TimeRange' not in d]
        else:
            pulses = False
    elif isinstance(pulses, str):
        pulses = [pulses]

    if recos is None:
        dpath = join(events_base, 'recos')
        if isdir(dpath):
            # TODO: make check a regex including colons, etc. so we don't
            # accidentally exclude a valid reco that starts with "slc"
            recos = []
            for fname in listdir(dpath):
                if fname[:3] in ("slc", "evt"):
                    continue
                fbase = splitext(fname)[0]
                if fbase.endswith(".llhp"):
                    continue
                recos.append(fbase)
        else:
            recos = False
    elif isinstance(recos, str):
        recos = [recos]

    if triggers is None:
        dpath = join(events_base, 'triggers')
        if isdir(dpath):
            triggers = [splitext(d)[0] for d in listdir(dpath)]
        else:
            triggers = False
    elif isinstance(triggers, str):
        triggers = [triggers]

    if hits is None:
        if pulses and len(pulses) == 1:
            hits = ['pulses', pulses[0]]
        elif photons and len(photons) == 1:
            hits = ['photons', photons[0]]
        else:
            hits = False
    elif isinstance(hits, str):
        hits = hits.split('/')

    if truth:
        file_iterator_tree['truth'] = iterate_file(
            fpath=join(events_base, 'truth.npy'), **slice_kw
        )
    if photons:
        photons = sorted(photons)
        file_iterator_tree['photons'] = iterators = OrderedDict()
        for photon_series in photons:
            iterators[photon_series] = iterate_file(
                fpath=join(events_base, 'photons', photon_series + '.pkl'), **slice_kw
            )
    if pulses:
        file_iterator_tree['pulses'] = iterators = OrderedDict()
        for pulse_series in sorted(pulses):
            iterators[pulse_series] = iterate_file(
                fpath=join(events_base, 'pulses', pulse_series + '.pkl'), **slice_kw
            )
            iterators[pulse_series + 'TimeRange'] = iterate_file(
                fpath=join(events_base,
                           'pulses',
                           pulse_series + 'TimeRange' + '.npy'),
                **slice_kw
            )
    if recos:
        file_iterator_tree['recos'] = iterators = OrderedDict()
        for reco in sorted(recos):
            iterators[reco] = iterate_file(
                fpath=join(events_base, 'recos', reco + '.npy'), **slice_kw
            )
    if triggers:
        file_iterator_tree['triggers'] = iterators = OrderedDict()
        for trigger_hier in sorted(triggers):
            iterators[trigger_hier] = iterate_file(
                fpath=join(events_base, 'triggers', trigger_hier + '.pkl'), **slice_kw
            )

    if hits and hits[0] == 'photons':
        angsens_model, _ = load_angsens_model(angsens_model)
    else:
        angsens_model = None

    start = 0 if start is None else start
    step = 1 if step is None else step

    event_idx = start
    while True:
        try:
            event = extract_next_event(file_iterator_tree)
        except StopIteration:
            break

        if hits:
            hits_, hits_indexer, hits_summary = get_hits(
                event=event, path=hits, angsens_model=angsens_model
            )
            event['hits'] = hits_
            event['hits_indexer'] = hits_indexer
            event['hits_summary'] = hits_summary

        yield event_idx, event

        event_idx += step


def iterate_file(fpath, start=0, stop=None, step=None):
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
            # Note that memory mapping the file is useful for not consuming too
            # much memory, and also might be essential in the future if we
            # write recos directly to the {reco}.npy file
            events = np.load(fpath, mmap_mode='r')
        except:
            print(fpath)
            raise
    else:
        raise ValueError(fpath)

    for event in events[slicer]:
        yield event


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
    if isinstance(path, str):
        path = [path]
    node = event
    for subpath in path:
        node = getitem(node, subpath)
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
        time_window_start = 0
        time_window_stop = 0
        if angsens_model is not None:
            if isinstance(angsens_model, str):
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
        sd_idx = const.get_sd_idx(string=string, dom=dom, pmt=pmt)
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

    total_num_hits = len(hits)
    total_num_doms_hit = len(hits_indexer)

    hits_summary = np.array(
        (
            earliest_hit_time,
            latest_hit_time,
            average_hit_time,
            total_charge,
            total_num_hits,
            total_num_doms_hit,
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
    for key, val in file_iterator_tree.items():
        if isinstance(val, Mapping):
            event[key] = extract_next_event(val)
        else:
            event[key] = next(val)
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
            '--angsens-model', required=True,
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
            '--step-length', type=float, default=1.0,
            help='''Step length used in the CLSim table generator.'''
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
            '--events-base', type=str,
            required=True,
            help='''i3 file or a directory containing Retro .npy/.pkl events
            files'''
        )
        group.add_argument(
            '--start', type=int, default=0,
            help='''Process events defined by slice [start:stop:step]. Default is 0.'''
        )
        group.add_argument(
            '--stop', type=int, default=None,
            help='''Process events defined by slice [start:stop:step]. Default is None,
            i.e., take events until they run out.'''
        )
        group.add_argument(
            '--step', type=int, default=None,
            help='''Process events defined by slice [start:stop:step]. Default is
            None, i.e., take every event from `start` through `stop - 1`.'''
        )
        group.add_argument(
            '--photons', action='append',
            help='''Photon series name. Repeat --photons to specify multiple
            photon series.'''
        )
        group.add_argument(
            '--pulses', action='append',
            help='''Name of pulse series to extract. Repeat --pulses to specify
            multiple pulse series.'''
        )
        group.add_argument(
            '--truth', action='store_true',
            help='''Whether to extract Monte Carlo truth information.'''
        )
        group.add_argument(
            '--recos', action='append',
            help='''Name of reconstruction to extract. Repeat --reco to extract
            multiple reconstructions.'''
        )
        group.add_argument(
            '--triggers', action='append',
            help='''Name of reconstruction to extract. Repeat --reco to extract
            multiple reconstructions.'''
        )
        group.add_argument(
            '--hits', default=None,
            help='''Path to item to use as "hits", e.g.
            "pulses/OfflinePulses".'''
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
