#!/usr/bin/env python
# pylint: disable=wrong-import-position

"""
Modify, analyze, and make distribution plots from data & MC events_array.npy,
doms_array.npy, and pulses_array.npy files (see
`data_mc_agreement__extract_pulses.py`).
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    "TRUE_TIME_INFO_T",
    "JIT_KW",
    "DATA_DIR_INFOS",
    "MC_DIR_INFOS",
    "MC_SET_SPECS",
    "NUM_BINS",
    "HIST_EDGES",
    "DOMS_PER_EVENT_DIG",
    "CHARGE_PER_EVENT_DIG",
    "PULSES_PER_EVENT_DIG",
    "CHARGE_PER_DOM_DIG",
    "PULSES_PER_DOM_DIG",
    "CHARGE_PER_PULSE_DIG",
    "TIME_DIFFS_WITHIN_EVENT_DIG",
    "TIME_DIFFS_WITHIN_DOM_DIG",
    "LABELS",
    "UNITS",
    "DISPLIM_Q",
    "DENOTE_Q",
    "LOG_X_PARAMS",
    "LEG_LOC",
    "REF_LOG_TICKLABELS",
    "REF_LOG_TICKS",
    "get_dom_region",
    "quantize",
    "generate_filter_func",
    "load_and_filter",
    "get_data_weight_func",
    "get_mc_weight_func",
    "binit",
    "create_histos",
    "get_true_time_relative_info",
    "integer_if_integral",
    "get_histo_fname_prefix",
    "load_histos",
    "plot",
    "parse_args",
    "main",
]

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
import numbers
from os.path import abspath, basename, dirname, isdir, isfile, join
import pickle
import sys
import time
import warnings

import numba
import numpy as np

RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
if __name__ == "__main__" and __package__ is None:
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import load_pickle
from retro.const import DC_STRS
from retro.utils.misc import expand, mkdir
from retro.utils.geom import generate_digitizer


TRUE_TIME_INFO_T = np.dtype(
    [
        ("true_energy", np.float32),
        ("weight", np.float32),
        ("q_fract", np.float32),
        ("t_fract", np.float32),
        ("dt", np.float32),
    ]
)

JIT_KW = dict(
    nopython=True, nogil=True, parallel=False, error_model="numpy", fastmath=True
)

DATA_DIR_INFOS = OrderedDict(
    [
        ((12, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.12")),
        ((13, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.13")),
        ((14, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.14")),
        ((15, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.15")),
        ((16, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.16")),
        ((17, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.17")),
        ((18, (1, 3)), dict(path="ana/LE/oscNext/pass2/data/level5_v01.03/IC86.18")),
    ]
)

MC_DIR_INFOS = OrderedDict(
    [
        # GENIE nue
        (
            (120000, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/120000", num_files=601),
        ),
        (
            (120001, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/120001", num_files=602),
        ),
        (
            (120002, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/120002", num_files=602),
        ),
        (
            (120003, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/120003", num_files=602),
        ),
        (
            (120004, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/120004", num_files=602),
        ),
        # GENIE numu
        (
            (140000, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/genie/level5_v01.03/140000", num_files=1494
            ),
        ),
        (
            (140001, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/genie/level5_v01.03/140001", num_files=1520
            ),
        ),
        (
            (140002, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/genie/level5_v01.03/140002", num_files=1520
            ),
        ),
        (
            (140003, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/genie/level5_v01.03/140003", num_files=1520
            ),
        ),
        (
            (140004, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/genie/level5_v01.03/140004", num_files=1520
            ),
        ),
        # GENIE nutau
        (
            (160000, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/160000", num_files=335),
        ),
        (
            (160001, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/160001", num_files=350),
        ),
        (
            (160002, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/160002", num_files=345),
        ),
        (
            (160003, (1, 3)),
            dict(path="ana/LE/oscNext/pass2/genie/level5_v01.03/160003", num_files=345),
        ),
        # muongun sets
        (
            (139011, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/muongun/level5_v01.03/139011", num_files=2996
            ),
        ),
        # Noise sets
        (
            (888003, (1, 3)),
            dict(
                path="ana/LE/oscNext/pass2/noise/level5_v01.03/888003/", num_files=5000
            ),
        ),
    ]
)

MC_SET_SPECS = OrderedDict(
    [
        (
            "test",
            dict(nue=(120000, (1, 3)), mu=(139011, (1, 3)), noise=(888003, (1, 3))),
        ),
        (
            "baseline",
            dict(
                nue=(120000, (1, 3)),
                numu=(140000, (1, 3)),
                nutau=(160000, (1, 3)),
                mu=(139011, (1, 3)),
                noise=(888003, (1, 3)),
            ),
        ),
        (
            "dom_eff_0.9",
            dict(
                nue=(120001, (1, 3)),
                numu=(140001, (1, 3)),
                nutau=(160001, (1, 3)),
                mu=(139011, (1, 3)),
                noise=(888003, (1, 3)),
            ),
        ),
        (
            "dom_eff_0.95",
            dict(
                nue=(120002, (1, 3)),
                numu=(140002, (1, 3)),
                nutau=(160002, (1, 3)),
                mu=(139011, (1, 3)),
                noise=(888003, (1, 3)),
            ),
        ),
        (
            "dom_eff_1.05",
            dict(
                nue=(120003, (1, 3)),
                numu=(140003, (1, 3)),
                nutau=(160003, (1, 3)),
                mu=(139011, (1, 3)),
                noise=(888003, (1, 3)),
            ),
        ),
        (
            "dom_eff_1.1",
            dict(
                nue=(120004, (1, 3)),
                numu=(140004, (1, 3)),
                nutau=(160003, (1, 3)),
                mu=(139011, (1, 3)),
                noise=(888003, (1, 3)),
            ),
        ),
    ]
)


NUM_BINS = 80

_HIST_EDGES = OrderedDict(
    [
        ("doms_per_event", np.logspace(np.log10(1), np.log10(320), NUM_BINS + 1)),
        ("charge_per_event", np.logspace(np.log10(1), np.log10(3200), NUM_BINS + 1)),
        ("pulses_per_event", np.logspace(np.log10(1), np.log10(1200), NUM_BINS + 1)),
        ("charge_per_dom", np.logspace(np.log10(0.05), np.log10(3200), NUM_BINS + 1)),
        ("pulses_per_dom", np.logspace(np.log10(1), np.log10(225), NUM_BINS + 1)),
        ("charge_per_pulse", np.logspace(np.log10(1e-2), np.log10(2200), NUM_BINS + 1)),
        ("time_diffs_within_event", np.linspace(0, 13000, NUM_BINS + 1)),
        ("time_diffs_within_dom", np.linspace(0, 13000, NUM_BINS + 1)),
    ]
)
HIST_EDGES = numba.typed.Dict.empty(
    key_type=numba.types.unicode_type, value_type=numba.types.float64[:]
)
for k, v in _HIST_EDGES.items():
    HIST_EDGES[k] = v

_DIG_KW = dict(clip=False, handle_under_overflow=False)
DOMS_PER_EVENT_DIG = generate_digitizer(HIST_EDGES["doms_per_event"], **_DIG_KW)
CHARGE_PER_EVENT_DIG = generate_digitizer(HIST_EDGES["charge_per_event"], **_DIG_KW)
PULSES_PER_EVENT_DIG = generate_digitizer(HIST_EDGES["pulses_per_event"], **_DIG_KW)

CHARGE_PER_DOM_DIG = generate_digitizer(HIST_EDGES["charge_per_dom"], **_DIG_KW)
PULSES_PER_DOM_DIG = generate_digitizer(HIST_EDGES["pulses_per_dom"], **_DIG_KW)

CHARGE_PER_PULSE_DIG = generate_digitizer(HIST_EDGES["charge_per_pulse"], **_DIG_KW)

TIME_DIFFS_WITHIN_EVENT_DIG = generate_digitizer(
    HIST_EDGES["time_diffs_within_event"], **_DIG_KW
)
TIME_DIFFS_WITHIN_DOM_DIG = generate_digitizer(
    HIST_EDGES["time_diffs_within_dom"], **_DIG_KW
)

LABELS = dict(
    nue=r"GENIE $\nu_e$",
    numu=r"GENIE $\nu_\mu$",
    nutau=r"GENIE $\nu_\tau$",
    mu=r"MuonGun",
    coszen=r"$\cos\theta_{\rm zen}$",
    zenith=r"$\theta_{\rm zen}$",
    azimuth=r"$\phi_{\rm az}$",
    angle=r"$\Delta\Psi$",
    doms_per_event="DOMs per event",
    time_diffs_within_event=r"$t_{\rm pulse} - t^{0, \, {\rm event}}_{\rm pulse}$",
    time_diffs_within_dom=r"$t_{\rm pulse} - t^{0, \, {\rm DOM}}_{\rm pulse}$",
    charge_per_event="Charge per event",
    charge_per_dom="Charge per DOM",
    charge_per_pulse="Charge per pulse",
    pulses_per_dom="Pulses per DOM",
    pulses_per_event="Pulses per event",
)
for _year in range(10, 21):
    LABELS[str(_year)] = "IC86.{}".format(_year)

UNITS = dict(
    x="m",
    y="m",
    z="m",
    time="ns",
    azimuth="rad",
    zenith="rad",
    energy="GeV",
    angle="deg",
    time_diffs_within_event="ns",
    time_diffs_within_dom="ns",
    charge_per_event="PE",
    charge_per_dom="PE",
    charge_per_pulse="PE",
)

DISPLIM_Q = [0, 100]
DENOTE_Q = [25, 75]

LOG_X_PARAMS = [
    "energy",
    "doms_per_event",
    "charge_per_event",
    "pulses_per_event",
    "charge_per_dom",
    "pulses_per_dom",
    "charge_per_pulse",
]

MIN_ENERGY = 0.1  # GeV

LEG_LOC = dict(
    x="upper left",
    y="upper left",
    z="upper right",
    time="upper right",
    azimuth="upper left",
    zenith="upper right",
    coszen="upper left",
    energy="upper left",
    hits_per_dom="upper right",
)

REF_LOG_TICKLABELS = [
    # "1/10",
    # "1/9",
    # "1/8",
    # "1/7",
    # "1/6",
    # "1/5",
    "1/4",
    # "1/3",
    "1/2",
    # "1/1.5",
    # "1/1.4",
    # "1/1.3",
    # "1/1.2",
    # "1/1.1",
    "1",
    # "1.1",
    # "1.2",
    # "1.3",
    # "1.4",
    # "1.5",
    "2",
    # "3",
    "4",
    # "5",
    # "6",
    # "7",
    # "8",
    # "9",
    # "10",
]

REF_LOG_TICKS = np.array(
    [eval(tl) for tl in REF_LOG_TICKLABELS]  # pylint: disable=eval-used
)


GEO = load_pickle(
    join(
        RETRO_DIR,
        "data",
        "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.pkl",
    )
)["geo"]


@numba.jit(cache=True, **JIT_KW)
def get_dom_region(dom):
    """
    Parameters
    ----------
    dom

    Returns
    -------
    is_dc : bool
    z_region : int in [0, 2]
        0 = below dust (z < -155 m)
        1 = within "dust layer" (-155 m <= z < 85 m)
        2 = above dust (z > 85 m)

    """
    dust_z0 = -155
    dust_z1 = 85

    is_dc = dom["string"] >= DC_STRS[0]

    z = GEO[dom["string"] - 1, dom["om"] - 1, 2]
    if z < dust_z0:
        z_region = 0
    elif z < dust_z1:
        z_region = 1
    else:
        z_region = 2

    return (is_dc, z_region)


@numba.jit(cache=True, **JIT_KW)
def quantize(x, qntm):
    """
    Parameters
    ----------
    x : scalar >= 0
    qntm : scalar > 0

    Returns
    -------
    q : scalar >= 0

    """
    return (np.float64(x) // qntm) * qntm + qntm / 2


def generate_filter_func(
    fixed_pulse_q,
    qntm,
    min_pulse_q,
    min_evt_p,
    min_evt_dt,
    max_evt_dt,
    # min_evt_t_fract,
    # max_evt_t_fract,
    # t_fract_window,
    # min_evt_q_t_qtl,
    # max_evt_q_t_qtl,
    min_dom_p,
    # min_dom_dt,
    max_dom_dt,
    integ_t,
    i3,
    dc,
    z_regions,
):
    """
    Parameters
    ----------
    fixed_pulse_q : float
    qntm : float
    min_pulse_q : float
    min_evt_p : int >= 1
    min_evt_dt : float >= 0
    max_evt_dt : float >= 0
    #min_evt_t_fract
    #max_evt_t_fract
    #t_fract_window
    #min_evt_q_t_qtl : 0 <= float < 1
    #max_evt_q_t_qtl : 0 < float <= 1
    min_dom_p : int >= 1
    #min_dom_dt : float >= 0
    max_dom_dt : float >= 0
        Time in ns. If set to 0, no max-delta-time limit is set for accepting
        pulses
    integ_t : float >= 0
        Integration time in ns. If 0, no integration is performed.
    i3 : bool
    dc : bool
    z_regions : int or tuple of int

    Returns
    -------
    filter_arrays : callable

    """
    assert qntm >= 0
    assert min_pulse_q >= 0
    assert min_evt_p >= 1
    assert 0 <= min_evt_dt <= max_evt_dt
    # assert 0 <= min_evt_t_fract <= max_evt_t_fract <= 1
    # assert t_fract_window >= 0
    # assert 0 <= min_evt_q_t_qtl < 1
    # assert 0 < max_evt_q_t_qtl <= 1
    assert min_dom_p >= 1
    # assert 0 <= min_dom_dt <= max_dom_dt
    assert max_dom_dt >= 0
    if isinstance(z_regions, int):
        z_regions = (z_regions,)
    z_regions = tuple(z_regions)

    @numba.jit(cache=False, **JIT_KW)
    def filter_arrays(events, doms, pulses):
        """Get all pulses either inside or outside deepcore

        Parameters
        ----------
        events
        doms
        pulses

        Returns
        -------
        new_events
        new_doms
        new_pulses

        """
        new_events = np.empty_like(events)
        new_doms = np.empty_like(doms)
        new_pulses = np.empty_like(pulses)

        total_num_events = 0
        total_num_hit_doms = 0
        total_num_pulses = 0

        for event in events:
            dom_idx0 = total_num_hit_doms

            event_num_hit_doms = 0
            event_num_pulses = 0
            event_charge = 0.0

            event_pulse_t0 = np.inf
            if min_evt_dt > 0 or max_evt_dt > 0:
                for dom in doms[
                    event["dom_idx0"] : event["dom_idx0"] + event["num_hit_doms"]
                ]:
                    event_pulse_t0 = min(
                        event_pulse_t0, pulses[dom["pulses_idx0"]]["time"]
                    )

            # if min_evt_t_fract > 0 or max_evt_t_fract > 0:
            #     event_first_dom = doms[int(event["dom_idx0"])]
            #     event_last_dom = doms[int(event["dom_idx0"] + event["num_hit_doms"] - 1)]
            #     event_first_pulse = pulses[int(event_first_dom["pulses_idx0"])]
            #     event_last_pulse = pulses[int(event_last_dom["pulses_idx0"] + event_last_dom["num_pulses"] - 1)]

            # lower_q_t_qtl_time = -np.inf
            # upper_q_t_qtl_time = -np.inf

            for dom in doms[
                event["dom_idx0"] : event["dom_idx0"] + event["num_hit_doms"]
            ]:
                is_dc, z_region = get_dom_region(dom)

                is_ic = not is_dc

                if not ((dc and is_dc or i3 and is_ic) and z_region in z_regions):
                    continue

                # `total_num_pulses` can increment in loop over pulses; need to know
                # value before loop, this is to be populated to `new_doms`
                # array (if the DOM is to be recorded)
                new_doms_pulses_idx0 = total_num_pulses

                dom_pulse_t0 = pulses[dom["pulses_idx0"]]["time"]

                dom_num_pulses = 0
                dom_charge = 0.0

                num_integ_pulses = 0
                last_integ_pulse_t0 = -np.inf
                integ_pulse_total_t = 0.0
                integ_pulse_total_q = 0.0
                integ_pulse_total_qt = 0.0

                for pulse in pulses[
                    dom["pulses_idx0"] : dom["pulses_idx0"] + dom["num_pulses"]
                ]:
                    # if min_dom_dt > 0:
                    #     if (pulse["time"] - dom_pulse_t0) < min_dom_dt:
                    #         continue

                    if max_dom_dt > 0:
                        if (pulse["time"] - dom_pulse_t0) >= max_dom_dt:
                            continue

                    if min_evt_dt > 0:
                        if (pulse["time"] - event_pulse_t0) < min_evt_dt:
                            continue

                    if max_evt_dt > 0:
                        if (pulse["time"] - event_pulse_t0) >= max_evt_dt:
                            continue

                    pulse_charge = pulse["charge"]
                    pulse_time = pulse["time"]

                    if qntm > 0:
                        pulse_charge = quantize(pulse_charge, qntm=qntm)

                    if min_pulse_q > 0:
                        if pulse_charge < min_pulse_q:
                            continue

                    if fixed_pulse_q > 0:
                        pulse_charge = fixed_pulse_q

                    if integ_t > 0:
                        if pulse_time - last_integ_pulse_t0 > integ_t:
                            # Record previous integrated pulse
                            if num_integ_pulses > 0:
                                new_pulses[total_num_pulses][
                                    "charge"
                                ] = integ_pulse_total_q
                                new_pulses[total_num_pulses]["time"] = (
                                    integ_pulse_total_qt / integ_pulse_total_q
                                )
                                dom_num_pulses += 1
                                total_num_pulses += 1
                                dom_charge += integ_pulse_total_q

                            # Start a new integrated pulse
                            num_integ_pulses += 1
                            last_integ_pulse_t0 = pulse_time
                            integ_pulse_total_t = pulse_time
                            integ_pulse_total_q = pulse_charge
                            integ_pulse_total_qt = pulse_charge * pulse_time
                        else:
                            integ_pulse_total_t += pulse_time
                            integ_pulse_total_q += pulse_charge
                            integ_pulse_total_qt = pulse_charge * pulse_time

                    else:
                        new_pulses[total_num_pulses]["charge"] = pulse_charge
                        new_pulses[total_num_pulses]["time"] = pulse_time

                        dom_num_pulses += 1
                        total_num_pulses += 1
                        dom_charge += pulse_charge

                if integ_pulse_total_q > 0:
                    new_pulses[total_num_pulses]["charge"] = integ_pulse_total_q
                    new_pulses[total_num_pulses]["time"] = (
                        integ_pulse_total_qt / integ_pulse_total_q
                    )
                    dom_num_pulses += 1
                    total_num_pulses += 1
                    dom_charge += integ_pulse_total_q

                if dom_num_pulses < min_dom_p:
                    # "rewind" array populated in the dom loop
                    total_num_pulses -= dom_num_pulses
                    continue

                new_doms[total_num_hit_doms]["string"] = dom["string"]
                new_doms[total_num_hit_doms]["om"] = dom["om"]
                new_doms[total_num_hit_doms]["pulses_idx0"] = new_doms_pulses_idx0
                new_doms[total_num_hit_doms]["num_pulses"] = dom_num_pulses
                new_doms[total_num_hit_doms]["charge"] = dom_charge

                event_num_hit_doms += 1
                event_num_pulses += dom_num_pulses
                event_charge += dom_charge

                total_num_hit_doms += 1

            if event_num_pulses < min_evt_p:
                # "rewind" all arrays populated in the event loop
                total_num_pulses -= event_num_pulses
                total_num_hit_doms -= event_num_hit_doms
                continue

            if event_num_hit_doms == 0:
                continue

            new_events[total_num_events : total_num_events + 1] = event
            new_events[total_num_events]["dom_idx0"] = dom_idx0
            new_events[total_num_events]["num_hit_doms"] = event_num_hit_doms
            new_events[total_num_events]["num_pulses"] = event_num_pulses
            new_events[total_num_events]["charge"] = event_charge

            total_num_events += 1

        new_events = new_events[:total_num_events]
        new_doms = new_doms[:total_num_hit_doms]
        new_pulses = new_pulses[:total_num_pulses]

        return new_events, new_doms, new_pulses

    return filter_arrays


def load_and_filter(
    set_key,
    root_data_dir,
    fixed_pulse_q,
    qntm,
    min_pulse_q,
    min_evt_p,
    min_evt_dt,
    max_evt_dt,
    # min_evt_t_fract,
    # max_evt_t_fract,
    # t_fract_window,
    min_dom_p,
    # min_dom_dt,
    max_dom_dt,
    integ_t,
    i3,
    dc,
    z_regions,
):
    """
    Parameters
    ----------
    set_key : key in either DATA_DIR_INFOS or MC_DIR_INFOS
    root_data_dir
    fixed_pulse_q
    qntm
    min_pulse_q
    min_evt_p
    min_evt_dt
    max_evt_dt
    #min_evt_t_fract
    #max_evt_t_fract
    #t_fract_window
    min_dom_p
    #min_dom_dt
    max_dom_dt
    integ_t
        Integration time in ns. If 0, no integration is performed.
    i3
    dc
    z_regions

    Returns
    -------
    events
    doms
    pulses

    """
    root_data_dir = expand(root_data_dir)
    assert i3 or dc
    assert set_key in MC_DIR_INFOS or set_key in DATA_DIR_INFOS

    filter_arrays = generate_filter_func(
        fixed_pulse_q=fixed_pulse_q,
        qntm=qntm,
        min_pulse_q=min_pulse_q,
        min_evt_p=min_evt_p,
        min_evt_dt=min_evt_dt,
        max_evt_dt=max_evt_dt,
        # min_evt_t_fract=min_evt_t_fract,
        # max_evt_t_fract=max_evt_t_fract,
        # t_fract_window=t_fract_window,
        # min_evt_q_t_qtl=min_evt_q_t_qtl,
        # max_evt_q_t_qtl=max_evt_q_t_qtl,
        min_dom_p=min_dom_p,
        # min_dom_dt=min_dom_dt,
        max_dom_dt=max_dom_dt,
        integ_t=integ_t,
        i3=i3,
        dc=dc,
        z_regions=z_regions,
    )

    if set_key in DATA_DIR_INFOS:
        is_mc = False
        dirpath = join(root_data_dir, DATA_DIR_INFOS[set_key]["path"])
        num_files = 1
        # season, proc_ver = set_key
        # label = "Data, 20{} season, proc ver {} : {}".format(season, proc_ver, dirpath)
    else:
        is_mc = True
        dirpath = join(root_data_dir, MC_DIR_INFOS[set_key]["path"])
        num_files = MC_DIR_INFOS[set_key]["num_files"]
        # mc_run, proc_ver = set_key
        # label = "MC, set {}, proc ver {}, num files={} : {}".format(
        #    mc_run, proc_ver, num_files, dirpath
        # )

    filter_args = tuple()
    filter_kwargs = dict()
    for name in ["events", "doms", "pulses"]:
        filter_kwargs[name] = np.load(join(dirpath, name + "_array.npy"), mmap_mode="r")

    result = filter_arrays(*filter_args, **filter_kwargs)
    if is_mc:
        result[0]["weight"] /= num_files

    return result


@numba.jit(cache=True, **JIT_KW)
def get_data_weight_func(event):  # pylint: disable=unused-argument
    """if `event` is data, weight is 1 (and there is no "weight" field in the dtype"""
    return 1


@numba.jit(cache=True, **JIT_KW)
def get_mc_weight_func(event):
    """if `event` is Monte Carlo, "weight" is a field in the array"""
    return event["weight"]


@numba.jit(cache=True, **JIT_KW)
def binit(val, weight, weight_sq, dig_func, edges, histo, histo_w2):
    """Histogram `val` within `edges` via `dig_func`, weighted by `weight` in
    `histo` histogram and weighted by `weight_sq` in `histo_w2`
    histogram.

    Parameters
    ----------
    val : scalar
    weight : scalar
    weight_sq : scalar
    dig_func : numba-callable
    edges : ndarray of monotonically increasing values
    histo : ndarray
    histo_w2 : ndarray

    """
    if edges[0] <= val < edges[-1]:
        idx = dig_func(val)
        histo[idx] += weight
        histo_w2[idx] += weight_sq


@numba.jit(cache=False, **JIT_KW)
def create_histos(events, doms, pulses, get_weight_func, edges):
    """
    Parameters
    ----------
    events : numpy ndarray of dtype
    doms : numpy ndarray of dtype
    pulses : numpy ndarray of dtype
    get_weight_func : njit-ed callable
    edges : numba.typed.Dict

    Returns
    -------
    histos : numba.typed.Dict
    histos_w2 : numba.typed.Dict
    total_weights : numba.typed.Dict
    total_weights_squared : numba.typed.Dict

    """
    histos = dict()
    histos_w2 = dict()

    total_weights = dict()
    total_weights_squared = dict()

    for stat, binning in edges.items():
        histos[stat] = np.zeros(shape=len(binning) - 1, dtype=np.float64)
        histos_w2[stat] = np.zeros(shape=len(binning) - 1, dtype=np.float64)
        total_weights[stat] = 0.0
        total_weights_squared[stat] = 0.0

    for event in events:
        weight = get_weight_func(event)
        weight_sq = np.square(weight)

        binit(
            val=event["num_hit_doms"],
            weight=weight,
            weight_sq=weight_sq,
            dig_func=DOMS_PER_EVENT_DIG,
            edges=edges["doms_per_event"],
            histo=histos["doms_per_event"],
            histo_w2=histos_w2["doms_per_event"],
        )
        total_weights["doms_per_event"] += weight
        total_weights_squared["doms_per_event"] += weight_sq

        binit(
            val=event["charge"],
            weight=weight,
            weight_sq=weight_sq,
            dig_func=CHARGE_PER_EVENT_DIG,
            edges=edges["charge_per_event"],
            histo=histos["charge_per_event"],
            histo_w2=histos_w2["charge_per_event"],
        )

        binit(
            val=event["num_pulses"],
            weight=weight,
            weight_sq=weight_sq,
            dig_func=PULSES_PER_EVENT_DIG,
            edges=edges["pulses_per_event"],
            histo=histos["pulses_per_event"],
            histo_w2=histos_w2["pulses_per_event"],
        )

        event_pulse_t0 = np.inf
        for dom in doms[event["dom_idx0"] : event["dom_idx0"] + event["num_hit_doms"]]:
            event_pulse_t0 = min(event_pulse_t0, pulses[dom["pulses_idx0"]]["time"])

        for dom in doms[event["dom_idx0"] : event["dom_idx0"] + event["num_hit_doms"]]:
            binit(
                val=dom["charge"],
                weight=weight,
                weight_sq=weight_sq,
                dig_func=CHARGE_PER_DOM_DIG,
                edges=edges["charge_per_dom"],
                histo=histos["charge_per_dom"],
                histo_w2=histos_w2["charge_per_dom"],
            )
            total_weights["charge_per_dom"] += weight
            total_weights_squared["charge_per_dom"] += weight_sq

            binit(
                val=dom["num_pulses"],
                weight=weight,
                weight_sq=weight_sq,
                dig_func=PULSES_PER_DOM_DIG,
                edges=edges["pulses_per_dom"],
                histo=histos["pulses_per_dom"],
                histo_w2=histos_w2["pulses_per_dom"],
            )

            dom_pulse_t0 = pulses[dom["pulses_idx0"]]["time"]

            for pulse in pulses[
                dom["pulses_idx0"] : dom["pulses_idx0"] + dom["num_pulses"]
            ]:
                binit(
                    val=pulse["charge"],
                    weight=weight,
                    weight_sq=weight_sq,
                    dig_func=CHARGE_PER_PULSE_DIG,
                    edges=edges["charge_per_pulse"],
                    histo=histos["charge_per_pulse"],
                    histo_w2=histos_w2["charge_per_pulse"],
                )
                total_weights["charge_per_pulse"] += weight
                total_weights_squared["charge_per_pulse"] += weight_sq

                binit(
                    val=pulse["time"] - dom_pulse_t0,
                    weight=weight,
                    weight_sq=weight_sq,
                    dig_func=TIME_DIFFS_WITHIN_DOM_DIG,
                    edges=edges["time_diffs_within_dom"],
                    histo=histos["time_diffs_within_dom"],
                    histo_w2=histos_w2["time_diffs_within_dom"],
                )

                binit(
                    val=pulse["time"] - event_pulse_t0,
                    weight=weight,
                    weight_sq=weight_sq,
                    dig_func=TIME_DIFFS_WITHIN_EVENT_DIG,
                    edges=edges["time_diffs_within_event"],
                    histo=histos["time_diffs_within_event"],
                    histo_w2=histos_w2["time_diffs_within_event"],
                )

    # -- Per-event stats all have same total weights -- #

    total_weights["charge_per_event"] = total_weights["doms_per_event"]
    total_weights_squared["charge_per_event"] = total_weights_squared["doms_per_event"]

    total_weights["pulses_per_event"] = total_weights["doms_per_event"]
    total_weights_squared["pulses_per_event"] = total_weights_squared["doms_per_event"]

    # -- Per-DOM stats all have same total weights -- #

    total_weights["pulses_per_dom"] = total_weights["charge_per_dom"]
    total_weights_squared["pulses_per_dom"] = total_weights_squared["charge_per_dom"]

    # -- Per-pulse stats all have same total weights -- #

    total_weights["time_diffs_within_dom"] = total_weights["charge_per_pulse"]
    total_weights_squared["time_diffs_within_dom"] = total_weights_squared[
        "charge_per_pulse"
    ]

    total_weights["time_diffs_within_event"] = total_weights["charge_per_pulse"]
    total_weights_squared["time_diffs_within_event"] = total_weights_squared[
        "charge_per_pulse"
    ]

    return histos, histos_w2, total_weights, total_weights_squared


def get_true_time_relative_info(events, doms, pulses):
    """
    Parameters
    ----------
    events : ndarray of dtype MC_DOMS_IDX_T
    doms : ndarray of dtype DOM_PULSES_IDX_T
    pulses : ndarray of dtype SIMPLE_PULSE_T

    Returns
    -------
    info : ndarray of dtype TRUE_TIME_INFO_T

    """
    info = np.empty(shape=len(events), dtype=TRUE_TIME_INFO_T)
    for event_idx, event in enumerate(events):
        event_first_dom = doms[event["dom_idx0"]]
        event_last_dom = doms[int(event["dom_idx0"] + event["num_hit_doms"] - 1)]

        event_pulses_start = int(event_first_dom["pulses_idx0"])
        event_pulses_stop = int(event_last_dom["pulses_idx0"] + event_last_dom["num_pulses"])

        event_pulses = pulses[event_pulses_start:event_pulses_stop]
        sorted_pulses = np.sort(event_pulses, order="time")
        cumulative_q = np.cumsum(sorted_pulses["charge"])

        cumulative_q_at_true_time = np.interp(
            x=event["true_time"], xp=sorted_pulses["time"], fp=cumulative_q
        )

        info[event_idx]["true_energy"] = event["true_energy"]
        info[event_idx]["weight"] = event["weight"]
        info[event_idx]["q_fract"] = cumulative_q_at_true_time / cumulative_q[-1]

        dt = event["true_time"] - sorted_pulses[0]["time"]
        pulses_width = sorted_pulses[-1]["time"] - sorted_pulses[0]["time"]
        if pulses_width == 0:
            info[event_idx]["t_fract"] = np.nan
        else:
            info[event_idx]["t_fract"] = dt / pulses_width
        info[event_idx]["dt"] = dt

    return info


def integer_if_integral(x):
    """Convert a float to int if it can be represented as an int"""
    return int(x) if x == int(x) else x


def get_histo_fname_prefix(processing_kw, set_key=None):
    """Get name of file that will/does contain histograms for a given data/MC
    set processed with processing_kw"""
    kw = deepcopy(processing_kw)
    if not isinstance(kw["z_regions"], int):
        kw["z_regions"] = ",".join(str(r) for r in kw["z_regions"])

    prefix = "__".join("{}={}".format(*it) for it in kw.items())
    if set_key is not None:
        prefix += "__set_key={}".format(str(set_key).replace(" ", ""))

    return prefix


def load_histos(histo_data_dir, processing_kw, set_key):
    """Load histograms (and weights totals)

    Parameters
    ----------
    histo_data_dir : str
        Dir containing histo data
    processing_kw : mapping
    set_key : tuple

    Returns
    -------
    OrderedDict
        Contains keys "histos", "histos_w2", "total_weights", and
        "total_weights_squared"; each is an OrderedDict keyed by statistic name
        and values are numpy arrays in the former case, and scalars in the
        latter.

    """
    histo_fname = (
        get_histo_fname_prefix(set_key=set_key, processing_kw=processing_kw) + ".pkl"
    )
    histo_fpath = join(histo_data_dir, histo_fname)
    print('loading "{}"'.format(histo_fpath))
    return load_pickle(histo_fpath)


def plot(histo_data_dir, histo_plot_dir, processing_kw, mc_set, only_seasons=None):
    """
    Parameters
    ----------

    Returns
    -------

    """
    import matplotlib as mpl

    if __name__ == "__main__" and __package__ is None:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    def setup_plots():
        width, height = 16, 7
        fig = plt.Figure(figsize=(width, height), dpi=50)
        gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0, figure=fig)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        fig = ax0.get_figure()
        fig.set_figwidth(width)
        fig.set_figheight(height)
        fig.set_dpi(120)
        plt.setp(ax0.get_xticklabels(), visible=False)
        return fig, (ax0, ax1)

    def augment(y):
        return np.concatenate([[y_n] * 2 for y_n in y])

    warnings.filterwarnings("ignore")
    try:
        mc_set_spec = MC_SET_SPECS[mc_set]
        if only_seasons is None:
            only_seasons = list(DATA_DIR_INFOS.keys())
        elif only_seasons in DATA_DIR_INFOS:
            only_seasons = [only_seasons]

        histo_data_dir = expand(histo_data_dir)
        if basename(histo_data_dir) != "histo_data":
            histo_data_dir = join(histo_data_dir, "histo_data")

        histo_plot_dir = expand(histo_plot_dir)
        if basename(histo_plot_dir) != "histo_plots":
            histo_plot_dir = join(histo_plot_dir, "histo_plots")
        mkdir(histo_plot_dir)

        mc_sets_d = OrderedDict()
        data_sets_d = OrderedDict()

        mc_data = OrderedDict()
        for mc_name, mc_key in mc_set_spec.items():
            mc_sets_d[mc_name] = mc_key
            mc_data[mc_name] = load_histos(
                histo_data_dir=histo_data_dir,
                processing_kw=processing_kw,
                set_key=mc_key,
            )

        data_data = OrderedDict()
        for season in only_seasons:
            season_num, _ = season
            season_num_str = "IC86.{:02d}".format(season_num)
            data_sets_d[season_num_str] = season
            data_data[season_num_str] = load_histos(
                histo_data_dir=histo_data_dir,
                processing_kw=processing_kw,
                set_key=season,
            )

        # Invert ordering hierarchy of dicts; stats_d is keyed by stat name
        stats_d = OrderedDict()
        for stat in next(iter(next(iter(mc_data.values())).values())).keys():
            stats_d[stat] = OrderedDict()
            for set_name, set_info in list(mc_data.items()) + list(data_data.items()):
                stats_d[stat][set_name] = dict(
                    histo=set_info["histos"][stat],
                    histo_w2=set_info["histos_w2"][stat],
                    total_weight=set_info["total_weights"][stat],
                    total_weight_sq=set_info["total_weights_squared"][stat],
                )

        # Make plots!
        for stat, stat_info in stats_d.items():
            fig, (ax0, ax1) = setup_plots()

            plabel = LABELS.get(stat, stat)
            ulabel = " ({})".format(UNITS[stat]) if stat in UNITS else ""

            total_mc = 0.0
            for mc_name in mc_data.keys():
                total_mc += stat_info[mc_name]["total_weight"]

            total_data = 0.0
            for data_name in data_data.keys():
                total_data += stat_info[data_name]["total_weight"]

            bins = HIST_EDGES[stat]
            aug_bins = np.concatenate([bins[i : i + 2] for i in range(len(bins))])[:-1]

            n_bins = len(bins) - 1

            # -- Plot MC -- #

            mc_cum_y = np.zeros(n_bins)
            mc_cum_w2 = np.zeros(n_bins)

            for mc_name in mc_data.keys():
                prev_mc_cum_y = deepcopy(mc_cum_y)
                mc_cum_y += stat_info[mc_name]["histo"]
                mc_cum_w2 += stat_info[mc_name]["histo_w2"]

                ax0.fill_between(
                    x=aug_bins,
                    y1=augment(prev_mc_cum_y),
                    y2=augment(mc_cum_y),
                    alpha=0.3,
                    edgecolor=(0, 0, 0, 0.5),
                    linewidth=0.5,
                    label=LABELS.get(mc_name, mc_name),
                )

            mc_err = np.sqrt(mc_cum_w2)
            ax0.fill_between(
                x=aug_bins,
                y1=augment(mc_cum_y - mc_err),
                y2=augment(mc_cum_y + mc_err),
                zorder=10,
                edgecolor="k",
                linewidth=0.75,
                facecolor="none",
                hatch="//",
                label=r"$\Sigma$ MC",
            )

            # -- Plot data -- #

            num_data_seasons = len(data_data)
            ndivs = num_data_seasons + 1 + 2
            if stat in LOG_X_PARAMS:
                log_bins = np.log(bins)
                width_log_bins = np.mean(np.diff(log_bins))
                dwlog = width_log_bins / ndivs
            else:
                width = np.mean(np.diff(bins))
                dw = width / ndivs

            data_cum_y = np.zeros(n_bins)
            data_w2_cum_y = np.zeros(n_bins)

            for data_set_num, data_name in enumerate(data_data.keys(), start=1 + 1):
                histo = stat_info[data_name]["histo"]
                histo_w2 = stat_info[data_name]["histo_w2"]
                season_total = stat_info[data_name]["total_weight"]

                data_cum_y += histo
                data_w2_cum_y += histo_w2

                y_normed = histo * total_mc / season_total
                data_err = np.sqrt(histo_w2) * total_mc / season_total

                if stat in LOG_X_PARAMS:
                    x = np.exp(log_bins[:-1] + data_set_num * dwlog)
                else:
                    x = bins[:-1] + data_set_num * dw

                ax0.errorbar(
                    x=x,
                    y=y_normed,
                    yerr=data_err,
                    linestyle="none",
                    marker="_",
                    markersize=3,
                    capsize=0,
                    label=LABELS.get(data_name, data_name),
                    zorder=20,
                )

                ratio = y_normed / mc_cum_y
                ratio_err = ratio * np.sqrt(
                    (mc_err / mc_cum_y) ** 2 + (data_err / y_normed) ** 2
                )

                ax1.errorbar(
                    x=x,
                    y=ratio,
                    yerr=ratio_err,
                    linestyle="none",
                    marker="_",
                    markersize=3,
                    capsize=0,
                    # label=LABELS.get(data_name, data_name),
                    zorder=20,
                )

            data_err = np.sqrt(data_cum_y) * total_mc / total_data
            data_cum_y_normed = data_cum_y * total_mc / total_data

            ax0.fill_between(
                x=aug_bins,
                y1=augment(data_cum_y_normed - data_err),
                y2=augment(data_cum_y_normed + data_err),
                color=(1, 0, 1),
                linewidth=0.75,
                alpha=0.4,
                zorder=12,
                label=r"$\Sigma$ data",
            )

            # -- Plot log_10(MC / data) ratio -- #

            ratio = data_cum_y_normed / mc_cum_y
            ratio_err = ratio * np.sqrt(
                (mc_err / mc_cum_y) ** 2 + (data_err / data_cum_y_normed) ** 2
            )

            ax1.fill_between(
                x=aug_bins,
                y1=augment(ratio - ratio_err),
                y2=augment(ratio + ratio_err),
                facecolor=(0, 0, 0, 0.3),
                edgecolor=(0, 0, 0, 0.7),
                linewidth=0.5,
                # facecolor=(0,0,0,0.4),
                # zorder=12,
                # label=r"$\Sigma$ data / $\Sigma$ MC",
            )

            ax1.step(x=bins, y=[ratio[0]] + ratio.tolist(), lw=1, color=(0, 0, 0))

            # -- Finalize plots -- #

            xlim = (bins[0], bins[-1])
            ax1.plot(xlim, [1, 1], color=(0, 0, 0), ls="-", lw=0.5, zorder=100)
            ax1.set_xlim(xlim)
            ax1.set_yscale("log")
            ax1.set_yticks([], minor=True)
            ax1.set_yticks([], minor=False)

            ax1.set_xlabel("{}{}".format(plabel, ulabel))
            ax1.set_ylabel("Data / MC (log scale)")

            ticklocs = REF_LOG_TICKS  # [idx0:idx1+1]
            ax1.set_yticks(ticklocs)
            ax1.set_yticklabels(REF_LOG_TICKLABELS, fontdict=dict())
            ax1.set_ylim(np.min(ticklocs), np.max(ticklocs))

            grid_kw = dict(linewidth=0.5, color=(0.7, 0.7, 0.7, 0.1), zorder=0)
            ax1.grid(True, which="both", axis="both", **grid_kw)

            # if stat in integral_params:
            #    xticks = np.arange(displimval_q0, displimval_q1 + 0.5, 1)
            #    ax1.set_xticks(xticks)
            #    ax1.set_xticklabels(["{:d}".format(int(xt)) for xt in xticks])

            loc = LEG_LOC.get(stat, "best")
            markerfirst = "left" in loc
            ncol = 2 if stat == "azimuth" else 1
            handles_, labels_ = ax0.get_legend_handles_labels()
            ordered_labels = (
                labels_[len(labels_) - len(data_data) - 1 :]
                + labels_[: -len(data_data) - 1][::-1]
            )
            ordered_handles = (
                handles_[len(handles_) - len(data_data) - 1 :]
                + handles_[: -len(data_data) - 1][::-1]
            )
            ax0.legend(
                handles=ordered_handles,
                labels=ordered_labels,
                loc=loc,
                frameon=False,
                markerfirst=markerfirst,
                ncol=ncol,
            )

            if stat in LOG_X_PARAMS:
                for ax in [ax0, ax1]:
                    ax.set_xscale("log")

            for log_yscale in [False, True]:
                if log_yscale:
                    ax0.set_yscale("log")
                else:
                    ax0.set_yscale("linear")

                ax0.autoscale(axis="y")
                ax0.grid(True, which="both", axis="x", **grid_kw)

                if log_yscale:
                    ax0.set_yticks([])
                    ax0.set_yscale("log")
                    ax0.set_ylabel("log scale")
                    ax0.set_yticklabels([])
                else:
                    ax0.set_ylim(0, ax0.get_ylim()[1])
                    ax0.set_yticklabels([])
                    ax0.set_ylabel("linear scale")

                # -- Save plots -- #

                if set(s[0] for s in only_seasons) == set(range(12, 18 + 1)):
                    only_seasons_str = ""
                else:
                    only_seasons_str = "__only_seasons=" + ",".join(
                        str(s[0]) for s in sorted(only_seasons)
                    )

                plt_basename = (
                    "{}{}__".format(mc_set, only_seasons_str)
                    + get_histo_fname_prefix(processing_kw=processing_kw)
                    + "__{}".format(stat)
                )
                fbasename = join(histo_plot_dir, plt_basename)
                if log_yscale:
                    extended_fbasename = fbasename + "__logy"
                else:
                    extended_fbasename = deepcopy(fbasename)

                fig.savefig(extended_fbasename + ".pdf")
                fig.savefig(extended_fbasename + ".png", dpi=120)
    finally:
        warnings.resetwarnings()


def parse_args(description=__doc__):
    """Command line interface"""
    parser = ArgumentParser(description=description)

    subparsers = parser.add_subparsers()

    populate_sp = subparsers.add_parser("populate")

    populate_sp.add_argument("--root-data-dir", type=str)
    populate_sp.add_argument("--set-key", type=str)

    plot_sp = subparsers.add_parser("plot")

    plot_sp.add_argument("--mc-set", type=str)
    plot_sp.add_argument("--only-seasons", type=str, default=None)
    plot_sp.add_argument("--histo-plot-dir", type=str)

    for subp in [populate_sp, plot_sp]:
        subp.add_argument("--histo-data-dir", type=str)
        subp.add_argument("--fixed-pulse-q", type=float, default=0)
        subp.add_argument("--qntm", type=float, default=0)
        subp.add_argument("--min-pulse-q", type=float, default=0)
        subp.add_argument("--min-evt-p", type=int, default=1)
        subp.add_argument("--min-evt-dt", type=float, default=0)
        subp.add_argument("--max-evt-dt", type=float, default=0)
        # subp.add_argument("--min-evt-t-fract", type=float, default=0)
        # subp.add_argument("--max-evt-t-fract", type=float, default=0)
        # subp.add_argument("--t-fract-window", type=float, default=0)
        # subp.add_argument("--min-evt-q-t-qtl", type=float, default=0)
        # subp.add_argument("--max-evt-q-t-qtl", type=float, default=1)
        subp.add_argument("--min-dom-p", type=int, default=1)
        # subp.add_argument("--min-dom-dt", type=float, default=0)
        subp.add_argument("--max-dom-dt", type=float, default=0)
        subp.add_argument("--integ-t", type=float, default=0)
        subp.add_argument("--no-i3", action="store_true")
        subp.add_argument("--no-dc", action="store_true")
        subp.add_argument("--z-regions", nargs="+", type=int)

    args = parser.parse_args()
    kwargs = vars(args)

    if "no_i3" in kwargs:
        kwargs["i3"] = not kwargs.pop("no_i3")
    if "no_dc" in kwargs:
        kwargs["dc"] = not kwargs.pop("no_dc")

    return kwargs


def main():
    """Scripty bits"""
    t0 = time.time()

    kwargs = parse_args()

    histo_data_dir = expand(kwargs.pop("histo_data_dir"))
    if basename(histo_data_dir) != "histo_data":
        histo_data_dir = join(histo_data_dir, "histo_data")

    proc_kw_keys = [
        "fixed_pulse_q",
        "qntm",
        "min_pulse_q",
        "min_evt_p",
        "min_evt_dt",
        "max_evt_dt",
        # "min_evt_t_fract",
        # "max_evt_t_fract",
        # "t_fract_window",
        # "min_evt_q_t_qtl",
        # "max_evt_q_t_qtl",
        "min_dom_p",
        # "min_dom_dt",
        "max_dom_dt",
        "integ_t",
        "i3",
        "dc",
        "z_regions",
    ]
    processing_kw = OrderedDict(
        (k, kwargs.pop(k)) for k in list(kwargs.keys()) if k in proc_kw_keys
    )

    if len(processing_kw["z_regions"]) == 1:
        processing_kw["z_regions"] = processing_kw["z_regions"][0]

    # Convert floats that are integral to ints for clean output
    for key, val in list(processing_kw.items()):
        if isinstance(val, bool) or not isinstance(val, numbers.Number):
            continue
        processing_kw[key] = integer_if_integral(val)

    if "mc_set" in kwargs:
        return plot(
            histo_data_dir=histo_data_dir,
            histo_plot_dir=kwargs["histo_plot_dir"],
            processing_kw=processing_kw,
            mc_set=kwargs["mc_set"],
            only_seasons=kwargs["only_seasons"],
        )

    # -- Else: populate histos for a single mc or data run -- #

    root_data_dir = expand(kwargs.pop("root_data_dir"))
    assert isdir(root_data_dir)

    set_key = eval(kwargs.pop("set_key"))  # pylint: disable=eval-used
    assert set_key in DATA_DIR_INFOS or set_key in MC_DIR_INFOS
    histo_fname = (
        get_histo_fname_prefix(set_key=set_key, processing_kw=processing_kw) + ".pkl"
    )
    if set_key in MC_DIR_INFOS:
        get_weight_func = get_mc_weight_func
    else:
        get_weight_func = get_data_weight_func

    histo_fpath = join(histo_data_dir, histo_fname)
    if isfile(histo_fpath):
        print('{} : Loading from file "{}"'.format(set_key, histo_fpath))
        print("{} : load_pickle     : {:.3f} sec".format(set_key, time.time() - t0))
        return load_pickle(histo_fpath)

    mkdir(histo_data_dir)
    print('{} : Histo vals will be saved to file "{}"'.format(set_key, histo_fpath))

    events, doms, pulses = load_and_filter(
        root_data_dir=root_data_dir, set_key=set_key, **processing_kw
    )
    t1 = time.time()
    print("{} : load_and_filter : {:.3f} sec".format(set_key, t1 - t0))

    histos, histos_w2, total_weights, total_weights_squared = create_histos(
        events=events,
        doms=doms,
        pulses=pulses,
        get_weight_func=get_weight_func,
        edges=HIST_EDGES,
    )
    t2 = time.time()
    print("{} : create_histos   : {:.3f} sec".format(set_key, t2 - t1))

    out_d = OrderedDict(
        [
            ("histos", OrderedDict(histos.items())),
            ("histos_w2", OrderedDict(histos_w2.items())),
            ("total_weights", OrderedDict(total_weights.items())),
            ("total_weights_squared", OrderedDict(total_weights_squared.items())),
        ]
    )
    pickle.dump(out_d, open(histo_fpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    t3 = time.time()
    print("{} : pickle.dump     : {:.3f} sec".format(set_key, t3 - t2))
    print("{} : total time      : {:.3f} sec".format(set_key, time.time() - t0))

    return out_d


if __name__ == "__main__":
    main()
