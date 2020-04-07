# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
namedtuples for interface simplicity and consistency
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'Event',
    'Hit',
    'Photon',
    'RetroPhotonInfo',
    'HypoPhotonInfo',
    'Cart2DCoord',
    'Cart3DCoord',
    'PolCoord',
    'SphCoord',
    'TimeCart3DCoord',
    'TimePolCoord',
    'TimeSphCoord',
    'OMKEY_T',
    'I3POSITION_T',
    'I3DIRECTION_T',
    'I3OMGEO_T',
    'I3DOMCALIBRATION_T',
    'DOMINFO_T',
    'EVT_DOM_INFO_T',
    'PULSE_T',
    'PHOTON_T',
    'HIT_T',
    'SD_INDEXER_T',
    'HITS_SUMMARY_T',
    'EVT_HIT_INFO_T',
    'ParticleType',
    "EM_CASCADE_PTYPES",
    "HADR_CASCADE_PTYPES",
    "CASCADE_PTYPES",
    "TRACK_PTYPES",
    "INVISIBLE_PTYPES",
    "ELECTRONS",
    "MUONS",
    "TAUS",
    "NUES",
    "NUMUS",
    "NUTAUS",
    "NEUTRINOS",
    'ParticleShape',
    'FitStatus',
    'LocationType',
    'TriggerConfigID',
    'TriggerTypeID',
    'TriggerSourceID',
    'TriggerSubtypeID',
    'ExtractionError',
    'OMType',
    'CableType',
    'DOMGain',
    'TrigMode',
    'LCMode',
    'ToroidType',
    'TRIGGER_T',
    'TRIGGERKEY_T',
    'I3TRIGGERREADOUTCONFIG_T',
    'I3TIME_T',
    'I3PARTICLEID_T',
    'I3PARTICLE_T',
    'FLAT_PARTICLE_T',
    'SRC_T',
    'TRACK_T',
    'CRAMERRAOPARAMS_T',
    'CramerRaoStatus',
    'DOMCALVERSION_T',
    'DSTPOSITION_T',
    'FLAT_I3DOMCALIBRATION_T',
    'FLAT_I3VEMCALIBRATION_T',
    'FLAT_PULSE_T',
    'FLAT_TRIGGER_T',
    'I3CLASTFITPARAMS_T',
    'I3CSCDLLHFITPARAMS_T',
    'I3DIPOLEFITPARAMS_T',
    'I3DIRECTHITSVALUES_T',
    'I3DOMSTATUS_T',
    'I3DST16_T',
    'I3EVENTHEADER_T',
    'I3FILLRATIOINFO_T',
    'I3FILTERRESULT_T',
    'I3FINITECUTS_T',
    'I3GENIERESULTDICT_SCALARS_T',
    'I3HITMULTIPLICITYVALUES_T',
    'I3HITSTATISTICSVALUES_T',
    'I3LINEFITPARAMS_T',
    'I3LOGLIKELIHOODFITPARAMS_T',
    'I3PORTIAEVENT_T',
    'I3RUSAGE_T',
    'I3SANTAFITPARAMS_T',
    'I3STARTSTOPPARAMS_T',
    'I3SUPERDSTTRIGGER_T',
    'I3TENSOROFINERTIAFITPARAMS_T',
    'I3TIMECHARACTERISTICSVALUES_T',
    'I3TIMEWINDOW_T',
    'I3TRACKCHARACTERISTICSVALUES_T',
    'I3VEMCALIBRATION_T',
    'I3VETO_T',
    'InteractionType',
    'LINEARFIT_T',
    'NEUTRINO_T',
    'OnOff',
    'PulseFlags',
    'SPECHARGEDISTRIBUTION_T',
    'START_AND_LENGTH_T',
    'START_STOP_T',
    'TAUPARAM_T',
    'INVALID_TRACK',
    'NO_TRACK',
    'CASCADE_T',
    'INVALID_CASCADE',
    'NO_CASCADE',
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

from collections import namedtuple
import enum
import numpy as np

Event = namedtuple( # pylint: disable=invalid-name
    typename='Event',
    field_names=('filename', 'event', 'uid', 'pulses', 'interaction',
                 'neutrino', 'track', 'cascade', 'ml_reco', 'spe_reco')
)

Hit = namedtuple(
    typename='Hit',
    field_names=('time', 'charge', 'width')
)

Photon = namedtuple(
    typename='Photon',
    field_names=('x', 'y', 'z', 'time', 'wavelength', 'coszen', 'azimuth')
)

RetroPhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='RetroPhotonInfo',
    field_names=('survival_prob', 'time_indep_survival_prob', 'theta', 'deltaphi', 'length')
)
"""Info contained in (original) retro tables: Photon survival probability
(survival_prob) and average photon direction and length (theta, deltaphi,
length). `deltaphi` is the direction in the azimuthal direction relative to
the bin center's azimuth (phi) direction. Note that directions are expected to
follow "standard" spherical coordinates where direction of vector is the
direciton in which it points, NOT the direction from which it comes (as is the
astro / IceCube convention). Intended to contain dictionaries with DOM depth
index as keys and arrays as values."""

HypoPhotonInfo = namedtuple( # pylint: disable=invalid-name
    typename='HypoPhotonInfo',
    field_names=('count', 'theta', 'phi', 'length')
)
"""Info contained in (original) retro tables: Photon survival probability
(survival_prob) and average photon direction and length (theta, phi, length).
Note that directions are expected to follow "standard" spherical coordinates
where direction of vector is the direciton in which it points, NOT the
direction from which it comes (as is the astro / IceCube convention). Intended
to contain dictionaries with DOM depth index as keys and arrays as values."""

Cart2DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Cart2DCoord',
    field_names=('x', 'y'))
"""Cartesian 2D coordinate: x, y."""

Cart3DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Cart3DCoord',
    field_names=('x', 'y', 'z'))
"""Cartesian 3D coordinate: x, y, z."""

PolCoord = namedtuple( # pylint: disable=invalid-name
    typename='PolCoord',
    field_names=('r', 'theta')
)
"""2D polar coordinate: r, theta."""

SphCoord = namedtuple( # pylint: disable=invalid-name
    typename='SphCoord',
    field_names=('r', 'theta', 'phi')
)
"""3D spherical coordinate: r, theta, and phi."""

TimeCart3DCoord = namedtuple( # pylint: disable=invalid-name
    typename='Time3DCartCoord',
    field_names=('time',) + Cart3DCoord._fields
)
"""Time and Cartesian 3D coordinate: t, x, y, z."""

TimePolCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimePolCoord',
    field_names=('time',) + PolCoord._fields
)
"""Time and polar coordinate: t, r, theta."""

TimeSphCoord = namedtuple( # pylint: disable=invalid-name
    typename='TimeSphCoord',
    field_names=('time',) + SphCoord._fields
)
"""Time and spherical coordinate: t, r, theta, phi."""


OMKEY_T = np.dtype(
    [
        ('string', np.int32),
        ('om', np.uint32),
        ('pmt', np.uint8),
    ]
)
"""icetray/public/icetray/OMKey.h"""


I3POSITION_T = np.dtype(
    [
        ('x', np.float64),
        ('y', np.float64),
        ('z', np.float64),
    ]
)
"""dataclasses/public/dataclasses/I3Position.h"""


I3DIRECTION_T = np.dtype(
    [
        ('zenith', np.float64),
        ('azimuth', np.float64),
    ]
)
"""dataclasses/public/dataclasses/I3Direction.h"""


I3OMGEO_T = np.dtype(
    [
        ('position', I3POSITION_T),
        ('direction', I3DIRECTION_T),
        ('omtype', np.uint8),
        ('omkey', OMKEY_T),
        ('area', np.float64),
    ]
)
"""dataclasses/public/dataclasses/geometry/I3OMGeo.h"""


DOMCALVERSION_T = np.dtype(
    [
        ('major', np.int8),
        ('minor', np.int8),
        ('rev', np.int8),
    ]
)


LINEARFIT_T = np.dtype(
    [
        ('slope', np.float64),
        ('intercept', np.float64),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


# TODO: QUADRATICFIT_T = np.dtype(
#    [
#        ('quadFitA', np.float64),
#        ('quadFitB', np.float64),
#        ('quadFitC', np.float64),
#    ]
# )
# """dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


SPECHARGEDISTRIBUTION_T = np.dtype(
    [
        ('exp1_amp', np.float64),
        ('exp1_width', np.float64),
        ('exp2_amp', np.float64),
        ('exp2_width', np.float64),
        ('gaus_amp', np.float64),
        ('gaus_mean', np.float64),
        ('gaus_width', np.float64),
        ('compensation_factor', np.float64),
        ('slc_gaus_mean', np.float64),
        ('is_valid', np.bool8),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


TAUPARAM_T = np.dtype(
    [
        ('p0', np.float64),
        ('p1', np.float64),
        ('p2', np.float64),
        ('p3', np.float64),
        ('p4', np.float64),
        ('p5', np.float64),
        ('tau_frac', np.float64),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


I3DOMCALIBRATION_T = np.dtype(
    [
        #('omkey', OMKEY_T),
        ('temperature', np.float64),  # Kelvin
        ('fadc_gain', np.float64),
        ('fadc_baseline_fit', LINEARFIT_T),
        ('fadc_beacon_baseline', np.float64),
        ('fadc_delta_t', np.float64),
        ('front_end_impedance', np.float64),
        ('tau_parameters', TAUPARAM_T),
        # where are ampGains_[3]?
        #('atwd_gain', <icecube.dataclasses._atwd_gain_proxy>),
        #('atwd_delta_t', <icecube.dataclasses._atwd_gain_proxy>),

        #('atwd_freq_fit', <icecube.dataclasses._atwd_freq_fit_proxy>),
        #('atwd_bin_calib_slope', <icecube.dataclasses._atwd_bin_calib_slope_proxy>),

        ('transit_time', LINEARFIT_T),
        ('hv_gain_fit', LINEARFIT_T),

        ('dom_cal_version', DOMCALVERSION_T),

        #('atwd_beacon_baseline', <icecube.dataclasses._atwd_beacon_baseline_proxy>),
        ('spe_disc_calib', LINEARFIT_T),
        ('mpe_disc_calib', LINEARFIT_T),
        ('pmt_disc_calib', LINEARFIT_T),

        ('relative_dom_eff', np.float64),

        ('dom_noise_rate', np.float64),  # 1/ns
        ('dom_noise_thermal_rate', np.float64),
        ('dom_noise_decay_rate', np.float64),
        ('dom_noise_scintillation_mean', np.float64),
        ('dom_noise_scintillation_sigma', np.float64),
        ('dom_noise_scintillation_hits', np.float64),

        ('combined_spe_charge_distribution', SPECHARGEDISTRIBUTION_T),

        ('mean_atwd_charge', np.float64),
        ('mean_fadc_charge', np.float64),

        ('toroid_type', np.uint8),  # icecube.dataclasses.ToroidType.NEW_TOROID),
        ('is_mean_atwd_charge_valid', np.bool8),
        ('is_mean_fadc_charge_valid', np.bool8),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3DOMCalibration.h"""


FLAT_I3DOMCALIBRATION_T = np.dtype(
    [
        ('omkey', OMKEY_T),
        ('dom_cal', I3DOMCALIBRATION_T),
    ]
)


I3VEMCALIBRATION_T = np.dtype(
    [
        ('pe_per_vem', np.float64),
        ('mu_peak_width', np.float64),
        ('hglg_cross_over', np.float64),
        ('corr_factor', np.float64),
    ]
)
"""dataclasses/public/dataclasses/calibration/I3VEMCalibration.h"""


FLAT_I3VEMCALIBRATION_T = np.dtype(
    [
        ('omkey', OMKEY_T),
        ('vem_cal', I3VEMCALIBRATION_T),
    ]
)


# TODO: I3CALIBRATION_T = np.dtype(
#     [
#         ('start_time', I3TIME_T),
#         ('end_time', I3TIME_T),
#         ('dom_cal', I3DOMCALIBRATIONMAP_T),
#         ('vem_cal', I3VEMCALIBRATIONMAP_T),
#     ]
# )
# """dataclasses/public/dataclasses/calibration/I3Calibration.h"""


I3DOMSTATUS_T = np.dtype(
    [
        ('omkey', OMKEY_T),
        ('cable_type', np.int8),  # icecube.dataclasses.CableType
        ('dac_fadc_ref', np.float64),
        ('dac_trigger_bias_0', np.float64),
        ('dac_trigger_bias_1', np.float64),
        ('delta_compress', np.int8),  # icecube.dataclasses.OnOff
        ('dom_gain_type', np.int8),  # icecube.dataclasses.DOMGain
        ('fe_pedestal', np.float64),
        # ('identity', <bound method I3DOMStatus.identity>) ... ???,
        ('lc_mode', np.int8),  # icecube.dataclasses.LCMode
        ('lc_span', np.uint32),
        ('lc_window_post', np.float64),
        ('lc_window_pre', np.float64),
        ('mpe_threshold', np.float64),
        ('n_bins_atwd_0', np.uint32),
        ('n_bins_atwd_1', np.uint32),
        ('n_bins_atwd_2', np.uint32),
        ('n_bins_atwd_3', np.uint32),
        ('n_bins_fadc', np.uint32),
        ('pmt_hv', np.float64),
        ('slc_active', np.bool8),
        ('spe_threshold', np.float64),
        ('status_atwd_a', np.int8),  # icecube.dataclasses.OnOff
        ('status_atwd_b', np.int8),  # icecube.dataclasses.OnOff
        ('status_fadc', np.int8),  # icecube.dataclasses.OnOff
        ('trig_mode', np.int8),  # icecube.dataclasses.TrigMode
        ('tx_mode', np.int8),  # icecube.dataclasses.LCMode
    ]
)

DOMINFO_T = np.dtype(
    [
        ('sd_idx', np.uint32),
        ('operational', np.bool),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('quantum_efficiency', np.float32),
        ('noise_rate_per_ns', np.float32)
    ]
)

EVT_DOM_INFO_T = np.dtype(
    [
        ('sd_idx', np.uint32),
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('quantum_efficiency', np.float32),
        ('noise_rate_per_ns', np.float32),
        ('table_idx', np.uint32),
        ('hits_start_idx', np.uint32),
        ('hits_stop_idx', np.uint32),
        ('total_observed_charge', np.float32),
    ]
)


PULSE_T = np.dtype(
    [
        ('time', np.float64),
        ('charge', np.float32),
        ('width', np.float32),
        ('flags', np.uint8),  # icecube.dataclasses.I3RecoPulse.PulseFlags
    ]
)
"""dataclasses/public/dataclasses/physics/I3RecoPulse.h"""


FLAT_PULSE_T = np.dtype([('key', OMKEY_T), ('pulse', PULSE_T)])


I3TIMEWINDOW_T = np.dtype([('start', np.float64), ('stop', np.float64)])
"""dataclasses/public/dataclasses/I3TimeWindow.h"""


PHOTON_T = np.dtype([
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('coszen', np.float32),
    ('azimuth', np.float32),
    ('wavelength', np.float32),
])

HIT_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32)
])

#EVENT_INDEXER_T = np.dtype([
#    ('first_idx', np.uint32),
#    ('num', np.uint32)
#])

SD_INDEXER_T = np.dtype([
    ('sd_idx', np.uint32),
    ('offset', np.uint64),
    ('num', np.uint32)
])

HITS_SUMMARY_T = np.dtype([
    ('earliest_hit_time', np.float32),
    ('latest_hit_time', np.float32),
    ('average_hit_time', np.float32),
    ('total_charge', np.float32),
    ('num_hits', np.uint32),
    ('num_doms_hit', np.uint32),
    ('time_window_start', np.float32),
    ('time_window_stop', np.float32)
])

EVT_HIT_INFO_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32),
    ('event_dom_idx', np.uint32),
])

class InteractionType(enum.IntEnum):
    """Neutrino interactions are either charged current (cc) or neutral current
    (nc); integer encodings are copied from the dominant IceCube software
    convention.
    """
    # pylint: disable=invalid-name
    undefined = 0
    CC = 1
    NC = 2


class OnOff(enum.IntEnum):
    """enum OnOff from public/dataclasses/status/I3DOMStatus.h

    Representable by np.int8.

    """
    # pylint: disable=invalid-name
    Unknown = -1
    Off = 0
    On = 1


class ParticleType(enum.IntEnum):
    """Particle types as found in an `I3Particle`.

    Only Requires int32 dtype for storage.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18;
    added (and so names might not be "standard"):
        K0, K0Bar, SigmaaCPP, SigmaCP

    """
    # pylint: disable=invalid-name

    # NB: These match the PDG codes. Keep it that way!
    unknown = 0
    Gamma = 22
    EPlus = -11
    EMinus = 11
    MuPlus = -13
    MuMinus = 13
    Pi0 = 111
    PiPlus = 211
    PiMinus = -211
    K0_Long = 130
    K0 = 311
    K0Bar = -311
    KPlus = 321
    KMinus = -321
    SigmaCPP = 4222 # charmed Sigma ++
    SigmaCP = 4212 # charmed Sigma +
    Neutron = 2112
    PPlus = 2212
    PMinus = -2212
    K0_Short = 310
    Eta = 221
    Lambda = 3122
    SigmaPlus = 3222
    Sigma0 = 3212
    SigmaMinus = 3112
    Xi0 = 3322
    XiMinus = 3312
    OmegaMinus = 3334
    NeutronBar = -2112
    LambdaBar = -3122
    SigmaMinusBar = -3222
    Sigma0Bar = -3212
    SigmaPlusBar = -3112
    Xi0Bar = -3322
    XiPlusBar = -3312
    OmegaPlusBar = -3334
    DPlus = 411
    DMinus = -411
    D0 = 421
    D0Bar = -421
    DsPlus = 431
    DsMinusBar = -431
    LambdacPlus = 4122
    WPlus = 24
    WMinus = -24
    Z0 = 23
    NuE = 12
    NuEBar = -12
    NuMu = 14
    NuMuBar = -14
    TauPlus = -15
    TauMinus = 15
    NuTau = 16
    NuTauBar = -16

    # Nuclei
    He3Nucleus = 1000020030
    He4Nucleus = 1000020040
    Li6Nucleus = 1000030060
    Li7Nucleus = 1000030070
    Be9Nucleus = 1000040090
    B10Nucleus = 1000050100
    B11Nucleus = 1000050110
    C12Nucleus = 1000060120
    C13Nucleus = 1000060130
    N14Nucleus = 1000070140
    N15Nucleus = 1000070150
    O16Nucleus = 1000080160
    O17Nucleus = 1000080170
    O18Nucleus = 1000080180
    F19Nucleus = 1000090190
    Ne20Nucleus = 1000100200
    Ne21Nucleus = 1000100210
    Ne22Nucleus = 1000100220
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Mg25Nucleus = 1000120250
    Mg26Nucleus = 1000120260
    Al26Nucleus = 1000130260
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    Si29Nucleus = 1000140290
    Si30Nucleus = 1000140300
    Si31Nucleus = 1000140310
    Si32Nucleus = 1000140320
    P31Nucleus = 1000150310
    P32Nucleus = 1000150320
    P33Nucleus = 1000150330
    S32Nucleus = 1000160320
    S33Nucleus = 1000160330
    S34Nucleus = 1000160340
    S35Nucleus = 1000160350
    S36Nucleus = 1000160360
    Cl35Nucleus = 1000170350
    Cl36Nucleus = 1000170360
    Cl37Nucleus = 1000170370
    Ar36Nucleus = 1000180360
    Ar37Nucleus = 1000180370
    Ar38Nucleus = 1000180380
    Ar39Nucleus = 1000180390
    Ar40Nucleus = 1000180400
    Ar41Nucleus = 1000180410
    Ar42Nucleus = 1000180420
    K39Nucleus = 1000190390
    K40Nucleus = 1000190400
    K41Nucleus = 1000190410
    Ca40Nucleus = 1000200400
    Ca41Nucleus = 1000200410
    Ca42Nucleus = 1000200420
    Ca43Nucleus = 1000200430
    Ca44Nucleus = 1000200440
    Ca45Nucleus = 1000200450
    Ca46Nucleus = 1000200460
    Ca47Nucleus = 1000200470
    Ca48Nucleus = 1000200480
    Sc44Nucleus = 1000210440
    Sc45Nucleus = 1000210450
    Sc46Nucleus = 1000210460
    Sc47Nucleus = 1000210470
    Sc48Nucleus = 1000210480
    Ti44Nucleus = 1000220440
    Ti45Nucleus = 1000220450
    Ti46Nucleus = 1000220460
    Ti47Nucleus = 1000220470
    Ti48Nucleus = 1000220480
    Ti49Nucleus = 1000220490
    Ti50Nucleus = 1000220500
    V48Nucleus = 1000230480
    V49Nucleus = 1000230490
    V50Nucleus = 1000230500
    V51Nucleus = 1000230510
    Cr50Nucleus = 1000240500
    Cr51Nucleus = 1000240510
    Cr52Nucleus = 1000240520
    Cr53Nucleus = 1000240530
    Cr54Nucleus = 1000240540
    Mn52Nucleus = 1000250520
    Mn53Nucleus = 1000250530
    Mn54Nucleus = 1000250540
    Mn55Nucleus = 1000250550
    Fe54Nucleus = 1000260540
    Fe55Nucleus = 1000260550
    Fe56Nucleus = 1000260560
    Fe57Nucleus = 1000260570
    Fe58Nucleus = 1000260580

    # The following are fake particles used in Icetray and have no official codes
    # The section abs(code) > 2000000000 is reserved for this kind of use
    CherenkovPhoton = 2000009900
    Nu = -2000000004
    Monopole = -2000000041
    Brems = -2000001001
    DeltaE = -2000001002
    PairProd = -2000001003
    NuclInt = -2000001004
    MuPair = -2000001005
    Hadrons = -2000001006
    ContinuousEnergyLoss = -2000001111
    FiberLaser = -2000002100
    N2Laser = -2000002101
    YAGLaser = -2000002201
    STauPlus = -2000009131
    STauMinus = -2000009132
    SMPPlus = -2000009500
    SMPMinus = -2000009501


EM_CASCADE_PTYPES = (
    ParticleType.EMinus,
    ParticleType.EPlus,
    ParticleType.Brems,
    ParticleType.DeltaE,
    ParticleType.PairProd,
    ParticleType.Gamma,
    ParticleType.Pi0,
)
"""Particle types parameterized as electromagnetic cascades,
from clsim/python/GetHybridParameterizationList.py"""


HADR_CASCADE_PTYPES = (
    ParticleType.Hadrons,
    ParticleType.Neutron,
    ParticleType.PiPlus,
    ParticleType.PiMinus,
    ParticleType.K0_Long,
    ParticleType.KPlus,
    ParticleType.KMinus,
    ParticleType.PPlus,
    ParticleType.PMinus,
    ParticleType.K0_Short,
    ParticleType.Eta,
    ParticleType.Lambda,
    ParticleType.SigmaPlus,
    ParticleType.Sigma0,
    ParticleType.SigmaMinus,
    ParticleType.Xi0,
    ParticleType.XiMinus,
    ParticleType.OmegaMinus,
    ParticleType.NeutronBar,
    ParticleType.LambdaBar,
    ParticleType.SigmaMinusBar,
    ParticleType.Sigma0Bar,
    ParticleType.SigmaPlusBar,
    ParticleType.Xi0Bar,
    ParticleType.XiPlusBar,
    ParticleType.OmegaPlusBar,
    ParticleType.DPlus,
    ParticleType.DMinus,
    ParticleType.D0,
    ParticleType.D0Bar,
    ParticleType.DsPlus,
    ParticleType.DsMinusBar,
    ParticleType.LambdacPlus,
    ParticleType.WPlus,
    ParticleType.WMinus,
    ParticleType.Z0,
    ParticleType.NuclInt,
    ParticleType.TauPlus,
    ParticleType.TauMinus,
)
"""Particle types parameterized as hadronic cascades,
from clsim/CLSimLightSourceToStepConverterPPC.cxx with addition of TauPlus and
TauMinus"""


CASCADE_PTYPES = EM_CASCADE_PTYPES + HADR_CASCADE_PTYPES
"""Particle types classified as either EM or hadronic cascades"""


TRACK_PTYPES = (ParticleType.MuPlus, ParticleType.MuMinus)
"""Particle types classified as tracks"""


INVISIBLE_PTYPES = (
    ParticleType.Neutron,  # long decay time exceeds trigger window
    ParticleType.K0,
    ParticleType.K0Bar,
    ParticleType.NuE,
    ParticleType.NuEBar,
    ParticleType.NuMu,
    ParticleType.NuMuBar,
    ParticleType.NuTau,
    ParticleType.NuTauBar,
)
"""Invisible particles (at least to low-energy IceCube triggers)"""

ELECTRONS = (ParticleType.EPlus, ParticleType.EMinus)
MUONS = (ParticleType.MuPlus, ParticleType.MuMinus)
TAUS = (ParticleType.TauPlus, ParticleType.TauMinus)
NUES = (ParticleType.NuE, ParticleType.NuEBar)
NUMUS = (ParticleType.NuMu, ParticleType.NuMuBar)
NUTAUS = (ParticleType.NuTau, ParticleType.NuTauBar)
NEUTRINOS = NUES + NUMUS + NUTAUS


class PulseFlags(enum.IntEnum):
    """Pulse flags.

    Values corresponding with even powers of 2

        LC = 1
        ATWD = 2
        FADC = 4

    can be used to define a bit mask (i.e., multiple of {LC, ATWD, FADC} can be
    simultaneously true).

    If the LC bit is true, then the pulse comes from a hard local coincidence
    (HLC) hit; otherwise, the hits are soft local coincidence (SLC). .. ::

        is_hlc = (pulses["flags"] & PulseFlags.LC).astype(bool)
        is_slc = np.logical_not((pulses["flags"] & PulseFlags.LC).astype(bool))

    Scraped from dataclasses/public/dataclasses/physics/I3RecoPulse.h, 2020-02-18
    """
    # pylint: disable=invalid-name
    LC = 1
    ATWD = 2
    LC_ATWD = 3
    FADC = 4
    LC_FADC = 5
    ATWD_FADC = 6
    LC_ATWD_FADC = 7


class ParticleShape(enum.IntEnum):
    """`I3Particle` property `shape`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """
    Null = 0
    Primary = 10
    TopShower = 20
    Cascade = 30
    CascadeSegment = 31
    InfiniteTrack = 40
    StartingTrack = 50
    StoppingTrack = 60
    ContainedTrack = 70
    MCTrack = 80
    Dark = 90


class FitStatus(enum.IntEnum):
    """`I3Particle` property `fit_status`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """
    # pylint: disable=invalid-name
    NotSet = -1
    OK = 0
    PositiveLLH = 1  # NOT present in IceCube / icetray software
    Skipped = 2  # NOT present in IceCube / icetray software
    GeneralFailure = 10
    InsufficientHits = 20
    FailedToConverge = 30
    MissingSeed = 40
    InsufficientQuality = 50


class LocationType(enum.IntEnum):
    """`I3Particle` property `location`.

    Scraped from dataclasses/public/dataclasses/physics/I3Particle.h, 2019-02-18
    """
    # pylint: disable=invalid-name
    Anywhere = 0
    IceTop = 10
    InIce = 20
    InActiveVolume = 30


class TriggerConfigID(enum.IntEnum):
    """Trigger common names mapped into TriggerConfigID (or config_id) in i3
    files.

    Note that this seems to be a really unique ID for a trigger, subsuming
    TriggerTypeID and SourceID.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    script to dump enumerated values & details of each is at ::

      http://code.icecube.wisc.edu/svn/projects/trigger-sim/trunk/resources/scripts/print_trigger_configuration.py

    run via .. ::

      $I3_SRC/trigger-sim/resources/scripts/print_trigger_configuration.py -g GCDFILE

    """
    # I added "NONE" (code -1) since GCD file(s) were found with
    # TriggerKey.source == 40 (GLOBAL), TriggerKey.type == 30 (THROUGHPUT), and
    # TriggerKey.subtype == 0 (NO_SUBTYPE) have TriggerKey.config_id of None
    NONE = -1

    SMT8_IN_ICE = 1006
    SMT3_DeepCore = 1011
    SMT6_ICE_TOP = 102
    SLOP = 24002
    Cluster = 1007
    Cylinder = 21001

    # Added for 2016-2017
    Cylinder_ICE_TOP = 21002

    # Unique to 2011
    SLOP_2011 = 22005

    # Unique to IC79
    SMT3_DeepCore_IC79 = 1010


class TriggerTypeID(enum.IntEnum):
    """Trigger TypeID:  Enumeration describing what "algorithm" issued a
    trigger. More details about a specific trigger can be stored in the
    I3TriggerStatus maps as part of the detector status.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    and enumerated values (and more comments on each type) are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """
    SIMPLE_MULTIPLICITY = 0
    CALIBRATION = 10
    MIN_BIAS = 20
    THROUGHPUT = 30
    TWO_COINCIDENCE = 40
    THREE_COINCIDENCE = 50
    MERGED = 70
    SLOW_PARTICLE = 80
    FRAGMENT_MULTIPLICITY = 105
    STRING = 120
    VOLUME = 125
    SPHERE = 127
    UNBIASED = 129
    SPASE_2 = 170
    UNKNOWN_TYPE = 180


class TriggerSourceID(enum.IntEnum):
    """Trigger SourceID: Enumeration describing what "subdetector" issued a trigger.

    See docs at ::

      http://software.icecube.wisc.edu/documentation/projects/trigger-sim/trigger_config_ids.html

    and enumerated values are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """
    IN_ICE = 0
    ICE_TOP = 10
    AMANDA_TWR_DAQ = 20
    EXTERNAL = 30
    GLOBAL = 40
    AMANDA_MUON_DAQ = 50
    SPASE = 70
    UNKNOWN_SOURCE = 80


class TriggerSubtypeID(enum.IntEnum):
    """Trigger SubtypeID: Enumeration describing how a software trigger was
    orginally "configured" within the TWR DAQ trigger system.

    Enumerated values are defined in ::

      http://code.icecube.wisc.edu/svn/projects/dataclasses/trunk/public/dataclasses/TriggerKey.h

    """
    # pylint: disable=invalid-name
    NO_SUBTYPE = 0
    M18 = 50
    M24 = 100
    T0 = 150
    LASER = 200
    UNKNOWN_SUBTYPE = 250


class ExtractionError(enum.IntEnum):
    """Error codes that can be set by retro/i3processing/extract_events.py"""
    NO_ERROR = 0
    NU_CC_LEPTON_SECONDARY_MISSING = 1
    NU_NC_OUTOING_NU_MISSING = 2


class OMType(enum.IntEnum):
    """`OMType` enum

    Note this currently requires only uint8, i.e., [0, 255], for storage.

    Scraped from dataclasses/public/dataclasses/geometry/I3OMGeo.h, 2019-06-26
    SVN rev 167541
    """
    # pylint: disable=invalid-name
    UnknownType = 0
    AMANDA = 10
    IceCube = 20
    IceTop = 30
    Scintillator = 40
    IceAct = 50
    # OMType > 100 are Gen2 R&D optical modules
    PDOM = 110
    DEgg = 120
    mDOM = 130
    WOM = 140
    FOM = 150


class CableType(enum.IntEnum):
    """icecube.dataclasses.CableType"""
    # pylint: disable=invalid-name
    UnknownCableType = -1
    Terminated = 0
    Unterminated = 1


class DOMGain(enum.IntEnum):
    """icecube.dataclasses.DOMGain"""
    # pylint: disable=invalid-name
    UnknownGainType = -1
    High = 0
    Low = 1


class TrigMode(enum.IntEnum):
    """icecube.dataclasses.TrigMode"""
    # pylint: disable=invalid-name
    UnknownTrigMode = -1
    TestPattern = 0
    CPU = 1
    SPE = 2
    Flasher = 3
    MPE = 4


class LCMode(enum.IntEnum):
    """icecube.dataclasses.LCMode"""
    # pylint: disable=invalid-name
    UnknownLCMode = -1
    LCOff = 0
    UpOrDown = 1
    Up = 2
    Down = 3
    UpAndDown = 4
    SoftLC = 5


class ToroidType(enum.IntEnum):
    """icecube.dataclasses.ToroidType"""
    # pylint: disable=invalid-name
    OLD_TOROID = 0
    NEW_TOROID = 1


class CramerRaoStatus(enum.IntEnum):
    """icecube.recclasses.CramerRaoStatus; see
    recclasses/public/recclasses/CramerRaoParams.h"""
    # pylint: disable=invalid-name
    NotSet = -1
    OK = 0
    MissingInput = 10
    SingularMatrix = 20
    InsufficientHits = 30
    OneStringEvent = 40
    OtherProblems = 50


TRIGGERKEY_T = np.dtype(
    [
        ('source', np.uint8),
        ('type', np.uint8),
        ('subtype', np.uint8),
        ('config_id', np.int32),
    ]
)
"""dataclasses/public/dataclasses/physics/TriggerKey.h"""


TRIGGER_T = np.dtype(
    [
        ('time', np.float32),
        ('length', np.float32),
        ('fired', np.bool),
        ('key', TRIGGERKEY_T),
    ]
)
"""dataclasses/public/dataclasses/physics/I3Trigger.h"""


FLAT_TRIGGER_T = np.dtype(
    [
        ("level", np.uint8),
        ("parent_idx", np.int8),
        ("trigger", TRIGGER_T),
    ]
)


I3TRIGGERREADOUTCONFIG_T = np.dtype(
    [
        ('readout_time_minus', np.float64),
        ('readout_time_plus', np.float64),
        ('readout_time_offset', np.float64),
    ]
)
"""icecube.dataclasses.I3TriggerReadoutConfig"""


I3TIME_T = np.dtype([("utc_year", np.int32), ("utc_daq_time", np.int64)])
"""I3Time is defined internally by the year and daqTime (tenths of ns since the
beginning of the year). See dataclasses/public/dataclasses/I3Time.h"""


I3EVENTHEADER_T = np.dtype(
    [
        ("run_id", np.uint32),
        ("sub_run_id", np.uint32),
        ("event_id", np.uint32),
        ("sub_event_id", np.uint32),
        ("sub_event_stream", np.dtype("S20")),
        ("state", np.uint8),
        ("start_time", I3TIME_T),
        ("end_time", I3TIME_T),
    ]
)
"""See: dataclasses/public/dataclasses/physics/I3EventHeader.h"""


I3RUSAGE_T = np.dtype(
    [
        ("SystemTime", np.float64),
        ("UserTime", np.float64),
        ("WallClockTime", np.float64),
    ]
)
"""icetray/public/icetray/I3PhysicsTimer.h"""


I3PARTICLEID_T = np.dtype(
    [
        ("majorID", np.uint64),
        ("minorID", np.int32),
    ]
)
"""dataclasses/public/dataclasses/physics/I3ParticleID.h"""


I3PARTICLE_T = np.dtype(
    [
        ("id", I3PARTICLEID_T),
        ("pdg_encoding", np.int32),
        ("shape", np.uint8),
        ("pos", I3POSITION_T),
        ("dir", I3DIRECTION_T),
        ("time", np.float64),
        ("energy", np.float64),
        ("length", np.float64),
        ("speed", np.float64),
        ("fit_status", np.int8),
        ("location_type", np.uint8),
    ]
)
"""dataclasses/public/dataclasses/physics/I3Particle.h"""


I3SUPERDSTTRIGGER_T = np.dtype([("time", np.float64), ("length", np.float64)])
"""dataclasses/public/dataclasses/payload/I3SuperDSTTrigger.h"""


I3DIPOLEFITPARAMS_T = np.dtype(
    [
        ("Magnet", np.float64),
        ("MagnetX", np.float64),
        ("MagnetY", np.float64),
        ("MagnetZ", np.float64),
        ("AmpSum", np.float64),
        ("NHits", np.int32),
        ("NPairs", np.int32),
        ("MaxAmp", np.float64),
    ]
)
"""recclasses/public/recclasses/I3DipoleFitParams.h"""


FLAT_PARTICLE_T = np.dtype(
    [
        ("level", np.uint8),
        ("parent_idx", np.int16),
        ("particle", I3PARTICLE_T),
    ]
)


I3GENIERESULTDICT_SCALARS_T = np.dtype(
    [
        ("iev", np.int32),
        ("neu", np.int32),
        ("tgt", np.int32),
        ("Z", np.int32),
        ("A", np.int32),
        ("hitnuc", np.int32),
        ("hitqrk", np.int32),
        ("resid", np.bool8),
        ("sea", np.bool8),
        ("qel", np.bool8),
        ("res", np.bool8),
        ("dis", np.bool8),
        ("coh", np.bool8),
        ("dfr", np.bool8),
        ("imd", np.bool8),
        ("nuel", np.bool8),
        ("em", np.bool8),
        ("cc", np.bool8),
        ("nc", np.bool8),
        ("charm", np.bool8),
        ("neut_code", np.int32),
        ("nuance_code", np.int32),
        ("wght", np.float64),
        ("xs", np.float64),
        ("ys", np.float64),
        ("ts", np.float64),
        ("Q2s", np.float64),
        ("Ws", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("t", np.float64),
        ("Q2", np.float64),
        ("W", np.float64),
        ("Ev", np.float64),
        ("pxv", np.float64),
        ("pyv", np.float64),
        ("pzv", np.float64),
        ("En", np.float64),
        ("pxn", np.float64),
        ("pyn", np.float64),
        ("pzn", np.float64),
        ("pdgl", np.int32),
        ("El", np.float64),
        ("KEl", np.float64),
        ("pxl", np.float64),
        ("pyl", np.float64),
        ("pzl", np.float64),
        ("nfp", np.int32),
        ("nfn", np.int32),
        ("nfpip", np.int32),
        ("nfpim", np.int32),
        ("nfpi0", np.int32),
        ("nfkp", np.int32),
        ("nfkm", np.int32),
        ("nfk0", np.int32),
        ("nfem", np.int32),
        ("nfother", np.int32),
        ("nip", np.int32),
        ("nin", np.int32),
        ("nipip", np.int32),
        ("nipim", np.int32),
        ("nipi0", np.int32),
        ("nikp", np.int32),
        ("nikm", np.int32),
        ("nik0", np.int32),
        ("niem", np.int32),
        ("niother", np.int32),
        ("ni", np.int32),
        #("pdgi", <class 'icecube.dataclasses.ListInt'>),
        #("resc", <class 'icecube.dataclasses.ListInt'>),
        #("Ei", <class 'icecube.dataclasses.ListDouble'>),
        #("pxi", <class 'icecube.dataclasses.ListDouble'>),
        #("pyi", <class 'icecube.dataclasses.ListDouble'>),
        #("pzi", <class 'icecube.dataclasses.ListDouble'>),
        ("nf", np.int32),
        #("pdgf", <class 'icecube.dataclasses.ListInt'>),
        #("Ef", <class 'icecube.dataclasses.ListDouble'>),
        #("KEf", <class 'icecube.dataclasses.ListDouble'>),
        #("pxf", <class 'icecube.dataclasses.ListDouble'>),
        #("pyf", <class 'icecube.dataclasses.ListDouble'>),
        #("pzf", <class 'icecube.dataclasses.ListDouble'>),
        ("vtxx", np.float64),
        ("vtxy", np.float64),
        ("vtxz", np.float64),
        ("vtxt", np.float64),
        ("calresp0", np.float64),
        ("xsec", np.float64),
        ("diffxsec", np.float64),
        ("prob", np.float64),
        ("tgtmass", np.float64),
    ]
)
"""genie-icetray/private/genie-icetray/ConvertToGST.cxx
Reading in Python requires `from icecube imoprt genie-icetray`"""


I3LINEFITPARAMS_T = np.dtype(
    [
        ("LFVel", np.float64),
        ("LFVelX", np.float64),
        ("LFVelY", np.float64),
        ("LFVelZ", np.float64),
        ("NHits", np.int32),
    ]
)
"""recclasses/public/recclasses/I3LineFitParams.h"""

I3FILLRATIOINFO_T = np.dtype(
    [
         ('mean_distance', np.float64),
         ('rms_distance', np.float64),
         ('nch_distance', np.float64),
         ('energy_distance', np.float64),
         ('fill_radius_from_rms', np.float64),
         ('fill_radius_from_mean', np.float64),
         ('fill_radius_from_mean_plus_rms', np.float64),
         ('fillradius_from_nch', np.float64),
         ('fill_radius_from_energy', np.float64),
         ('fill_ratio_from_rms', np.float64),
         ('fill_ratio_from_mean', np.float64),
         ('fill_ratio_from_mean_plus_rms', np.float64),
         ('fillratio_from_nch', np.float64),
         ('fill_ratio_from_energy', np.float64),
         ('hit_count', np.int32),
    ]
)
"""recclasses/public/recclasses/I3FillRatioInfo.h"""


I3FINITECUTS_T = np.dtype(
    [
        ('Length', np.float64),
        ('endFraction', np.float64),
        ('startFraction', np.float64),
        ('Sdet', np.float64),
        ('finiteCut', np.float64),
        ('DetectorLength', np.float64),
    ]
)
"""recclasses/public/recclasses/I3FiniteCuts.h"""


CRAMERRAOPARAMS_T = np.dtype(
    [
         ('cramer_rao_theta', np.float64),
         ('cramer_rao_phi', np.float64),
         ('variance_theta', np.float64),
         ('variance_phi', np.float64),
         ('variance_x', np.float64),
         ('variance_y', np.float64),
         ('covariance_theta_phi', np.float64),
         ('covariance_theta_x', np.float64),
         ('covariance_theta_y', np.float64),
         ('covariance_phi_x', np.float64),
         ('covariance_phi_y', np.float64),
         ('covariance_x_y', np.float64),
         # ('cramer_rao_theta_corr', nan, float),  # obsolete
         # ('cramer_rao_phi_corr', nan, float),  # obsolete
         # ('llh_est', nan, float),  # obsolete
         # ('rllh_est', nan, float),  # obsolete
         ('status', np.int8),  # enum CramerRaoStatus
    ]
)
"""recclasses/public/recclasses/CramerRaoParams.h"""


I3DIRECTHITSVALUES_T = np.dtype(
    [
        ('n_dir_strings', np.uint32),
        ('n_dir_doms', np.uint32),
        ('n_dir_pulses', np.uint64),
        ('q_dir_pulses', np.float64),
        ('n_early_strings', np.uint32),
        ('n_early_doms', np.uint32),
        ('n_early_pulses', np.uint64),
        ('q_early_pulses', np.float64),
        ('n_late_strings', np.uint32),
        ('n_late_doms', np.uint32),
        ('n_late_pulses', np.uint64),
        ('q_late_pulses', np.float64),
        ('dir_track_length', np.float64),
        ('dir_track_hit_distribution_smoothness', np.float64),
    ]
)
"""recclasses/public/recclasses/I3DirectHitsValues.h"""


I3HITSTATISTICSVALUES_T = np.dtype(
    [
        ('cog', I3POSITION_T),
        ('cog_z_sigma', np.float64),
        ('min_pulse_time', np.float64),
        ('max_pulse_time', np.float64),
        ('q_max_doms', np.float64),
        ('q_tot_pulses', np.float64),
        ('z_min', np.float64),
        ('z_max', np.float64),
        ('z_mean', np.float64),
        ('z_sigma', np.float64),
        ('z_travel', np.float64),
    ]
)
"""recclasses/public/recclasses/I3HitStatisticsValues.h"""


I3HITMULTIPLICITYVALUES_T = np.dtype(
    [
        ('n_hit_strings', np.uint32),
        ('n_hit_doms', np.uint32),
        ('n_hit_doms_one_pulse', np.uint32),
        ('n_pulses', np.uint64),
    ]
)
"""recclasses/public/recclasses/I3HitMultiplicityValues.h"""


I3CLASTFITPARAMS_T = I3TENSOROFINERTIAFITPARAMS_T = np.dtype(
    [
        ('mineval', np.float64),
        ('evalratio', np.float64),
        ('eval2', np.float64),
        ('eval3', np.float64),
    ]
)
"""recclasses/public/recclasses/I3TensorOfInertiaFitParams.h
and recclasses/public/recclasses/I3CLastFitParams.h"""


I3VETO_T = np.dtype(
    [
        ('nUnhitTopLayers', np.int16),
        ('nLayer', np.int16),
        ('earliestLayer', np.int16),
        ('earliestOM', np.int16),
        ('earliestContainment', np.int16),
        ('latestLayer', np.int16),
        ('latestOM', np.int16),
        ('latestContainment', np.int16),
        ('mostOuterLayer', np.int16),
        ('depthHighestHit', np.float64),
        ('depthFirstHit', np.float64),
        ('maxDomChargeLayer', np.int16),
        ('maxDomChargeString', np.int16),
        ('maxDomChargeOM', np.int16),
        ('nDomsBeforeMaxDOM', np.int16),
        ('maxDomChargeLayer_xy', np.int16),
        ('maxDomChargeLayer_z', np.int16),
        ('maxDomChargeContainment', np.int16),
    ]
)
"""recclasses/public/recclasses/I3Veto.h"""


I3CSCDLLHFITPARAMS_T = np.dtype(
    [
        ('HitCount', np.int32),
        ('HitOmCount', np.int32),
        ('UnhitOmCount', np.int32),
        ('Status', np.int32),
        ('ErrT', np.float64),
        ('ErrX', np.float64),
        ('ErrY', np.float64),
        ('ErrZ', np.float64),
        ('ErrTheta', np.float64),
        ('ErrPhi', np.float64),
        ('ErrEnergy', np.float64),
        ('NegLlh', np.float64),
        ('ReducedLlh', np.float64),
    ]
)
"""recclasses/public/recclasses/I3CscdLlhFitParams.h"""


I3PORTIAEVENT_T = np.dtype(
    [
        ('TotalBestNPE', np.float64),
        ('TotalAtwdNPE', np.float64),
        ('TotalFadcNPE', np.float64),
        ('TotalNch', np.int32),
        ('AtwdNch', np.int32),
        ('FadcNch', np.int32),

        ('TotalBestNPEbtw', np.float64),
        ('TotalAtwdNPEbtw', np.float64),
        ('TotalFadcNPEbtw', np.float64),
        ('TotalNchbtw', np.int32),
        ('AtwdNchbtw', np.int32),
        ('FadcNchbtw', np.int32),

        ('FirstPulseOMKey', OMKEY_T),
        ('LastPulseOMKey', OMKEY_T),
        ('LargestNPEOMKey', OMKEY_T),

        ('FirstPulseOMKeybtw', OMKEY_T),
        ('LastPulseOMKeybtw', OMKEY_T),
    ]
)
"""recclasses/public/recclasses/I3PortiaEvent.h"""


I3STARTSTOPPARAMS_T = np.dtype(
    [
        ('LLHStartingTrack', np.float64),
        ('LLHStoppingTrack', np.float64),
        ('LLHInfTrack', np.float64),
    ]
)
"""recclasses/public/recclasses/I3StartStopParams.h"""


I3TRACKCHARACTERISTICSVALUES_T = np.dtype(
    [
        ('avg_dom_dist_q_tot_dom', np.float64),
        ('empty_hits_track_length', np.float64),
        ('track_hits_separation_length', np.float64),
        ('track_hits_distribution_smoothness', np.float64),
    ]
)
"""recclasses/public/recclasses/I3TrackCharacteristicsValues.h"""


I3FILTERRESULT_T = np.dtype(
    [('condition_passed', np.bool8), ('prescale_passed', np.bool8)]
)
"""dataclasses/public/dataclasses/physics/I3FilterResult.h"""


DSTPOSITION_T = np.dtype([('x', np.int8), ('y', np.int8), ('z', np.int8)])
"""recclasses/public/recclasses/I3DST.h"""


I3DST16_T = np.dtype(
    [
        ('n_string', np.uint8),
        ('cog', DSTPOSITION_T),
        ('n_dir', np.uint8),
        ('ndom', np.uint16),  # `nchannel_` in .h file
        ('reco_label', np.uint8),
        ('time', np.uint64),
        ('trigger_tag', np.uint16)
    ]
)
"""recclasses/public/recclasses/I3DST16.h"""


I3SANTAFITPARAMS_T = np.dtype(
    [
        ('zc', np.float64),
        ('tc', np.float64),
        ('dc', np.float64),
        ('uz', np.float64),
        ('chi2', np.float64),
        ('chi2_simple', np.float64),
        ('zc_error', np.float64),
        ('tc_error', np.float64),
        ('dc_error', np.float64),
        ('uz_error', np.float64),
        ('dof', np.int32),
        ('string', np.int32),
        ('n_calls', np.int32),
        ('fit_time', np.float64),
        ('fit_status', np.int8),
        ('fit_type', np.int8),
        # ('zenith', nan, float),  # derived from uz: u_z = -cos(zenith)
    ]
)
"""oscNext_meta/trunk/santa/public/santa/I3SantaFitParams.h"""


I3TIMECHARACTERISTICSVALUES_T = np.dtype(
    [
        ('timelength_fwhm', np.float64),
        ('timelength_last_first', np.int32),
        ('timelength_maxgap', np.int32),
        ('zpattern', np.int32),
    ]
)
"""recclasses/public/recclasses/I3TimeCharacteristicsValues.h"""


I3LOGLIKELIHOODFITPARAMS_T = np.dtype(
    [
        ('logl', np.float64),
        ('rlogl', np.float64),
        ('ndof', np.int32),
        ('nmini', np.int32),
    ]
)
"""gulliver.I3LogLikelihoodFitParams"""


START_AND_LENGTH_T = np.dtype(
    [
        ('start', np.uint64),
        ('length', np.uint16),
    ]
)


START_STOP_T = np.dtype(
    [
        ('start', np.uint64),
        ('stop', np.uint64),
    ]
)


SRC_T = np.dtype([
    ('kind', np.uint32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('photons', np.float32),
    ('dir_costheta', np.float32),
    ('dir_sintheta', np.float32),
    ('dir_phi', np.float32),
    ('dir_cosphi', np.float32),
    ('dir_sinphi', np.float32),
], align=True)
"""Each source point is described by (up to) these fields (e.g., SRC_OMNI
doesn't care what dir_* fields are)"""


NEUTRINO_T = np.dtype([
    ('pdg_encoding', np.int32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('zenith', np.float32),
    ('coszen', np.float32),
    ('azimuth', np.float32),
    ('directionality', np.float32),
    ('energy', np.float32),
    ('length', np.float32),
])


TRACK_T = np.dtype([
    ('pdg_encoding', np.int32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('zenith', np.float32),
    ('coszen', np.float32),
    ('azimuth', np.float32),
    ('directionality', np.float32),
    ('energy', np.float32),
    ('length', np.float32),
    ('stochastic_loss', np.float32),
    ('vis_em_equiv_stochastic_loss', np.float32),
])


INVALID_TRACK = np.full(shape=1, fill_value=np.nan, dtype=TRACK_T)
INVALID_TRACK['pdg_encoding'] = ParticleType.unknown

NO_TRACK = np.full(shape=1, fill_value=np.nan, dtype=TRACK_T)
NO_TRACK['pdg_encoding'] = ParticleType.unknown
NO_TRACK['energy'] = 0
NO_TRACK['vis_em_equiv_stochastic_loss'] = 0


CASCADE_T = np.dtype([
    ('pdg_encoding', np.int32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('zenith', np.float32),
    ('coszen', np.float32),
    ('azimuth', np.float32),
    ('directionality', np.float32),
    ('energy', np.float32),
    ('hadr_fraction', np.float32),
    ('em_equiv_energy', np.float32),
    ('hadr_equiv_energy', np.float32),
])

INVALID_CASCADE = np.full(shape=1, fill_value=np.nan, dtype=CASCADE_T)
INVALID_CASCADE['pdg_encoding'] = ParticleType.unknown

NO_CASCADE = np.full(shape=1, fill_value=np.nan, dtype=CASCADE_T)
NO_CASCADE['pdg_encoding'] = ParticleType.unknown
NO_CASCADE['energy'] = 0
NO_CASCADE['em_equiv_energy'] = 0
NO_CASCADE['hadr_equiv_energy'] = 0
NO_CASCADE['hadr_fraction'] = np.nan
