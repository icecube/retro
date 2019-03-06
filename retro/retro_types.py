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
    'DOM_INFO_T',
    'EVT_DOM_INFO_T',
    'PULSE_T',
    'PHOTON_T',
    'HIT_T',
    'SD_INDEXER_T',
    'HITS_SUMMARY_T',
    'EVT_HIT_INFO_T',
    'SPHER_T',
    'ParticleType',
    'ParticleShape',
    'FitStatus',
    'LocationType',
    'TriggerConfigID',
    'TriggerTypeID',
    'TriggerSourceID',
    'TriggerSubtypeID',
    'ExtractionError',
    'TRIGGER_T',
    'SRC_T',
    'TRACK_T',
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
        ('string', np.uint16),
        ('dom', np.uint16),
    ]
)

DOM_INFO_T = np.dtype(
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

EVT_DOM_INFO_T = np.dtype([
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
])

PULSE_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32),
    ('width', np.float32),
])

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
    ('total_num_hits', np.uint32),
    ('total_num_doms_hit', np.uint32),
    ('time_window_start', np.float32),
    ('time_window_stop', np.float32)
])

EVT_HIT_INFO_T = np.dtype([
    ('time', np.float32),
    ('charge', np.float32),
    ('event_dom_idx', np.uint32),
])

SPHER_T = np.dtype([
    ('zen', np.float32),
    ('az', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('sinzen', np.float32),
    ('coszen', np.float32),
    ('sinaz', np.float32),
    ('cosaz', np.float32),
])
"""type to store spherical coordinates and handy quantities"""


class InteractionType(enum.IntEnum):
    """Neutrino interactions are either charged current (cc) or neutral current
    (nc); integer encodings are copied from the dominant IceCube software
    convention.
    """
    # pylint: disable=invalid-name
    undefined = 0
    CC = 1
    NC = 2


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


TRIGGER_T = np.dtype([
    ('type', np.uint8),
    ('subtype', np.uint8),
    ('source', np.uint8),
    ('config_id', np.int32),
    ('fired', np.bool),
    ('time', np.float32),
    ('length', np.float32)
])


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


TRACK_T = np.dtype([
    ('pdg', np.int32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('zenith', np.float32),
    ('azimuth', np.float32),
    ('directionality', np.float32),
    ('energy', np.float32),
    ('length', np.float32),
    ('stochastic_loss', np.float32),
    ('vis_em_equiv_stochastic_loss', np.float32),
])

INVALID_TRACK = np.full(shape=1, fill_value=np.nan, dtype=TRACK_T)
INVALID_TRACK['pdg'] = ParticleType.unknown

NO_TRACK = np.full(shape=1, fill_value=np.nan, dtype=TRACK_T)
NO_TRACK['pdg'] = ParticleType.unknown
NO_TRACK['energy'] = 0
NO_TRACK['vis_em_equiv_stochastic_loss'] = 0


CASCADE_T = np.dtype([
    ('pdg', np.int32),
    ('time', np.float32),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('zenith', np.float32),
    ('azimuth', np.float32),
    ('directionality', np.float32),
    ('energy', np.float32),
    ('hadr_fraction', np.float32),
    ('em_equiv_energy', np.float32),
    ('hadr_equiv_energy', np.float32),
])

INVALID_CASCADE = np.full(shape=1, fill_value=np.nan, dtype=CASCADE_T)
INVALID_CASCADE['pdg'] = ParticleType.unknown

NO_CASCADE = np.full(shape=1, fill_value=np.nan, dtype=CASCADE_T)
NO_CASCADE['pdg'] = ParticleType.unknown
NO_CASCADE['energy'] = 0
NO_CASCADE['em_equiv_energy'] = 0
NO_CASCADE['hadr_equiv_energy'] = 0
NO_CASCADE['hadr_fraction'] = np.nan
