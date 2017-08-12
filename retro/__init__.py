"""
Basic module-wide definitions and simple types (namedtuples).
"""


from collections import namedtuple

import numpy as np


__all__ = ['FTYPE', 'SPEED_OF_LIGHT_M_PER_NS', 'TWO_PI', 'PI_BY_TWO',
           'HypoParams8D', 'HypoParams10D', 'HYPO_PARAMS_T', 'TrackParams',
           'Event', 'Pulses', 'PhotonInfo', 'BinningCoords']


FTYPE = np.float32

SPEED_OF_LIGHT_M_PER_NS = FTYPE(0.299792458)
"""Speed of light in units of m/ns"""

TWO_PI = FTYPE(2*np.pi)
PI_BY_TWO = FTYPE(np.pi / 2)

HypoParams8D = namedtuple(typename='HypoParams8D', # pylint: disable=invalid-name
                          field_names=('t', 'x', 'y', 'z', 'track_zenith',
                                       'track_azimuth', 'track_energy',
                                       'cascade_energy'))

HypoParams10D = namedtuple(typename='HypoParams10D', # pylint: disable=invalid-name
                           field_names=('t', 'x', 'y', 'z', 'track_azimuth',
                                        'track_zenith', 'track_energy',
                                        'cascade_azimuth', 'cascade_zenith',
                                        'cascade_energy'))

HYPO_PARAMS_T = HypoParams8D

TrackParams = namedtuple(typename='TrackParams', # pylint: disable=invalid-name
                         field_names=('t', 'x', 'y', 'z', 'azimuth', 'zenith',
                                      'length'))


Event = namedtuple(typename='Event', # pylint: disable=invalid-name
                   field_names=('event', 'pulses', 'interaction', 'neutrino',
                                'track', 'cascade', 'ml_reco', 'spe_reco'))

Pulses = namedtuple(typename='Pulses', # pylint: disable=invalid-name
                    field_names=('strings', 'oms', 'times', 'charges'))

PhotonInfo = namedtuple(typename='PhotonInfo', # pylint: disable=invalid-name
                        field_names=('count', 'theta', 'phi', 'length'))
"""Intended to contain dictionaries with DOM depth number as keys"""

BinningCoords = namedtuple(typename='BinningCoords', # pylint: disable=invalid-name
                           field_names=('t', 'r', 'theta', 'phi'))


