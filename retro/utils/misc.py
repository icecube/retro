# pylint: disable=wrong-import-position, invalid-name

"""
Miscellaneous utilites: file I/O, hashing, etc.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    expand
    mkdir
    wstdout
    wstderr
    force_little_endian
    hash_obj
    test_hash_obj
    get_file_md5
    convert_to_namedtuple
'''.split()

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


import base64
from collections import Iterable, Mapping, Sequence
import cPickle as pickle
import errno
import hashlib
from numbers import Number
from os import makedirs
from os.path import abspath, dirname, expanduser, expandvars
import struct
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import HYPO_PARAMS_T, const, types


ZSTD_EXTENSIONS = ('zstd', 'zstandard', 'zst')
"""Extensions recognized as zstandard-compressed files"""

COMPR_EXTENSIONS = ZSTD_EXTENSIONS
"""Extensions recognized as a compressed file"""


def expand(p):
    """Fully expand a path.

    Parameters
    ----------
    p : string
        Path to expand

    Returns
    -------
    e : string
        Expanded path

    """
    return abspath(expanduser(expandvars(p)))


def mkdir(d, mode=0o0770):
    """Simple wrapper around os.makedirs to create a directory but not raise an
    exception if the dir already exists

    Parameters
    ----------
    d : string
        Directory path
    mode : integer
        Permissions on created directory; see os.makedirs for details.
    warn : bool
        Whether to warn if directory already exists.

    """
    try:
        makedirs(d, mode=mode)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


# TODO: add other compression algos (esp. bz2 and gz)
def get_decompressd_fobj(fpath):
    """Open a file directly if uncompressed or decompress if zstd compression
    has been applied.

    Parameters
    ----------
    fpath : string

    Returns
    -------
    fobj : file-like object

    """
    fpath = abspath(retro.expand(fpath))
    if not isfile(fpath):
        raise ValueError('Not a file: `fpath`="{}"'.format(fpath))
    _, ext = splitext(fpath)
    ext = ext.lstrip('.').lower()
    if ext in retro.ZSTD_EXTENSIONS:
        # -c sends decompressed output to stdout
        proc = Popen(['zstd', '-d', '-c', fpath], stdout=PIPE)
        # Read from stdout
        (proc_stdout, _) = proc.communicate()
        # Give the string from stdout a file-like interface
        fobj = StringIO(proc_stdout)
    elif ext in ('fits',):
        fobj = open(fpath, 'rb')
    else:
        raise ValueError('Unhandled extension "{}"'.format(ext))
    return fobj


def wstdout(s):
    """Write `s` to stdout and flush the buffer"""
    sys.stdout.write(s)
    sys.stdout.flush()


def wstderr(s):
    """Write `s` to stderr and flush the buffer"""
    sys.stderr.write(s)
    sys.stderr.flush()


def force_little_endian(x):
    """Convert a numpy ndarray to little endian if it isn't already.

    E.g., use when loading from FITS files since that spec is big endian, while
    most CPUs (e.g. Intel) are little endian.

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    x : numpy.ndarray
        Same as input `x` with byte order swapped if necessary.

    """
    if np.isscalar(x):
        return x
    if x.dtype.byteorder == '>':
        x = x.byteswap().newbyteorder()
    return x


def hash_obj(obj, prec=None, fmt='hex'):
    """Hash an object (recursively), sorting dict keys first and (optionally)
    rounding floating point values to ensure consistency between invocations on
    effectively-equivalent objects.

    Parameters
    ----------
    obj
        Object to hash

    prec : None, np.float32, or np.float64
        Precision to enforce on numeric values

    fmt : string, one of {'int', 'hex', 'base64'}
        Format to use for hash

    Returns
    -------
    hash_val : int or string
        Hash value, where type is determined by `fmt`

    """
    if isinstance(obj, Mapping):
        hashable = []
        for key in sorted(obj.keys()):
            hashable.append((key, hash_obj(obj[key], prec=prec)))
        hashable = pickle.dumps(hashable)
    elif isinstance(obj, np.ndarray):
        if prec is not None:
            obj = obj.astype(prec)
        hashable = obj.tobytes()
    elif isinstance(obj, Number):
        if prec is not None:
            obj = prec(obj)
        hashable = obj
    elif isinstance(obj, basestring):
        hashable = obj
    elif isinstance(obj, Iterable):
        hashable = tuple(hash_obj(x, prec=prec) for x in obj)
    else:
        raise ValueError('`obj`="{}" is unhandled type {}'
                         .format(obj, type(obj)))

    if not isinstance(hashable, basestring):
        hashable = pickle.dumps(hashable, pickle.HIGHEST_PROTOCOL)

    md5hash = hashlib.md5(hashable)
    if fmt == 'hex':
        hash_val = md5hash.hexdigest()#[:16]
    elif fmt == 'int':
        hash_val, = struct.unpack('<q', md5hash.digest()[:8])
    elif fmt == 'base64':
        hash_val = base64.b64encode(md5hash.digest()[:8], '+-')
    else:
        raise ValueError('Unrecognized `fmt`: "%s"' % fmt)

    return hash_val


def test_hash_obj():
    """Unit tests for `hash_obj` function"""
    obj = {'x': {'one': {1:[1, 2, {1:1, 2:2}], 2:2}, 'two': 2}, 'y': {1:1, 2:2}}
    print('base64:', hash_obj(obj, fmt='base64'))
    print('int   :', hash_obj(obj, fmt='int'))
    print('hex   :', hash_obj(obj, fmt='hex'))
    obj = {
        'r_binning_kw': {'n_bins': 200, 'max': 400, 'power': 2, 'min': 0},
        't_binning_kw': {'n_bins': 300, 'max': 3000, 'min': 0},
        'costhetadir_binning_kw': {'n_bins': 40, 'max': 1, 'min': -1},
        'costheta_binning_kw': {'n_bins': 40, 'max': 1, 'min': -1},
        'deltaphidir_binning_kw': {'n_bins': 80, 'max': 3.141592653589793,
                                   'min': -3.141592653589793},
        'tray_kw_to_hash': {'DisableTilt': True, 'PhotonPrescale': 1,
                            'IceModel': 'spice_mie',
                            'Zenith': 3.141592653589793, 'NEvents': 1,
                            'Azimuth': 0.0, 'Sensor': 'none',
                            'PhotonSource': 'retro'}
    }
    print('hex   :', hash_obj(obj, fmt='hex'))
    obj['r_binning_kw']['n_bins'] = 201
    print('hex   :', hash_obj(obj, fmt='hex'))


def get_file_md5(fpath, blocksize=2**20):
    """Get a file's MD5 checksum.

    Code from stackoverflow.com/a/1131255

    Parameters
    ----------
    fpath : string
        Path to file

    blocksize : int
        Read file in chunks of this many bytes

    Returns
    -------
    md5sum : string
        32-characters representing hex MD5 checksum

    """
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            md5.update(buf)
    return md5.hexdigest()


def convert_to_namedtuple(val, nt_type):
    """Convert ``val`` to a namedtuple of type ``nt_type``.

    If ``val`` is:
    * ``nt_type``: return without conversion
    * Mapping: instantiate an ``nt_type`` via ``**val``
    * Iterable or Sequence: instantiate an ``nt_type`` via ``*val``

    Parameters
    ----------
    val : nt_type, Mapping, Iterable, or Sequence
        Value to be converted

    nt_type : namedtuple type
        Namedtuple type (class)

    Returns
    -------
    nt_val : nt_type
        ``val`` converted to ``nt_type``

    Raises
    ------
    TypeError
        If ``val`` is not one of the above-specified types.

    """
    if isinstance(val, nt_type):
        return val

    if isinstance(val, Mapping):
        return nt_type(**val)

    if isinstance(val, (Iterable, Sequence)):
        return nt_type(*val)

    raise TypeError('Cannot convert %s to %s' % (type(val), nt_type))


# -- Retro-specific functions -- #


def event_to_hypo_params(event, hypo_params_t=HYPO_PARAMS_T):
    """Convert an event to hypothesis params, for purposes of defining "truth"
    hypothesis.

    For now, only works with HypoParams8D.

    Parameters
    ----------
    event : likelihood.Event namedtuple

    hypo_params_t : retro.types.HypoParams8D or retro.types.HypoParams8D

    Returns
    -------
    params : hypo_params_t namedtuple

    """
    assert hypo_params_t is types.HypoParams8D

    track_energy = event.track.energy
    cascade_energy = event.cascade.energy
    #if event.interaction == 1: # charged current
    #    track_energy = event.neutrino.energy
    #    cascade_energy = 0
    #else: # neutral current (2)
    #    track_energy = 0
    #    cascade_energy = event.neutrino.energy

    hypo_params = hypo_params_t(
        t=event.neutrino.t,
        x=event.neutrino.x,
        y=event.neutrino.y,
        z=event.neutrino.z,
        track_azimuth=event.neutrino.azimuth,
        track_zenith=event.neutrino.zenith,
        track_energy=track_energy,
        cascade_energy=cascade_energy
    )

    return hypo_params


def hypo_to_track_params(hypo_params):
    """Extract track params from hypo params.

    Parameters
    ----------
    hypo_params : HYPO_PARAMS_T namedtuple

    Returns
    -------
    track_params : retro.types.TrackParams namedtuple

    """
    track_params = types.TrackParams(
        t=hypo_params.t,
        x=hypo_params.x,
        y=hypo_params.y,
        z=hypo_params.z,
        zenith=hypo_params.track_zenith,
        azimuth=hypo_params.track_azimuth,
        energy=hypo_params.track_energy
    )
    return track_params


def generate_anisotropy_str(anisotropy):
    """Generate a string from anisotropy specification parameters.

    Parameters
    ----------
    anisotropy : None or tuple of values

    Returns
    -------
    anisotropy_str : string

    """
    if anisotropy is None:
        anisotropy_str = 'none'
    else:
        anisotropy_str = '_'.join(str(param) for param in anisotropy)
    return anisotropy_str


def generate_unique_ids(events):
    """Generate unique IDs from `event` fields because people are lazy
    inconsiderate assholes.

    Parameters
    ----------
    events : array of int

    Returns
    -------
    uids : array of int

    """
    uids = (
        events
        + 1e7 * np.cumsum(np.concatenate(([0], np.diff(events) < 0)))
    ).astype(int)
    return uids


def get_primary_interaction_str(event):
    """Produce simple string representation of event's primary neutrino and
    interaction type (if present).

    Parameters
    ----------
    event : Event namedtuple

    Returns
    -------
    flavintstr : string

    """
    pdg = int(event.neutrino.pdg)
    inter = int(event.interaction)
    return const.PDG_INTER_STR[(pdg, inter)]


def get_primary_interaction_tex(event):
    """Produce latex representation of event's primary neutrino and interaction
    type (if present).

    Parameters
    ----------
    event : Event namedtuple

    Returns
    -------
    flavinttex : string

    """
    pdg = int(event.neutrino.pdg)
    inter = int(event.interaction)
    return const.PDG_INTER_TEX[(pdg, inter)]


if __name__ == '__main__':
    test_hash_obj()
