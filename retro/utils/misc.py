# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

"""
Miscellaneous utilites: file I/O, hashing, etc.
"""

from __future__ import absolute_import, division, print_function

__all__ = [
    'ZSTD_EXTENSIONS',
    'COMPR_EXTENSIONS',
    'LazyLoader',
    'expand',
    'mkdir',
    'get_decompressd_fobj',
    'wstdout',
    'wstderr',
    'force_little_endian',
    'hash_obj',
    'test_hash_obj',
    'get_file_md5',
    'sort_dict',
    'convert_to_namedtuple',
    'check_kwarg_keys',
    'validate_and_convert_enum',
    'hrlist2list',
    'hr_range_formatter',
    'list2hrlist',
    'generate_anisotropy_str',
    'generate_unique_ids',
    'get_primary_interaction_str',
    'get_primary_interaction_tex',
    'get_partial_match_expr',
    'AZ_REGEX',
    'ZEN_REGEX',
    'deduce_sph_pairs',
    'RE_INVALID_CHARS',
    'RE_LEADING_INVALID',
    'make_valid_python_name',
    'join_struct_arrays',
    'NSORT_RE',
    'nsort_key_func',
    'set_explicit_dtype',
    'dict2struct',
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

import base64
from collections import Iterable, OrderedDict, Mapping, Sequence
import errno
import hashlib
from numbers import Integral, Number
from os import makedirs
from os.path import abspath, dirname, expanduser, expandvars, isfile, splitext
import pickle
import re
import struct
from subprocess import Popen, PIPE
import sys

import enum
import numpy as np
from six import BytesIO, string_types
from six.moves import map, range

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import const


ZSTD_EXTENSIONS = ('zstd', 'zstandard', 'zst')
"""Extensions recognized as zstandard-compressed files"""

COMPR_EXTENSIONS = ZSTD_EXTENSIONS
"""Extensions recognized as a compressed file"""


class LazyLoader(object):
    """Lazily load a pickled datasource only when its data or SHA-256 sum is
    requested.

    Parameters
    ----------
    datasource : str
        Path to the pickle file; must exist at time of instantiation (though
        not guaranteed to exist at first access time)

    """
    def __init__(self, datasource):
        self._datasource = abspath(expanduser(expandvars(datasource)))
        self._data = None
        self._sha256 = None
        self._is_loaded = False

    def _load_data(self):
        sdata = open(self.datasource).read()
        self._sha256 = hashlib.sha256(sdata).hexdigest()
        self._data = pickle.loads(sdata)

    @property
    def datasource(self):
        """Absolute path to file from which data is loaded"""
        return self._datasource

    @property
    def data(self):
        """Un-pickled contents of the file"""
        if not self._is_loaded:
            self._load_data()
        return self._data

    @property
    def sha256(self):
        """SHA256 sum of the file"""
        if not self._is_loaded:
            self._load_data()
        return self._sha256


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
    fpath = abspath(expand(fpath))
    if not isfile(fpath):
        raise ValueError('Not a file: `fpath`="{}"'.format(fpath))
    _, ext = splitext(fpath)
    ext = ext.lstrip('.').lower()
    if ext in ZSTD_EXTENSIONS:
        # -c sends decompressed output to stdout
        proc = Popen(['zstd', '-d', '-c', fpath], stdout=PIPE)
        # Read from stdout
        (proc_stdout, _) = proc.communicate()
        # Give the string from stdout a file-like interface
        fobj = BytesIO(proc_stdout)
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
    elif isinstance(obj, str):
        hashable = obj
    elif isinstance(obj, Iterable):
        hashable = tuple(hash_obj(x, prec=prec) for x in obj)
    else:
        raise ValueError('`obj`="{}" is unhandled type {}'
                         .format(obj, type(obj)))

    if not isinstance(hashable, str):
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


def sort_dict(d):
    """Return an OrderedDict like `d` but with sorted keys.

    Parameters
    ----------
    d : mapping

    Returns
    -------
    od : OrderedDict

    """
    return OrderedDict([(k, d[k]) for k in sorted(d.keys())])


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


def check_kwarg_keys(required_keys, provided_kwargs, meta_name, message_pfx):
    """Check that provided kwargs' keys match exactly those required.

    Raises
    ------
    TypeError
        if there are too few or too many keys provided

    """
    provided_keys = provided_kwargs.keys()

    provided_set = set(provided_keys)
    required_set = set(required_keys)

    missing_set = required_set.difference(provided_set)
    excess_set = provided_set.difference(required_set)
    missing_keys = [k for k in required_keys if k in missing_set]
    excess_keys = [k for k in provided_keys if k in excess_set]

    kwarg_error_strings = []
    if missing_keys:
        kwarg_error_strings.append("missing {} {}".format(meta_name, missing_keys))
    if excess_keys:
        kwarg_error_strings.append("excess {} {}".format(meta_name, excess_keys))
    if kwarg_error_strings:
        raise TypeError("{} ".format(message_pfx) + " and ".join(kwarg_error_strings))


def validate_and_convert_enum(val, enum_type, none_evaluates_to=None):
    """Validate `val` and, if valid, convert to the `enum_type` specified.

    Validation proceeds via the following rules:
        * If `val` is None and `none_evaluates_to` is specified, return the value
          of `none_evaluates_to`
        * If `val` is an enum, only accept it if it is of type `enum_type`.
        * If `val` is a string, lookup the corresponding enum in `enum_type` by
          attribute name (trying also lowercase and uppercase versions of `val`
        * Otherwise, attempt to extract the enum corresponding to `val` by
          calling the `enum_type` with `val`, i.e., ``enum_type(val)``

    Parameters
    ----------
    val : numeric, string, or `enum_type`
    enum_type : enum
    none_evaluates_to : enum, optional

    Returns
    -------
    enum : `enum_type`

    """
    if val is None:
        val = none_evaluates_to

    if isinstance(type(val), enum.EnumMeta) and not isinstance(val, enum_type):
        raise TypeError(
            "if enum, `val` must be a {}; got {} instead".format(enum_type, type(val))
        )

    if isinstance(val, string_types):
        if hasattr(enum_type, val):
            val = getattr(enum_type, val)
        elif hasattr(enum_type, val.lower()):
            val = getattr(enum_type, val.lower())
        elif hasattr(enum_type, val.upper()):
            val = getattr(enum_type, val.upper())

    val = enum_type(val)

    return val


WHITESPACE_RE = re.compile(r'\s')

NUMBER_RESTR = r'((?:-|\+){0,1}[0-9.]+(?:e(?:-|\+)[0-9.]+){0,1})'

HRGROUP_RESTR = (
    NUMBER_RESTR
    + r'(?:-' + NUMBER_RESTR
    + r'(?:\:' + NUMBER_RESTR + r'){0,1}'
    + r'){0,1}'
)

HRGROUP_RE = re.compile(HRGROUP_RESTR, re.IGNORECASE)
"""RE str for matching signed, unsigned, and sci.-not. ("1e10") numbers."""

IGNORE_CHARS_RE = re.compile(r'[^0-9e:.,;+-]', re.IGNORECASE)


def _hrgroup2list(hrgroup):
    def isint(num):
        """Test whether a number is *functionally* an integer"""
        try:
            return int(num) == np.float32(num)
        except ValueError:
            return False

    def num_to_float_or_int(num):
        """Return int if number is effectively int, otherwise return float"""
        try:
            if isint(num):
                return int(num)
        except (ValueError, TypeError):
            pass
        return np.float32(num)

    # Strip all whitespace, brackets, parens, and other ignored characters from
    # the group string
    hrgroup = IGNORE_CHARS_RE.sub('', hrgroup)
    if (hrgroup is None) or (hrgroup == ''):
        return []
    num_str = HRGROUP_RE.match(hrgroup).groups()
    range_start = num_to_float_or_int(num_str[0])

    # If no range is specified, just return the number
    if num_str[1] is None:
        return [range_start]

    range_stop = num_to_float_or_int(num_str[1])
    if num_str[2] is None:
        step_size = 1 if range_stop >= range_start else -1
    else:
        step_size = num_to_float_or_int(num_str[2])
    all_ints = isint(range_start) and isint(step_size)

    # Make an *INCLUSIVE* list (as best we can considering floating point mumbo
    # jumbo)
    n_steps = np.clip(
        np.floor(np.around(
            (range_stop - range_start)/step_size,
            decimals=12,
        )),
        a_min=0, a_max=np.inf
    )
    lst = np.linspace(range_start, range_start + n_steps*step_size, n_steps+1)
    if all_ints:
        lst = lst.astype(np.int)

    return lst.tolist()


def hrlist2list(hrlst):
    """Convert human-readable string specifying a list of numbers to a Python
    list of numbers.

    Parameters
    ----------
    hrlist : string

    Returns
    -------
    lst : list of numbers

    """
    groups = re.split(r'[,; _]+', WHITESPACE_RE.sub('', hrlst))
    lst = []
    if not groups:
        return lst
    for group in groups:
        lst.extend(_hrgroup2list(group))
    return lst


def hr_range_formatter(start, end, step):
    """Format a range (sequence) in a simple and human-readable format by
    specifying the range's starting number, ending number (inclusive), and step
    size.

    Parameters
    ----------
    start, end, step : numeric

    Notes
    -----
    If `start` and `end` are integers and `step` is 1, step size is omitted.

    The format does NOT follow Python's slicing syntax, in part because the
    interpretation is meant to differ; e.g.,
        '0-10:2' includes both 0 and 10 with step size of 2
    whereas
        0:10:2 (slicing syntax) excludes 10

    Numbers are converted to integers if they are equivalent for more compact
    display.

    Examples
    --------
    >>> hr_range_formatter(start=0, end=10, step=1)
    '0-10'
    >>> hr_range_formatter(start=0, end=10, step=2)
    '0-10:2'
    >>> hr_range_formatter(start=0, end=3, step=8)
    '0-3:8'
    >>> hr_range_formatter(start=0.1, end=3.1, step=1.0)
    '0.1-3.1:1'

    """
    if int(start) == start:
        start = int(start)
    if int(end) == end:
        end = int(end)
    if int(step) == step:
        step = int(step)
    if int(start) == start and int(end) == end and step == 1:
        return '{}-{}'.format(start, end)
    return '{}-{}:{}'.format(start, end, step)


def list2hrlist(lst):
    """Convert a list of numbers to a compact and human-readable string.

    Parameters
    ----------
    lst : sequence

    Notes
    -----
    Adapted to make scientific notation work correctly from [1].

    References
    ----------
    [1] http://stackoverflow.com/questions/9847601 user Scott B's adaptation to
        Python 2 of Rik Poggi's answer to his question

    Examples
    --------
    >>> list2hrlist([0, 1])
    '0,1'
    >>> list2hrlist([0, 3])
    '0,3'
    >>> list2hrlist([0, 1, 2])
    '0-2'
    >>> utils.list2hrlist([0.1, 1.1, 2.1, 3.1])
    '0.1-3.1:1'
    >>> list2hrlist([0, 1, 2, 4, 5, 6, 20])
    '0-2,4-6,20'

    """
    if isinstance(lst, Number):
        lst = [lst]
    lst = sorted(lst)
    rtol = np.finfo(np.float32).resolution  # pylint: disable=no-member
    n = len(lst)
    result = []
    scan = 0
    while n - scan > 2:
        step = lst[scan + 1] - lst[scan]
        if not np.isclose(lst[scan + 2] - lst[scan + 1], step, rtol=rtol):
            result.append(str(lst[scan]))
            scan += 1
            continue
        for j in range(scan+2, n-1):
            if not np.isclose(lst[j+1] - lst[j], step, rtol=rtol):
                result.append(hr_range_formatter(lst[scan], lst[j], step))
                scan = j+1
                break
        else:
            result.append(hr_range_formatter(lst[scan], lst[-1], step))
            return ','.join(result)
    if n - scan == 1:
        result.append(str(lst[scan]))
    elif n - scan == 2:
        result.append(','.join(map(str, lst[scan:])))

    return ','.join(result)


# -- Retro-specific functions -- #


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


def get_partial_match_expr(word, minchars):
    """Generate regex sub-expression for matching first `minchars` of a word or
    `minchars` + 1 of the word, or `minchars` + 2, ....

    Adapted from user sberry at https://stackoverflow.com/a/13405331

    Parameters
    ----------
    word : str
    minchars : int

    Returns
    -------
    expr : str

    """
    return '|'.join(word[:i] for i in range(len(word), minchars - 1, -1))


AZ_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("azimuthal", 2)))

ZEN_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("zenith", 3)))

#COSZEN_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("coszenith", 4)))
#
#CZEN_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("czenith", 2)))
#
#COSINEZEN_REGEX = re.compile(r"(.*)({})(.*)".format(get_partial_match_expr("cosinezenith", 7)))


def deduce_sph_pairs(param_names):
    """Attempt to deduce which param names represent (azimuth, zenith) pairs.

    Parameters
    ----------
    param_names : iterable of strings

    Returns
    -------
    sph_pairs : tuple of 2-tuples of strings
        Sub-tuples are sorted to ensure result is the same regardless of the order
        of `param_names`

    Notes
    -----
    Works by looking for the prefix and suffix (if any) surrounding
      "az", "azi", ..., or "azimuthal"
    to match the prefix and suffix (if any) surrounding
      "zen", "zeni", ..., or "zenith"

    Examples
    --------
    >>> deduce_sph_pairs(("x", "azimuth", "zenith", "cascade_azimuth", "cascade_zenith"))
    (('azimuth', 'zenith'), ('cascade_azimuth', 'cascade_zenith'))

    """
    az_pfxs_and_sfxs = OrderedDict()
    zen_pfxs_and_sfxs = OrderedDict()
    for param_name in param_names:
        match = AZ_REGEX.match(param_name)
        if match:
            groups = match.groups()
            az_pfxs_and_sfxs[param_name] = (groups[0], groups[2])
            continue
        match = ZEN_REGEX.match(param_name)
        if match:
            groups = match.groups()
            zen_pfxs_and_sfxs[param_name] = (groups[0], groups[2])
            continue

    sph_pairs = []
    matched_params = []
    for az_param_name, az_pfx_and_sfx in az_pfxs_and_sfxs.items():
        zen_match_found = False
        for zen_param_name, zen_pfx_and_sfx in zen_pfxs_and_sfxs.items():
            if zen_pfx_and_sfx == az_pfx_and_sfx and zen_param_name != az_param_name:
                zen_match_found = True
                pair = (az_param_name, zen_param_name)
                sph_pairs.append(pair)
                matched_params.extend(pair)
        if not zen_match_found:
            print(
                'Warning: azimuthal parameter "{}" has no {{cos}}zenith counterpart'
                .format(az_param_name)
            )

    for zen_param_name in zen_pfxs_and_sfxs.keys():
        if zen_param_name not in matched_params:
            print(
                'Warning: {{cos}}zenith parameter "{}" has no azimuthal counterpart'
                .format(zen_param_name)
            )

    return tuple(sorted(sph_pairs))


RE_INVALID_CHARS = re.compile('[^0-9a-zA-Z_]')
"""Invalid Python name if chars after first aren't one of 0-9, a-z, A-Z, or
underscore"""

RE_LEADING_INVALID = re.compile('^[^a-zA-Z_]+')
"""Invalid Python name if first char isn't a-z or A-Z"""


def make_valid_python_name(name):
    """Make a name a valid Python identifier by removing illegal chars.

    From user Triptych at http://stackoverflow.com/questions/3303312

    Parameters
    ----------
    name : string

    Returns
    -------
    valid_name : string

    """
    # Remove invalid characters
    name = RE_INVALID_CHARS.sub('', name)
    # Remove leading characters until we find a letter or underscore
    name = RE_LEADING_INVALID.sub('', name)
    return name


def join_struct_arrays(arrays):
    """"Horizontal" join of structured arrays: Combine into one structured
    array whose fields are taken from each component array.

    Code from user Sven Marnach, https://stackoverflow.com/a/5355974

    Parameters
    ----------
    arrays : sequence of numpy.arrays with struct dtypes

    Returns
    -------
    array : numpy.array with struct dtype

    """
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset : offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


NSORT_RE = re.compile(r'(\d+)')


def nsort_key_func(s):
    """Use as the `key` argument to the `sorted` function or `sort` method.

    Code adapted from nedbatchelder.com/blog/200712/human_sorting.html#comments

    Examples
    --------
    >>> l = ['f1.10.0.txt', 'f1.01.2.txt', 'f1.1.1.txt', 'f9.txt', 'f10.txt']
    >>> sorted(l, key=nsort_key_func)
    ['f1.1.1.txt', 'f1.01.2.txt', 'f1.10.0.txt', 'f9.txt', 'f10.txt']

    """
    spl = NSORT_RE.split(s)
    key = []
    for non_number, number in zip(spl[::2], spl[1::2]):
        key.append(non_number)
        key.append(int(number))
    return key


def set_explicit_dtype(x):
    """Force `x` to have a numpy type if it doesn't already have one.

    Parameters
    ----------
    x : numpy-typed object, bool, integer, float
        If not numpy-typed, type is attempted to be inferred. Currently only
        bool, int, and float are supported, where bool is converted to
        np.bool8, integer is converted to np.int64, and float is converted to
        np.float64. This ensures that full precision for all but the most
        extreme cases is maintained for inferred types.

    Returns
    -------
    x : numpy-typed object

    Raises
    ------
    TypeError
        In case the type of `x` is not already set or is not a valid inferred
        type. As type inference can yield different results for different
        inputs, rather than deal with everything, explicitly failing helps to
        avoid inferring the different instances of the same object differently
        (which will cause a failure later on when trying to concatenate the
        types in a larger array).

    """
    if hasattr(x, "dtype"):
        return x

    # bools are numbers.Integral, so test for bool first
    if isinstance(x, bool):
        return np.bool8(x)

    if isinstance(x, Integral):
        x_new = np.int64(x)
        assert x_new == x
        return x_new

    if isinstance(x, Number):
        x_new = np.float64(x)
        assert x_new == x
        return x_new

    if isinstance(x, string_types):
        x_new = np.string0(x)
        assert x_new == x
        return x_new

    raise TypeError("Type of argument is invalid: {}".format(type(x)))


def dict2struct(d):
    """Convert a dict with string keys and numpy-typed values into a numpy
    array with struct dtype.

    Parameters
    ----------
    d : OrderedMapping
        The dict's keys are the names of the fields (strings) and the dict's
        values are numpy-typed objects.

    Returns
    -------
    array : numpy.array of struct dtype

    """
    dt_spec = OrderedDict()
    for key, val in d.items():
        d[key] = typed_val = set_explicit_dtype(val)
        dt_spec[key] = typed_val.dtype
    array = np.array(tuple(d.values()), dtype=dt_spec.items())
    return array


if __name__ == '__main__':
    test_hash_obj()
