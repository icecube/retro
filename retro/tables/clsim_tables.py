# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Class for using a set of "raw" 5D (r, costheta, t, costhetadir, deltaphidir)
CLSim-produced Retro tables
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    MY_CLSIM_TABLE_KEYS
    TABLE_NORM_KEYS
    CLSIM_TABLE_FNAME_PROTO
    CLSIM_TABLE_FNAME_RE
    CLSIM_TABLE_METANAME_PROTO
    CLSIM_TABLE_METANAME_RE
    interpret_clsim_table_fname
    generate_time_indep_table
    get_table_norm
    load_clsim_table_minimal
    load_clsim_table
    CLSimTables
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

from collections import OrderedDict
from os.path import abspath, basename, dirname, isdir, isfile, join
import re
import sys
from time import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro import DEBUG
from retro.const import SPEED_OF_LIGHT_M_PER_NS, PI, TWO_PI
from retro.tables.pexp_5d import TBL_KIND_CLSIM, generate_pexp_5d_function
from retro.utils.geom import spherical_volume
from retro.utils.misc import (
    expand, force_little_endian, get_decompressd_fobj, wstderr
)


MY_CLSIM_TABLE_KEYS = [
    'table_shape', 'n_photons', 'phase_refractive_index', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges', 'costhetadir_bin_edges',
    'deltaphidir_bin_edges', 'table', #'t_indep_table'
]

TABLE_NORM_KEYS = [
    'n_photons', 'phase_refractive_index', 'step_length', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges'
]
"""All besides 'quantum_efficiency' and 'angular_acceptance_fract'"""

CLSIM_TABLE_FNAME_PROTO = [
    (
        'retro_nevts1000_{string}_DOM{depth_idx:d}.fits.*'
    ),
    (
        'clsim_table'
        '_set_{hash_val:s}'
        '_string_{string}'
        '_depth_{depth_idx:d}'
        '_seed_{seed}'
        '.fits'
    )
]
"""String templates for CLSim ("raw") retro tables. Note that `string` can
either be a specific string number OR either "ic" or "dc" indicating a generic
DOM of one of these two types located at the center of the detector, where z
location is averaged over all DOMs. `seed` can either be an integer or a
human-readable range (e.g. "0-9" for a table that combines toegether seeds, 0,
1, ..., 9)"""

CLSIM_TABLE_FNAME_RE = [
    re.compile(
        r'''
        retro
        _nevts(?P<n_events>[0-9]+)
        _(?P<string>[0-9a-z]+)
        _DOM(?P<depth_idx>[0-9]+)
        \.fits
        ''', re.IGNORECASE | re.VERBOSE
    ),
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _string_(?P<string>[0-9a-z]+)
        _depth_(?P<depth_idx>[0-9]+)
        _seed_(?P<seed>[0-9]+)
        \.fits
        ''', re.IGNORECASE | re.VERBOSE
    )
]

CLSIM_TABLE_METANAME_PROTO = [
    'clsim_table_set_{hash_val:s}_meta.json'
]

CLSIM_TABLE_METANAME_RE = [
    re.compile(
        r'''
        clsim_table
        _set_(?P<hash_val>[0-9a-f]+)
        _meta
        \.json
        ''', re.IGNORECASE | re.VERBOSE
    )
]


def interpret_clsim_table_fname(fname):
    """Get fields from fname and interpret these (e.g. by covnerting into
    appropriate Python types).

    The fields are parsed into the following types / values:
        - fname_version : int
        - hash_val : None or str
        - string : str (one of {'ic', 'dc'}) or int
        - depth_idx : int
        - seed : None, str (exactly '*'), int, or list of ints
        - n_events : None or int

    Parameters
    ----------
    fname : string

    Returns
    -------
    info : dict

    Raises
    ------
    ValueError
        If ``basename(fname)`` does not match the regexes
        ``CLSIM_TABLE_FNAME_RE``

    """
    from pisa.utils.format import hrlist2list

    fname = basename(fname)
    fname_version = None
    for fname_version in range(len(CLSIM_TABLE_FNAME_RE) - 1, -1, -1):
        match = CLSIM_TABLE_FNAME_RE[fname_version].match(fname)
        if match:
            break
    if not match:
        raise ValueError(
            'File basename "{}" does not match regex {} or any legacy regexes'
            .format(fname, CLSIM_TABLE_FNAME_RE[-1].pattern)
        )
    info = match.groupdict()
    info['fname_version'] = fname_version

    try:
        info['string'] = int(info['string'])
    except ValueError:
        assert isinstance(info['string'], basestring)
        assert info['string'].lower() in ['ic', 'dc']
        info['string'] = info['string'].lower()

    if fname_version == 1:
        info['seed'] = None
        info['hash_val'] = None
    elif fname_version == 2:
        info['n_events'] = None
        try:
            info['seed'] = int(info['seed'])
        except ValueError:
            if info['seed'] != '*':
                info['seed'] = hrlist2list(info['seed'])

    info['depth_idx'] = int(info['depth_idx'])

    ordered_keys = ['fname_version', 'hash_val', 'string', 'depth_idx', 'seed',
                    'n_events']
    ordered_info = OrderedDict()
    for key in ordered_keys:
        ordered_info[key] = info[key]

    return ordered_info


def generate_time_indep_table(table, quantum_efficiency,
                              angular_acceptance_fract, step_length=None):
    """
    Parameters
    ----------
    table : mapping
    quantum_efficiency : float in (0, 1]
    angular_acceptance_fract : float in (0, 1]
    step_length : float in (0, 1]
        Required if 'step_length' is not a key in `table`.

    Returns
    -------
    t_indep_table : numpy.ndarray of size (n_r, n_costheta, n_costhetadir, n_deltaphidir)
    t_indep_table_norm : scalar
        This norm factor must be applied to the table in order to convert its
        entries to survival probabilities.

    """
    if 'step_length' in table:
        if step_length is not None:
            assert step_length == table['step_length']
        else:
            step_length = table['step_length']

    if step_length is None:
        raise ValueError('Need either an value for the `step_length` argument'
                         ' or "step_length"  in `table`.')

    table_norm = get_table_norm(
        quantum_efficiency=quantum_efficiency,
        angular_acceptance_fract=angular_acceptance_fract,
        step_length=step_length,
        **{k: table[k] for k in TABLE_NORM_KEYS if k != 'step_length'}
    )

    # Indices in the Einstein sum are
    #   table:
    #     r=radius, c=costheta, t=time, q=costhetadir, p=deltaphidir
    #   norm:
    #     r=radius, t=time
    #   output:
    #     r=radius, c=costheta, q=costhetadir, p=deltaphidir
    t_indep_table = np.einsum(
        'rctqp,rt->rcqp',
        table['table'][1:-1, 1:-1, 1:-1, 1:-1, 1:-1],
        table_norm,
        optimize=False # can change when we know if this helps
    )

    t_indep_table_norm = quantum_efficiency * angular_acceptance_fract

    return t_indep_table, t_indep_table_norm


def get_table_norm(
        n_photons, phase_refractive_index, step_length, r_bin_edges,
        costheta_bin_edges, t_bin_edges, quantum_efficiency,
        angular_acceptance_fract, norm_version
    ):
    """Get the normalization array to use a raw CLSim table with Retro reco.

    Note that the `norm` array returned is meant to _multiply_ the counts in
    the raw CLSim table to obtain a survival probability.

    Parameters
    ----------
    n_photons : int > 0
        Number of photons thrown in the simulation.

    phase_refractive_index : float > 0
        Phase refractive index in the medium.

    step_length : float > 0
        Step length used in CLSim tabulator, in units of meters. (Hard-coded to
        1 m in CLSim, but this is a paramter that ultimately could be changed.)

    r_bin_edges : 1D numpy.ndarray, ascending values => 0 (meters)
        Radial bin edges in units of meters.

    costheta_bin_edges : 1D numpy.ndarray, ascending values in [-1, 1]
        Cosine of the zenith angle bin edges; all must be equally spaced.

    t_bin_edges : 1D numpy.ndarray, ascending values > 0 (nanoseconds)
        Time bin eges in units of nanoseconds.

    quantum_efficiency : float in (0, 1], optional
        Average DOM quantum efficiency for converting photons to
        photo electrons. Note that any shape to the quantum efficiency should
        already be accounted for by simulating photons according to the
        shape of that distribution. If not specific, defaults to 1.

    angular_acceptance_fract : float in (0, 1], optional
        Average DOM angular acceptance fraction, which modifies the
        "efficiency" beyond that accounted for by `quantum_efficiency`.
        Note that any shape to the angular accptance should already be
        accounted for by simulating photons according to the
        shape of that distribution. If not specified, defaults to 1.

    norm_version : string

    Returns
    -------
    table_norm : numpy.ndarray of shape (n_r_bins, n_t_bins), values >= 0
        The normalization is a function of both r- and t-bin (we assume
        costheta binning is "regular"). To obtain a survival probability,
        multiply the value in the CLSim table's bin by the appropriate
        `table_norm` entry. I.e.:
        ``survival_prob = raw_bin_val * table_norm[r_bin_idx, t_bin_idx]``.

    """
    n_costheta_bins = len(costheta_bin_edges) - 1

    r_bin_widths = np.diff(r_bin_edges)
    costheta_bin_widths = np.diff(costheta_bin_edges)
    t_bin_widths = np.diff(t_bin_edges)

    # We need costheta bins to all have same width for the logic below to hold
    costheta_bin_width = np.mean(costheta_bin_widths)
    assert np.allclose(costheta_bin_widths, costheta_bin_width), costheta_bin_widths

    constant_part = (
        # Number of photons, divided equally among the costheta bins
        1 / (n_photons / n_costheta_bins)

        # Correction for quantum efficiency of the DOM
        * angular_acceptance_fract

        # Correction for additional loss of sensitivity due to angular
        # acceptance model
        * quantum_efficiency
    )

    # A photon is tabulated every step_length meters; we want the
    # average probability in each bin, so the count in the bin must be
    # divided by the number of photons in the bin times the number of
    # times each photon is counted.
    speed_of_light_in_medum = ( # units = m/ns
        SPEED_OF_LIGHT_M_PER_NS / phase_refractive_index
    )

    # t bin edges are in ns and speed_of_light_in_medum is m/ns
    t_bin_widths_in_m = t_bin_widths * speed_of_light_in_medum

    # TODO: Note that the actual counts will be rounded down (or are one fewer
    # than indicated by this ratio if the ratio comres out to an integer). We
    # don't account for this here, but we _might_ want to if it seems that this
    # will make a difference (e.g. for small bins). Of course if the number
    # comes out to zero, then... should we clip the lower-bound to 1? Go with
    # the fraction we come up with here (since the first step is randomized)?
    # And in fact, since the first step is randomized, it seems this ratio
    # should be fine as is; but it's the 2D toy simulation that indicates that
    # maybe the `floor` might be necessary. Someday we should really sort this
    # whole normalization thing out!
    counts_per_r = r_bin_widths / step_length
    counts_per_t = t_bin_widths_in_m / step_length

    inner_edges = r_bin_edges[:-1]
    outer_edges = r_bin_edges[1:]

    radial_midpoints = 0.5 * (inner_edges + outer_edges)
    inner_radius = np.where(inner_edges == 0, 0.01* radial_midpoints, inner_edges)
    avg_radius = 3/4 * (outer_edges**4 - inner_edges**4) / (outer_edges**3 - inner_edges**3)

    surf_area_at_avg_radius = avg_radius**2 * TWO_PI * costheta_bin_width
    surf_area_at_inner_radius = inner_radius**2 * TWO_PI * costheta_bin_width
    surf_area_at_midpoints = radial_midpoints**2 * TWO_PI * costheta_bin_width

    bin_vols = np.abs(spherical_volume(rmin=inner_edges, rmax=outer_edges, dcostheta=-costheta_bin_width, dphi=TWO_PI))

    # Take the smaller of counts_per_r and counts_per_t
    table_step_length_norm = np.minimum.outer(counts_per_r, counts_per_t) # pylint: disable=no-member
    assert table_step_length_norm.shape == (counts_per_r.size, counts_per_t.size)

    if norm_version == 'avgsurfarea':
        radial_norm = 1 / surf_area_at_avg_radius
        #radial_norm = 1 / surf_area_at_inner_radius

        # Shape of the radial norm is 1D: (n_r_bins,)
        table_norm = 1/(1.01987*0.43741) * 2*constant_part * table_step_length_norm * radial_norm[:, np.newaxis]

    elif norm_version == 'binvol':
        radial_norm = 1 / bin_vols
        table_norm = 1/(0.10943*0.78598) * constant_part * table_step_length_norm * radial_norm[:, np.newaxis]

    # copied norm from Philipp / generate_t_r_theta_table :
    elif norm_version == 'pde':
        table_norm = np.outer(
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            4 * PI / surf_area_at_midpoints,
            np.full(
                shape=(len(t_bin_edges) - 1,),
                fill_value=(
                    1
                    / n_photons
                    / (SPEED_OF_LIGHT_M_PER_NS / phase_refractive_index)
                    / np.mean(t_bin_widths)
                    * angular_acceptance_fract
                    * quantum_efficiency
                    * n_costheta_bins
                )
            )
        )

    elif norm_version == 'wtf':
        not_really_volumes = surf_area_at_midpoints * (outer_edges - inner_edges)
        not_really_volumes *= costheta_bin_width

        table_norm = np.outer(
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            1/(1.17059*1.03695) * 4 / not_really_volumes,
            np.full(
                shape=(len(t_bin_edges) - 1,),
                fill_value=(
                    1
                    / n_photons
                    / (SPEED_OF_LIGHT_M_PER_NS / phase_refractive_index)
                    / np.mean(t_bin_widths)
                    * angular_acceptance_fract
                    * quantum_efficiency
                    * n_costheta_bins
                )
            )
        )

    elif norm_version == 'wtf2':
        not_really_volumes = surf_area_at_midpoints * (outer_edges - inner_edges)
        not_really_volumes *= costheta_bin_width

        table_norm = np.outer(
            # NOTE: pi factor needed to get agreement with old code (why?);
            # 4 is needed for new clsim tables (why?)
            1/(3.21959*1.03695*1.02678) * 4 / not_really_volumes,
            np.full(shape=(len(t_bin_edges) - 1,), fill_value=constant_part)
        )

    else:
        raise ValueError('unhandled `norm_version` "{}"'.format(norm_version))

    return table_norm


def load_clsim_table_minimal(fpath, step_length=None, mmap=False,
                             gen_t_indep=False):
    """Load a CLSim table from disk (optionally compressed with zstd).

    Similar to the `load_clsim_table` function but the full table, including
    under/overflow bins, is kept and no normalization or further processing is
    performed on the table data besides populating the ouptput OrderedDict.

    Parameters
    ----------
    fpath : string
        Path to file to be loaded. If the file has extension 'zst', 'zstd', or
        'zstandard', the file will be decompressed using the `python-zstandard`
        Python library before passing to `pyfits` for interpreting.

    step_length : float > 0 in units of meters, optional
        Required if computing the `t_indep_table` (if `gen_t_indep` is True).

    mmap : bool, optional
        Whether to memory map the table (if it's stored in a directory
        containing .npy files).

    gen_t_indep : bool, optional
        Generate the time-independent table if it does not exist.

    Returns
    -------
    table : OrderedDict
        Items include
        - 'table_shape' : tuple of int
        - 'table' : np.ndarray
        - 't_indep_table' : np.ndarray
        - 'n_photons' :
        - 'phase_refractive_index' :
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'table' :
        - 't_indep_table' :

    """
    table = OrderedDict()

    fpath = expand(fpath)

    if DEBUG:
        wstderr('Loading table from {} ...\n'.format(fpath))

    if isdir(fpath):
        t0 = time()
        indir = fpath
        if mmap:
            mmap_mode = 'r'
        else:
            mmap_mode = None
        for key in MY_CLSIM_TABLE_KEYS + ['t_indep_table']:
            fpath = join(indir, key + '.npy')
            if DEBUG:
                wstderr('    loading {} from "{}" ...'.format(key, fpath))
            t1 = time()
            if isfile(fpath):
                table[key] = np.load(fpath, mmap_mode=mmap_mode)
            elif key != 't_indep_table':
                raise ValueError(
                    'Could not find file "{}" for loading table key "{}"'
                    .format(fpath, key)
                )
            if DEBUG:
                wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))
        if step_length is not None and 'step_length' in table:
            assert step_length == table['step_length']
        if DEBUG:
            wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))
        return table

    if not isfile(fpath):
        raise ValueError('Table does not exist at path "{}"'.format(fpath))

    if mmap:
        print('WARNING: Cannot memory map a fits or compressed fits file;'
              ' ignoring `mmap=True`.')

    import pyfits
    t0 = time()
    fobj = get_decompressd_fobj(fpath)
    try:
        pf_table = pyfits.open(fobj)

        table['table_shape'] = pf_table[0].data.shape # pylint: disable=no-member
        table['n_photons'] = force_little_endian(
            pf_table[0].header['_i3_n_photons'] # pylint: disable=no-member
        )
        table['phase_refractive_index'] = force_little_endian(
            pf_table[0].header['_i3_n_phase'] # pylint: disable=no-member
        )
        if step_length is not None:
            table['step_length'] = step_length

        n_dims = len(table['table_shape'])
        if n_dims == 5:
            # Space-time dimensions
            table['r_bin_edges'] = force_little_endian(
                pf_table[1].data # meters # pylint: disable=no-member
            )
            table['costheta_bin_edges'] = force_little_endian(
                pf_table[2].data # pylint: disable=no-member
            )
            table['t_bin_edges'] = force_little_endian(
                pf_table[3].data # nanoseconds # pylint: disable=no-member
            )

            # Photon directionality
            table['costhetadir_bin_edges'] = force_little_endian(
                pf_table[4].data # pylint: disable=no-member
            )
            table['deltaphidir_bin_edges'] = force_little_endian(
                pf_table[5].data # pylint: disable=no-member
            )

        else:
            raise NotImplementedError(
                '{}-dimensional table not handled'.format(n_dims)
            )

        table['table'] = force_little_endian(pf_table[0].data) # pylint: disable=no-member

        if gen_t_indep and 't_indep_table' not in table:
            table['t_indep_table'] = generate_time_indep_table(
                table=table,
                quantum_efficiency=1,
                angular_acceptance_fract=1
            )

        wstderr('    (load took {} s)\n'.format(np.round(time() - t0, 3)))

    finally:
        del pf_table
        if hasattr(fobj, 'close'):
            fobj.close()
        del fobj

    return table


def load_clsim_table(fpath, step_length, angular_acceptance_fract,
                     quantum_efficiency):
    """Load a CLSim table from disk (optionally compressed with zstd).

    Parameters
    ----------
    fpath : string
        Path to file to be loaded. If the file has extension 'zst', 'zstd', or
        'zstandard', the file will be decompressed using the `python-zstandard`
        Python library before passing to `pyfits` for interpreting.

    Returns
    -------
    table : OrderedDict
        Items include
        - 'table_shape' : tuple of int
        - 'table' : np.ndarray
        - 't_indep_table' : np.ndarray
        - 'n_photons' :
        - 'phase_refractive_index' :

        If the table is 5D, items also include
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'table_norm'

    """
    table = OrderedDict()

    table = load_clsim_table_minimal(fpath=fpath, step_length=step_length)
    table['table_norm'] = get_table_norm(
        angular_acceptance_fract=angular_acceptance_fract,
        quantum_efficiency=quantum_efficiency,
        step_length=step_length,
        **{k: table[k] for k in TABLE_NORM_KEYS if k != 'step_length'}
    )
    table['t_indep_table_norm'] = quantum_efficiency * angular_acceptance_fract

    wstderr('Interpreting table...\n')
    t0 = time()
    n_dims = len(table['table_shape'])

    # Cut off first and last bin in each dimension (underflow and
    # overflow bins)
    slice_wo_overflow = (slice(1, -1),) * n_dims
    wstderr('    slicing to remove underflow/overflow bins...')
    t0 = time()
    table_wo_overflow = table['table'][slice_wo_overflow]
    wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    wstderr('    slicing and summarizing underflow and overflow...')
    t0 = time()
    underflow, overflow = [], []
    for n in range(n_dims):
        sl = tuple([slice(1, -1)]*n + [0] + [slice(1, -1)]*(n_dims - 1 - n))
        underflow.append(table['table'][sl].sum())

        sl = tuple([slice(1, -1)]*n + [-1] + [slice(1, -1)]*(n_dims - 1 - n))
        overflow.append(table['table'][sl].sum())
    wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    table['table'] = table_wo_overflow
    table['underflow'] = np.array(underflow)
    table['overflow'] = np.array(overflow)

    return table


class CLSimTables(object):
    """
    Class to interact with and obtain photon survival probabilities from a set
    of CLSim tables.

    Parameters
    ----------
    geom : shape-(n_strings, n_doms, 3) array
        x, y, z coordinates of all DOMs, in meters relative to the IceCube
        coordinate system

    rde : shape-(n_strings, n_doms) array
        Relative DOM efficiencies (this accounts for quantum efficiency). Any
        DOMs with either 0 or NaN rde will be disabled and return 0's for
        expected photon counts.

    noise_rate_hz : shape-(n_strings, n_doms) array
        Noise rate for each DOM, in Hz.

    use_directionality : bool
        Enable or disable directionality when computing expected photons at
        the DOMs

    num_phi_samples : int > 0
        If using directionality, set how many samples around the Cherenkov cone
        are used to find which (costhetadir, deltaphidir) bins to use to
        compute expected photon survival probability.

    ckv_sigma_deg : float >= 0
        If using directionality, Gaussian-smear the Cherenkov angle by this
        meany degrees by randomly distributing the phi samples. Higher
        `ckv_sigma_deg` could necessitate higher `num_phi_samples` to get an
        accurate "smearing."

    norm_version : string
        (Temporary) Which version of the norm to use. Only for experimenting,
        and will be removed once we figure the norm out.

    """
    def __init__(
            self, geom, rde, noise_rate_hz, use_directionality,
            num_phi_samples, ckv_sigma_deg, norm_version
        ):
        assert len(geom.shape) == 3
        self.geom = geom
        self.use_directionality = use_directionality
        self.num_phi_samples = num_phi_samples
        self.ckv_sigma_deg = ckv_sigma_deg
        self.norm_version = norm_version

        zero_mask = rde == 0
        nan_mask = np.isnan(rde)
        inf_mask = np.isinf(rde)
        num_zero = np.count_nonzero(zero_mask)
        num_nan = np.count_nonzero(nan_mask)
        num_inf = np.count_nonzero(inf_mask)

        if num_nan or num_inf or num_zero:
            print(
                "WARNING: RDE is zero for {} DOMs, NaN for {} DOMs and +/-inf"
                " for {} DOMs.\n"
                "These DOMs will be disabled and return 0's forexpected photon"
                " computations."
                .format(num_zero, num_nan, num_inf)
            )
        mask = zero_mask | nan_mask | inf_mask

        self.operational_doms = ~mask
        self.rde = np.ma.masked_where(mask, rde)
        self.quantum_efficiency = 0.25 * self.rde
        self.noise_rate_hz = np.ma.masked_where(mask, noise_rate_hz)

        self.tables = {}
        self.string_aggregation = None
        self.depth_aggregation = None
        self.pexp_func = None
        self.binning_info = None

    def load_table(
            self, fpath, string, dom, step_length, angular_acceptance_fract,
            mmap
        ):
        """Load a table into the set of tables.

        Parameters
        ----------
        fpath : string
            Path to the table .fits file or table directory (in the case of the
            Retro-formatted directory with .npy files).

        string : int in [1, 86] or str in {'ic', 'dc', or 'all'}

        dom : int in [1, 60] or str == 'all'

        step_length : float > 0
            The stepLength parameter (in meters) used in CLSim tabulator code
            for tabulating a single photon as it travels. This is a hard-coded
            paramter set to 1 meter in the trunk version of the code, but it's
            something we might play with to optimize table generation speed, so
            just be warned that this _can_ change.

        angular_acceptance_fract : float in (0, 1]
            Constant normalization factor to apply to correct for the integral
            of the angular acceptance curve used in the simulation that
            produced this table.

        mmap : bool
            Whether to attempt to memory map the table (only applicable for
            Retro npy-files-in-a-dir tables).

        """
        single_dom_spec = True
        if isinstance(string, basestring):
            string = string.strip().lower()
            assert string in ['ic', 'dc', 'all']
            agg_mode = 'all' if string == 'all' else 'subdetector'
            if self.string_aggregation is None:
                self.string_aggregation = agg_mode
            assert agg_mode == self.string_aggregation
            single_dom_spec = False
        else:
            if self.string_aggregation is None:
                self.string_aggregation = False
            # `False` is ok but `None` is not ok
            assert self.string_aggregation == False # pylint: disable=singleton-comparison
            assert 1 <= string <= 86

        if isinstance(dom, basestring):
            dom = dom.strip().lower()
            assert dom == 'all'
            if self.depth_aggregation is None:
                self.depth_aggregation = True
            assert self.depth_aggregation
            single_dom_spec = False
        else:
            if self.depth_aggregation is None:
                self.depth_aggregation = False
            # `False` is ok but `None` is not ok
            assert self.depth_aggregation == False # pylint: disable=singleton-comparison
            assert 1 <= dom <= 60

        assert step_length > 0
        assert 0 < angular_acceptance_fract <= 1

        if single_dom_spec and not self.operational_doms[string - 1, dom - 1]:
            print(
                'WARNING: String {}, DOM {} is not operational, skipping'
                ' loading the corresponding table'.format(string, dom)
            )
            return

        table = load_clsim_table_minimal(
            fpath=fpath,
            step_length=step_length,
            mmap=mmap,
            gen_t_indep=True
        )

        table['step_length'] = step_length
        table['table_norm'] = get_table_norm(
            angular_acceptance_fract=angular_acceptance_fract,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: table[k] for k in TABLE_NORM_KEYS}
        )
        table['t_indep_table_norm'] = angular_acceptance_fract

        pexp_5d, _ = generate_pexp_5d_function(
            table=table,
            table_kind=TBL_KIND_CLSIM,
            use_directionality=self.use_directionality,
            num_phi_samples=self.num_phi_samples,
            ckv_sigma_deg=self.ckv_sigma_deg
        )

        # NOTE: original tables have underflow (bin 0) and overflow (bin -1)
        # bins, so whole-axis slices must exclude the first and last bins.
        self.tables[(string, dom)] = (
            pexp_5d,
            table['t_indep_table'],
            table['t_indep_table_norm'],
            table['table'][1:-1, 1:-1, 1:-1, 1:-1, 1:-1],
            table['table_norm'],
        )

    def get_photon_expectation(
            self, pinfo_gen, hit_time, time_window, string, dom
        ):
        """
        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy.ndarray
            Info about photons generated photons by the event hypothesis.

        hit_time : float, units of ns
        time_window : float, units of ns
        string : int in [1, 86]
        dom : int in [1, 60]

        Returns
        -------
        total_photon_count, expected_photon_count : float
            See pexp_t_r_theta

        """
        # `string` and `dom` are 1-indexed but array indices are 0-indexed
        string_idx, dom_idx = string - 1, dom - 1
        if not self.operational_doms[string_idx, dom_idx]:
            return 0, 0

        dom_coord = self.geom[string_idx, dom_idx]
        dom_quantum_efficiency = self.quantum_efficiency[string_idx, dom_idx]
        dom_noise_rate_hz = self.noise_rate_hz[string_idx, dom_idx]

        if self.string_aggregation == 'all':
            string = 'all'
        elif self.string_aggregation == 'subdetector':
            if string < 79:
                string = 'ic'
            else:
                string = 'dc'

        if self.depth_aggregation:
            dom = 'all'

        #print('string =', string, 'dom =', dom)

        (pexp_5d,
         t_indep_table,
         t_indep_table_norm,
         table,
         table_norm) = self.tables[(string, dom)]

        return pexp_5d(
            pinfo_gen=pinfo_gen,
            hit_time=hit_time,
            time_window=time_window,
            dom_coord=dom_coord,
            noise_rate_hz=dom_noise_rate_hz,
            quantum_efficiency=dom_quantum_efficiency,
            table=table,
            table_norm=table_norm,
            #t_indep_table=t_indep_table,
            #t_indep_table_norm=t_indep_table_norm,
        )
