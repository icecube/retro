# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, too-many-instance-attributes, too-many-locals, line-too-long

"""
Classes for reading and getting info from Retro tables.
"""


from __future__ import absolute_import, division, print_function


__all__ = '''
    MACHINE_EPS
    MY_CLSIM_TABLE_KEYS
    TABLE_NORM_KEYS
    open_table_file
    get_table_norm
    generate_time_indep_table
    load_clsim_table_minimal
    load_clsim_table
    load_t_r_theta_table
    generate_pexp_5d_function
    pexp_t_r_theta
    pexp_xyz
    CLSimTable
    CLSimTables
    DOMTimePolarTables
    TDICartTable
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
from copy import deepcopy
from glob import glob
from os import remove
from os.path import abspath, basename, dirname, isdir, isfile, join, splitext
from StringIO import StringIO
from subprocess import Popen, PIPE
import sys
from time import time
import math

import numpy as np

from pisa.utils.format import hrlist2list

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro
from retro import PI, TWO_PI
from retro.ckv import survival_prob_from_cone, survival_prob_from_smeared_cone
from retro.generate_t_r_theta_table import generate_t_r_theta_table


MACHINE_EPS = 1e-16

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

CKV_TABLE_KEYS = [
    'n_photons', 'phase_refractive_index', 'r_bin_edges',
    'costheta_bin_edges', 't_bin_edges', 'costhetadir_bin_edges',
    'deltaphidir_bin_edges', 'ckv_table', #'t_indep_ckv_table'
]

TBL_KIND_CLSIM = 0
TBL_KIND_CKV = 1


def open_table_file(fpath):
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
    assert isfile(fpath)
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
        retro.SPEED_OF_LIGHT_M_PER_NS / phase_refractive_index
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

    # Take the smaller of counts_per_r and counts_per_t
    table_step_length_norm = constant_part * np.minimum.outer(counts_per_r, counts_per_t) # pylint: disable=no-member
    assert table_step_length_norm.shape == (counts_per_r.size, counts_per_t.size)

    if norm_version == 'avgsurfarea':
        radial_norm = 1 / surf_area_at_avg_radius
        #radial_norm = 1 / surf_area_at_inner_radius

        # Shape of the radial norm is 1D: (n_r_bins,)
        table_norm = table_step_length_norm * radial_norm[:, np.newaxis]

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
                    / (retro.SPEED_OF_LIGHT_M_PER_NS / phase_refractive_index)
                    / np.mean(t_bin_widths)
                    * angular_acceptance_fract
                    * quantum_efficiency
                    * n_costheta_bins
                )
            )
        )
    else:
        raise ValueError('unhandled `norm_version` "{}"'.format(norm_version))

    return table_norm


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


def load_ckv_table(fpath, mmap, step_length=None):
    """Load a Cherenkov table from disk.

    Parameters
    ----------
    fpath : string
        Path to directory containing the table's .npy files.

    mmap : bool
        Whether to memory map the table (if it's stored in a directory
        containing .npy files).

    step_length : float > 0 in units of meters, optional
        Required if computing the `t_indep_table` (if `gen_t_indep` is True).

    Returns
    -------
    table : OrderedDict
        Items are
        - 'n_photons' :
        - 'phase_refractive_index' :
        - 'r_bin_edges' :
        - 'costheta_bin_edges' :
        - 't_bin_edges' :
        - 'costhetadir_bin_edges' :
        - 'deltaphidir_bin_edges' :
        - 'ckv_table' : np.ndarray
        - 't_indep_ckv_table' : np.ndarray

    """
    fpath = retro.expand(fpath)
    table = OrderedDict()

    if retro.DEBUG:
        retro.wstderr('Loading table from {} ...\n'.format(fpath))

    assert isdir(fpath)
    t0 = time()
    indir = fpath

    if mmap:
        mmap_mode = 'r'
    else:
        mmap_mode = None

    for key in CKV_TABLE_KEYS: # TODO: + ['t_indep_ckv_table']:
        fpath = join(indir, key + '.npy')
        if retro.DEBUG:
            retro.wstderr('    loading {} from "{}" ...'.format(key, fpath))

        t1 = time()
        if isfile(fpath):
            table[key] = np.load(fpath, mmap_mode=mmap_mode)
        elif key != 't_indep_ckv_table':
            raise ValueError(
                'Could not find file "{}" for loading table key "{}"'
                .format(fpath, key)
            )

        if retro.DEBUG:
            retro.wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))

    if step_length is not None and 'step_length' in table:
        assert step_length == table['step_length']

    if retro.DEBUG:
        retro.wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))

    return table


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

    fpath = retro.expand(fpath)

    if retro.DEBUG:
        retro.wstderr('Loading table from {} ...\n'.format(fpath))

    if isdir(fpath):
        t0 = time()
        indir = fpath
        if mmap:
            mmap_mode = 'r'
        else:
            mmap_mode = None
        for key in MY_CLSIM_TABLE_KEYS + ['t_indep_table']:
            fpath = join(indir, key + '.npy')
            if retro.DEBUG:
                retro.wstderr('    loading {} from "{}" ...'.format(key, fpath))
            t1 = time()
            if isfile(fpath):
                table[key] = np.load(fpath, mmap_mode=mmap_mode)
            elif key != 't_indep_table':
                raise ValueError(
                    'Could not find file "{}" for loading table key "{}"'
                    .format(fpath, key)
                )
            if retro.DEBUG:
                retro.wstderr(' ({} ms)\n'.format(np.round((time() - t1)*1e3, 3)))
        if step_length is not None and 'step_length' in table:
            assert step_length == table['step_length']
        if retro.DEBUG:
            retro.wstderr('  Total time to load: {} s\n'.format(np.round(time() - t0, 3)))
        return table

    if mmap:
        print('WARNING: Cannot memory map a fits or compressed fits file;'
              ' ignoring `mmap=True`.')

    import pyfits
    t0 = time()
    fobj = open_table_file(fpath)
    try:
        pf_table = pyfits.open(fobj)

        table['table_shape'] = pf_table[0].data.shape # pylint: disable=no-member
        table['n_photons'] = retro.force_little_endian(
            pf_table[0].header['_i3_n_photons'] # pylint: disable=no-member
        )
        table['phase_refractive_index'] = retro.force_little_endian(
            pf_table[0].header['_i3_n_phase'] # pylint: disable=no-member
        )
        if step_length is not None:
            table['step_length'] = step_length

        n_dims = len(table['table_shape'])
        if n_dims == 5:
            # Space-time dimensions
            table['r_bin_edges'] = retro.force_little_endian(
                pf_table[1].data # meters # pylint: disable=no-member
            )
            table['costheta_bin_edges'] = retro.force_little_endian(
                pf_table[2].data # pylint: disable=no-member
            )
            table['t_bin_edges'] = retro.force_little_endian(
                pf_table[3].data # nanoseconds # pylint: disable=no-member
            )

            # Photon directionality
            table['costhetadir_bin_edges'] = retro.force_little_endian(
                pf_table[4].data # pylint: disable=no-member
            )
            table['deltaphidir_bin_edges'] = retro.force_little_endian(
                pf_table[5].data # pylint: disable=no-member
            )

        else:
            raise NotImplementedError(
                '{}-dimensional table not handled'.format(n_dims)
            )

        table['table'] = retro.force_little_endian(pf_table[0].data) # pylint: disable=no-member

        if gen_t_indep and 't_indep_table' not in table:
            table['t_indep_table'] = generate_time_indep_table(
                table=table,
                quantum_efficiency=1,
                angular_acceptance_fract=1
            )

        retro.wstderr('    (load took {} s)\n'.format(np.round(time() - t0, 3)))

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

    retro.wstderr('Interpreting table...\n')
    t0 = time()
    n_dims = len(table['table_shape'])

    # Cut off first and last bin in each dimension (underflow and
    # overflow bins)
    slice_wo_overflow = (slice(1, -1),) * n_dims
    retro.wstderr('    slicing to remove underflow/overflow bins...')
    t0 = time()
    table_wo_overflow = table['table'][slice_wo_overflow]
    retro.wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    retro.wstderr('    slicing and summarizing underflow and overflow...')
    t0 = time()
    underflow, overflow = [], []
    for n in range(n_dims):
        sl = tuple([slice(1, -1)]*n + [0] + [slice(1, -1)]*(n_dims - 1 - n))
        underflow.append(table['table'][sl].sum())

        sl = tuple([slice(1, -1)]*n + [-1] + [slice(1, -1)]*(n_dims - 1 - n))
        overflow.append(table['table'][sl].sum())
    retro.wstderr(' ({} ms)\n'.format(np.round((time() - t0)*1e3)))

    table['table'] = table_wo_overflow
    table['underflow'] = np.array(underflow)
    table['overflow'] = np.array(overflow)

    return table


def load_t_r_theta_table(fpath, depth_idx, scale=1, exponent=1,
                         photon_info=None):
    """Extract info from a file containing a (t, r, theta)-binned Retro table.

    Parameters
    ----------
    fpath : string
        Path to FITS file corresponding to the passed ``depth_idx``.

    depth_idx : int
        Depth index (e.g. from 0 to 59)

    scale : float
        Scaling factor to apply to the photon survival probability from the
        table, e.g. for quantum efficiency. This is applied _before_
        `exponent`. See `Notes` for more info.

    exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs. This is
        applied to each DOM's table _after_ `scale`. See `Notes` for more
        info.

    photon_info : None or retro.RetroPhotonInfo namedtuple of dicts
        If None, creates a new RetroPhotonInfo namedtuple with empty dicts to
        fill. If one is provided, the existing component dictionaries are
        updated.

    Returns
    -------
    photon_info : retro.RetroPhotonInfo namedtuple of dicts
        Tuple fields are 'survival_prob', 'theta', 'phi', and 'length'. Each
        dict is keyed by `depth_idx` and values are the arrays loaded
        from the FITS file.

    bin_edges : retro.TimeSphCoord namedtuple
        Each element of the tuple is an array of bin edges.

    Notes
    -----
    The parameters `scale` and `exponent` modify a table's probability `P` by::

        P = 1 - (1 - P*scale)**exponent

    This allows for `scale` (which must be from 0 to 1) to be used for e.g.
    quantum efficiency--which always reduces the detection probability--and
    `exponent` (which must be 0 or greater) to be used as a systematic that
    modifies the post-`scale` probabilities up and down while keeping them
    valid (i.e., between 0 and 1). Larger values of `scale` (i.e., closer to 1)
    indicate a more efficient DOM. Likewise, values of `exponent` greater than
    one scale up the DOM efficiency, while values of `exponent` between 0 and 1
    scale the efficiency down.

    """
    # pylint: disable=no-member
    import pyfits

    assert 0 <= scale <= 1
    assert exponent >= 0

    if photon_info is None:
        empty_dicts = []
        for _ in retro.RetroPhotonInfo._fields:
            empty_dicts.append({})
        photon_info = retro.RetroPhotonInfo(*empty_dicts)

    with pyfits.open(retro.expand(fpath)) as table:
        data = retro.force_little_endian(table[0].data)

        if scale == exponent == 1:
            photon_info.survival_prob[depth_idx] = data
        else:
            photon_info.survival_prob[depth_idx] = (
                1 - (1 - data * scale)**exponent
            )

        photon_info.theta[depth_idx] = retro.force_little_endian(table[1].data)

        photon_info.deltaphi[depth_idx] = retro.force_little_endian(table[2].data)

        photon_info.length[depth_idx] = retro.force_little_endian(table[3].data)

        # Note that we invert (reverse and multiply by -1) time edges; also,
        # no phi edges are defined in these tables.
        data = retro.force_little_endian(table[4].data)
        t = - data[::-1]

        r = retro.force_little_endian(table[5].data)

        # Previously used the following to get "agreement" w/ raw photon sim
        #r_volumes = np.square(0.5 * (r[1:] + r[:-1]))
        #r_volumes = (0.5 * (r[1:] + r[:-1]))**2 * (r[1:] - r[:-1])
        r_volumes = 0.25 * (r[1:]**3 - r[:-1]**3)

        photon_info.survival_prob[depth_idx] /= r_volumes[np.newaxis, :, np.newaxis]

        photon_info.time_indep_survival_prob[depth_idx] = np.sum(
            photon_info.survival_prob[depth_idx], axis=0
        )

        theta = retro.force_little_endian(table[6].data)

        bin_edges = retro.TimeSphCoord(
            t=t, r=r, theta=theta, phi=np.array([], dtype=t.dtype)
        )

    return photon_info, bin_edges


def generate_pexp_5d_function(
        table, table_kind, use_directionality, num_phi_samples, ckv_sigma_deg
    ):
    """Generate a numba-compiled function for computing expected photon counts
    at a DOM, where the table's binning info is used to pre-compute various
    constants for the compiled function.

    Parameters
    ----------
    table : mapping
        As returned by `load_clsim_table_minimal`

    table_kind

    use_directionality : bool, optional
        If the source photons have directionality, use it in computing photon
        expectations at the DOM.

    num_phi_samples : int
        Number of samples in the phi_dir to average over bin counts.
        (Irrelevant if `use_directionality` is False.)

    ckv_sigma_deg : float
        Standard deviation in degrees for Cherenkov angle. (Irrelevant if
        `use_directionality` is False).

    Returns
    -------
    pexp_5d : callable
    binning_info : dict
        Binning parameters that uniquely identify the binning from the table.

    """
    r_min = np.min(table['r_bin_edges'])
    # Ensure r_min is zero; this removes need for lower-bound checks and a
    # subtraction
    assert r_min == 0
    r_max = np.max(table['r_bin_edges'])
    rsquared_min = r_min*r_min
    rsquared_max = r_max*r_max
    r_power = retro.infer_power(table['r_bin_edges'])
    inv_r_power = 1 / r_power
    n_r_bins = len(table['r_bin_edges']) - 1
    table_dr_pwr = (r_max - r_min)**inv_r_power / n_r_bins

    n_costheta_bins = len(table['costheta_bin_edges']) - 1
    table_dcostheta = 2 / n_costheta_bins

    t_min = np.min(table['t_bin_edges'])
    # Ensure t_min is zero; this removes need for lower-bound checks and a
    # subtraction
    assert t_min == 0
    t_max = np.max(table['t_bin_edges'])
    n_t_bins = len(table['t_bin_edges']) - 1
    table_dt = (t_max - t_min) / n_t_bins

    assert table['costhetadir_bin_edges'][0] == -1
    assert table['costhetadir_bin_edges'][-1] == 1
    n_costhetadir_bins = len(table['costhetadir_bin_edges']) - 1
    table_dcosthetadir = 2 / n_costhetadir_bins
    assert np.allclose(np.diff(table['costhetadir_bin_edges']), table_dcosthetadir)
    last_costhetadir_bin_idx = n_costhetadir_bins - 1

    assert table['deltaphidir_bin_edges'][0] == 0
    assert np.isclose(table['deltaphidir_bin_edges'][-1], PI)
    n_deltaphidir_bins = len(table['deltaphidir_bin_edges']) - 1
    table_dphidir = PI / n_deltaphidir_bins
    assert np.allclose(np.diff(table['deltaphidir_bin_edges']), table_dphidir)
    last_deltaphidir_bin_idx = n_deltaphidir_bins - 1

    binning_info = dict(
        r_min=r_min, r_max=r_max, n_r_bins=n_r_bins, r_power=r_power,
        n_costheta_bins=n_costheta_bins,
        t_min=t_min, t_max=t_max, n_t_bins=n_t_bins,
        n_costhetadir_bins=n_costhetadir_bins,
        n_deltaphidir_bins=n_deltaphidir_bins,
        deltaphidir_one_sided=True
    )

    random_delta_thetas = np.array([])
    if ckv_sigma_deg > 0:
        rand = np.random.RandomState(0)
        random_delta_thetas = rand.normal(
            loc=0,
            scale=np.deg2rad(ckv_sigma_deg),
            size=num_phi_samples
        )

    #@profile
    @retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
    def pexp_5d(
            pinfo_gen, hit_time, time_window, dom_coord, quantum_efficiency,
            noise_rate_hz, table, table_norm#, t_indep_table, t_indep_table_norm
        ):
        """For a set of generated photons `pinfo_gen`, compute the expected
        photons in a particular DOM at `hit_time` and the total expected
        photons, independent of time.

        This function utilizes the relative space-time coordinates _and_
        directionality of the generated photons (via "raw" 5D CLSim tables) to
        determine how many photons are expected to arrive at the DOM.

        Retro DOM tables applied to the generated photon info `pinfo_gen`,
        and the total expected photon count (time integrated) -- the
        normalization of the pdf.

        Parameters
        ----------
        pinfo_gen : shape (N, 8) ndarray
            Information about the photons generated by the hypothesis.

        hit_time : float, units of ns
            Time at which the DOM recorded a hit (or multiple simultaneous
            hits). Use np.nan to indicate no hit occurred.

        time_window : float, units of ns
            The entire tine window of data considered, used to arrive at
            expected total noise hits (along with `noise_rate_hz`).

        dom_coord : shape (3,) ndarray
            DOM (x, y, z) coordinate in meters (in terms of the IceCube
            coordinate system).

        quantum_efficiency : float in (0, 1]
            Scale factor that reduces detected photons due to average quantum
            efficiency of the DOM.

        noise_rate_hz : float
            Noise rate for the DOM, in Hz.

        table : shape (n_r, n_costheta, n_t, n_costhetadir, n_deltaphidir) ndarray
            Time-dependent photon survival probability table.

        table_norm : shape (n_r, n_t) ndarray
            Normalization to apply to `table`, which is assumed to depend on
            both r- and t-dimensions and therefore is an array.

        t_indep_table : shape (n_r, n_costheta, n_costhetadir, n_deltaphidir) ndarray
            Time-independent photon survival probability table.

        t_indep_table_norm : float
            r- and t-dependent normalization is assumed to already have been
            applied to generate the t_indep_table, leaving only a possible
            constant normalization that must still be applied (e.g.
            ``quantum_efficiency*angular_acceptance_fract``).

        Returns
        -------
        photons_at_all_times : float
            Total photons due to the hypothesis expected to arrive at the
            specified DOM for _all_ times.

        photons_at_hit_time : float
            Total photons due to the hypothesis expected to arrive at the
            specified DOM at the time the DOM recorded the hit.

        """
        # NOTE: on optimization:
        # * np.square(x) is slower than x*x by a few percent (maybe within tolerance, though)

        # Initialize accumulators (using double precision)

        photons_at_all_times = np.float64(0.0)
        photons_at_hit_time = np.float64(0.0)

        # Initialize "prev_*" vars

        prev_r_bin_idx = -1
        prev_costheta_bin_idx = -1
        prev_t_bin_idx = -1
        prev_costhetadir_bin_idx = -1
        prev_deltaphidir_bin_idx = -1
        if use_directionality:
            prev_pdir_r = np.nan
        else:
            pdir_r = 0.0
            new_pdir_r = False

        # Initialize cached values to nan since it's a bug if these are not
        # computed at least the first time through and this will help ensure
        # that such a bug shows itself

        this_table_norm = np.nan

        # Loop over the entries (one per row)

        for pgen_idx in range(pinfo_gen.shape[0]):
            # Info about the generated photons
            t, x, y, z, p_count, pdir_x, pdir_y, pdir_z = pinfo_gen[pgen_idx, :]

            #print('t={}, x={}, y={}, z={}, p_count={}, pdir_x={}, pdir_y={}, pdir_z={}'
            #      .format(t, x, y, z, p_count, pdir_x, pdir_y, pdir_z))

            # A photon that starts immediately in the past (before the DOM was hit)
            # will show up in the raw CLSim Retro DOM tables in bin 0; the
            # further in the past the photon started, the higher the time bin
            # index.
            dt = hit_time - t
            dx = dom_coord[0] - x
            dy = dom_coord[1] - y
            dz = dom_coord[2] - z

            #print('dt={}, dx={}, dy={}, dz={}'.format(dt, dx, dy, dz))

            rhosquared = dx*dx + dy*dy
            rsquared = rhosquared + dz*dz

            # Continue if photon is outside the radial binning limits
            if rsquared >= rsquared_max or rsquared < rsquared_min:
                #print('XX CONTINUE: outside r binning')
                continue

            r = math.sqrt(rsquared)
            r_bin_idx = int(r**inv_r_power // table_dr_pwr)
            costheta_bin_idx = int((1 - dz/r) // table_dcostheta)

            #print('r={}, r_bin_idx={}, costheta_bin_idx={}'.format(r, r_bin_idx, costheta_bin_idx))

            if r_bin_idx == prev_r_bin_idx:
                new_r_bin = False
            else:
                new_r_bin = True
                prev_r_bin_idx = r_bin_idx

            if costheta_bin_idx == prev_costheta_bin_idx:
                new_costheta_bin = False
            else:
                new_costheta_bin = True
                prev_costheta_bin_idx = costheta_bin_idx

            if use_directionality:
                pdir_rhosquared = pdir_x*pdir_x + pdir_y*pdir_y
                pdir_r = math.sqrt(pdir_rhosquared + pdir_z*pdir_z)

                #print('pdir_rhosquared={}, pdir_r={}'.format(pdir_rhosquared, pdir_r))

                if pdir_r != prev_pdir_r:
                    new_pdir_r = True
                    prev_pdir_r = pdir_r
                else:
                    new_pdir_r = False

            # TODO: handle special cases for pdir_r:
            #
            #   pdir_r == 1 : Line emitter
            #   pdir_r  > 1 : Gaussian profile with stddev (pdir_r - 1)
            #
            # Note that while pdir_r == 1 is a special case of both Cherenkov
            # emission and Gaussian-profile emission, both of those are very
            # computationally expensive compared to a simple
            # perfectly-directional source, so we should handle all three
            # separately.

            if pdir_r == 0.0: # isotropic emitter
                pass
                #if new_pdir_r or new_r_bin or new_costheta_bin:
                #    this_photons_at_all_times = np.mean(
                #        t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                #    )

            elif pdir_r < 1.0: # Cherenkov emitter
                # Note that for these tables, we have to invert the photon
                # direction relative to the vector from the DOM to the photon's
                # vertex since simulation has photons going _away_ from the DOM
                # that in reconstruction will hit the DOM if they're moving
                # _towards_ the DOM.

                pdir_rho = math.sqrt(pdir_rhosquared)

                # Zenith angle is indep. of photon position relative to DOM
                pdir_costheta = pdir_z / pdir_r
                pdir_sintheta = pdir_rho / pdir_r

                rho = math.sqrt(rhosquared)

                # \Delta\phi depends on photon position relative to the DOM...

                # Below is the projection of pdir into the (x, y) plane and the
                # projection of that onto the vector in that plane connecting
                # the photon source to the DOM. We get the cosine of the angle
                # between these vectors by solving the identity
                #   `a dot b = |a| |b| cos(deltaphi)`
                # for cos(deltaphi).
                #
                if pdir_rho <= MACHINE_EPS or rho <= MACHINE_EPS:
                    pdir_cosdeltaphi = 1.0
                    pdir_sindeltaphi = 0.0
                else:
                    pdir_cosdeltaphi = (
                        pdir_x/pdir_rho * dx/rho + pdir_y/pdir_rho * dy/rho
                    )
                    # Note that the max and min here here in case numerical
                    # precision issues cause the dot product to blow up.
                    pdir_cosdeltaphi = min(1, max(-1, pdir_cosdeltaphi))
                    pdir_sindeltaphi = math.sqrt(1 - pdir_cosdeltaphi * pdir_cosdeltaphi)

                #print('pdir_cosdeltaphi={}, pdir_sindeltaphi={}'
                #      .format(pdir_cosdeltaphi, pdir_sindeltaphi))

                # Cherenkov angle is encoded as the projection of a length-1
                # vector going in the Ckv direction onto the charged particle's
                # direction. Ergo, in the length of the pdir vector is the
                # cosine of the ckv angle.
                ckv_costheta = pdir_r
                ckv_theta = math.acos(ckv_costheta)

                #print('ckv_theta={}'.format(ckv_theta*180/PI))

                if table_kind == TBL_KIND_CLSIM:
                    if ckv_sigma_deg > 0:
                        pass
                        #this_photons_at_all_times, _a, _b = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                        #    theta=ckv_theta,
                        #    num_phi=num_phi_samples,
                        #    rot_costheta=pdir_costheta,
                        #    rot_sintheta=pdir_sintheta,
                        #    rot_cosphi=pdir_cosdeltaphi,
                        #    rot_sinphi=pdir_sindeltaphi,
                        #    directional_survival_prob=(
                        #        t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                        #    ),
                        #    num_costheta_bins=n_costhetadir_bins,
                        #    num_deltaphi_bins=n_deltaphidir_bins,
                        #    random_delta_thetas=random_delta_thetas
                        #)
                    else:
                        ckv_sintheta = math.sqrt(1 - ckv_costheta*ckv_costheta)
                        #this_photons_at_all_times, _a, _b = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                        #    costheta=ckv_costheta,
                        #    sintheta=ckv_sintheta,
                        #    num_phi=num_phi_samples,
                        #    rot_costheta=pdir_costheta,
                        #    rot_sintheta=pdir_sintheta,
                        #    rot_cosphi=pdir_cosdeltaphi,
                        #    rot_sinphi=pdir_sindeltaphi,
                        #    directional_survival_prob=(
                        #        t_indep_table[r_bin_idx, costheta_bin_idx, :, :]
                        #    ),
                        #    num_costheta_bins=n_costhetadir_bins,
                        #    num_deltaphi_bins=n_deltaphidir_bins,
                        #)

                elif table_kind == TBL_KIND_CKV:
                    costhetadir_bin_idx = int((pdir_costheta + 1) // table_dcosthetadir)
                    # Make upper edge inclusive
                    if costhetadir_bin_idx > last_costhetadir_bin_idx:
                        costhetadir_bin_idx = last_costhetadir_bin_idx

                    if costhetadir_bin_idx == prev_costhetadir_bin_idx:
                        new_costhetadir_bin = False
                    else:
                        new_costhetadir_bin = True
                        prev_costhetadir_bin_idx = costhetadir_bin_idx

                    pdir_deltaphi = math.acos(pdir_cosdeltaphi)
                    deltaphidir_bin_idx = int(pdir_deltaphi // table_dphidir)
                    # Make upper edge inclusive
                    if deltaphidir_bin_idx > last_deltaphidir_bin_idx:
                        deltaphidir_bin_idx = last_deltaphidir_bin_idx

                    if deltaphidir_bin_idx == prev_deltaphidir_bin_idx:
                        new_deltaphidir_bin = False
                    else:
                        new_deltaphidir_bin = True
                        prev_deltaphidir_bin_idx = deltaphidir_bin_idx

                    if new_r_bin or new_costheta_bin or new_costhetadir_bin or new_deltaphidir_bin:
                        new_r_ct_ctd_or_dpd_bin = True
                        #this_photons_at_all_times = t_indep_table[r_bin_idx, costheta_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx]
                    else:
                        new_r_ct_ctd_or_dpd_bin = False
                else:
                    raise ValueError('Unknown table kind.')

            elif pdir_r == 1.0: # line emitter; can't do this with Ckv table!
                raise NotImplementedError('Line emitter not handled.')

            else: # Gaussian emitter; can't do this with Ckv table!
                raise NotImplementedError('Gaussian emitter not handled.')

            #photons_at_all_times += p_count * t_indep_table_norm * this_photons_at_all_times

            #print('photons_at_all_times={}'.format(photons_at_all_times))

            # Causally impossible? (Note the comparison is written such that it
            # will evaluate to True if hit_time is NaN.)
            if not t <= hit_time:
                #print('XX CONTINUE: noncausal')
                continue

            # Is relative time outside binning?
            if dt >= t_max:
                #print('XX CONTINUE: outside t binning')
                continue

            t_bin_idx = int(dt // table_dt)

            #print('t_bin_idx={}'.format(t_bin_idx))

            if t_bin_idx == prev_t_bin_idx:
                new_t_bin = False
            else:
                new_t_bin = True
                prev_t_bin_idx = t_bin_idx

            if new_r_bin or new_t_bin:
                this_table_norm = table_norm[r_bin_idx, t_bin_idx]

            if pdir_r == 0.0: # isotropic emitter
                if new_pdir_r or new_r_bin or new_costheta_bin or new_t_bin:
                    this_photons_at_hit_time = np.mean(
                        table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                    )
            elif pdir_r < 1.0: # Cherenkov emitter
                if table_kind == TBL_KIND_CLSIM:
                    if ckv_sigma_deg > 0:
                        this_photons_at_hit_time, _c, _d = survival_prob_from_smeared_cone( # pylint: disable=unused-variable, invalid-name
                            theta=ckv_theta,
                            num_phi=num_phi_samples,
                            rot_costheta=pdir_costheta,
                            rot_sintheta=pdir_sintheta,
                            rot_cosphi=pdir_cosdeltaphi,
                            rot_sinphi=pdir_sindeltaphi,
                            directional_survival_prob=(
                                table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                            ),
                            num_costheta_bins=n_costhetadir_bins,
                            num_deltaphi_bins=n_deltaphidir_bins,
                            random_delta_thetas=random_delta_thetas
                        )
                    else:
                        this_photons_at_hit_time, _c, _d = survival_prob_from_cone( # pylint: disable=unused-variable, invalid-name
                            costheta=ckv_costheta,
                            sintheta=ckv_sintheta,
                            num_phi=num_phi_samples,
                            rot_costheta=pdir_costheta,
                            rot_sintheta=pdir_sintheta,
                            rot_cosphi=pdir_cosdeltaphi,
                            rot_sinphi=pdir_sindeltaphi,
                            directional_survival_prob=(
                                table[r_bin_idx, costheta_bin_idx, t_bin_idx, :, :]
                            ),
                            num_costheta_bins=n_costhetadir_bins,
                            num_deltaphi_bins=n_deltaphidir_bins,
                        )
                    #print('this_photons_at_hit_time={}'.format(this_photons_at_hit_time))
                elif table_kind == TBL_KIND_CKV:
                    if new_r_ct_ctd_or_dpd_bin or new_t_bin:
                        this_photons_at_hit_time = table[r_bin_idx, costheta_bin_idx, t_bin_idx, costhetadir_bin_idx, deltaphidir_bin_idx]

            elif pdir_r == 1.0: # line emitter
                raise NotImplementedError('Line emitter not handled.')
            else: # Gaussian emitter
                raise NotImplementedError('Gaussian emitter not handled.')

            photons_at_hit_time += p_count * this_table_norm * this_photons_at_hit_time
            #print('photons_at_hit_time={}'.format(photons_at_hit_time))
            #print('XX FINISHED LOOP')

        photons_at_all_times = quantum_efficiency * photons_at_all_times + noise_rate_hz * time_window * 1e-9
        photons_at_hit_time = quantum_efficiency * photons_at_hit_time + noise_rate_hz * table_dt * 1e-9

        return photons_at_all_times, photons_at_hit_time

    return pexp_5d, binning_info


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def pexp_t_r_theta(pinfo_gen, hit_time, dom_coord, survival_prob,
                   time_indep_survival_prob, t_min, t_max, n_t_bins, r_min,
                   r_max, r_power, n_r_bins, n_costheta_bins):
    """Compute expected photons in a DOM based on the (t,r,theta)-binned
    Retro DOM tables applied to a the generated photon info `pinfo_gen`,
    and the total expected photon count (time integrated) -- the normalization
    of the pdf.

    Parameters
    ----------
    pinfo_gen : shape (N, 8) numpy ndarray, dtype float64
    hit_time : float
    dom_coord : shape (3,) numpy ndarray, dtype float64
    survival_prob
    time_indep_survival_prob
    t_min : float
    t_max : float
    n_t_bins : int
    r_min : float
    r_max : float
    r_power : float
    n_r_bins : int
    n_costheta_bins : int

    Returns
    -------
    total_photon_count, expected_photon_count : (float, float)

    """
    table_dt = (t_max - t_min) / n_t_bins
    table_dcostheta = 2. / n_costheta_bins
    expected_photon_count = 0.
    total_photon_count = 0.
    inv_r_power = 1. / r_power
    table_dr_pwr = (r_max-r_min)**inv_r_power / n_r_bins

    rsquared_max = r_max*r_max
    rsquared_min = r_min*r_min

    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable

        # A photon that starts immediately in the past (before the DOM was hit)
        # will show up in the Retro DOM tables in the _last_ bin.
        # Therefore, invert the sign of the t coordinate and index sequentially
        # via e.g. -1, -2, ....
        dt = t - hit_time
        dx = x - dom_coord[0]
        dy = y - dom_coord[1]
        dz = z - dom_coord[2]

        rsquared = dx**2 + dy**2 + dz**2
        # we can already continue before computing the bin idx
        if rsquared > rsquared_max:
            continue
        if rsquared < rsquared_min:
            continue

        r = math.sqrt(rsquared)

        #spacetime_sep = SPEED_OF_LIGHT_M_PER_NS*dt - r
        #if spacetime_sep < 0 or spacetime_sep >= retro.POL_TABLE_RMAX:
        #    print('spacetime_sep:', spacetime_sep)
        #    print('retro.MAX_POL_TABLE_SPACETIME_SEP:', retro.POL_TABLE_RMAX)
        #    continue

        r_bin_idx = int((r-r_min)**inv_r_power / table_dr_pwr)
        #print('r_bin_idx: ',r_bin_idx)
        #if r_bin_idx < 0 or r_bin_idx >= n_r_bins:
        #    #print('r at ',r,'with idx ',r_bin_idx)
        #    continue

        costheta_bin_idx = int((1 -(dz / r)) / table_dcostheta)
        #print('costheta_bin_idx: ',costheta_bin_idx)
        #if costheta_bin_idx < 0 or costheta_bin_idx >= n_costheta_bins:
        #    print('costheta out of range! This should not happen')
        #    continue

        # time indep.
        time_indep_count = (
            p_count * time_indep_survival_prob[r_bin_idx, costheta_bin_idx]
        )
        total_photon_count += time_indep_count

        # causally impossible
        if hit_time < t:
            continue

        t_bin_idx = int(np.floor((dt - t_min) / table_dt))
        #print('t_bin_idx: ',t_bin_idx)
        #if t_bin_idx < -n_t_bins or t_bin_idx >= 0:
        #if t_bin_idx < 0 or t_bin_idx >= -retro.POL_TABLE_DT:
        if t_bin_idx > n_t_bins or t_bin_idx < 0:
            #print('t')
            #print('t at ',t,'with idx ',t_bin_idx)
            continue

        #print(t_bin_idx, r_bin_idx, thetabin_idx)
        #raise Exception()
        surviving_count = (
            p_count * survival_prob[t_bin_idx, r_bin_idx, costheta_bin_idx]
        )

        #print(surviving_count)

        # TODO: Include simple ice photon prop asymmetry here? Might need to
        # use both phi angle relative to DOM _and_ photon directionality
        # info...

        expected_photon_count += surviving_count

    return total_photon_count, expected_photon_count


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def pexp_xyz(pinfo_gen, x_min, y_min, z_min, nx, ny, nz, binwidth,
             survival_prob, avg_photon_x, avg_photon_y, avg_photon_z,
             use_directionality):
    """Compute the expected number of detected photons in _all_ DOMs at _all_
    times.

    Parameters
    ----------
    pinfo_gen :
    x_min, y_min, z_min :
    nx, ny, nz :
    binwidth :
    survival_prob :
    avg_photon_x, avg_photon_y, avg_photon_z :
    use_directionality : bool

    """
    expected_photon_count = 0.0
    for pgen_idx in range(pinfo_gen.shape[0]):
        t, x, y, z, p_count, p_x, p_y, p_z = pinfo_gen[pgen_idx, :] # pylint: disable=unused-variable
        x_idx = int(np.round((x - x_min) / binwidth))
        if x_idx < 0 or x_idx >= nx:
            continue
        y_idx = int(np.round((y - y_min) / binwidth))
        if y_idx < 0 or y_idx >= ny:
            continue
        z_idx = int(np.round((z - z_min) / binwidth))
        if z_idx < 0 or z_idx >= nz:
            continue
        sp = survival_prob[x_idx, y_idx, z_idx]
        surviving_count = p_count * sp

        # TODO: Incorporate photon direction info
        if use_directionality:
            raise NotImplementedError('Directionality cannot be used yet')

        expected_photon_count += surviving_count

    return expected_photon_count


class CKVTables(object):
    """
    Class to interact with and obtain photon survival probabilities from a set
    of Retro 5D Cherenkov tables.

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

    norm_version : string
        (Temporary) Which version of the norm to use. Only for experimenting,
        and will be removed once we figure the norm out.

    """
    def __init__(
            self, geom, rde, noise_rate_hz, use_directionality, norm_version
        ):
        assert len(geom.shape) == 3
        self.geom = geom
        self.use_directionality = use_directionality
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
            Path to the directory containing the table's .npy files.

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
            Whether to attempt to memory map the table.

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

        table = load_ckv_table(
            fpath=fpath,
            step_length=step_length,
            mmap=mmap
        )

        table['step_length'] = step_length
        table['table_norm'] = get_table_norm(
            angular_acceptance_fract=angular_acceptance_fract,
            quantum_efficiency=1,
            norm_version=self.norm_version,
            **{k: table[k] for k in TABLE_NORM_KEYS}
        )
        #table['t_indep_table_norm'] = angular_acceptance_fract

        pexp_5d, _ = generate_pexp_5d_function(
            table=table,
            table_kind=TBL_KIND_CKV,
            use_directionality=self.use_directionality,
            num_phi_samples=0,
            ckv_sigma_deg=0
        )

        # NOTE: original tables have underflow (bin 0) and overflow (bin -1)
        # bins, so whole-axis slices must exclude the first and last bins.
        self.tables[(string, dom)] = (
            pexp_5d,
            #table['t_indep_ckv_table'],
            #table['t_indep_table_norm'],
            table['ckv_table'],
            table['table_norm'],
        )

    #@profile
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
         #t_indep_ckv_table,
         #t_indep_table_norm,
         ckv_table,
         table_norm) = self.tables[(string, dom)]

        return pexp_5d(
            pinfo_gen=pinfo_gen,
            hit_time=hit_time,
            time_window=time_window,
            dom_coord=dom_coord,
            noise_rate_hz=dom_noise_rate_hz,
            quantum_efficiency=dom_quantum_efficiency,
            table=ckv_table,
            table_norm=table_norm,
            #t_indep_table=t_indep_ckv_table,
            #t_indep_table_norm=t_indep_table_norm,
        )


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
            t_indep_table=t_indep_table,
            t_indep_table_norm=t_indep_table_norm,
        )


# TODO: remove this class!
class CLSimTable(object):
    """Load and use information from a single "raw" individual-DOM
    (time, r, theta, thetadir, deltaphidir)-binned Retro table.

    Note that this is the table generated by CLSim, prior to any manipulations
    performed for using the table with Retro.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    string : int
        Indexed from 1

    depth_idx : int

    angular_acceptance_fract : float
        Normalization for angular acceptance being less than 1

    naming_version : int or None
        Version of naming for CLSim table (original is 0). Passing None uses
        the latest version. Note that any derived tables use the latest naming
        version regardless of what is passed here.

    """
    def __init__(self, fpath=None, tables_dir=None, hash_val=None, string=None,
                 depth_idx=None, seed=None,
                 angular_acceptance_fract=0.338019664877, naming_version=None):

        print('WARNING: This class is deprecated and will be removed soon!')

        # Translation and validation of args
        assert 0 < angular_acceptance_fract <= 1
        self.angular_acceptance_fract = angular_acceptance_fract

        if naming_version is None:
            naming_version = len(retro.CLSIM_TABLE_FNAME_PROTO) - 1
        fname_proto = retro.CLSIM_TABLE_FNAME_PROTO[naming_version]

        if fpath is None:
            tables_dir = retro.expand(tables_dir)
            assert isdir(tables_dir)
            self.tables_dir = tables_dir
            self.hash_val = hash_val
            self.seed = seed

            if isinstance(string, basestring):
                self.string = string.strip().lower()
                assert self.string in ('ic', 'dc')
                self.subdet = self.string
                self.string_idx = None

            else:
                self.string = string
                self.string_idx = self.string - 1

                if self.string_idx < 79:
                    self.subdet = 'ic'
                else:
                    self.subdet = 'dc'

            self.depth_idx = depth_idx

            fname = fname_proto.format(
                hash_val=hash_val,
                string=self.string,
                depth_idx=self.depth_idx,
                seed=self.seed
            )
            self.fpath = join(self.tables_dir, fname)

        else:
            assert tables_dir is None
            assert hash_val is None
            assert string is None
            assert depth_idx is None

            self.fpath = fpath
            self.tables_dir = dirname(fpath)
            info = retro.interpret_clsim_table_fname(fpath)
            self.string = info['string']
            if isinstance(self.string, int):
                self.string_idx = self.string - 1
            else:
                self.string_idx = None
            self.hash_val = info['hash_val']
            self.depth_idx = info['depth_idx']
            self.seed = info['seed']

        assert self.subdet in ('ic', 'dc')
        if self.subdet == 'ic':
            self.quantum_efficiency = retro.IC_DOM_QUANT_EFF
        elif self.subdet == 'dc':
            self.quantum_efficiency = retro.DC_DOM_QUANT_EFF

        self.dtp_fname_proto = retro.RETRO_DOM_TABLE_FNAME_PROTO[-1]

        table_info = load_clsim_table(
            self.fpath, step_length=1,
            angular_acceptance_fract=self.angular_acceptance_fract,
            quantum_efficiency=self.quantum_efficiency
        )

        self.table = table_info.pop('table')
        self.table_shape = table_info.pop('table_shape')
        self.n_dims = len(self.table_shape)
        self.n_photons = table_info.pop('n_photons')
        self.phase_refractive_index = table_info.pop('phase_refractive_index')
        self.angular_acceptance_fract = angular_acceptance_fract

        self.r_bin_edges = table_info.pop('r_bin_edges')
        self.t_bin_edges = table_info.pop('t_bin_edges')
        self.costheta_bin_edges = table_info.pop('costheta_bin_edges')

        self.theta_bin_edges = np.arccos(self.costheta_bin_edges) # radians
        self.costheta_centers = retro.linear_bin_centers(self.costheta_bin_edges)
        self.theta_centers = np.arccos(self.costheta_centers) # radians

        t_bin_widths = np.diff(self.t_bin_edges)
        assert np.allclose(t_bin_widths, t_bin_widths[0])
        self.t_bin_width = np.mean(t_bin_widths)

        if self.n_dims == 5:
            self.costhetadir_bin_edges = table_info.pop('costhetadir_bin_edges')
            self.deltaphidir_bin_edges = table_info.pop('deltaphidir_bin_edges') # rad
            self.costhetadir_centers = retro.linear_bin_centers(self.costhetadir_bin_edges)
            self.thetadir_bin_edges = np.arccos(self.costhetadir_bin_edges) # rad
            self.thetadir_centers = np.arccos(self.costhetadir_centers) # rad
            self.deltaphidir_centers = retro.linear_bin_centers(self.deltaphidir_bin_edges) # rad
        else:
            raise NotImplementedError(
                'Can only work with CLSim tables with 5 dimensions; got %d'
                % self.n_dims
            )

        self.table_norm = table_info.pop('table_norm')

        del table_info

        #self.norm = (
        #    1
        #    / self.n_photons
        #    / (retro.SPEED_OF_LIGHT_M_PER_NS / self.phase_refractive_index
        #       * self.t_bin_width)
        #    * self.angular_acceptance_fract
        #    * (len(self.costheta_bin_edges) - 1)
        #)

        # The photon direction is tabulated in dimensions 3 and 4
        #self.survival_prob = self.data.sum(axis=(3, 4)) * self.table_norm

    def export_t_r_theta_table(self, outdir=None, overwrite=True):
        """Distill binned photon directionality information into a single
        vector per bin and force azimuthal symmetry to reduce the table from a
        5D histogram of photon counts binned in
        (t, r, costheta, phi, thetadir, deltaphidir) to a 3D histogram binned in
        (t, r, costheta) where each bin contains a probability and an average
        direction vector.

        The resulting file will be placed in the same directory as the source
        table and the file name will be the source filename suffixed by
        "_r_cz_t_angles" (prior to the extension).

        Parameters
        ----------
        outdir : string, optional
            If specified, store the DOM-time-polar table into this directory.
            Otherwise, if not specified, the table is stored in the same
            directory as the source table.

        """
        import pyfits

        if outdir is None:
            outdir = self.tables_dir
        outdir = retro.expand(outdir)

        new_fname = self.dtp_fname_proto.format(depth_idx=self.depth_idx)
        new_fpath = join(outdir, new_fname)

        if not isdir(outdir):
            retro.mkdir(outdir)

        if isfile(new_fpath):
            if overwrite:
                retro.wstderr('WARNING: overwriting existing file at "%s"\n' % new_fpath)
                remove(new_fpath)
            else:
                retro.wstderr(
                    'ERROR: There is an existing file at "%s"; not'
                    ' proceeding.\n' % new_fpath
                )
                return

        survival_probs, average_thetas, average_phis, lengths = \
            generate_t_r_theta_table(
                table=self.table,
                n_photons=self.n_photons,
                phase_refractive_index=self.phase_refractive_index,
                t_bin_width=self.t_bin_width,
                angular_acceptance_fract=self.angular_acceptance_fract,
                thetadir_centers=self.thetadir_centers,
                deltaphidir_centers=self.deltaphidir_centers,
                theta_bin_edges=self.theta_bin_edges
            )
        objects = [
            pyfits.PrimaryHDU(survival_probs),
            pyfits.ImageHDU(average_thetas.astype(np.float32)),
            pyfits.ImageHDU(average_phis.astype(np.float32)),
            pyfits.ImageHDU(lengths.astype(np.float32)),
            pyfits.ImageHDU(self.t_bin_edges.astype(np.float32)),
            pyfits.ImageHDU(self.r_bin_edges.astype(np.float32)),
            pyfits.ImageHDU(self.theta_bin_edges[::-1].astype(np.float32))
        ]

        hdulist = pyfits.HDUList(objects)
        hdulist.writeto(new_fpath)


class DOMTimePolarTables(object):
    """Load and use information from individual-dom (t,r,theta)-binned Retro
    tables.

    Parameters
    ----------
    tables_dir : string

    hash_val : None or string
        Hash string identifying the source Retro tables to use.

    geom : shape (n_strings, n_depths, 3) numpy ndarray, dtype float64

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts.

    ic_exponent, dc_exponent : float >= 0, optional
        Modify probabilties in the table by ``prob = 1 - (1 - prob)**exponent``
        to allow for up- and down-scaling the efficiency of the DOMs.
        `ic_exponent` is applied to IceCube (non-DeepCore) DOMs and
        `dc_exponent` is applied to DeepCore DOMs. Note that this is applied to
        each DOM's table after the appropriate quantum efficiency scale factor
        has already been applied (quantum efficiency is applied as a simple
        multiplier; see :attr:`retro.IC_DOM_QUANT_EFF` and
        :attr:`retro.DC_DOM_QUANT_EFF`).

    naming_version : int or None
        Version of naming for single-DOM+directionality tables (original is 0).
        Passing None uses the latest version. Note that any derived tables use
        the latest naming version regardless of what is passed here.

    """
    def __init__(self, tables_dir, hash_val, geom, use_directionality,
                 ic_exponent=1, dc_exponent=1, naming_version=None):
        # Translation and validation of args
        tables_dir = retro.expand(tables_dir)
        assert isdir(tables_dir)
        assert len(geom.shape) == 3
        assert isinstance(use_directionality, bool)
        assert ic_exponent >= 0
        assert dc_exponent >= 0
        if naming_version is None:
            naming_version = len(retro.RETRO_DOM_TABLE_FNAME_PROTO) - 1
        self.naming_version = naming_version
        self.dom_table_fname_proto = retro.RETRO_DOM_TABLE_FNAME_PROTO[naming_version]

        self.tables_dir = tables_dir
        self.hash_val = hash_val
        self.geom = geom
        self.use_directionality = use_directionality
        self.ic_exponent = ic_exponent
        self.dc_exponent = dc_exponent
        self.tables = {'ic': {}, 'dc': {}}
        self.bin_edges = {'ic': {}, 'dc': {}}

    def load_table(self, string, dom, force_reload=False):
        """Load a table from disk into memory.

        Parameters
        ----------
        string : int in [1, 86]
            Indexed from 1, currently 1-86. Strings 1-78 are "regular" IceCube
            strings, while strings 79-86 are DeepCore strings. (It should be
            noted, though, that strings 79 and 80 are considered in-fill
            strings, with "a mix of high quantum-efficiency and standard DOMs";
            which are which is _not_ taken into consideration in the software
            yet.)

        dom : int in [1, 60]
            Indexed from 0, currently 0-59

        force_reload : bool

        """
        if string < 79:
            subdet = 'ic'
            dom_quant_eff = retro.IC_DOM_QUANT_EFF
            exponent = self.ic_exponent
        else:
            subdet = 'dc'
            dom_quant_eff = retro.DC_DOM_QUANT_EFF
            exponent = self.dc_exponent

        if not force_reload and dom in self.tables[subdet]:
            return

        depth_idx = dom - 1
        if self.naming_version == 0:
            fpath = join(
                self.tables_dir,
                self.dom_table_fname_proto.format(
                    string=subdet.upper(), depth_idx=depth_idx
                )
            )
        elif self.naming_version == 1:
            raise NotImplementedError()
            #fpath = join(
            #    self.tables_dir,
            #    self.dom_table_fname_proto.format(
            #        hash_val=self.hash_val,
            #        string=subdet,
            #        depth_idx=depth_idx,
            #        seed=seed, # TODO
            #    )
            #)
        else:
            raise NotImplementedError()

        photon_info, bin_edges = load_t_r_theta_table(
            fpath=fpath,
            depth_idx=depth_idx,
            scale=dom_quant_eff,
            exponent=exponent
        )

        #length = photon_info.length[depth_idx]
        #deltaphi = photon_info.deltaphi[depth_idx]
        self.tables[subdet][depth_idx] = retro.RetroPhotonInfo(
            survival_prob=photon_info.survival_prob[depth_idx],
            time_indep_survival_prob=photon_info.time_indep_survival_prob[depth_idx],
            theta=photon_info.theta[depth_idx],
            deltaphi=photon_info.deltaphi[depth_idx],
            length=(photon_info.length[depth_idx]
                    * np.cos(photon_info.deltaphi[depth_idx]))
        )

        self.bin_edges[subdet][depth_idx] = bin_edges

    def load_tables(self):
        """Load all tables"""
        # TODO: parallelize the loading of each table to reduce CPU overhead
        # time (though most time I expect to be disk-read times, this could
        # still help speed the process up)
        for string in range(1, 86+1):
            for dom in range(1, 60+1):
                self.load_table(string=string, dom=dom, force_reload=False)

    def get_photon_expectation(self, pinfo_gen, hit_time, string, dom,
                               use_directionality=None):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy ndarray, dtype float64

        use_directionality : None or bool
            Whether to use photon directionality informatino in hypo / table to
            modify expected surviving photon counts. If specified, overrides
            argument passed at class instantiation time. Otherwise, that value
            for `use_directionality` is used.

        Returns
        -------
        total_photon_count, expected_photon_count : (float, float)
            Total expected surviving photon count

        """
        if use_directionality is None:
            use_directionality = self.use_directionality

        string_idx = string - 1
        depth_idx = dom - 1

        dom_coord = self.geom[string_idx, depth_idx]
        if string < 79:
            subdet = 'ic'
        else:
            subdet = 'dc'
        table = self.tables[subdet][depth_idx]
        bin_edges = self.bin_edges[subdet][depth_idx]
        survival_prob = table.survival_prob
        time_indep_survival_prob = table.time_indep_survival_prob
        return pexp_t_r_theta(pinfo_gen=pinfo_gen,
                              hit_time=hit_time,
                              dom_coord=dom_coord,
                              survival_prob=survival_prob,
                              time_indep_survival_prob=time_indep_survival_prob,
                              t_min=bin_edges.t[0],
                              t_max=bin_edges.t[-1],
                              n_t_bins=len(bin_edges.t)-1,
                              r_min=bin_edges.r[0],
                              r_max=bin_edges.r[-1],
                              r_power=2,
                              n_r_bins=len(bin_edges.r)-1,
                              n_costheta_bins=len(bin_edges.theta)-1)


# TODO: convert to using exponent rather than scale (scale will be applied via
# dom_quant_eff when generating the TDI table in the first place; at this
# stage, we want to go either up or down with probabilities, so a single
# exponent should be appropriate, while a scale factor can exceed 1 for
# probabilities).
class TDICartTable(object):
    """Load and use information from a time- and DOM-independent Cartesian
    (x, y, z)-binned Retro table.

    The parameters used to generate the table are passed at instantiation of
    this class to determine which table(s) to load when a table is requested
    (multiple "tables" are loaded if a single table is generated from multiple
    smaller tiles meant to be stitched together).

    Parameters
    ----------
    tables_dir : string

    proto_tile_hash : string
        Hash value used to locate files in the `tables_dir` which contain tiles
        relevant to the table being loaded.

    scale : float from 0 to 1, optional
        Scale factor by which to multiply the detection probabilities in the
        table.

    subvol : None or sequence of 3 2-element sequences, optional
        Specify (min, max) values for the x-, y-, and z-dimensions to load only
        this portion of the large table. If None, load the entire table

    use_directionality : bool
        Whether to use photon directionality information from the hypothesis
        and table to modify the expected surviving photon counts. Note that if
        directionality is not to be used, the corresponding tables will not be
        loaded, resulting in ~1/4 the memory footprint.

    """
    def __init__(self, tables_dir, proto_tile_hash, subvol=None, scale=1,
                 use_directionality=True):
        # Translation and validation of args
        tables_dir = retro.expand(tables_dir)
        assert isdir(tables_dir)
        assert isinstance(use_directionality, bool)
        assert isinstance(proto_tile_hash, basestring)
        assert scale > 0

        self.tables_dir = tables_dir
        self.use_directionality = use_directionality
        self.proto_tile_hash = proto_tile_hash
        self.scale = scale

        self.survival_prob = None
        self.avg_photon_x = None
        self.avg_photon_y = None
        self.avg_photon_z = None

        self.x_min, self.y_min, self.z_min = None, None, None
        self.x_max, self.y_max, self.z_max = None, None, None
        self.nx, self.ny, self.nz = None, None, None # pylint: disable=invalid-name
        self.nx_tiles, self.ny_tiles, self.nz_tiles = None, None, None
        self.nx_per_tile, self.ny_per_tile, self.nz_per_tile = None, None, None

        self.tables_meta = None

        proto_table_fpath = glob(join(
            retro.expand(self.tables_dir),
            'retro_tdi_table_%s_*survival_prob.fits' % self.proto_tile_hash
        ))
        if not proto_table_fpath:
            raise ValueError('Could not find the prototypical table.')
        proto_table_fpath = proto_table_fpath[0]
        proto_meta = self.get_table_metadata(proto_table_fpath)
        if not proto_meta:
            raise ValueError('Could not figure out metadata from\n%s'
                             % proto_table_fpath)
        self.proto_meta = proto_meta

        # Some "universal" metadata can be gotten from the proto table
        self.binmap_hash = proto_meta['binmap_hash']
        self.geom_hash = proto_meta['geom_hash']
        self.dom_tables_hash = proto_meta['dom_tables_hash']
        self.time_indices = proto_meta['time_indices']
        self.times_str = proto_meta['times_str']
        self.x_tile_width = proto_meta['x_width']
        self.y_tile_width = proto_meta['y_width']
        self.z_tile_width = proto_meta['z_width']
        self.binwidth = proto_meta['binwidth']
        self.anisotropy = proto_meta['anisotropy']

        if subvol is not None:
            raise NotImplementedError()
            sv_x0, sv_x1 = subvol[0][0], subvol[0][1]
            sv_y0, sv_y1 = subvol[1][0], subvol[1][1]
            sv_z0, sv_z1 = subvol[2][0], subvol[2][1]
            assert sv_x1 - sv_x0 >= self.binwidth
            assert sv_y1 - sv_y0 >= self.binwidth
            assert sv_z1 - sv_z0 >= self.binwidth
            sv_x0_idx = (sv_x0 - proto_meta['x_min']) / self.binwidth
            sv_y0_idx = (sv_y0 - proto_meta['y_min']) / self.binwidth
            sv_z0_idx = (sv_z0 - proto_meta['z_min']) / self.binwidth
            assert abs(np.round(sv_x0_idx) - sv_x0_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_y0_idx) - sv_y0_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_z0_idx) - sv_z0_idx) * self.binwidth < 1e-6
            sv_x1_idx = (sv_x1 - proto_meta['x_min']) / self.binwidth
            sv_y1_idx = (sv_y1 - proto_meta['y_min']) / self.binwidth
            sv_z1_idx = (sv_z1 - proto_meta['z_min']) / self.binwidth
            assert abs(np.round(sv_x1_idx) - sv_x1_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_y1_idx) - sv_y1_idx) * self.binwidth < 1e-6
            assert abs(np.round(sv_z1_idx) - sv_z1_idx) * self.binwidth < 1e-6
        self.subvol = subvol

        self.tables_loaded = False
        self.load_tables()

    @staticmethod
    def get_table_metadata(fpath):
        """Interpret a Retro TDI table filename or path, returning the critical
        parameters used in generating that table.

        Parameters
        ----------
        fpath : string
            Path to the file or simply the filename

        Returns
        -------
        info : None or dict
            None is returned if the file name/path does not match the format
            for a TDI table.

        """
        fname = basename(fpath)
        match = retro.TDI_TABLE_FNAME_RE[-1].match(fname)
        if match is None:
            return None
        meta = match.groupdict()
        for key, value in meta.items():
            if key.endswith('min') or key.endswith('max') or key == 'binwidth':
                meta[key] = float(meta[key])
            elif key == 'n_phibins':
                meta[key] = int(meta[key])
            elif key.endswith('scale'):
                meta[key] = float(meta[key])
            elif key.endswith('hash'):
                if value.lower() == 'none':
                    meta[key] = None
            elif key == 'times_str':
                if value == 'all':
                    meta['time_indices'] = slice(None)
                else:
                    meta['time_indices'] = hrlist2list(value)
            elif key == 'anisotropy':
                if value.lower() == 'none':
                    meta[key] = None
                # TODO: implement any other anisotropy spec xlation here
        meta['x_width'] = meta['x_max'] - meta['x_min']
        meta['y_width'] = meta['y_max'] - meta['y_min']
        meta['z_width'] = meta['z_max'] - meta['z_min']
        return meta

    def load_tables(self, force_reload=False):
        """Load all tables that match the `proto_tile_hash`; if multiple tables
        match, then stitch these together into one large TDI table."""
        if self.tables_loaded and not force_reload:
            return
        import pyfits

        t0 = time()

        x_ref = self.proto_meta['x_min']
        y_ref = self.proto_meta['y_min']
        z_ref = self.proto_meta['z_min']

        must_match = [
            'binmap_hash', 'geom_hash', 'dom_tables_hash', 'time_indices',
            'binwidth', 'anisotropy',
            # For simplicity, assume all tiles have equal widths. If there's a
            # compelling reason to use something more complicated, we could
            # implement it... but I see no reason to do so now
            'x_width', 'y_width', 'z_width'
        ]

        # Work with "survival_prob" table filepaths, which generalizes to all
        # table filepaths (so long as they exist)
        fpaths = glob(join(
            retro.expand(self.tables_dir),
            'retro_tdi_table_*survival_prob.fits'
        ))

        lowermost_corner = np.array([np.inf]*3)
        uppermost_corner = np.array([-np.inf]*3)
        to_load_meta = {}
        for fpath in fpaths:
            meta = self.get_table_metadata(fpath)
            if meta is None:
                continue

            is_match = True
            for key in must_match:
                if meta[key] != self.proto_meta[key]:
                    is_match = False

            if not is_match:
                continue

            # Make sure that the corner falls on the reference grid (within
            # micrometer precision)
            x_float_idx = (meta['x_min'] - x_ref) / self.x_tile_width
            y_float_idx = (meta['y_min'] - y_ref) / self.y_tile_width
            z_float_idx = (meta['z_min'] - z_ref) / self.x_tile_width
            indices_widths = (
                [x_float_idx, y_float_idx, z_float_idx],
                [self.x_tile_width, self.y_tile_width, self.z_tile_width]
            )
            for float_idx, tile_width in zip(indices_widths):
                if abs(np.round(float_idx) - float_idx) * tile_width >= 1e-6:
                    continue

            # Extend the limits of the tiled volume to include this tile
            lower_corner = [meta['x_min'], meta['y_min'], meta['z_min']]
            upper_corner = [meta['x_max'], meta['y_max'], meta['z_max']]
            lowermost_corner = np.min([lowermost_corner, lower_corner], axis=0)
            uppermost_corner = np.max([uppermost_corner, upper_corner], axis=0)

            # Store the metadata by relative tile index
            rel_idx = tuple(int(np.round(i))
                            for i in (x_float_idx, y_float_idx, z_float_idx))
            to_load_meta[rel_idx] = meta

        x_min, y_min, z_min = lowermost_corner
        x_max, y_max, z_max = uppermost_corner

        # Figure out how many tiles we _should_ have
        nx_tiles = int(np.round((x_max - x_min) / self.x_tile_width))
        ny_tiles = int(np.round((y_max - y_min) / self.y_tile_width))
        nz_tiles = int(np.round((z_max - z_min) / self.z_tile_width))
        n_tiles = nx_tiles * ny_tiles * nz_tiles
        if len(to_load_meta) < n_tiles:
            raise ValueError('Not enough tiles found! Cannot fill the extents'
                             ' of the outermost extents of the volume defined'
                             ' by the tiles found.')
        elif len(to_load_meta) > n_tiles:
            print(self.proto_meta['tdi_hash'])
            print('x:', self.proto_meta['x_min'], self.proto_meta['x_max'],
                  self.proto_meta['x_width'])
            print('y:', self.proto_meta['y_min'], self.proto_meta['y_max'],
                  self.proto_meta['y_width'])
            print('z:', self.proto_meta['z_min'], self.proto_meta['z_max'],
                  self.proto_meta['z_width'])
            print('')
            for v in to_load_meta.values():
                print(v['tdi_hash'])
                print('x:', v['x_min'], v['x_max'], v['x_width'])
                print('y:', v['y_min'], v['y_max'], v['y_width'])
                print('z:', v['z_min'], v['z_max'], v['z_width'])
                print('')
            raise ValueError(
                'WTF? How did we get here? to_load_meta = %d, n_tiles = %d'
                % (len(to_load_meta), n_tiles)
            )

        # Figure out how many bins in each dimension fill the volume
        nx = int(np.round(nx_tiles * self.x_tile_width / self.binwidth))
        ny = int(np.round(ny_tiles * self.y_tile_width / self.binwidth))
        nz = int(np.round(nz_tiles * self.z_tile_width / self.binwidth))

        # Number of bins per dimension in the tile
        nx_per_tile = int(np.round(self.x_tile_width / self.binwidth))
        ny_per_tile = int(np.round(self.y_tile_width / self.binwidth))
        nz_per_tile = int(np.round(self.z_tile_width / self.binwidth))

        # Create empty arrays to fill
        survival_prob = np.empty((nx, ny, nz), dtype=np.float32)
        if self.use_directionality:
            avg_photon_x = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_y = np.empty((nx, ny, nz), dtype=np.float32)
            avg_photon_z = np.empty((nx, ny, nz), dtype=np.float32)
        else:
            avg_photon_x, avg_photon_y, avg_photon_z = None, None, None

        anisotropy_str = retro.generate_anisotropy_str(self.anisotropy)

        tables_meta = {} #[[[None]*nz_tiles]*ny_tiles]*nx_tiles
        for meta in to_load_meta.values():
            tile_x_idx = int(np.round(
                (meta['x_min'] - x_min) / self.x_tile_width
            ))
            tile_y_idx = int(np.round(
                (meta['y_min'] - y_min) / self.y_tile_width
            ))
            tile_z_idx = int(np.round(
                (meta['z_min'] - z_min) / self.z_tile_width
            ))

            x0_idx = int(np.round((meta['x_min'] - x_min) / self.binwidth))
            y0_idx = int(np.round((meta['y_min'] - y_min) / self.binwidth))
            z0_idx = int(np.round((meta['z_min'] - z_min) / self.binwidth))

            bin_idx_range = (slice(x0_idx, x0_idx + nx_per_tile),
                             slice(y0_idx, y0_idx + ny_per_tile),
                             slice(z0_idx, z0_idx + nz_per_tile))

            kwargs = deepcopy(meta)
            kwargs.pop('table_name')

            to_fill = [('survival_prob', survival_prob)]
            if self.use_directionality:
                to_fill.extend([
                    ('avg_photon_x', avg_photon_x),
                    ('avg_photon_y', avg_photon_y),
                    ('avg_photon_z', avg_photon_z)
                ])

            for table_name, table in to_fill:
                fpath = join(
                    self.tables_dir,
                    retro.TDI_TABLE_FNAME_PROTO[-1].format(
                        table_name=table_name, anisotropy_str=anisotropy_str,
                        **kwargs
                    ).lower()
                )

                with pyfits.open(fpath) as fits_table:
                    data = retro.force_little_endian(fits_table[0].data) # pylint: disable=no-member

                if self.scale != 1 and table_name == 'survival_prob':
                    data = 1 - (1 - data)**self.scale

                table[bin_idx_range] = data

            tables_meta[(tile_x_idx, tile_y_idx, tile_z_idx)] = meta

        # Since we have made it to the end successfully, it is now safe to
        # store the above-computed info to the object for later use
        self.nx, self.ny, self.nz = nx, ny, nz
        self.nx_tiles = nx_tiles
        self.ny_tiles = ny_tiles
        self.nz_tiles = nz_tiles
        self.n_bins = self.nx * self.ny * self.nz
        self.n_tiles = self.nx_tiles * self.ny_tiles * self.nz_tiles
        self.x_min, self.y_min, self.z_min = x_min, y_min, z_min
        self.x_max, self.y_max, self.z_max = x_max, y_max, z_max

        self.survival_prob = survival_prob
        self.avg_photon_x = avg_photon_x
        self.avg_photon_y = avg_photon_y
        self.avg_photon_z = avg_photon_z

        self.tables_meta = tables_meta
        self.tables_loaded = True

        if self.n_tiles == 1:
            tstr = 'tile'
        else:
            tstr = 'tiles'
        print('Loaded %d %s spanning'
              ' x ∈ [%.2f, %.2f) m,'
              ' y ∈ [%.2f, %.2f) m,'
              ' z ∈ [%.2f, %.2f) m;'
              ' bins are (%.3f m)³'
              % (self.n_tiles, tstr, self.x_min, self.x_max, self.y_min,
                 self.y_max, self.z_min, self.z_max, self.binwidth))
        print('Time to load: {} s'.format(np.round(time() - t0, 3)))

    def get_photon_expectation(self, pinfo_gen):
        """Get the expectation for photon survival.

        Parameters
        ----------
        pinfo_gen : shape (N, 8) numpy ndarray, dtype float64

        Returns
        -------
        expected_photon_count : float
            Total expected surviving photon count

        """
        if not self.tables_loaded:
            raise Exception("Tables haven't been loaded")

        kwargs = dict(
            pinfo_gen=pinfo_gen,
            x_min=self.x_min, y_min=self.y_min, z_min=self.z_min,
            nx=self.nx, ny=self.ny, nz=self.nz,
            binwidth=self.binwidth,
            survival_prob=self.survival_prob,
            avg_photon_x=self.avg_photon_x,
            avg_photon_y=self.avg_photon_y,
            avg_photon_z=self.avg_photon_z,
            use_directionality=self.use_directionality
        )
        photon_expectation = pexp_xyz(**kwargs)

        return photon_expectation

    #def plot_slices(self, x_slice=slice(None), y_slice=slice(None),
    #                z_slice=slice(None)):
    #    # Formulate a slice through the table to look at
    #    slx = slice(dom_x_idx - ncells,
    #                dom_x_idx + ncells,
    #                1)
    #    sly = slice(dom_y_idx - ncells,
    #                dom_y_idx + ncells,
    #                1)
    #    slz = dom_z_idx
    #    sl = (x_slice, y_slice, slz)

    #    # Slice the x and y directions
    #    pxsl = binned_px[sl]
    #    pysl = binned_py[sl]

    #    xmid = (xlims[0] + x_bw/2.0 + x_bw * np.arange(nx))[x_slice]
    #    ymid = (ylims[0] + y_bw/2.0 + y_bw * np.arange(ny))[y_slice]
    #    zmid = zlims[0] + z_bw/2.0 + z_bw * dom_z_idx

    #    x_inner_lim = (xmid.min() - x_bw/2.0, xmid.max() + x_bw/2.0)
    #    y_inner_lim = (ymid.min() - y_bw/2.0, ymid.max() + y_bw/2.0)
    #    X, Y = np.meshgrid(xmid, ymid, indexing='ij')

    #    fig = plt.figure(1, figsize=(10, 10), dpi=72)
    #    fig.clf()
    #    ax = fig.add_subplot(111)

    #    ax.plot(
    #        dom_x, dom_y,
    #        'ro', ms=8, lw=0.5,
    #        label='Actual DOM location'
    #    )
    #    ax.plot(
    #        xlims[0] + x_os_bw*dom_x_os_idx,
    #        ylims[0] + y_os_bw*dom_y_os_idx,
    #        'go', ms=8, lw=0.5,
    #        label='DOM location used for binning'
    #    )
    #    ax.quiver(
    #        X, Y, pxsl, pysl,
    #        label='Binned average photon direction'
    #    )

    #    ax.axis('image')
    #    ax.set_xlabel('x (m)')
    #    ax.set_ylabel('y (m)')

    #    ax.set_xticks(np.arange(xlims[0], xlims[1]+x_bw, x_bw), minor=False)
    #    ax.grid(which='major', b=True)
    #    if x_oversample > 1:
    #        ax.set_xticks(
    #            np.arange(x_inner_lim[0]+x_os_bw, x_inner_lim[1], x_os_bw),
    #            minor=True
    #        )
    #        ax.grid(which='minor', b=True, ls=':', alpha=0.6)

    #    if y_oversample > 1:
    #        ax.set_yticks(
    #            np.arange(y_inner_lim[0]+y_os_bw, y_inner_lim[1], y_os_bw),
    #            minor=True
    #        )
    #        ax.grid(which='minor', b=True, ls=':', alpha=0.6)

    #    ax.set_xlim(x_inner_lim)
    #    ax.set_ylim(y_inner_lim)
    #    ax.legend(loc='upper left', fancybox=True, framealpha=0.9)
    #    ax.set_title('Detail of table, XY-slice through center of DOM')
    #    fig.savefig('xyslice_detail.png', dpi=300)
    #    fig.savefig('xyslice_detail.pdf')

    #def plot_projections(self):
    #    pass
