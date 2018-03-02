# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, redefined-outer-name

"""
Convert raw Retro 5D tables (which represent survival probabilities for light
traveling in a particular direction) to tables for Cherenkov emitters with a
particular direction.

Output tables will be in .npy-files-in-a-directory format for easy memory
mapping.
"""

from __future__ import absolute_import, division, print_function

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
from os import remove
from os.path import abspath, dirname, expanduser, expandvars, isdir, isfile, join
import re
import sys
import math

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
import retro


__all__ = [
    'generate_ckv_table'
]


# TODO: allow different directional binning in output table

def generate_ckv_table(
        table, beta, samples_per_bin, num_cone_samples, outdir=None,
        mmap=True
    ):
    """
    Parameters
    ----------
    table : string or mapping
        If string, path to table file (or directory in the case of npy tables).
        A mapping is assumed to be a table loaded as by
        `retro.table_readers.load_clsim_table_minimal`.

    beta : float in [0, 1]
        Beta factor, i.e. velocity of the charged particle divided by the speed
        of light in vacuum: `v/c`.

    samples_per_bin : int > 0
        Sample from each directional bin (costhetadir and deltaphidir) this
        many times. Increase to obtain a more accurate average over the range
        of directions that the resulting ckv-emitter-direction can take within
        the same output (directional) bin. Note that there is no unique
        information given by sampling (more than once) in the spatial
        dimensions, so these dimensions ignore `samples_per_bin`. Therefore,
        the computational cost is `samples_per_bin**2`.

    num_cone_samples : int > 0
        Number of samples around the circumference of the Cherenkov cone.

    outdir : string or None
        If a string, use this directory to place the .npy file containing the
        ckv table. If `outdir` is None and `table` is a .npy-file-directory,
        this directory is used for `outdir`. If `outdir` is None and `table` is
        the path to a .fits file, `outdir` is the same name but with the .fits
        extension stripped. If `outdir` is None and `table` is a mapping, a
        ValueError is raised.
        npy-file-directory will be placed.

    mmap : bool, optional
        Whether to (attempt to) memory map the source `table` (if `table` is a
        string pointing to the file/directory). Default is `True`, as tables
        can easily exceed the memory capacity of a machine.

    """
    input_filename = None
    if isinstance(table, basestring):
        input_filename = expanduser(expandvars(table))
        table = retro.table_readers.load_clsim_table_minimal(input_filename, mmap=mmap)

    if input_filename is None and outdir is None:
        raise ValueError('You must provide an `outdir` if `table` is a python'
                         ' object (i.e. not a file or directory path).')

    # Store original table to keep binning info, etc.
    full_table = table

    r_bin_edges = full_table['r_bin_edges']
    costheta_bin_edges = full_table['costheta_bin_edges']
    t_bin_edges = full_table['t_bin_edges']
    costhetadir_bin_edges = full_table['costhetadir_bin_edges']
    deltaphidir_bin_edges = full_table['deltaphidir_bin_edges']

    n_r_bins = len(r_bin_edges) - 1
    n_costheta_bins = len(costheta_bin_edges) - 1
    n_t_bins = len(t_bin_edges) - 1

    # NOTE: we are making output binning same as input binning.

    n_phase = table['phase_refractive_index']
    cos_theta_ckv = 1 / (n_phase * beta)
    if cos_theta_ckv > 1:
        raise ValueError(
            'Particle moving at beta={} in medium with n_phase={} does not'
            ' produce Cherenkov light!'.format(beta, n_phase)
        )
    theta_ckv = np.arccos(cos_theta_ckv)
    sin_theta_ckv = np.sin(theta_ckv)

    # Extract just the "useful" part of the table, i.e., exclude under/overflow
    # bins.
    table = table['table'][(slice(1, -1),)*5]

    if outdir is None:
        if isdir(input_filename):
            outdir = input_filename
        elif isfile(input_filename):
            outdir = input_filename.rstrip('.fits')
            assert outdir != input_filename, str(input_filename)
    else:
        outdir = expanduser(expandvars(outdir))
        if not isdir(outdir):
            retro.mkdir(outdir)
    outdir = expanduser(expandvars(outdir))
    ckv_table_fpath = join(outdir, 'ckv_table.npy')
    retro.mkdir(outdir)

    # Allocate memory-mapped file
    ckv_table = np.lib.format.open_memmap(
        filename=ckv_table_fpath,
        mode='w+',
        dtype=np.float32,
        shape=table.shape
    )
    try:
        convolve_table_w_ckv_cone(
            src=table,
            dst=ckv_table,
            cos_ckv=cos_theta_ckv,
            sin_ckv=sin_theta_ckv,
            n_r=n_r_bins,
            n_ct=n_costheta_bins,
            n_t=n_t_bins,
            ctdir_bin_edges=costhetadir_bin_edges,
            dpdir_bin_edges=deltaphidir_bin_edges,
            num_cone_samples=num_cone_samples,
            samples_per_bin=samples_per_bin
        )
    except:
        del ckv_table
        remove(ckv_table_fpath)
        raise

    return ckv_table


@retro.numba_jit(**retro.DFLT_NUMBA_JIT_KWARGS)
def convolve_table_w_ckv_cone(
        src, dst, cos_ckv, sin_ckv, n_r, n_ct, n_t, ctdir_bin_edges,
        dpdir_bin_edges, num_cone_samples, samples_per_bin
    ):
    n_ctdir = len(ctdir_bin_edges) - 1
    n_dpdir = len(dpdir_bin_edges) - 1

    ctdir_bw = (ctdir_bin_edges[-1] - ctdir_bin_edges[0]) / n_ctdir
    dpdir_bw = (dpdir_bin_edges[-1] - dpdir_bin_edges[0]) / n_dpdir

    ctdir_samp_step = ctdir_bw / samples_per_bin
    dpdir_samp_step = dpdir_bw / samples_per_bin

    samples_shape = (samples_per_bin, samples_per_bin)

    # Cosine and sine of thetadir
    ctd_samples = np.empty(shape=samples_shape, dtype=np.float64)
    std_samples = np.empty(shape=samples_shape, dtype=np.float64)

    # Cosine and sine of deltaphidir
    cdpd_samples = np.empty(shape=samples_shape, dtype=np.float64)
    sdpd_samples = np.empty(shape=samples_shape, dtype=np.float64)

    for ctdir_idx in range(n_ctdir):
        ctd0 = ctdir_idx*ctdir_bw + ctdir_bw / 2

        for dpdir_idx in range(n_dpdir):
            dpd0 = dpdir_idx*dpdir_bw + dpdir_samp_step / 2

            for ctdir_subidx in range(samples_per_bin):
                ctd_samp = ctd0 + ctdir_subidx * ctdir_samp_step
                std_samp = math.sqrt(1 - ctd_samp*ctd_samp)

                for dpdir_subidx in range(samples_per_bin):
                    dpd_samp = dpd0 + dpdir_subidx * dpdir_samp_step
                    sdpd_samp = math.sin(dpd_samp)
                    cdpd_samp = math.cos(dpd_samp)

                    ctd_samples[ctdir_subidx, dpdir_subidx] = ctd_samp
                    std_samples[ctdir_subidx, dpdir_subidx] = std_samp
                    cdpd_samples[ctdir_subidx, dpdir_subidx] = cdpd_samp
                    cdpd_samples[ctdir_subidx, dpdir_subidx] = sdpd_samp

            src_dir_indices, weights = retro.ckv.get_cone_map(
                costheta=cos_ckv,
                sintheta=sin_ckv,
                num_phi=num_cone_samples,
                axis_costheta=ctd_samples,
                axis_sintheta=std_samples,
                axis_cosphi=cdpd_samples,
                axis_sinphi=sdpd_samples,
                num_costheta_bins=n_ctdir,
                num_deltaphi_bins=n_dpdir
            )

            for r_idx in range(n_r):
                for ct_idx in range(n_ct):
                    for t_idx in range(n_t):
                        avg = 0.0
                        for src_dir_idx, weight in zip(src_dir_indices, weights):
                            avg += weight * src[(r_idx, ct_idx, t_idx) + src_dir_idx]
                        dst[r_idx, ct_idx, t_idx, ctdir_idx, dpdir_idx] = avg


def parse_args(description=__doc__):
    """Parse command line arguments"""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--tables', required=True, nargs='+',
        help='''npy-table directories and/or .fits table files'''
    )
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='''Cherenkov emitter beta factor (v / c).'''
    )
    parser.add_argument(
        '--n_phase', type=float, default=1.33,
        help='''Phase velocity of light in the medium.'''
    )
    parser.add_argument(
        '--outdir', default=None,
        help='''Directory in which to store the resulting table
        directory(ies).'''
    )

    args = parser.parse_args()

    # Construct the output filename if none is provided
    if args.outfile is None:
        args.outfile = re.sub(r'_photons.pkl', '_photon_histos.pkl', args.photons)

    return args


if __name__ == '__main__':
    ckv_table = generate_ckv_table(**vars(parse_args())) # pylint: disable=invalid-name
