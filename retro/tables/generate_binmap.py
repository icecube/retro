# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Functions for mapping spherical bins to Cartesian bins.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    generate_binmap_meta
    generate_binmap
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
from os.path import abspath, dirname, isdir, isfile, join
import cPickle as pickle
import sys
import time

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.geom import powerspace, spherical_volume
from retro.utils.misc import hash_obj
from retro.tables.sphbin2cartbin import sphbin2cartbin


# TODO: does anisotropy need to be considered in the functions defined here?


def generate_binmap_meta(r_max, r_power, n_rbins, n_costhetabins, n_phibins,
                         cart_binwidth, oversample, antialias):
    """Generate metadata dict for spherical to Cartesian bin mapping, including
    the file name, hash string, and a dict with all of the parameters that
    contributed to these which can be passed via ``**binmap_kw`` to the
    `sphbin2cartbin` function.

    Parameters
    ----------
    r_max : float
        Maximum radius in Retro (t,r,theta)-binned DOM table (meters)

    r_power : float
        Binning in radial direction is regular in the inverse of this power.
        I.e., every element of `np.diff(r**(1/r_power))` is equal.

    n_rbins, n_costhetabins, n_phibins : int

    cart_binwidth : float
        Cartesian bin widths, same in x, y, and z (meters)

    oversample : int
        Oversample factor, same in x, y, and z

    antialias : int
        Antialias factor

    Returns
    -------
    metadata : OrderedDict
        Contains following items:
            'fname' : string
                File name for the specified bin mapping
            'hash' : length-8 string
                Hex digits represented as a string.
            'kwargs' : OrderedDict
                The keyword args used for the hash.

    """
    kwargs = OrderedDict([
        ('r_max', r_max),
        ('r_power', r_power),
        ('n_rbins', n_rbins),
        ('n_costhetabins', n_costhetabins),
        ('n_phibins', n_phibins),
        ('cart_binwidth', cart_binwidth),
        ('oversample', oversample),
        ('antialias', antialias)
    ])

    binmap_hash = hash_obj(kwargs, fmt='hex')

    print('kwargs:', kwargs)

    fname = (
        'sph2cart_binmap'
        '_%s'
        '_nr{n_rbins:d}_ncostheta{n_costhetabins:d}_nphi{n_phibins:d}'
        '_rmax{r_max:f}_rpwr{r_power}'
        '_bw{cart_binwidth:.6f}'
        '_os{oversample:d}'
        '_aa{antialias:d}'
        '.pkl'.format(**kwargs)
    ) % binmap_hash

    metadata = OrderedDict([
        ('fname', fname),
        ('hash', binmap_hash),
        ('kwargs', kwargs)
    ])

    return metadata


def generate_binmap(r_max, r_power, n_rbins, n_costhetabins, n_phibins,
                    cart_binwidth, oversample, antialias, tables_dir,
                    recompute):
    """Generate mapping from polar binning (assumed to be symmetric about
    Z-axis) to Cartesian 3D binning.

    The heart of the functionality is implemented in
    `retro.sphbin2cartbin.sphbin2cartbin`, while this function implements
    loading already-computed mappings and storing the results to disk.

    Parameters
    ----------
    r_max : float > 0
    r_power : float != 0
    n_rbins, n_costhetabins, n_phibins : int >= 1
    cart_binwidth : float > 0
    oversample : int >= 1
    antialias : int between 1 and 50
    tables_dir : string
    recompute : bool

    Returns
    -------
    ind_arrays
    vol_arrays
    meta
        Output of `generate_binmap_meta`

    """
    assert isdir(tables_dir)
    r_edges = powerspace(0, r_max, n_rbins + 1, r_power)
    theta_edges = np.arccos(np.linspace(1, -1, n_costhetabins + 1))

    r_mesh, theta_mesh = np.meshgrid(r_edges, theta_edges, indexing='ij')
    exact_vols = []
    for ri in range(n_rbins):
        sub_exact_vols = []
        for ti in range(int(np.ceil(n_costhetabins / 2.0))):
            rs = r_mesh[ri:ri+2, ti:ti+2]
            ts = theta_mesh[ri:ri+2, ti:ti+2]
            dcostheta = np.abs(np.diff(np.cos([ts.max(), ts.min()])))
            exact_vol = spherical_volume(rmin=rs.max(), rmax=rs.min(),
                                         dcostheta=dcostheta, dphi=np.pi/2)
            sub_exact_vols.append(exact_vol)
        exact_vols.append(sub_exact_vols)
    exact_vols = np.array(exact_vols)

    meta = generate_binmap_meta(
        r_max=r_max, r_power=r_power,
        n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
        cart_binwidth=cart_binwidth, oversample=oversample, antialias=antialias
    )
    fpath = join(tables_dir, meta['fname'])

    print('Binmap kwargs:', meta['kwargs'])

    if not recompute and isfile(fpath):
        sys.stdout.write('Loading binmap from file\n  "%s"\n' % fpath)
        sys.stdout.flush()

        t0 = time.time()
        data = pickle.load(file(fpath, 'rb'))
        ind_arrays = data['ind_arrays']
        vol_arrays = data['vol_arrays']
        t1 = time.time()
        print('  Time to load bin mapping from pickle: {} ms'
              .format(np.round((t1 - t0)*1000, 3)))

    else:
        sys.stdout.write('  Computing bin mapping...\n')
        sys.stdout.flush()

        t0 = time.time()
        ind_arrays, vol_arrays = sphbin2cartbin(**meta['kwargs'])
        t1 = time.time()
        print('    Time to compute bin mapping: {} ms'
              .format(np.round((t1 - t0)*1000, 3)))

        print('  Writing bin mapping to pickle file\n  "%s"' % fpath)
        data = OrderedDict([
            ('kwargs', meta['kwargs']),
            ('ind_arrays', ind_arrays),
            ('vol_arrays', vol_arrays)
        ])
        pickle.dump(data, file(fpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        t2 = time.time()
        print('    Time to pickle bin mapping: {} ms'
              .format(np.round((t2 - t1)*1000, 3)))

    print('')

    binned_vol = np.sum([va.sum() for va in vol_arrays])
    exact_vol = spherical_volume(rmin=0, rmax=r_max, dcostheta=-1, dphi=np.pi/2)
    print('  Exact vol = %f, binned vol = %f (%e fract error)'
          % (exact_vol, binned_vol, (binned_vol-exact_vol)/exact_vol))

    ind_bin_vols = np.array([va.sum() for va in vol_arrays])
    fract_err = ind_bin_vols/exact_vols.flat - 1
    abs_fract_err = np.abs(fract_err)
    worst_abs_fract_err = np.max(abs_fract_err)
    flat_idx = np.where(abs_fract_err == worst_abs_fract_err)[0][0]
    r_idx, costheta_idx = divmod(flat_idx, int(np.ceil(n_costhetabins/2)))
    print('  Worst single-bin fract err: %e;'
          'r_idx=%d, costheta_idx=%d;'
          'binned vol=%e, exact vol=%e'
          % (worst_abs_fract_err, r_idx, costheta_idx, ind_bin_vols[flat_idx],
             exact_vols[r_idx, costheta_idx]))

    return ind_arrays, vol_arrays, meta
