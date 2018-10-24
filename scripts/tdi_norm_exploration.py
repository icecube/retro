#!/usr/bin/env python
# pylint: disable=wrong-import-order, wrong-import-position, invalid-name, no-member, line-too-long

"""
Test single-DOM expected light (i.e. per TDI tile)
"""

from __future__ import absolute_import, division, print_function

__author__ = 'J.L. Lanfranchi'
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
from os import makedirs
from os.path import abspath, dirname, isdir, join
import pickle
import sys
import time

import numpy as np
import pyfits

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(abspath(__file__)))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.i3info.angsens_model import load_angsens_model
from retro.i3info.extract_gcd import extract_gcd
from retro.init_obj import setup_discrete_hypo
from retro.retro_dom_pdfs import SIMULATIONS
from retro.utils.ckv import convolve_dirmap
from retro.utils.geom import generate_digitizer
from retro.utils.misc import force_little_endian, wstdout

# -- User-defined constants -- #

TDI_TILE_ROOTDIR = '/gpfs/scratch/jll1062'
TILESET_HASH = '873a6a13'
TILT = True
ANISOTROPY = False
SIMDIR = '/gpfs/group/dfc13/default/sim/retro'
SIM = SIMULATIONS['lea_horizontal_muon']
GCD_FILE = 'GeoCalibDetectorStatus_IC86.2017.Run129700_V0.pkl'
BETA = 1.0
CHECK_DOMS = 'str86'

print(
    'doms: {}, sim: {}, tilt: {}, anisotropy: {}'
    .format(CHECK_DOMS, SIM, TILT, ANISOTROPY)
)


# -- Derived constants -- #

tilt_onoff = 'on' if TILT else 'off'
anisotropy_onoff = 'on' if ANISOTROPY else 'off'

basedirname = 'tdi_{}_tilt_{}_anisotropy_{}'.format(
    TILESET_HASH,
    tilt_onoff,
    anisotropy_onoff,
)
tdi_tile_dir = join(TDI_TILE_ROOTDIR, basedirname)

# -- Create output dir -- #

outdir = join('tdi_norm_exploration', basedirname)
if not isdir(outdir):
    makedirs(outdir)

# -- Load info -- #

gcd_info = extract_gcd(gcd_file=GCD_FILE)

fwdsim_histos = pickle.load(file(join(
    SIMDIR,
    SIM['fwd_sim_histo_file'],
)))

spec_fname = (
    'tdi_tiles_to_produce.{hash}.tilt_{tilt_onoff}_anisotropy_{anisotropy_onoff}.txt'
    .format(hash=TILESET_HASH, tilt_onoff=tilt_onoff, anisotropy_onoff=anisotropy_onoff)
)
specs = []
for line in file(join(tdi_tile_dir, spec_fname), 'r').readlines():
    spec = OrderedDict()
    (tile, string, dom, seed, n_events, x_min, x_max, n_x, y_min, y_max, n_y,
     z_min, z_max, n_z, n_costhetadir, n_phidir) = (
        line.strip().split()
    )
    spec['tile'] = int(tile)
    spec['string'] = int(string)
    spec['dom'] = int(dom)
    spec['omkey'] = (int(string), int(dom))
    spec['omkey'] = (int(string), int(dom))
    spec['seed'] = int(seed)
    spec['n_events'] = int(n_events)
    spec['x_min'] = float(x_min)
    spec['x_max'] = float(x_max)
    spec['n_x'] = int(n_x)
    spec['y_min'] = float(y_min)
    spec['y_max'] = float(y_max)
    spec['n_y'] = int(n_y)
    spec['z_min'] = float(z_min)
    spec['z_max'] = float(z_max)
    spec['n_z'] = int(n_z)
    spec['costhetadir_min'] = -1
    spec['costhetadir_max'] = 1
    spec['n_costhetadir'] = int(n_costhetadir)
    spec['phidir_min'] = -np.pi
    spec['phidir_max'] = np.pi
    spec['n_phidir'] = int(n_phidir)
    specs.append(spec)

hypo_handler = setup_discrete_hypo(
    cascade_kernel=None,
    track_kernel='table_e_loss',
    track_time_step=3,
)

all_sources = hypo_handler.get_generic_sources(SIM['mc_true_params'])
assert len(all_sources) > 0
print('len(all_sources):', len(all_sources))


n_costhetadir = specs[0]['n_costhetadir']
costhetadir_bin_edges = np.linspace(
    start=specs[0]['costhetadir_min'],
    stop=specs[0]['costhetadir_max'],
    num=n_costhetadir,
)

n_phidir = specs[0]['n_phidir']
phidir_bin_edges = np.linspace(
    start=specs[0]['phidir_min'],
    stop=specs[0]['phidir_max'],
    num=n_phidir,
)
costhetadir_dig = generate_digitizer(costhetadir_bin_edges, clip=True)
phidir_dig = generate_digitizer(phidir_bin_edges, clip=True)
dirmap = np.empty(shape=(n_costhetadir, n_phidir))
n_dir_bins = n_costhetadir * n_phidir


avg_angsens = None
results = OrderedDict()
for tilenum, spec in enumerate(specs):
    omkey = string, dom = spec['omkey']

    if CHECK_DOMS == 'str86':
        if string != 86:
            continue
    elif CHECK_DOMS == 'dcsubset':
        if string < 79 or dom < 34 or dom > 38:
            continue
    else:
        raise ValueError('CHECK_DOMS value "{}" unhandled'.format(CHECK_DOMS))

    wstdout('tile {:4d} om ({:2d}, {:2d}):'.format(spec['tile'], string, dom))
    if not fwdsim_histos['results'].has_key(omkey):
        wstdout(' no results.\n')
        continue

    t0 = time.time()

    quantum_efficiency = 0.25 * gcd_info['rde'][string - 1, dom - 1]
    if quantum_efficiency == 0:
        wstdout(' QE=0.\n')
        continue

    fwdsim_histo = fwdsim_histos['results'][omkey]

    # Subtract off noise floor & sum to find time-independent total signal
    fwdsim_total = np.sum(fwdsim_histo - np.min(fwdsim_histo))

    tile_fname = (
        'clsim_table_set'
        '_{hash}_tile_{tile}_string_{string}_dom_{dom}'
        '_seed_{seed}_n_{n_events}.fits'
        .format(hash=TILESET_HASH, **spec)
    )
    tile_fits = pyfits.open(
        join(tdi_tile_dir, tile_fname),
        memmap=True,
    )
    header = tile_fits[0].header
    tile = tile_fits[0].data

    # Get rid of underflow and overflow bins
    tile = tile[(slice(1, -1),) * tile.ndim]

    # Convert to little-endian for numba codes (plus native format on Intel)
    tile = force_little_endian(tile)

    if np.product(tile.shape) == 0:
        wstdout(' tile failed!.\n')
        continue

    n_photons = header['_i3_n_photons']
    n_phase = header['_i3_n_phase']
    cos_ckv = 1 / (n_phase * BETA)
    sin_ckv = np.sin(np.arccos(cos_ckv))

    if avg_angsens is None:
        angsens_model = [
            k.replace('_i3_angsens_', '')
            for k in header.keys() if k.startswith('_i3_angsens_')
        ][0]
        _, avg_angsens = load_angsens_model(angsens_model)

    del tile_fits

    digitizers = []
    cart_bin_vol = 1.0
    for dim in ('x', 'y', 'z'):
        n_bins = spec['n_' + dim]
        start = spec[dim + '_min']
        stop = spec[dim + '_max']
        bin_edges = np.linspace(start, stop, n_bins + 1)
        cart_bin_vol *= (stop - start) / n_bins
        try:
            digitizers.append(
                (
                    generate_digitizer(bin_edges, clip=False),
                    n_bins,
                )
            )
        except:
            wstdout(
                '\ndim={}, n_bins={}, start={}, stop={}, bin_edges={}\n'
                .format(dim, n_bins, start, stop, bin_edges)
            )
            raise

    norm = n_dir_bins / n_photons * quantum_efficiency * avg_angsens / cart_bin_vol

    previous_cart_idxs = None
    dom_tindep_exp = 0.0
    wstdout(' ')
    for source in all_sources:
        wstdout('.')
        cart_idxs = []
        for dimnum, dim in enumerate(('x', 'y', 'z')):
            dig, nbins = digitizers[dimnum]
            idx = dig(source[dim])
            if idx < 0 or idx >= nbins:
                break
            cart_idxs.append(idx)
        if len(cart_idxs) != 3:
            wstdout('\b_')
            continue
        cart_idxs = tuple(cart_idxs)

        if cart_idxs != previous_cart_idxs:
            previous_cart_idxs = cart_idxs
            raw_dirmap = tile[cart_idxs]
            if np.sum(raw_dirmap) == 0:
                continue
            wstdout('\bC')
            convolve_dirmap(
                dirmap=raw_dirmap,
                out=dirmap,
                cos_ckv=cos_ckv,
                sin_ckv=sin_ckv,
                costhetadir_bin_edges=costhetadir_bin_edges,
                phidir_bin_edges=phidir_bin_edges,
                num_cone_samples=100,
                oversample=5,
            )
            wstdout('\bc')
            if np.sum(dirmap) == 0:
                continue
        else:
            wstdout('\b-')

        dir_costheta = -source['dir_costheta']
        dir_phi = source['dir_phi'] + np.pi
        if dir_phi > np.pi:
            dir_phi -= 2*np.pi
        elif dir_phi < -np.pi:
            dir_phi += 2*np.pi

        costhetadir_idx = costhetadir_dig(dir_costheta)
        azdir_idx = phidir_dig(dir_phi)

        dom_tindep_exp += norm * source['photons'] * dirmap[costhetadir_idx, azdir_idx]

    if fwdsim_total != 0:
        fracterr = 100*(dom_tindep_exp/fwdsim_total - 1)
    else:
        fracterr = np.nan
    wstdout(
        ' fwd, tile: {:.2e}, {:.2e} -> {:+8.1f}% fracterr'
        .format(fwdsim_total, dom_tindep_exp, fracterr)
    )
    results[omkey] = (fwdsim_total, dom_tindep_exp)
    wstdout(' ... {:.1f} sec\n'.format(time.time() - t0))
