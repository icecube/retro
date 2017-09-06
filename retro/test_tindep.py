#!/usr/bin/env python
# coding: utf-8
# pylint: disable=invalid-name


from __future__ import absolute_import, print_function

from collections import OrderedDict
import cPickle as pickle
import os
from os.path import abspath, dirname, isfile, join
import sys
import time

import numpy as np
import pyfits

os.sys.path.append(dirname(dirname(abspath('__file__'))))
from retro import (powerspace, spherical_volume, extract_photon_info, pol2cart,
                   sph2cart)
import retro.shift_and_bin
import retro.sphbin2cartbin

from pisa.utils.hash import hash_obj
from pisa.utils.format import list2hrlist
from genericUtils import timediffstamp
from plotGoodies import removeBorder


#tables_dir = '/home/justin/src/retro/retro/data/tables'
#tables_dir = '/data/icecube/retro_tables/full1000'
tables_dir = '/fastio/icecube/retro_tables/full1000'
geom_fpath = '/home/justin/src/retro/retro/data/geo_array.npy'

r_max = 400
r_power = 2
n_rbins = 200
n_costhetabins = 40
n_phibins = 80
n_tbins = 300

# Whole detector
#xlims = (-900, 950)
#ylims = (-900, 800)
#zlims = (-750, 600)

# DeepCore (below dust layer) and tighter extents in x, y as well
#xlims = (-750, 750)
#ylims = (-750, 600)
#zlims = (-600, -10)

# Tighter volume around DeepCore (200 m past a DOM in any dim, rounded to
# nearest 50 m)
#xlims = (-200, 300)
#ylims = (-300, 250)
#zlims = (-700, 50)

# 50x50x50 m tiles...
xlims = (-200, -150)
ylims = (-300, -250)
zlims = (-700, -650)


x_bw = y_bw = z_bw = 1
x_oversample = y_oversample = z_oversample = 2
antialias_factor = 1

nx = int((xlims[1] - xlims[0]) / x_bw)
ny = int((ylims[1] - ylims[0]) / y_bw)
nz = int((zlims[1] - zlims[0]) / z_bw)
xyz_shape = (nx, ny, nz)
print('xyz_shape:', xyz_shape)

string_indices = slice(None)
depth_slice = slice(None)
#depth_slice = slice(48, 48+3)
#string_indices = np.array([25, 26, 34, 35, 36, 44, 45])
#depth_slice = slice(48, 48+1)
#string_indices = np.array([35])

t_slice = slice(None)
t_etc_slice = [t_slice, slice(None), slice(None)]


geom = np.atleast_3d(np.load(geom_fpath))
geom_hash = hash_obj(geom, hash_to='hex', full_hash=True)[:8]
num_strings = geom.shape[0]
num_depths = geom.shape[1]
num_doms = num_strings * num_depths
print('num depths:', num_doms, 'num strings:', num_strings, 'total num doms:', num_doms)
depth_indices = np.atleast_1d(range(60)[depth_slice])

subdet_doms = {'ic': [], 'dc': []}
dc_string_indices = range(79, 86)
for string_idx in np.atleast_1d(np.arange(86)[string_indices]):
    dom_coords = geom[string_idx:string_idx+1, depth_slice, :]
    if string_idx in dc_string_indices:
        subdet_doms['dc'].append(dom_coords)
    else:
        subdet_doms['ic'].append(dom_coords)
for subdet in subdet_doms.keys():
    dom_string_list = subdet_doms[subdet]
    if not dom_string_list:
        subdet_doms.pop(subdet)
    else:
        subdet_doms[subdet] = np.concatenate(dom_string_list, axis=0)
all_doms = np.atleast_3d(geom[string_indices, depth_slice, :])


r_edges = powerspace(0, r_max, n_rbins + 1, r_power)
theta_edges = np.arccos(np.linspace(1, -1, n_costhetabins + 1))

R, THETA = np.meshgrid(r_edges, theta_edges, indexing='ij')
coords = []
exact_vols = []
for ri in range(n_rbins):
    subcoords = []
    sub_exact_vols = []
    for ti in range(int(np.ceil(n_costhetabins / 2.0))):
        rs = R[ri:ri+2, ti:ti+2]
        ts = THETA[ri:ri+2, ti:ti+2]
        bin_corner_coords = zip(rs.flat, ts.flat)
        dcostheta = np.abs(np.diff(np.cos([ts.max(), ts.min()])))
        exact_vol = spherical_volume(rmin=rs.max(), rmax=rs.min(), dcostheta=dcostheta, dphi=np.pi/2)
        sub_exact_vols.append(exact_vol)
    exact_vols.append(sub_exact_vols)
exact_vols = np.array(exact_vols)


binmapping_kw = dict(
    r_max=r_max, r_power=r_power,
    n_rbins=n_rbins, n_costhetabins=n_costhetabins, n_phibins=n_phibins,
    x_bw=x_bw, y_bw=y_bw, z_bw=z_bw,
    x_oversample=x_oversample, y_oversample=y_oversample, z_oversample=z_oversample,
    antialias_factor=antialias_factor
)

grid_hash = hash_obj(binmapping_kw, hash_to='hex', full_hash=True)[:8]

fname = (
    'sph2cart_map'
    '_%s'
    '_nr{n_rbins:d}_ncostheta{n_costhetabins:d}_nphi{n_phibins:d}'
    '_rmax{r_max:f}_rpwr{r_power}'
    '_xbw{x_bw:f}_ybw{y_bw:f}_zbw{z_bw:f}'
    '_xos{x_oversample:d}_yos{y_oversample:d}_zos{z_oversample:d}'
    '_aa{antialias_factor:d}'
    '.pkl'.format(**binmapping_kw)
) % grid_hash
fpath = join(tables_dir, fname)

print('params:')
print(binmapping_kw)

#if False:
if isfile(fpath):
    sys.stdout.write('loading arrays from file...\n')
    sys.stdout.flush()

    t0 = time.time()
    dump_data = pickle.load(file(fpath, 'rb'))
    ind_arrays = dump_data['ind_arrays']
    vol_arrays = dump_data['vol_arrays']
    t1 = time.time()
    print('time to load from pickle:', timediffstamp(t1 - t0))

else:
    sys.stdout.write('computing arrays...\n')
    sys.stdout.flush()

    t0 = time.time()
    ind_arrays, vol_arrays = retro.sphbin2cartbin.sphbin2cartbin(**binmapping_kw)
    t1 = time.time()
    print('time to compute:', timediffstamp(t1 - t0))

    dump_data = OrderedDict([
        ('kwargs', binmapping_kw),
        ('ind_arrays', ind_arrays),
        ('vol_arrays', vol_arrays)
    ])
    pickle.dump(dump_data, file(fpath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    t2 = time.time()
    print('time to pickle the results:', timediffstamp(t2 - t1))
print('')


binned_vol = np.sum([va.sum() for va in vol_arrays])
exact_vol = spherical_volume(rmin=0, rmax=r_max, dcostheta=-1, dphi=np.pi/2)
print('exact vol = %f, binned vol = %f (%e fract error)'
      % (exact_vol, binned_vol, (binned_vol-exact_vol)/exact_vol))


ind_bin_vols = np.array([va.sum() for va in vol_arrays])
fract_err = ind_bin_vols/exact_vols.flat - 1
abs_fract_err = np.abs(fract_err)
worst_abs_fract_err = np.max(abs_fract_err)
flat_idx = np.where(abs_fract_err == worst_abs_fract_err)[0][0]
r_idx, costheta_idx = divmod(flat_idx, int(np.ceil(n_costhetabins/2)))
print('worst single-bin fract err: %e; r_idx=%d, costheta_idx=%d; binned vol=%e, exact vol=%e'
      % (worst_abs_fract_err, r_idx, costheta_idx, ind_bin_vols[flat_idx], exact_vols[r_idx, costheta_idx]))

all_t_bins = list(range(n_tbins))
remaining_t_bins = np.array(all_t_bins)[t_slice].tolist()
if all_t_bins == remaining_t_bins:
    t_slice_str = ''
else:
    t_slice_str = '_tbins' + list2hrlist(remaining_t_bins)

table_basename_kw = dict(
    geom_hash=geom_hash, grid_hash=grid_hash, t_slice_str=t_slice_str,
    xmin=xlims[0], xmax=xlims[1],
    ymin=ylims[0], ymax=ylims[1],
    zmin=zlims[0], zmax=zlims[1],
    nx=nx, ny=ny, nz=nz, nphi=n_phibins
)

td_indep_table_hash = hash_obj(table_basename_kw, hash_to='hex', full_hash=True)[:8]

fbasename = (
    'retro_tdi_table'
    '_%s'
    '_doms_{geom_hash:s}'
    '_grid_{grid_hash:s}'
    '{t_slice_str:s}'
    '_lims{xmin:.2f}_{xmax:.2f}x{ymin:.2f}_{ymax:.2f}x{zmin:.2f}_{zmax:.2f}'
    '_nbins{nx:d}x{ny:d}x{nz:d}'
    '_nphi{nphi}'.format(**table_basename_kw)
) % td_indep_table_hash

names = [
    'survival_prob',
    'avg_photon_x',
    'avg_photon_y',
    'avg_photon_z'
]
recompute = False
for name in names:
    fpath = join(tables_dir, fbasename + '_' + name + '.fits')
    #if True:
    if not isfile(fpath):
        print('could not find table, will (re)compute\n%s\n' % fpath)
        recompute = True
        break

# TODO: bake this entire thing a function, and make these return values
if not recompute:
    for name in names:
        fpath = join(tables_dir, fbasename + '_' + name + '.fits')
        with pyfits.open(fpath) as fits_file:
            tmp = fits_file[0].data
        if name == 'survival_prob':
            binned_sp = tmp
        elif name == 'avg_photon_x':
            binned_px = tmp
        elif name == 'avg_photon_y':
            binned_py = tmp
        elif name == 'avg_photon_z':
            binned_pz = tmp
        del tmp
    sys.exit()

# Instantiate accumulation arrays
binned_spv = np.zeros((nx*ny*nz), dtype=np.float64)
binned_px_spv = np.zeros((nx*ny*nz), dtype=np.float64)
binned_py_spv = np.zeros((nx*ny*nz), dtype=np.float64)
binned_pz_spv = np.zeros((nx*ny*nz), dtype=np.float64)
binned_one_minus_sp = np.ones((nx*ny*nz), dtype=np.float64)

t00 = time.time()
for subdet, subdet_dom_coords in subdet_doms.items():
    print('subdet:', subdet)
    print('subdet_dom_coords.shape:', subdet_dom_coords.shape)
    for rel_ix, depth_index in enumerate(depth_indices):
        print('depth_index:', depth_index)
        dom_coords = subdet_dom_coords[:, rel_ix, :]
    
        t0 = time.time()
        table_fname = (
            'retro_nevts1000'
            '_{subdet:s}'
            '_DOM{depth_index:d}'
            '_r_cz_t_angles'
            '.fits'.format(
                subdet=subdet.upper(), depth_index=depth_index
            )
        )
        photon_info, bin_edges = extract_photon_info(
            fpath=join(tables_dir, table_fname),
            dom_depth_index=depth_index
        )
        t1 = time.time()
        print('time to load the retro table:', timediffstamp(t1 - t0))
    
        sp = photon_info.survival_prob[depth_index]
        plength = photon_info.length[depth_index]
        ptheta = photon_info.theta[depth_index]
        pdeltaphi = photon_info.deltaphi[depth_index]
    
        plength *= np.cos(pdeltaphi)
        pz = plength * np.cos(ptheta)
        prho = plength * np.sin(ptheta)
    
        t_indep_sp = 1 - np.prod(1 - sp[t_slice], axis=0)
    
        mask = t_indep_sp != 0
        scale = 1 / sp.sum(axis=0)[mask]
    
        t_indep_pz = np.zeros_like(t_indep_sp)
        t_indep_prho = np.zeros_like(t_indep_sp)
    
        t_indep_pz[mask] = (pz[t_slice] * sp[t_slice]).sum(axis=0)[mask] * scale
        t_indep_prho[mask] = (prho[t_slice] * sp[t_slice]).sum(axis=0)[mask] * scale
    
        t2 = time.time()
        print('time to marginalize out time dim, 1 depth:', timediffstamp(t2 - t1))
    
        retro.shift_and_bin.shift_and_bin(
            ind_arrays=ind_arrays,
            vol_arrays=vol_arrays,
            dom_coords=dom_coords,
            survival_prob=t_indep_sp,
            prho=t_indep_prho,
            pz=t_indep_pz,
            nr=n_rbins,
            ntheta=n_costhetabins,
            r_max=r_max,
            binned_spv=binned_spv,
            binned_px_spv=binned_px_spv,
            binned_py_spv=binned_py_spv,
            binned_pz_spv=binned_pz_spv,
            binned_one_minus_sp=binned_one_minus_sp,
            nx=nx,
            ny=ny,
            nz=nz,
            x_min=xlims[0],
            y_min=ylims[0],
            z_min=zlims[0],
            x_bw=x_bw,
            y_bw=y_bw,
            z_bw=z_bw,
            x_oversample=x_oversample,
            y_oversample=y_oversample,
            z_oversample=z_oversample
        )
        t3 = time.time()
        print('time to shift and bin:', timediffstamp(t3 - t2))
        print('')

print('Total time to shift and bin:', timediffstamp(t3 - t00))
print('')

binned_sp = (1 - binned_one_minus_sp).reshape(xyz_shape)
del binned_one_minus_sp

mask = binned_spv != 0
binned_px_spv[mask] /= binned_spv[mask]
binned_py_spv[mask] /= binned_spv[mask]
binned_pz_spv[mask] /= binned_spv[mask]
del mask

# Rename so as to not mislead
binned_px = binned_px_spv.reshape(xyz_shape)
binned_py = binned_py_spv.reshape(xyz_shape)
binned_pz = binned_pz_spv.reshape(xyz_shape)
del binned_px_spv, binned_py_spv, binned_pz_spv

t4 = time.time()
print('time to normalize histograms:', timediffstamp(t4 - t3))
print('')


arrays_names = [
    (binned_sp, 'survival_prob'),
    (binned_px, 'avg_photon_x'),
    (binned_py, 'avg_photon_y'),
    (binned_pz, 'avg_photon_z')
]
for array, name in arrays_names:
    fname = '%s_%s.fits' % (fbasename, name)
    fpath = join(tables_dir, fname)
    hdulist = pyfits.HDUList([
        pyfits.PrimaryHDU(array.astype(np.float32)),
        pyfits.ImageHDU(xyz_shape),
        pyfits.ImageHDU(np.array([xlims, ylims, zlims])),
        pyfits.ImageHDU(all_doms)
    ])
    print('Saving %s to file\n%s\n' % (name, fpath))
    hdulist.writeto(fpath, clobber=True)
t5 = time.time()
print('time to save tables to disk:', timediffstamp(t5 - t4))
print('')


print('TOTAL RUN TIME:', timediffstamp(t5 - t00))


