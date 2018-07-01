'''
Script to assign a template to each table bin and save a mixed type table with (n_photons, idx) for every bin
'''

import numpy as np
import os.path
import sys
from numba import jit, float64, int32, float32, guvectorize, SmartArray
import cPickle as pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser()
parser.add_argument('--cluster-idx', type=int,
                    help='cluster index (0,59)')
parser.add_argument('--nfs', action='store_true',
                    help='also acces over NFS')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite existing file')
args = parser.parse_args()


#custom dtype
tabledt = np.dtype([('index', np.uint16), ('weight', np.float32)])

# vectorized chi2 calculator
@guvectorize(['(f4[:,:],f4[:,:],f4[:])'], '(a,b),(a,b)->()', target='cuda')
def tot_chi2(dir_map, templates, chi2s):
    for j in range(dir_map.shape[0]):
        for k in range(dir_map.shape[1]):
            A = dir_map[j, k]
            B = templates[j, k]
            tot = A + B
            if tot == 0.:
                continue
            chi2s[0] += (A-B)**2/tot


def find_best_template(dir_map, templates):
    '''
    Find the best template for a given directionality map
    
    dir_map : array of size (m,n)
    templates : array of size (k,m,n)
    returns int in range(k)
    '''
    if np.sum(dir_map) == 0.:
        return 0, 0.
    chi2s = np.zeros(templates.get('host').shape[0], np.float32)
    tot_chi2(dir_map, templates.get('gpu'), out=chi2s)
    idx = np.argmin(chi2s)
    return idx, chi2s[idx]


def fill_index_table(table_5d_normed, templates):
    index_table = np.zeros(table_5d_normed.shape[:3], dtype=np.uint16)
    chi2s = np.zeros(table_5d_normed.shape[:3], dtype=np.float32)
    for i in range(index_table.shape[0]):
        print i
        for j in range(index_table.shape[1]):
            for k in range(index_table.shape[2]):
		idx, chi2 = find_best_template(table_5d_normed[i,j,k], templates)
                index_table[i,j,k] = idx
		chi2s[i,j,k] = chi2
    return index_table, chi2s



@jit(nopython=True, nogil=True, cache=True)
def fill_template_map(index_table, table_3d):
    template_map = np.zeros(table_5d_normed.shape[:3], dtype=tabledt)
    for i in range(template_map.shape[0]):
        for j in range(template_map.shape[1]):
            for k in range(template_map.shape[2]):
                template_map[i,j,k]['index'] = index_table[i,j,k]
                template_map[i,j,k]['weight'] = table_3d[i,j,k]
    return template_map

print 'table cluster %s'%(args.cluster_idx)

path = 'tilt_on_anisotropy_on_noazimuth_80/cl%s/'%(args.cluster_idx)

if os.path.isfile('/data/icecube/retro_tables/' + path + 'ckv_template_map.npy'):
    if args.overwrite:
        print 'overwritting existing file'
    else:
        print 'file exists, abort'
        sys.exit()

# try files on fastio first
fname = '/fastio/icecube/retro/tables/' + path + 'ckv_table.npy'
if not os.path.isfile(fname):
    fname = '/fastio2/icecube/retro/tables/' + path + 'ckv_table.npy'
if not os.path.isfile(fname):
    if args.nfs:
        fname = '/data/icecube/retro_tables/' + path + 'ckv_table.npy'
    else:
        sys.exit()

table_5d = np.load(fname)
print 'table loaded'
table_3d = np.sum(table_5d, axis=(3,4))

# normalize the tables such that all directionality maps sum to 1
table_5d_normed = np.nan_to_num(table_5d / table_3d[:,:,:,np.newaxis,np.newaxis])
del table_5d
print 'table normed'

templates = np.load('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/final_templates.npy')
templates = templates.astype(np.float32)
templates = SmartArray(templates)

index_table, chi2s = fill_index_table(table_5d_normed, templates)
print 'indices found'

template_map = fill_template_map(index_table, table_3d)



np.save('/data/icecube/retro_tables/'+path+'ckv_template_map.npy', template_map)
np.save('/data/icecube/retro_tables/'+path+'template_chi2s.npy', chi2s)
