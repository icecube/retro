from __future__ import division, print_function
'''
Script to find principal components of all directionality maps across all tables

Maps with fever than 1000 hits are not considered
'''
import numpy as np
import os
from sklearn.decomposition import IncrementalPCA as PCA
from numba import jit
import cPickle as pickle


from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('-d', '--dir', type=str,
                    metavar='directory', default=None,
                    help='parent directory of the tables')
parser.add_argument('-n', '--num-pca', type=int, default=100,
                    help='number of PCA components')


args = parser.parse_args()

pca = PCA(n_components = args.num_pca)
increment = 0

for item in os.listdir(args.dir):
    path = os.path.join(args.dir, item)
    if os.path.isdir(path):
        table = os.path.join(path, 'ckv_table.npy')
        if os.path.isfile(table):
            print('table %s'%table)
            print('PCA increment %i'%(increment))
            table_5d = np.load(table)
            print('table loaded')
            # create 3d table
            table_3d = np.sum(table_5d, axis=(3,4))
            table_5d_normed = np.nan_to_num(table_5d / table_3d[:,:,:,np.newaxis,np.newaxis])
            print('table normed')
            # prepare data structure
            # reject low stats samples (these contian a lot of specific shapes just due to too low statistics. we don't want to add these shapes to our template library)
            mask = table_3d < 1000.
            # reduced data
            data = np.empty((np.product(table_5d.shape[:3])-np.sum(mask),np.product(table_5d.shape[3:])))
            counter = 0
            for i,m in np.ndenumerate(mask):
                if not m:
                    data[counter] = table_5d_normed[i].ravel()
                    counter += 1
            print('data created')
            pca.partial_fit(data)
            print('PCA fitted')
            outfile = os.path.join(path, 'pca_increment_%s'%increment)
            with open(outfile,'wb') as f:
                pickle.dump(pca, f)
            del table_5d_normed
            del table_3d
            del table_5d
            del data
            increment += 1
