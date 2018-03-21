'''
Script to find principal components of all directionality maps across all tables

Maps with fever than 1000 hits are not considered
'''
import numpy as np
from sklearn.decomposition import IncrementalPCA as PCA
from numba import jit
import cPickle as pickle

pca = PCA(n_components = 100)

for string in ['dc', 'ic']:
    for depth_idx in range(60):
        print 'det %s table %s'%(string, depth_idx)
        table_5d = np.load('/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_%s_depth_%s/ckv_table.npy'%(string, depth_idx))
        print 'table loaded'
        # create 3d table
        table_3d = np.sum(table_5d, axis=(3,4))
        table_5d_normed = np.nan_to_num(table_5d / table_3d[:,:,:,np.newaxis,np.newaxis])
        print 'table normed'
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
        print 'data created'
        pca.partial_fit(data)
        print 'PCA fitted'
        with open('pca_after_%s%s.pkl'%(string,depth_idx),'wb') as f:
            pickle.dump(pca, f)
        del table_5d_normed
        del table_3d
        del table_5d
        del data
