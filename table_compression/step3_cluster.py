'''
Script to run the k-menas clustering algorithm on the PCA reduced tables
'''

import numpy as np
import os.path
import sys
#from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans as KMeans
from numba import jit
import cPickle as pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser()
parser.add_argument('--n-clusters', type=int,
                    default=4000,
                    help='number of clusters')
args = parser.parse_args()

all_tables = []
length_list = []
for cluster_idx in range(80):
    all_tables.append(np.load('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/pca_reduced_table.npy'%(cluster_idx),  mmap_mode='r'))
    # figure out how many element does each table have
    length_list.append(all_tables[-1].shape[0])

length_list = np.array(length_list)

print 'creating data...'
data = np.concatenate(all_tables)
print 'data created'
del all_tables

n_clusters = args.n_clusters

k_means = KMeans(n_clusters=args.n_clusters,
                 ##n_jobs=-1,
                 #precompute_,
                 max_no_improvement=1000,
                 compute_labels=False,
                 random_state=0,
                 batch_size=100000,
                 verbose=1,
                 #copy_x=False,
                 n_init = 100,
                 max_iter = 10,
                 )
k_means.fit(data)
print 'clustering done'
# save everything
with open('kmeans_%i_clusters_spice_lea_tables.pkl'%args.n_clusters, 'wb') as f:
    pickle.dump(k_means, f)

#print 'predicting'

#labels = k_means.predict(data)
#np.save('kmeans_%i_clusters_labels.npy'%args.n_clusters, labels)
#np.save('kmeans_%i_clusters_length_list'%args.n_clusters, length_list)
