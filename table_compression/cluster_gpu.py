'''
Script to run the k-menas clustering algorithm on the PCA reduced tables
'''

import numpy as np
import os.path
import sys
from libKMCUDA import kmeans_cuda


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser()
parser.add_argument('--n-clusters', type=int,
                    default=4000,
                    help='number of clusters')
args = parser.parse_args()

#all_tables = []
#length_list = []
#for string in ['ic','dc']:
#    for depth_idx in range(60):
#        all_tables.append(np.load('/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_%s_depth_%s/pca_reduced_table.npy'%(string, depth_idx),  mmap_mode='r'))
#        # figure out how many element does each table have
#        length_list.append(all_tables[-1].shape[0])
#
#length_list = np.array(length_list)
#
#print('creating data...')
#data = np.concatenate(all_tables)
#print('data created')
#data = data.astype(np.float32)
#np.save('cluster_data.fp32.npy', data)
#data = np.load('cluster_data.fp16.npy')
data = np.load('cluster_data.fp32.npy')
data = data[:100000]
np.save('testdata_fp32.npy', data)
np.save('testdata_fp16.npy', data.astype(np.float16))

data = data.astype(np.float16)
#data = data.astype(np.float32)

#data = data.astype(np.float16)
#np.save('cluster_data.fp16.npy', data)
print('data loaded')

n_clusters = args.n_clusters

centroids, assignments = kmeans_cuda(data, args.n_clusters, verbosity=1, seed=3, device=5, yinyang_t=0, tolerance=0.05)
#centroids, assignments = kmeans_cuda(data, args.n_clusters, verbosity=2, seed=3, device=1, yinyang_t=0, tolerance=0.5, init='random')

print('clustering done')
np.save('kmcuda_%i_clusters_centroids.npy'%args.n_clusters, centroids)
np.save('kmcuda_%i_clusters_assignements_list'%args.n_clusters, assignments)
