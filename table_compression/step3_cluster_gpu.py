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
parser.add_argument('-d', '--dir', type=str,
                    metavar='directory', default=None,
                    help='parent directory of the tables')
args = parser.parse_args()

if not os.path.isfile(os.path.join(args.dir, 'cluster_data.fp32.npy')):
    all_tables = []
    length_list = []

    for item in os.listdir(args.dir):
        path = os.path.join(args.dir, item)
        if os.path.isdir(path):
            table = os.path.join(path, 'pca_reduced_table.npy')
            if os.path.isfile(table):
                all_tables.append(np.load(table,  mmap_mode='r'))
                # figure out how many element does each table have
                length_list.append(all_tables[-1].shape[0])

    length_list = np.array(length_list)

    print('creating data...')
    data = np.concatenate(all_tables)
    print('data created')
    data = data.astype(np.float32)
    np.save(os.path.join(args.dir, 'cluster_data.fp32.npy'), data)
else:
    print('loading exisiting data')
    data = np.load(os.path.join(args.path, 'cluster_data.fp32.npy'))

# random subsample
#print('subsampling')
#data = data[np.random.choice(data.shape[0], 41000000, replace=False), :]

#data = data[:100000]

#data = data.astype(np.float16)
#data = data.astype(np.float32)

print('data loaded')

n_clusters = args.n_clusters

centroids, assignments = kmeans_cuda(data, args.n_clusters, verbosity=2, seed=3, device=5, yinyang_t=0, tolerance=0.05)
#centroids, assignments = kmeans_cuda(data, args.n_clusters, verbosity=2, seed=3, device=1, yinyang_t=0, tolerance=0.5, init='random')

print('clustering done')
np.save(os.path.join(args.dir, 'kmcuda_%i_clusters_centroids_rand.npy'%args.n_clusters), centroids)
np.save(os.path.join(args.dir, 'kmcuda_%i_clusters_assignements_rand.npy'%args.n_clusters), assignments)
