'''
Script to apply k-means to tables, saveing raw templates
'''

import numpy as np
import os.path
import sys
from sklearn.cluster import KMeans
from numba import jit
import cPickle as pickle

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser()
parser.add_argument('-d', '--dir', type=str,
                    metavar='directory', default=None,
                    help='parent directory of the tables')
parser.add_argument('--cluster-idx', type=int,
                    help='cluster index (0,79)')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite existing file')
parser.add_argument('--centroids', type=str, default=None,
                    help='centroids file')
args = parser.parse_args()

# load the latest, greates kmeans centroids
centroids = np.load(args.centroids)
centroids = np.nan_to_num(centroids)
k_means = KMeans(centroids.shape[0])
k_means.cluster_centers_ = centroids
#with open('kmeans_4000_clusters.pkl', 'rb') as f:
#    k_means = pickle.load(f)

print 'table cluster %s'%(args.cluster_idx)

fname = os.path.join(args.dir, 'cl%s/ckv_table.npy'%(args.cluster_idx))
pcaname = os.path.join(args.dir, 'cl%s/pca_reduced_table.npy'%(args.cluster_idx))
outname = os.path.join(args.dir, 'cl%s/templates.npy'%(args.cluster_idx))

if os.path.isfile(outname):
    if args.overwrite:
        print('overwritting existing file')
    else:
        print('file exists, abort')
        sys.exit()

table_5d = np.load(fname)
print 'table loaded'
# create 3d table
table_3d = np.sum(table_5d, axis=(3,4))
print '3d table created'
# reject low stats samples (these contian a lot of specific shapes just due to too low statistics. we don't want to add these shapes to our template library)
mask = table_3d < 1000.
print 'mask created'

reduced_data = np.load(pcaname)
print 'reduced data loaded'

length = np.product(table_5d.shape[:3])-np.sum(mask)

assert(length == reduced_data.shape[0]), 'data missmatch!'

# compute indices
indices = np.empty((length, 3), dtype=np.uint32)
counter = 0
for i,m in np.ndenumerate(mask):
    if not m:
        indices[counter] = i
        counter += 1
del table_3d
print 'indices created'

# compute labels:
labels = k_means.predict(reduced_data)
print 'labels computed'

# n_clusters x 2d map shape
templates = np.zeros((k_means.cluster_centers_.shape[0], table_5d.shape[3], table_5d.shape[4]), dtype=np.float32)

# fill in tanplates
for i in xrange(length):
    templates[labels[i]] = templates[labels[i]] + table_5d[indices[i,0], indices[i,1], indices[i,2]]

print 'templates computed'

np.save(outname, templates)
