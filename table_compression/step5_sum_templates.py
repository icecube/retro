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
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite existing file')
args = parser.parse_args()


first = True
for cluster_idx in range(60):
    if os.path.isfile('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/templates.npy'%(cluster_idx)):
        template = np.load('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/cl%s/templates.npy'%(cluster_idx))

        if first:
            templates = np.zeros(template.shape)
            first = False

        # add
        templates = templates + template
    else:
        print 'templates missing for table cluster %s'%(cluster_idx)

print 'all templates summed'

# now followed by some numpy acrobatics

# sum
n_templates = np.sum(templates, axis=(1,2))

#normalize:
for i in range(templates.shape[0]):
    if n_templates[i] > 0:
        templates[i] /= n_templates[i]
    else:
        print 'template %i is zero - substituting with flat template'%i
        templates[i] = np.ones_like(templates[i])
        templates[i] /= np.sum(templates[i])

# sort, why not (largest first)
sorted_indices = np.argsort(n_templates)
templates = templates[sorted_indices[::-1]]

#save
np.save('/data/icecube/retro_tables/tilt_on_anisotropy_on_noazimuth_80/final_templates.npy', templates)
