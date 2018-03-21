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

for string in ['ic', 'dc']:
    for depth_idx in range(60):
        if os.path.isfile('/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_%s_depth_%s/templates.npy'%(string, depth_idx)):
            template = np.load('/data/icecube/retro_tables/large_5d_notilt_combined/large_5d_notilt_string_%s_depth_%s/templates.npy'%(string, depth_idx))

            if first:
                templates = np.zeros(template.shape)
                first = False

            # add
            templates = templates + template
        else:
            print 'templates missing for string %s depth %s'%(string, depth_idx)

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
np.save('final_templates.npy', templates)
