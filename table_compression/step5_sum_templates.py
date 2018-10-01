from __future__ import print_function, division
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
args = parser.parse_args()


first = True
for item in os.listdir(args.dir):
    path = os.path.join(args.dir, item)
    if os.path.isdir(path):
        tf = os.path.join(path, 'templates.npy')
        if os.path.isfile(tf):
            template = np.load(tf)
            print('adding templates from %s'%path)

            if first:
                templates = np.zeros(template.shape)
                first = False

            # add
            templates = templates + template
        else:
            print('templates missing for %s'%path)

print('all templates summed')

# now followed by some numpy acrobatics

# sum
n_templates = np.sum(templates, axis=(1,2))

#normalize:
for i in range(templates.shape[0]):
    if n_templates[i] > 0:
        templates[i] /= n_templates[i]
    else:
        print('template %i is zero - substituting with flat template'%i)
        templates[i] = np.ones_like(templates[i])
        templates[i] /= np.sum(templates[i])

# sort, why not (largest first)
sorted_indices = np.argsort(n_templates)
templates = templates[sorted_indices[::-1]]

#save
np.save(os.path.join(args.dir, 'ckv_dir_templates.npy'), templates)
