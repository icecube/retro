#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position, invalid-name

from __future__ import absolute_import, division, print_function

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import pandas as pd
import seaborn as sns
sns.set(style="ticks", color_codes=True)
from scipy import stats

from pymultinest.analyse import Analyzer


llhs = []
bestfits = []

event = 2

for i in range(1000):
    #if os.path.exists('/gpfs/scratch/pde3/retro/test_event%i/log/run_%04d.log'%(event,i)):
    if os.path.exists('/gpfs/scratch/pde3/retro/log_cscd/run_%04d.log'%(i)):
        try:
            #a = Analyzer(8, outputfiles_basename="/gpfs/scratch/pde3/retro/test_event%i/out/tol0.1_evt%i-"%(event,i))
            a=Analyzer(8, outputfiles_basename="/gpfs/scratch/pde3/retro/out_cscd/tol0.1_evt%i-"%(i))
            bestfit_params=a.get_best_fit()
            llhs.append(bestfit_params['log_likelihood'])
            bestfits.append(bestfit_params['parameters'])
        except IOError:
            pass

llhs = np.array(llhs)
bestfits = np.array(bestfits)

names = ['time', 'x', 'y', 'z', 'zenith', 'azimuth', 'energy', 'track_fraction']
units = ['ns', 'm', 'm', 'm', 'rad', 'rad', 'GeV', None]
#event 1:
#if event == 1:
#    truth = {'time':0, 'x':70, 'y':-10, 'z':-250, 'zenith':1.27, 'azimuth':0.7, 'energy':20, 'track_fraction':1}
#event 2:
#elif event == 2:
#    truth = {'time':0, 'x':90, 'y':-90, 'z':-450, 'zenith':0.84, 'azimuth':0., 'energy':20, 'track_fraction':0}
truth = {}
truth['time'] = 0
truth['x'] = 0
truth['y'] = 0
truth['z'] = -400
truth['zenith'] = np.pi
truth['azimuth'] = 0
truth['energy'] = 20
truth['track_fraction'] = 0

data = {}

for i, name in enumerate(names):
    data[name] = bestfits[:,i]
data['llh'] = llhs

df = pd.DataFrame(data)

# exclude outliers that are 5 or more sigmas away
df = df[(np.abs(stats.zscore(df)) < 5).all(axis=1)]

dot_size = max(min(7, int(1000/len(df.index))), 1)

def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = %.2f\np0 = %.1e"%(r, p),
                xy=(.1, .8), xycoords=ax.transAxes)


def add_text(x, **kwargs):
    ax = plt.gca()
    m = np.mean(x)
    med = np.median(x)
    unc = np.std(x)/np.sqrt(len(x))
    std = np.std(x)
    a_text = AnchoredText('median = %.2f\nmean = %.2f +/- %.2f\nstd = %.2f'%(med, m, unc, std), loc=2, frameon=False)
    ax.add_artist(a_text)

def add_lines(*args, **kwargs):
    ax = plt.gca()
    for i,x in enumerate(args):
        if truth.has_key(x.name):
            if i == 0:
                ax.axvline(truth[x.name], color='r', linewidth=2)
            else:
                ax.axhline(truth[x.name], color='r', linewidth=2)

g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.regplot, scatter_kws={'s':dot_size})
g.map_lower(sns.kdeplot, shade=True, cmap='Blues', shade_lowest=False)
g.map_upper(corrfunc)
g.map_lower(add_lines)
g.map_diag(plt.hist)
g.map_diag(add_lines)
g.map_diag(add_text)

#plt.savefig('reco_event%i.png'%event)
#plt.savefig('reco_event%i.pdf'%event)
plt.savefig('reco_cscd.png')
plt.savefig('reco_cscd.pdf')

#bins = 24
#
#for i, name in enumerate(names):
#    points = bestfits[:,i]
#    plt.hist(points, bins=bins, alpha=0.7)
#    label = name
#    if units[i] is not None:
#        label+= ' (%s)'%units[i]
#    plt.gca().set_xlabel(label)
#    a_text = AnchoredText('mean: %.2f, std_dev: %.2f'%(points.mean(), points.std()), loc=2)
#    plt.gca().add_artist(a_text)
#    plt.savefig('reco_%s.png'%name)
#    plt.clf()

#plt.hist(llhs, bins=bins, alpha=0.7)
#plt.gca().set_xlabel('llh')
#plt.savefig('reco_llh.png')
