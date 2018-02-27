#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import
"""
read in an i3 file from CLsim and generate 'oversampled' photon arrival
distributions per DOM
"""

from __future__ import absolute_import, division, print_function

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this i3_file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


import sys
import numpy as np
import pickle

from icecube import dataclasses # pylint: disable=import-error, unused-import
from icecube import dataio # pylint: disable=import-error
from icecube.clsim import I3Photon

from icecube.clsim.GetIceCubeDOMAcceptance import *

import matplotlib.pyplot as plt


# dict to store the photons
raw_data = {}
# dict for histograms (to later save)
histos = {}
out = {}

fname = sys.argv[1]

dom_acceptance = GetIceCubeDOMAcceptance()

i3_file = dataio.I3File(fname)

n_bins = 200
bins = np.linspace(0, 2000, n_bins+1)

counter = 0

while i3_file.more():
    frame = i3_file.pop_frame()
    if not frame.Has('photons'):
        continue
    #if counter > 10000:
    #    break
    counter += 1
    if counter%100==0:
        print(counter)
    p = frame['photons']
    for DOM in p:
        DOM_key = DOM[0]
        times = []
        weights = []
        if not raw_data.has_key(DOM_key):
            #raw_data[DOM_key] = []
            raw_data[DOM_key] = {'time':[], 'weight':[]}
        for hit in DOM[1]:
            raw_data[DOM_key]['time'].append(hit.time)
            raw_data[DOM_key]['weight'].append(hit.weight*dom_acceptance.GetValue(hit.wavelength))
            #times.append(hit.time)
            #weights.append(hit.weight*dom_acceptance.GetValue(hit.wavelength))
        #raw_data[DOM_key]['time'] = np.array(raw_data[DOM_key]['time'])
        #raw_data[DOM_key]['weight'] = np.array(raw_data[DOM_key]['weight'])
        #c, _ = np.histogram(raw_data[DOM_key]['time'], bins=n_bins, range=range)
        #y, _ = np.histogram(times, bins=bins, weights=weights)
        #raw_data[DOM_key].append(y)
        #raw_data[DOM_key]['weight'].append(y)
        #raw_data[DOM_key]['counts'] += np.nan_to_num(w/c)

for key, d in raw_data.items():
    string = int(key.string)
    dom = int(key.om)
    #data = np.array(d)
    #print(data)
    #norm = np.nan_to_num(data/data)
    #h = np.sum(data, axis=0) / np.sum(norm, axis=0)
    #h = np.nan_to_num(h)
    #print(h)
    #h = np.true_divide(data.sum(axis=0),(data!=0).sum(axis=0))
    #h = np.nan_to_num(h)
    time = np.array(d['time']).flatten()
    weight = np.array(d['weight']).flatten()
    #print(time)
    hist, _ = np.histogram(time, bins=bins, weights=weight)
    #print(hist)
    #hist /= d['counts']
    hist /= counter
    #hist = np.nan_to_num(hist)
    #print(hist)
    if not histos.has_key(string):
        histos[string] = {}
    histos[string][dom] = hist
    #if not out.has_key(string):
    #    out[string] = {}
    #if not out[string].has_key(dom):
    #    out[string][dom] = {}
    #out[string][dom]['time'] = 0.5*(bins[1:] + bins[:-1])
    #out[string][dom]['weight'] = h

outfile = open(fname.rstrip('.i3.bz2') + '.pkl', 'wb')
pickle.dump(histos, outfile)
#pickle.dump(out, outfile)
outfile.close()

    # plotting
    #if len(weight) > counter/4:
    #    plt.hist(time, bins=200, weights=weight, range=(0,2000))
    #    #plt.hist(time, bins=40, range=(0,2000))
    #    plt.savefig('dom_pdfs/%s_%s.png'%(fname.rstrip('.i3.bz2'),key))
    #    plt.clf()
