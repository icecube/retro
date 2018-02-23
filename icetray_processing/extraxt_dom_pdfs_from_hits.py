#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import
"""
read in an i3 file contain SRT hit series and generate 'oversampled' hit
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

import matplotlib.pyplot as plt


# dict to store the photons
raw_data = {}
# dict for histograms (to later save)
histos = {}
out = {}

#pulse_series = 'SRTInIcePulses_90_700_1'
pulse_series = 'SplitUncleanedInIcePulses'

fname = sys.argv[1]
i3_file = dataio.I3File(fname)

n_bins = 200
bins = np.linspace(0, 2000, n_bins+1)

counter = 0

while i3_file.more():
    frame = i3_file.pop_frame()
    if not frame.Has(pulse_series):
        continue
    #if counter > 100:
    #    break
    counter += 1
    if counter%100==0:
        print(counter)
    hits = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, pulse_series)
    timeshift = frame['TimeShift'].value
    for DOM in hits:
        DOM_key = DOM[0]
        times = []
        weights = []
        if not raw_data.has_key(DOM_key):
            #raw_data[DOM_key] = []
            raw_data[DOM_key] = {'time':[], 'weight':[]}
        for hit in DOM[1]:
            raw_data[DOM_key]['time'].append(hit.time + timeshift)
            raw_data[DOM_key]['weight'].append(hit.charge)

for key, d in raw_data.items():
    string = int(key.string)
    dom = int(key.om)
    time = np.array(d['time']).flatten()
    weight = np.array(d['weight']).flatten()
    hist, _ = np.histogram(time, bins=bins, weights=weight)
    hist /= counter
    if not histos.has_key(string):
        histos[string] = {}
    histos[string][dom] = hist

outfile = open(fname.rstrip('.i3.bz2') + '_%s.pkl'%pulse_series, 'wb')
pickle.dump(histos, outfile)
outfile.close()

    # plotting
    #if len(weight) > counter/4:
    #    plt.hist(time, bins=200, weights=weight, range=(0,2000))
    #    #plt.hist(time, bins=40, range=(0,2000))
    #    plt.savefig('dom_pdfs/%s_%s.png'%(fname.rstrip('.i3.bz2'),key))
    #    plt.clf()
