#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import
"""
read in an i3 file from CLsim and output a pkl file
containing the photon series
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
raw_data_list = []

fname = sys.argv[1]

dom_acceptance = GetIceCubeDOMAcceptance()

i3_file = dataio.I3File(fname)

counter = 0

while i3_file.more():
    frame = i3_file.pop_frame()
    if not frame.Has('photons'):
        continue
    if counter > 1000:
        break
    counter += 1
    raw_data_list.append({})
    raw_data = raw_data_list[-1]
    if counter%100==0:
        print(counter)
    p = frame['photons']
    for DOM in p:
        DOM_key = DOM[0]
        string = int(DOM_key.string)
        dom = int(DOM_key.om)
        # create dicts
        if not raw_data.has_key(string):
            raw_data[string] = {}
        if not raw_data[string].has_key(dom):
            raw_data[string][dom] = {'time':[], 'weight':[]}

        for hit in DOM[1]:
            raw_data[string][dom]['time'].append(hit.time)
            raw_data[string][dom]['weight'].append(hit.weight*dom_acceptance.GetValue(hit.wavelength))
        raw_data[string][dom]['time'] = np.array(raw_data[string][dom]['time'])
        raw_data[string][dom]['weight'] = np.array(raw_data[string][dom]['weight'])

outfile = open(fname.rstrip('.i3.bz2') + '_events.pkl', 'wb')
pickle.dump(raw_data_list, outfile)
outfile.close()
