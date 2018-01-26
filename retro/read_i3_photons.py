#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import
"""
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

from icecube import dataclasses # pylint: disable=import-error, unused-import
from icecube import dataio # pylint: disable=import-error
from icecube.clsim import I3Photon

from icecube.clsim.GetIceCubeDOMAcceptance import *

import matplotlib.pyplot as plt


histos = {}

fname = sys.argv[1]

dom_acceptance = GetIceCubeDOMAcceptance()

i3_file = dataio.I3File(fname)

counter = 0

while i3_file.more():
    frame = i3_file.pop_frame()
    #if counter > 1000:
    #    break
    counter += 1
    if 'photons' in frame.keys():
        p = frame['photons']
        for DOM in p:
            DOM_key = DOM[0]
            if not histos.has_key(DOM_key):
                histos[DOM_key] = {'time':[], 'weight':[]}
            for hit in DOM[1]:
                histos[DOM_key]['time'].append(hit.time)
                histos[DOM_key]['weight'].append(hit.weight*dom_acceptance.GetValue(hit.wavelength))

for key, histo in histos.items():
    time = np.array(histo['time'])
    weight = np.array(histo['weight'])/counter
    if len(weight) > counter/4:
        plt.hist(time, bins=200, weights=weight, range=(0,2000))
        #plt.hist(time, bins=40, range=(0,2000))
        plt.savefig('dom_pdfs/%s_%s.png'%(fname.rstrip('.i3.bz2'),key))
        plt.clf()
