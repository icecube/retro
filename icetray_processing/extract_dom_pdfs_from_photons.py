#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import, invalid-name

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


from collections import OrderedDict
from os.path import basename, expandvars
import sys
import pickle
import re

import numpy as np
from scipy.interpolate import interp1d

from icecube import dataclasses, simclasses, dataio # pylint: disable=import-error, unused-import
#from icecube.clsim.GetIceCubeDOMAcceptance import *


# dict to store the photons
raw_data = OrderedDict([('doms', OrderedDict()), ('meta', OrderedDict())])

fname = sys.argv[1]

#dom_acceptance = GetIceCubeDOMAcceptance()

i3_file = dataio.I3File(fname, 'r')

n_bins = 250
bins = np.linspace(0, 3000, n_bins+1)

counter = 0
while i3_file.more():
    frame = i3_file.pop_frame()
    if not frame.Has('photons'):
        continue

    counter += 1
    if counter % 100 == 0:
        sys.stdout.write('.')
        sys.stdout.flush()

    photon_series = frame['photons']

    for dom_key, photon_info in photon_series:
        if dom_key not in raw_data['doms']:
            raw_data['doms'][dom_key] = OrderedDict([
                ('time', []),
                ('wavelength', []),
                ('coszen', []),
                #('original_weight', [])
            ])

        for photon in photon_info:
            raw_data['doms'][dom_key]['time'].append(photon.time)
            raw_data['doms'][dom_key]['wavelength'].append(photon.wavelength)
            raw_data['doms'][dom_key]['coszen'].append(np.cos(photon.dir.zenith))
            #raw_data['doms'][dom_key]['original_weight'].append(photon.weight)

            # This only serves to make the weights the same (and approx. 1)...
            #raw_data['doms'][dom_key]['weight'].append(
            #    photon.weight*dom_acceptance.GetValue(photon.wavelength)
            #)

if counter >= 100:
    sys.stdout.write('\n')


# Convert I3OMKey object to integers so we can ditch the albatross known as
# IceCube software
raw_data['doms'] = {
    (int(k.string), int(k.om)): v for k, v in raw_data['doms'].items()
}

# Sorted dict by (string, dom) such that output is guaranteed to be consistent
sorted_data = OrderedDict()
for dom in sorted(raw_data['doms'].keys()):
    data_dict = raw_data['doms'][dom]
    for k in data_dict.keys():
        data_dict[k] = np.array(data_dict[k])
    sorted_data[dom] = data_dict
raw_data['doms'] = sorted_data

#==============================================================================
# Add weights that transform the samples into a realistic distribution of
# photon detection probabilities
#==============================================================================

# Since we assume the generated photons will account for absolute energy
# contained within the "useful" range of wavelengths, we normalize the Ckv
# wavelength distribution to peak at 1 and all other probabilities are reduced
# from that. "Overall" normalization factor is, therefore, contained in the
# number of photons a hypothesis says it produces.

# Shortest wavelength (and therefore the point at which to normalize to 1) is
# assumed to be 260 nm.
wl_ckv_accept = np.loadtxt(
    '../data/sampled_cherenkov_distr_and_dom_acceptance_vs_wavelength.csv',
    delimiter=','
)
wavelengths = wl_ckv_accept[:, 0]
ckv_distr = wl_ckv_accept[:, 1]
ckv_distr_linterp = interp1d(
    x=wavelengths,
    y=ckv_distr,
    kind='linear'
)
ckv_at_260 = ckv_distr_linterp(260)
ckv_distr_linterp = interp1d(
    x=wavelengths,
    y=ckv_distr / ckv_at_260,
    kind='linear'
)


fname_base = basename(fname)
match = re.match(r'.*holeice_([^_]*)', fname_base)
if match is None:
    hole_ice_model = 'as.h2-50cm'
else:
    hole_ice_model = match.groups()[0]

# Note the first number in the file is a number approximately equal (but
# greater than) the peak in the distribution, so is useless for us.
poly_coeffs = np.loadtxt(expandvars(
    '$I3_SRC/ice-models/resources/models/angsens/' + hole_ice_model
))[1:]

# We want coszen = -1 to correspond to upgoing particles, but angular
# sensitivity is given w.r.t. the DOM axis (which points "down" towards earth,
# and therefore is rotated 180-deg). So mirror the coszen polynomial about cz=0
# by negating the odd coefficients.
flipped_coeffs = np.empty_like(poly_coeffs)
flipped_coeffs[0::2] = poly_coeffs[0::2]
flipped_coeffs[1::2] = -poly_coeffs[0::2]

angsens_poly = np.polynomial.Polynomial(flipped_coeffs, domain=(-1, 1))


for key, data_dict in raw_data['doms'].items():
    wl = data_dict['wavelength'] * 1e9
    try:
        ckv_wt = ckv_distr_linterp(wl)
    except:
        print(np.min(wl), np.max(wl))
        raise

    cz = data_dict['coszen']
    try:
        angsens_wt = angsens_poly(cz)
    except:
        print(np.min(cz), np.max(cz))
        raise

    data_dict['weight'] = ckv_wt * angsens_wt

    for k, array in data_dict.items():
        data_dict[k] = array.astype(np.float32)

raw_data['meta']['num_sims'] = counter
pickle.dump(
    raw_data,
    open(fname_base + '_photon_raw_data.pkl', 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL
)

#histos = OrderedDict()
#for key, data in raw_data.items():
#    if isinstance(key, basestring):
#        continue
#    hist, _ = np.histogram(data['time'], bins=bins, weights=data['weight'])
#    hist /= counter
#    histos[key] = hist
#
#basename = fname.rstrip('.bz2').rstrip('.i3')
#pickle.dump(
#    histos,
#    open(basename + '_photon_histos.pkl', 'wb'),
#    protocol=pickle.HIGHEST_PROTOCOL
#)
