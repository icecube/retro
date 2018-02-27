#!/usr/bin/env python
# pylint: disable=wrong-import-position, wildcard-import, invalid-name

"""
Read in an i3 file from CLsim and generate a pickle file containing the photon
info (a dict containing time, wavelength, coszen, and weight for each DOM).
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
from os.path import basename, dirname, expanduser, expandvars, join
import sys
import pickle
import re

import numpy as np
from scipy.interpolate import interp1d

from icecube import dataclasses, simclasses, dataio # pylint: disable=import-error, unused-import

RETRO_DIR = dirname(dirname(__file__))
if not RETRO_DIR:
    RETRO_DIR = '../..'

# dict to store the extracted info
raw_data = OrderedDict([('doms', OrderedDict()), ('meta', OrderedDict())])

fname = sys.argv[1]

orig_fname = fname
fname = expanduser(expandvars(fname))
fname_base = basename(fname).rstrip('.bz2').rstrip('.i3')
outfpath = fname_base + '_photon_raw_data.pkl'
print('Extracting photons from "{}"\nand writing info to "{}"'
      .format(orig_fname, outfpath))

i3_file = dataio.I3File(fname, 'r')

num_sims = 0
while i3_file.more():
    frame = i3_file.pop_frame()
    if not frame.Has('photons'):
        continue

    num_sims += 1
    if num_sims % 100 == 0:
        sys.stdout.write('.')
        sys.stdout.flush()

    photon_series = frame['photons']

    for dom_key, photon_info in photon_series:
        if dom_key not in raw_data['doms']:
            raw_data['doms'][dom_key] = OrderedDict([
                ('time', []),
                ('wavelength', []),
                ('coszen', []),
            ])

        for photon in photon_info:
            raw_data['doms'][dom_key]['time'].append(photon.time)
            raw_data['doms'][dom_key]['wavelength'].append(photon.wavelength)
            raw_data['doms'][dom_key]['coszen'].append(np.cos(photon.dir.zenith))

if num_sims >= 100:
    sys.stdout.write('\n')


# Convert I3OMKey object to integers so we can ditch the albatross known as
# IceCube software
raw_data['doms'] = {
    (int(k.string), int(k.om)): v for k, v in raw_data['doms'].items()
}

# Sorted dict by (string, dom) such that output file is guaranteed to be
# consistent
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
    join(
        RETRO_DIR,
        'data',
        'sampled_cherenkov_distr_and_dom_acceptance_vs_wavelength.csv'
    ),
    delimiter=','
)
wavelengths = wl_ckv_accept[:, 0]
ckv_distr = wl_ckv_accept[:, 1]
ckv_distr_linterp = interp1d(
    x=wavelengths,
    y=ckv_distr,
    kind='linear'
)


match = re.match(r'.*holeice_([^_]*)', fname_base)
if match is None:
    print('Could not find hole ice model in filename; using "as.h2-50cm"')
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
flipped_coeffs[1::2] = -poly_coeffs[1::2]

angsens_poly = np.polynomial.Polynomial(flipped_coeffs, domain=(-1, 1))


for key, data_dict in raw_data['doms'].items():
    cz = data_dict['coszen']
    try:
        # Note that angular sensitivity _will_ modify the total number of
        # photons detected, so we do not divide by the sum of weights here.
        angsens_wt = angsens_poly(cz)
    except:
        print(np.min(cz), np.max(cz))
        raise

    #data_dict['weight'] = ckv_wt * angsens_wt / num_sims
    data_dict['weight'] = angsens_wt / num_sims

    for k, array in data_dict.items():
        data_dict[k] = array.astype(np.float32)

raw_data['meta']['num_sims'] = num_sims
pickle.dump(
    raw_data,
    open(outfpath, 'wb'),
    protocol=pickle.HIGHEST_PROTOCOL
)
