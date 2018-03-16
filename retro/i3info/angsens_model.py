# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Apply weighting to photons based on an angular acceptance model. Note that this
model can include the effects of hole ice.
"""

from __future__ import absolute_import, division, print_function

__all__ = '''
    load_angsens_model
'''.split()

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
from os.path import basename, dirname, isfile, join, realpath
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(realpath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.utils.misc import expand


def load_angsens_model(model):
    """Load an angular sensitivity model.

    Note that this tries to look in current directory, in Retro data
    directory, and in the $I3_SRC directory--if it is defined--for the model
    named `model` _and_ `model` prefixed by "as." if it is not already.

    Returns
    -------
    angsens_poly : numpy.polynomial.Polynomial
    avg_angsens : float

    """
    models = [model]
    if not basename(model).startswith('as.'):
        models += join(dirname(model) + 'as.' + basename(model))

    possible_dirs = [
        '.',
        join(RETRO_DIR, 'data'),
    ]
    if 'I3_SRC' in os.environ:
        possible_dirs.append(
            join('$I3_SRC/ice-models/resources/models/angsens/'.split('/'))
        )

    possible_paths = []
    for model_name in models:
        possible_paths.extend(join(d, model_name) for d in possible_dirs)

    coeffs_loaded = False
    for path in possible_paths:
        path = expand(path)
        if not isfile(path):
            continue
        # The first number in the file is approximately equal (but greater
        # than) the peak in the distribution, used for scaling before
        # rejection sampling, so is useless for us (and makes simulation
        # less efficient).
        poly_coeffs = np.loadtxt(path)[1:]
        coeffs_loaded = True
        break

    if not coeffs_loaded:
        raise ValueError('Could not load hole ice model at any of\n{}'
                         .format(possible_paths))

    # We want coszen = -1 to correspond to upgoing particles, but angular
    # sensitivity is given w.r.t. the DOM axis (which points "down" towards
    # earth, and therefore is rotated 180-deg). So rotate the coszen
    # polynomial about cz=0 by negating the odd coefficients (coeffs are in
    # ascending powers of "x").
    flipped_coeffs = np.empty_like(poly_coeffs)
    flipped_coeffs[0::2] = poly_coeffs[0::2]
    flipped_coeffs[1::2] = -poly_coeffs[1::2]
    angsens_poly = np.polynomial.Polynomial(flipped_coeffs, domain=(-1, 1))

    integral_poly = angsens_poly.integ(m=1)
    avg_angsens = (integral_poly(1) - integral_poly(-1)) / 2

    return angsens_poly, avg_angsens
