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
from os.path import abspath, dirname
import sys

import numpy as np
from scipy import interpolate

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.hypo.discrete_muon_kernels import (
    const_energy_loss_muon, table_energy_loss_muon
)
from retro.const import (TRACK_M_PER_GEV,
                         TRACK_PHOTONS_PER_M)
from retro.retro_types import HypoParams8D


hypo_params = HypoParams8D(
        t=0,
        x=0,
        y=0,
        z=0,
        track_azimuth=0,
        track_zenith=np.pi / 2,
        track_energy=10,
        cascade_energy=5
)

#dedx = table_energy_loss_muon(hypo_params, dt=1.0)
#print('Time values for non-constant dedx function:')
#print(dedx[:,0])
#const = const_energy_loss_muon(hypo_params, dt=1.0)
#print('Time values for constant dedx function:')
#print(const[:,0])
#
#print('x values for non-constant dedx function:')
#print(dedx[:,1])
#print('x values for constant dedx function:')
#print(const[:,1])

for i in range(1, -4, -1):
    dedx = table_energy_loss_muon(hypo_params, dt = 10 ** i)
    print(dedx[-1, 0])
    print(dedx[-1, 1])
