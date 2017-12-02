from __future__ import absolute_import, division, print_function

import math
import numpy as np
import csv
from scipy import interpolate
import os
from os.path import abspath, dirname

if __name__ == '__main__' and __package__ is None:
    os.sys.path.append(dirname(dirname(abspath(__file__))))

from discrete_muon_kernels import (const_energy_loss_muon, table_energy_loss_muon)
from retro import (SPEED_OF_LIGHT_M_PER_NS, TRACK_M_PER_GEV,
                   TRACK_PHOTONS_PER_M, HypoParams8D)

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
