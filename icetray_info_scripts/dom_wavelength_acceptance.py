#!/usr/bin/env python


from __future__ import absolute_import, division, print_function

from icecube.clsim.GetIceCubeDOMAcceptance import *
from icecube import icetray, dataclasses
import numpy as np


def getPhaseRefIndex(wavelength):
    x = wavelength/1000 # wavelength in micrometer
    return 1.55749 - 1.57988*x + 3.99993*x**2 - 4.68271*x**3 + 2.09354*x**4

def Cherenkov_dN_dXdwlen(wlen, beta=1):
    value = (np.pi/(137*(wlen**2)))*(1 - 1/((beta*getPhaseRefIndex(wlen))**2))
    return np.where(value>0, value, 0)


def dom_wavelength_acceptance():
    dom_acceptance = GetIceCubeDOMAcceptance()
    acceptance = []
    cherenkov = []
    wavelengths = np.arange(265, 680, 5)
    for wlen in wavelengths:
        acceptance.append(dom_acceptance.GetValue(wlen*I3Units.nanometer) * Cherenkov_dN_dXdwlen(wlen))
        cherenkov.append(Cherenkov_dN_dXdwlen(wlen))
    integral = np.trapz(y=acceptance, x=wavelengths)
    integral_cherenkov = np.trapz(y=cherenkov, x=wavelengths)
    acceptance /= integral
    wavelength_acceptance = np.array(zip(wavelengths, acceptance))
    print('    wavelength (nm)  acceptance')
    print(wavelength_acceptance)
    print('integral =', integral)
    print('cherenkov integral = ', integral_cherenkov)
    print('fraction (accept./cherenk.) = ', integral/integral_cherenkov)
    header = (
        'Sampled DOM wavelength acceptance\n'
        'wavelength (nm), acceptance'
    )
    fpath = 'sampled_dom_wavelength_acceptance.csv'
    np.savetxt(fpath, wavelength_acceptance, delimiter=',', header=header)
    print('Saved sampled wavelength acceptance to "{}"'.format(fpath))


if __name__ == '__main__':
    dom_wavelength_acceptance()
