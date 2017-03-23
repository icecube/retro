from icecube.clsim.GetIceCubeDOMAcceptance import *
from icecube import icetray, dataclasses
import numpy as np
import scipy, scipy.integrate

def getPhaseRefIndex(wavelength):
    x = wavelength/1000.# wavelength in micrometer
    return 1.55749 - 1.57988*x + 3.99993*x**2. - 4.68271*x**3. + 2.09354*x**4.

def Cherenkov_dN_dXdwlen(wlen, beta=1.):
    value = (np.pi/(137.*(wlen**2.)))*(1. - 1./((beta*getPhaseRefIndex(wlen))**2.))
    return np.where(value>0.,value,0.)

dom_acceptance = GetIceCubeDOMAcceptance()
acceptance = []
wavelengths = np.arange(265, 680, 5)
for wlen in wavelengths:
    acceptance.append(dom_acceptance.GetValue(wlen*I3Units.nanometer) * Cherenkov_dN_dXdwlen(wlen))
integral = scipy.integrate.trapz(y=acceptance, x=wavelengths)
acceptance /= integral
for x,y in zip(wavelengths, acceptance):
    print x,y
