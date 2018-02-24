#!/usr/bin/env python


from __future__ import absolute_import, division, print_function

import numpy as np

from icecube.clsim.GetIceCubeDOMAcceptance import I3Units, GetIceCubeDOMAcceptance
from icecube import icetray, dataclasses


def get_phase_ref_index(wlen):
    """
    Parameters
    ----------
    wlen
        Wavelength in units of nm

    """
    # convert wavelength to micrometers
    wl_um = wlen/1000
    return 1.55749 - 1.57988*wl_um + 3.99993*wl_um**2 - 4.68271*wl_um**3 + 2.09354*wl_um**4


def cherenkov_dN_dXdwlen(wlen, beta=1):
    """
    Parameters
    ----------
    wlen
        Wavelength in units of nm

    beta
        Beta factor of particle emitting the Cherenkov light

    """
    value = (np.pi / (137 * wlen**2)) * (1 - 1/((beta * get_phase_ref_index(wlen))**2))
    return np.where(value>0, value, 0)


def dom_wavelength_acceptance(wlens=np.arange(265, 680, 5), beta=1):
    """
    Parameters
    ----------
    wlens : iterable
        Wavelenghts, in nm

    beta : float
        Beta factor of particle emitting the Cherenkov light

    """
    dom_acceptance = GetIceCubeDOMAcceptance()
    acceptance = []
    cherenkov = []
    for wlen in wlens:
        acceptance.append(dom_acceptance.GetValue(wlen*I3Units.nanometer))
        cherenkov.append(cherenkov_dN_dXdwlen(wlen=wlen, beta=beta))
    combined = [a*c for a, c in zip(acceptance, cherenkov)]

    integral_combined = np.trapz(y=combined, x=wlens)
    integral_acceptance = np.trapz(y=acceptance, x=wlens)
    integral_cherenkov = np.trapz(y=cherenkov, x=wlens)
    combined /= integral_combined
    wavelength_combined = np.array(zip(wlens, combined))
    wavelength_ckv_accept = np.array(zip(wlens, cherenkov, acceptance))

    print('    wavelength (nm)  acceptance*cherenkov')
    print(wavelength_combined)
    print('integral(acceptance):           {:.8e}'.format(integral_acceptance))
    print('integral(Cherenkov):            {:.8e}'.format(integral_cherenkov))
    print('integral(acceptance*Cherenkov): {:.8e}'.format(integral_combined))
    print('fraction (combined. / cherenk.) = ', integral_combined/integral_cherenkov)

    header = (
        'Sampled DOM wavelength acceptance\n'
        'wavelength (nm), acceptance'
    )
    fpath = 'sampled_dom_wavelength_acceptance.csv'
    np.savetxt(fpath, wavelength_combined, delimiter=',', header=header)
    print('Saved sampled wavelength acceptance to "{}"'.format(fpath))

    header = (
        'Sampled Cherenkov distribution and DOM wavelength acceptance\n'
        'wavelength (nm), Chernkov distr (normalized), acceptance'
    )
    fpath = 'sampled_cherenkov_distr_and_dom_acceptance_vs_wavelength.csv'
    np.savetxt(fpath, wavelength_ckv_accept, delimiter=',', header=header)
    print('Saved wavelen + Ckv distr + DOM accept to "{}"'.format(fpath))


if __name__ == '__main__':
    dom_wavelength_acceptance()
