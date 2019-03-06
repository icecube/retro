#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position

"""
Display photon production info for tracks and cascades, as parameterized in
IceCube software
"""

from __future__ import absolute_import, division, print_function

__all__ = ['main']

from os.path import abspath, dirname, expandvars
import sys

import numpy as np

if __name__ == '__main__' and __package__ is None:
    RETRO_DIR = dirname(dirname(dirname(abspath(__file__))))
    if RETRO_DIR not in sys.path:
        sys.path.append(RETRO_DIR)
from retro.const import NOMINAL_ICE_DENSITY


def main():
    from icecube.clsim.traysegments.common import parseIceModel
    from icecube.clsim import NumberOfPhotonsPerMeter
    from icecube import clsim
    #from icecube.icetray import I3Units

    mediumProperties = parseIceModel(
        expandvars("$I3_SRC/clsim/resources/ice/spice_mie"),
        disableTilt=True,
    )
    domAcceptance = clsim.GetIceCubeDOMAcceptance()
    ppmt = []
    for i in range(1000):
        try:
            photons_per_m_trck = NumberOfPhotonsPerMeter(
                mediumProperties.GetPhaseRefractiveIndex(i),
                domAcceptance,
                mediumProperties.GetMinWavelength(),
                mediumProperties.GetMaxWavelength()
            )
        except RuntimeError:
            break
        ppmt.append(photons_per_m_trck)

    #density = mediumProperties.GetMediumDensity() * (I3Units.cm3 / I3Units.g)
    #"""Density in units of g/cm^3"""

    ppmt = np.array(ppmt)
    print(
        'photons_per_m_trck: min={}, mean={}, median={}, max={}'.format(
            np.min(ppmt), np.mean(ppmt), np.median(ppmt), np.max(ppmt)
        )
    )
    #densities = np.array(densities)
    #print(np.min(densities), np.mean(densities), np.median(densities), np.max(densities))


    # PPC parametrerization
    #
    # Following adapted slightly from C++ source at
    #   private/clsim/I3CLSimLightSourceToStepConverterPPC.cxx:
    #
    # For a layer in the ice i:
    # meanPhotonsPerMeter = NumberOfPhotonsPerMeter(
    #     *(mediumProperties_->GetPhaseRefractiveIndex(i)),
    #     *(wlenBias_),
    #     mediumProperties_->GetMinWavelength(),
    #     mediumProperties_->GetMaxWavelength()
    # );
    # const double meanNumPhotons = f*meanPhotonsPerMeter*nph*E;
    # (f = 1 unless emScaleSigma != 0; or else it's a random normal in [0, 1];
    # therefore, assume it's 1)
    photons_per_gev_em_cscd = 5.21 * 0.924 / NOMINAL_ICE_DENSITY * photons_per_m_trck

    # TODO: what factor to use here?
    #photons_per_gev_had_cscd = photons_per_gev_em_cscd * ???
    m_per_gev_trck = 15 / 3.33
    photons_per_gev_trck = photons_per_m_trck * m_per_gev_trck

    print('Medium density is reported to be %.15f g/cm^3' % NOMINAL_ICE_DENSITY)
    print('')
    print('Muon track:')
    print('  %10.3f photons per m' % photons_per_m_trck)
    print('  %10.3f photons per GeV' % photons_per_gev_trck)
    print('  %10.3f m per GeV' % m_per_gev_trck)
    print('')
    print('Electromagnetic cascade:')
    print('  %10.3f photons per GeV' % photons_per_gev_em_cscd)
    print('')
    #print('Hadronic cascade:')
    #print('  %10.3f photons per GeV' % photons_per_gev_had_cscd)
    #print('')
    print('10 GeV EM cascade      : %10.3f photons'
          % (10*photons_per_gev_em_cscd))
    #print('10 GeV hadronic cascade: %10.3f photons'
    #      % (10*photons_per_gev_had_cscd))
    print('10 GeV muon track      : %10.3f photons'
          % (10*photons_per_m_trck))


if __name__ == '__main__':
    main()
