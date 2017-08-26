#!/usr/bin/env python

"""
Display photon production info for tracks and cascades, as parameterized in
IceCube software
"""


from __future__ import absolute_import, division, print_function

from icecube.clsim.traysegments.common import parseIceModel
from icecube.clsim import NumberOfPhotonsPerMeter
from icecube import clsim
from os.path import expandvars
from icecube.icetray import I3Units


mediumProperties = parseIceModel(
    expandvars("$I3_SRC/clsim/resources/ice/spice_mie"),
    disableTilt=True
)
domAcceptance = clsim.GetIceCubeDOMAcceptance()
photons_per_m_trck = NumberOfPhotonsPerMeter(
    mediumProperties.GetPhaseRefractiveIndex(0),
    domAcceptance,
    mediumProperties.GetMinWavelength(),
    mediumProperties.GetMaxWavelength()
)

density = mediumProperties.GetMediumDensity() * (I3Units.cm3 / I3Units.g)
"""Density in units of g/cm^3"""

# PPC parametrerization
photons_per_gev_em_cscd = 5.21 * 0.924 / density

# TODO: what factor to use here?
#photons_per_gev_had_cscd = photons_per_gev_em_cscd * ???
m_per_gev_trck = 15 / 3.33
photons_per_gev_trck = photons_per_m_trck * m_per_gev_trck


if __name__ == '__main__':
    print('Medium density is reported to be %.5f g/cm^3' % density)
    print('')
    print('Muon track:')
    print('  %10.3f photons per m' % photons_per_m_trck)
    print('  %10.3f photons per GeV' % photons_per_gev_trck)
    print('  %10.3f m per GeV' % m_per_gev_trck)
    print('')
    print('Electromagnetic cascade:')
    print('  %10.3f photons per GeV' % photons_per_gev_em_cscd )
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
