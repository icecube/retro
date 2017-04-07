from icecube.clsim.traysegments.common import parseIceModel
from icecube.clsim import NumberOfPhotonsPerMeter
from icecube import clsim
from os.path import expandvars
from icecube.icetray import I3Units

mediumProperties = parseIceModel(expandvars("$I3_SRC/clsim/resources/ice/spice_mie"), disableTilt=True)
domAcceptance = clsim.GetIceCubeDOMAcceptance()
photons_per_meter = NumberOfPhotonsPerMeter(mediumProperties.GetPhaseRefractiveIndex(0),
                                            domAcceptance,
                                            mediumProperties.GetMinWavelength(),
                                            mediumProperties.GetMaxWavelength()
                                            )
density = mediumProperties.GetMediumDensity()

# PPC parametrerization
nph=5.21*(0.924)/density
nphot10GeV = nph*photons_per_meter*10*I3Units.g/I3Units.cm3
print '10 GeV cascd = ',nphot10GeV
print '10 GeV trck = ', photons_per_meter * (10 * 15 / 3.33)
