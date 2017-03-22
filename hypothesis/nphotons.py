from icecube.clsim.traysegments.common import parseIceModel
from icecube.clsim import NumberOfPhotonsPerMeter
from icecube import clsim
from os.path import expandvars

mediumProperties = parseIceModel(expandvars("$I3_SRC/clsim/resources/ice/spice_mie"), disableTilt=True)
print mediumProperties.GetMinWavelength()
print mediumProperties.GetMaxWavelength()

flatAcceptance = clsim.I3CLSimFunctionConstant(1.)
domAcceptance = clsim.GetIceCubeDOMAcceptance()

meanPhotonsPerMeterInLayer_flat = []
meanPhotonsPerMeterInLayer_dom = []

for  i in range(mediumProperties.GetLayersNum()):
    nPhot_flat = NumberOfPhotonsPerMeter(mediumProperties.GetPhaseRefractiveIndex(i),
                                   flatAcceptance,
                                   mediumProperties.GetMinWavelength(),
                                   mediumProperties.GetMaxWavelength()
                                   )
    nPhot_dom = NumberOfPhotonsPerMeter(mediumProperties.GetPhaseRefractiveIndex(i),
                                   domAcceptance,
                                   mediumProperties.GetMinWavelength(),
                                   mediumProperties.GetMaxWavelength()
                                   )

    meanPhotonsPerMeterInLayer_flat.append(nPhot_flat)
    meanPhotonsPerMeterInLayer_dom.append(nPhot_dom)

print meanPhotonsPerMeterInLayer_flat
print meanPhotonsPerMeterInLayer_dom
