from icecube.clsim import GetIceCubeDOMAngularSensitivity
from os.path import expandvars
import numpy as np
import scipy, scipy.integrate
ang = GetIceCubeDOMAngularSensitivity(holeIce=expandvars("$I3_SRC/ice-models/resources/models/angsens/as.h2-50cm"))
print ang.GetValue(-1)
print ang.GetValue(0)
print ang.GetValue(1)
coszen = np.linspace(-1, 1, 101)
acceptance = []
for cz in coszen:
    acceptance.append(ang.GetValue(cz))
print coszen
print acceptance
integral = scipy.integrate.trapz(y=acceptance, x=coszen) /2.
print integral
