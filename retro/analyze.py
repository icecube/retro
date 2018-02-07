import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pymultinest.analyse import Analyzer

llhs = []
bestfits = []

#for i in range(10):
for i in [273]:
    try:
        a = Analyzer(8, outputfiles_basename = "/gpfs/scratch/pde3/retro/out/tol0.1_evt%i-"%i)
        bestfit_params = a.get_best_fit()
        llhs.append(bestfit_params['log_likelihood'])
        bestfits.append(bestfit_params['parameters'])
    except IOError:
        pass

llhs = np.array(llhs)
bestfits = np.array(bestfits)

names = ['t', 'x', 'y', 'z', 'zenith', 'azimuth', 'energy', 'track_fraction']

for i, name in enumerate(names):
    points = bestfits[:,i]
    plt.hist(points)
    plt.gca().set_xlabel(name)
    plt.savefig('reco_%s.png'%name)
    plt.clf()

