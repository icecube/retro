import numpy as np

def PowerAxis(minval, maxval, n_bins, power):
    l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
    bin_edges = np.power(l, power)
    return bin_edges
