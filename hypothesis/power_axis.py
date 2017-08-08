import numpy as np

def PowerAxis(minval, maxval, n_bins, power):
    '''
    creates bin edges (similar to np.linspace) that are in higher powers (e.g. quadratic, cubic, quartic)
    minval : minimum value for the binning
    maxval : maximum value for the binning
    n_bins : number of bins to be created (n_bins + 1 bin edges will be returned)
    power : power (e.g. 2, 3, 4...) that the binning should be based in
    '''
    l = np.linspace(np.power(minval, 1./power), np.power(maxval, 1./power), n_bins+1)
    bin_edges = np.power(l, power)
    return bin_edges
