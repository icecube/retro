"""Class to handle sparse elements in n-dimensional array"""


from __future__ import absolute_import, division


class Sparse(object):
    """Class for n-dimensional sparse array objects using Python dict.

    Parameters
    ----------
    shape
    default
    dtype
        Datatype, as passed to np.array(..., dtype=<dtype>)

    """
    def __init__(self, shape, default=0, dtype=float):
        self._default = dtype(default) #default value of non-assigned elements
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self._data = {}

    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self._data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self._data.get(index, self._default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if self._data.has_key(index):
            del self._data[index]

    def __iter__(self):
        for key, val in self._data.iteritems():
            yield key, val
