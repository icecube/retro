import numpy as np

'''
class to handle sparse elements in nd array'''

class sparse(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, shape, default=0., dtype=float):
        
        self.__default = default #default value of non-assigned elements
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.__data = {}


    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.__data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self.__data.get(index,self.__default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if self.__data.has_key(index):
            del(self.__data[index])

    def __iter__(self):
        for key, val in self.__data.items():
            yield key, val
