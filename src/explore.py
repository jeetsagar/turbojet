#!python3

import h5py
import numpy as np

def print_keys(filename):
    with h5py.File(filename, 'r') as hdf:
        keylist = list(hdf.keys())
        varkeys = [i for i in keylist if i.endswith('var')]

        for k in varkeys:
            vararr = np.array(hdf.get(k))
            varlist = list(np.array(vararr, dtype='U20'))
            print(k, varlist)
