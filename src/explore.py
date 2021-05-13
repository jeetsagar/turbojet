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

        A_dev = np.array(hdf.get('A_dev'))
        A_test = np.array(hdf.get('A_test'))

    dev_units = np.array(A_dev[:, 0], dtype=np.int32)
    test_units = np.array(A_test[:, 0], dtype=np.int32)
    dev_unique = np.unique(dev_units)
    test_unique = np.unique(test_units)
    dev_hist = np.histogram(dev_units, dev_unique)
    test_hist = np.histogram(test_units, test_unique)
    print('dev: ', dev_hist[1], dev_hist[0])
    print('test: ', test_hist[1], test_hist[0])
