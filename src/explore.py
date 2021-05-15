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

def check_dev_data(filename):
    with h5py.File(filename, 'r') as hdf:
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary
    return None


def check_test_data(filename):
    with h5py.File(filename, 'r') as hdf:
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary
    return None


def get_valid_dataset(filename, unit=2):
    with h5py.File(filename, 'r') as hdf:
        W_dev = np.array(hdf.get('W_dev'))
        X_s_dev = np.array(hdf.get('X_s_dev'))
        A_dev = np.array(hdf.get('A_dev'))
        Y_dev = np.array(hdf.get('Y_dev'))
    unit_array = np.array(A_dev[:, 0], dtype=np.int32)
    dev_data = np.concatenate((W_dev, X_s_dev, Y_dev), axis=1)
    unit_data = dev_data[unit_array == unit]
    return unit_data
