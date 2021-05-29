#!python3

"""functions for exploring the data"""

import h5py
import pathlib
import numpy as np


def get_filenames(parent='../../../data_set/'):
    parentdir = pathlib.Path(parent)
    flist = list(parentdir.glob('*.h5'))
    flist.sort()
    return flist


def print_keys(filename):
    with h5py.File(filename, 'r') as hdf:
        keylist = list(hdf.keys())
        varkeys = [i for i in keylist if i.endswith('var')]

        for k in varkeys:
            vararr = np.array(hdf.get(k))
            varlist = list(np.array(vararr, dtype='U20'))
            print(k, varlist, len(varlist))

        A_dev = np.array(hdf.get('A_dev'))
        A_test = np.array(hdf.get('A_test'))

    dev_units = np.array(A_dev[:, 0], dtype=np.int32)
    test_units = np.array(A_test[:, 0], dtype=np.int32)
    dev_unique = np.unique(dev_units)
    test_unique = np.unique(test_units)

    dev_bins = np.append(dev_unique, dev_unique[-1] + 1)
    test_bins = np.append(test_unique, test_unique[-1]+1)
    dev_hist, _ = np.histogram(dev_units, dev_bins)
    test_hist, _ = np.histogram(test_units, test_bins)

    print('dev: ', dev_unique, dev_hist)
    print('test: ', test_unique, test_hist)


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


if __name__ == '__main__':
    data_dir = '../../../data_set/'
    flist = get_filenames(data_dir)
    fpath = np.random.choice(flist)
    print(fpath.name)
    print_keys(fpath)
    for fpath in flist[:-1]:
        print(fpath.name)
        print_keys(fpath)
