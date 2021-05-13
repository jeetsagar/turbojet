#!python3

import h5py
# import torch
import numpy as np
import pandas as pd

from explore import print_keys


# from torch.utils.data import Dataset, DataLoader


def normalize_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_denom = (x_max - x_min)
    x_denom[x_denom == 0] = 1
    x_norm = -1 + (2 * (x - x_min) / x_denom)
    return x_norm


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


class DevDataset:

    def __init__(self, filename, unit=2):
        with h5py.File(filename, 'r') as hdf:
            W_dev = np.array(hdf.get('W_dev'))
            X_s_dev = np.array(hdf.get('X_s_dev'))
            A_dev = np.array(hdf.get('A_dev'))
            Y_dev = np.array(hdf.get('Y_dev'))
        unit_array = np.array(A_dev[:, 0], dtype=np.int32)

        dev_data = np.concatenate((W_dev, X_s_dev), axis=1)
        unit_data = dev_data[unit_array == unit]
        unit_target = Y_dev[unit_array == unit]

        self.normed_data = normalize_data(unit_data)
        self.target = unit_target
        self.window = 50
        self.length = self.normed_data.shape[0] - self.window

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.normed_data[index:index+self.window, :]
        target = self.target[index+self.window]
        return data, target


if __name__ == '__main__':
    filename = '../../data_set/N-CMAPSS_DS02-006.h5'
    print_keys(filename)
    ds = DevDataset(filename)
    print(len(ds))
