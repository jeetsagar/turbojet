#!python3

"""preparing the dataset for pytorch"""

import h5py
import torch
import bisect
import numpy as np

from explore import print_keys

from torch.utils.data import Dataset, DataLoader


def normalize_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_denom = (x_max - x_min)
    x_denom[x_denom == 0] = 1
    x_norm = -1 + (2 * (x - x_min) / x_denom)
    return x_norm


def load_traindata(params):
    dataset = UnitDataset(params.traindata, params.units, mode='dev')
    train_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, pin_memory=False)
    return train_loader


def load_testdata(params):
    dataset = UnitDataset(params.testdata, params.units, mode='test')
    test_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False, pin_memory=False)
    return test_loader


class UnitDataset(Dataset):

    def __init__(self, filepath, units, mode='dev'):
        self.window = 50

        with h5py.File(filepath, 'r') as hdf:
            W_array = np.array(hdf.get(f'W_{mode}'))
            X_s_array = np.array(hdf.get(f'X_s_{mode}'))
            A_array = np.array(hdf.get(f'A_{mode}'))
            Y_array = np.array(hdf.get(f'Y_{mode}'))
        unit_array = np.array(A_array[:, 0], dtype=np.int32)

        existing_units = list(np.unique(unit_array))
        if units:
            units = units[0]
            self.units = list(set(units).intersection(set(existing_units)))
            self.units.sort()
        else:
            self.units = existing_units
        self.num_units = len(self.units)

        dev_data = np.concatenate((W_array, X_s_array), axis=1)
        dev_data = normalize_data(dev_data)

        self.data_list = []
        self.target_list = []
        self.length_list = []
        self.total_length = 0

        for unit in self.units:
            unit_ind = (unit_array == unit)

            unit_data = dev_data[unit_ind]
            unit_target = Y_array[unit_ind]
            unit_target = unit_target[self.window:]

            # using a subset of the data for testing
            # unit_data = unit_data[:1024+self.window]
            # unit_target = unit_target[:1024]

            # remove the transpose() call when using tensorflow
            # tensorflow uses channels last, but pytorch uses channels first
            data_tensor = torch.Tensor(unit_data).transpose(0, 1)
            target_tensor = torch.Tensor(unit_target)
            self.data_list.append(data_tensor)
            self.target_list.append(target_tensor)

            target_length = target_tensor.shape[0]

            self.total_length += target_length
            self.length_list.append(target_length)

        self.total_elem = list(np.cumsum(self.length_list))

    def _get_index(self, n):
        n = n + 1
        n = max(1, min(self.total_length, n))
        i = bisect.bisect_left(self.total_elem, n)
        if i == 0:
            j = n - 1
        else:
            m = self.total_elem[i-1]
            j = n - m - 1
        return i, j

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        i, j = self._get_index(index)
        data = self.data_list[i][:, j:j+self.window]
        target = self.target_list[i][j]
        return data, target


if __name__ == '__main__':
    fpath = '../../../data_set/N-CMAPSS_DS02-006.h5'
    # print_keys(fpath)
    # ds = UnitDataset(fpath, [])
    # a, b = ds[0]
    # print(a.shape, b.shape)
    # print(ds.units)
    # print(ds.num_units)
    # print(ds.length_list)
    # print(len(ds))

    ds = UnitDataset(fpath, [[14]], mode='test')
    td = DataLoader(ds, batch_size=4, shuffle=False, pin_memory=False)

    for i, (j, k) in enumerate(td):
        if i > 1:
            break
        print(j.shape)
        print(j.dtype)
        print(k.shape)
        print(k.dtype)
