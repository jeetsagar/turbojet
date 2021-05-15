#!python3

import h5py
import torch
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
    datapath = params.traindata
    unit = params.unit
    batch_size = params.batch_size
    dataset = DevDataset(datapath, unit)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader


class DevDataset(Dataset):

    def __init__(self, filepath, unit=2):
        with h5py.File(filepath, 'r') as hdf:
            W_dev = np.array(hdf.get('W_dev'))
            X_s_dev = np.array(hdf.get('X_s_dev'))
            A_dev = np.array(hdf.get('A_dev'))
            Y_dev = np.array(hdf.get('Y_dev'))
        unit_array = np.array(A_dev[:, 0], dtype=np.int32)

        dev_data = np.concatenate((W_dev, X_s_dev), axis=1)
        unit_ind = (unit_array == unit)

        unit_data = dev_data[unit_ind]
        unit_data = normalize_data(unit_data)
        unit_target = Y_dev[unit_ind]

        # remove the transpose() call when using tensorflow
        # tensorflow uses channels last, but pytorch uses channels first
        self.source = torch.Tensor(unit_data).transpose(0, 1)
        self.target = torch.Tensor(unit_target)

        self.window = 50
        self.length = self.source.shape[1] - self.window

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.source[:, index:index+self.window]
        target = self.target[index+self.window]
        return data, target


if __name__ == '__main__':
    fpath = '../../data_set/N-CMAPSS_DS02-006.h5'
    print_keys(fpath)
    ds = DevDataset(fpath)
    print(len(ds))
    a, b = ds[0]
    print(a.shape)
    print(b.shape)
