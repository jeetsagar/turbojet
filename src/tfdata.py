#!python3
"""
File To preprocess and prepare the data as tensorflow generator
"""
import h5py
import bisect
import numpy as np
import tensorflow as tf


# Normalisation preprocess [-1,1]
def normalize_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_denom = (x_max - x_min)
    x_denom[x_denom == 0] = 1
    x_norm = -1 + (2 * (x - x_min) / x_denom)
    return x_norm


# Class to read any dataset
class DataProvider:

    def __init__(self, filepath, units, mode):
        self.window = 50
        if mode == 1:
            with h5py.File(filepath, 'r') as hdf:
                W_dev = np.array(hdf.get('W_dev'))
                X_s_dev = np.array(hdf.get('X_s_dev'))
                X_sk_dev = np.array(hdf.get('X_sk_dev'))
                X_vk_dev = np.array(hdf.get('X_vk_dev'))
                Tk_dev = np.array(hdf.get('Tk_dev'))
                A_dev = np.array(hdf.get('A_dev'))
                Y_dev = np.array(hdf.get('Y_dev'))
        else:
            with h5py.File(filepath, 'r') as hdf:
                W_dev = np.array(hdf.get('W_test'))
                X_s_dev = np.array(hdf.get('X_s_test'))
                X_sk_dev = np.array(hdf.get('X_sk_test'))
                X_vk_dev = np.array(hdf.get('X_vk_test'))
                Tk_dev = np.array(hdf.get('Tk_test'))
                A_dev = np.array(hdf.get('A_test'))
                Y_dev = np.array(hdf.get('Y_test'))

        unit_array = np.array(A_dev[:, 0], dtype=np.int32)

        existing_units = list(np.unique(unit_array))
        if units:
            units = units[0]
            self.units = list(set(units).intersection(set(existing_units)))
            self.units.sort()
        else:
            self.units = existing_units
        self.num_units = len(self.units)

        dev_data = np.concatenate((W_dev, X_s_dev, X_sk_dev, X_vk_dev, Tk_dev), axis=1)

        dev_data = normalize_data(dev_data)

        self.data_list = []
        self.target_list = []
        self.length_list = []
        self.total_length = 0

        for unit in self.units:
            unit_ind = (unit_array == unit)

            unit_data = dev_data[unit_ind]
            unit_target = Y_dev[unit_ind]
            unit_target = unit_target[self.window:]

            # remove the transpose() call when using tensorflow
            # tensorflow uses channels last, but pytorch uses channels first
            data_tensor = unit_data
            target_tensor = unit_target
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
            m = self.total_elem[i - 1]
            j = n - m - 1
        return i, j

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        i, j = self._get_index(index)
        data = self.data_list[i][j:j + self.window, :]
        target = self.target_list[i][j]
        data = np.expand_dims(data, axis=0)
        target = np.expand_dims(target, axis=0)
        return data, target


# Prepare dataset as generator
def generate_data(filepath, units, mode):
    ds = DataProvider(filepath, units, mode)
    for i in range(ds.total_length):
        data, value = ds[i]
        yield data, value


# 50 here is number of timestamps and 49 is total attributes (after kalman) changes to 35 without kalman
def get_dataset(filepath, units, mode):
    return tf.data.Dataset.from_generator(generate_data, args=[filepath, units, mode],
                                          output_types=(tf.float32, tf.float32), output_shapes=([1, 50, 49], [1, ]))


if __name__ == '__main__':
    fname = '../../data_set/N-CMAPSS_DS02-006.h5'
    a = DataProvider(fname, [], "dev")
    b, c = a[0]
    print(b.shape, c.shape)
    tf_ds = get_dataset(fname, [], "dev")
