#!python3

import h5py
import bisect
import numpy as np
import tensorflow as tf


def normalize_data(x):
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_denom = (x_max - x_min)
    x_denom[x_denom == 0] = 1
    x_norm = -1 + (2 * (x - x_min) / x_denom)
    return x_norm


class DataProvider:

    def __init__(self, filepath, units, mode):
        self.window = 50
        if mode==1:
            with h5py.File(filepath, 'r') as hdf:
                W_dev = np.array(hdf.get('W_dev'))
                X_s_dev = np.array(hdf.get('X_s_dev'))
                X_v_dev = np.array(hdf.get('X_v_dev'))
                T_dev = np.array(hdf.get('T_dev'))
                A_dev = np.array(hdf.get('A_dev'))
                Y_dev = np.array(hdf.get('Y_dev'))
        else:
            with h5py.File(filepath, 'r') as hdf:
                W_dev = np.array(hdf.get('W_test'))
                X_s_dev = np.array(hdf.get('X_s_test'))
                X_v_dev = np.array(hdf.get('X_v_test'))
                T_dev = np.array(hdf.get('T_test'))
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

        dev_data = np.concatenate((W_dev, X_s_dev,X_v_dev,T_dev[:, [-1,-2,-4]]), axis=1)
        # dev_data = np.concatenate((W_dev, X_s_dev), axis=1)


        # print(np.shape(dev_data))
        # exit()
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

            # using a subset of the data for testing
            # unit_data = unit_data[:1024+self.window]
            # unit_target = unit_target[:1024]

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


def generate_data(filepath, units,mode):
    ds = DataProvider(filepath, units,mode)
    for i in range(ds.total_length):
        data, value = ds[i]
        yield data, value


def get_dataset(filepath, units,mode):
    # return tf.data.Dataset.from_generator(generate_data, args=[filepath, units],output_signature=(tf.TensorSpec(shape=(1, 50, 18), dtype=tf.float32),tf.TensorSpec(shape=(1, 1), dtype=tf.float32)))
    return tf.data.Dataset.from_generator(generate_data, args=[filepath,units,mode], output_types=(tf.float32,tf.float32), output_shapes=([1,50,35],[1,1]))
# training_dataset = tf.data.Dataset.from_generator(
#      raw_data_gen, 
#      args=(1), 
#      output_types=(tf.float32, tf.uint8), 
#      output_shapes=([None, 1], [None]))

if __name__ == '__main__':
    fname = '../../data_set/N-CMAPSS_DS02-006.h5'
    a = DataProvider(fname, [],"dev")
    b, c = a[0]
    print(b.shape, c.shape)
    tf_ds = get_dataset(fname, [],"dev")
