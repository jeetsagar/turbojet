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

    def __init__(self, filepath, units, augmentPhy, mode):
        self.window = 50

        if mode==1:
            suffix = "dev"
        else:
            suffix = "test"

        if augmentPhy:
            with h5py.File(filepath,'r') as hdf:
                W_in = np.array(hdf.get("W_"+suffix))
                X_s_in = np.array(hdf.get("X_s_"+suffix))
                X_sk_in = np.array(hdf.get("X_sk_"+suffix))
                X_vk_in = np.array(hdf.get("X_vk_"+suffix))
                Tk_in = np.array(hdf.get("Tk_"+suffix))
                A_in = np.array(hdf.get("A_"+suffix))
                Y_in = np.array(hdf.get("Y_"+suffix))
        else:
            with h5py.File(filepath,'r') as hdf:
                W_in = np.array(hdf.get("W_"+suffix))
                X_s_in = np.array(hdf.get("X_s_"+suffix))
                X_v_in = np.array(hdf.get("X_v_"+suffix))
                T_in = np.array(hdf.get("T_"+suffix))
                A_in = np.array(hdf.get("A_"+suffix))
                Y_in = np.array(hdf.get("Y_"+suffix))
                
        unit_array = np.array(A_in[:, 0], dtype=np.int32)

        existing_units = list(np.unique(unit_array))

        self.units = existing_units 
        self.num_units = len(self.units)

        if augmentPhy:
            data_in = np.concatenate((W_in,X_s_in,X_sk_in,X_vk_in,Tk_in), axis=1)
        else:
            data_in = np.concatenate((W_in,X_s_in,X_v_in,T_in[:, [-1,-2,-4]]), axis=1)

        data_in = normalize_data(data_in)

        self.data_list = []
        self.target_list = []
        self.length_list = []
        self.total_length = 0

        for unit in self.units:
            unit_ind = (unit_array == unit)

            unit_data = data_in[unit_ind]
            unit_target = Y_in[unit_ind]
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


def generate_data(filepath,units,augmentPhy,mode):
    ds = DataProvider(filepath, units,augmentPhy,mode)
    for i in range(ds.total_length):
        data, value = ds[i]
        yield data, value


def get_dataset(filepath,units,augmentPhy,mode):
    # return tf.data.Dataset.from_generator(generate_data, args=[filepath, units],output_signature=(tf.TensorSpec(shape=(1, 50, 18), dtype=tf.float32),tf.TensorSpec(shape=(1, 1), dtype=tf.float32)))
    if augmentPhy:
        return tf.data.Dataset.from_generator(generate_data, args=[filepath,units,augmentPhy,mode], output_types=(tf.float32,tf.float32), output_shapes=([1,50,49],[1,]))
    else:
        return tf.data.Dataset.from_generator(generate_data, args=[filepath,units,augmentPhy,mode], output_types=(tf.float32,tf.float32), output_shapes=([1,50,35],[1,1]))
