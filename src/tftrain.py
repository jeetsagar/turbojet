#!python3

import tensorflow as tf

from tfdata import get_dataset
from tensorflow.keras import layers

fpath = '../../data_set/N-CMAPSS_DS02-006.h5'

tf_ds = get_dataset(fpath, [])
tf_ds = tf_ds.batch(256)

phm_model = tf.keras.Sequential([
    layers.Conv1D(filters=20, kernel_size=9, padding="same", activation="relu", input_shape=(1, 50, 18)),
    layers.Conv1D(filters=20, kernel_size=9, padding="same", activation="relu"),
    layers.Conv1D(filters=1, kernel_size=9, padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dense(50,"relu"),
    layers.Dense(1,"relu")
])

phm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam())

print('\n\nmodel has been compiled\n\n')

phm_model.fit(tf_ds, epochs=10)
