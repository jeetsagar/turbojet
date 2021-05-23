#!python3

"""
Training code file
"""

import tensorflow as tf

from tfdata import get_dataset
from tensorflow.keras import layers

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# fpath = './N-CMAPSS_DS02-006.h5'
# Get UKF data
fpath = './DS02-Kalman.h5'

# Prepare dataset
tf_ds = get_dataset(fpath, [], 1)
tf_ds = tf_ds.shuffle(3)

tf_ds = tf_ds.batch(512)

# 1-D CNN (49 attributes with UKF, 35 without UKF)
phm_model = tf.keras.Sequential([
    layers.Conv1D(filters=10, kernel_size=10, padding="same", activation="relu", input_shape=(1, 50, 49)),
    layers.Conv1D(filters=10, kernel_size=10, padding="same", activation="relu"),
    layers.Conv1D(filters=1, kernel_size=10, padding="same", activation="relu"),
    layers.Flatten(),
    layers.Dense(50, "relu"),
    layers.Dense(1)
])

# Compile model
phm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True))

# Uncomment to resume training from previous checkpoint
# phm_model.load_weights("./Model_inKalmanFirst.h5")

filepath = "./Model_inKalmanFirst.h5"
# Save checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbos=0, save_best_only=False,
                                                save_freq='epoch')
# Train
phm_model.fit(tf_ds, epochs=100, callbacks=[checkpoint])

seq_model = tf.keras.Sequential([
    layers.Conv1D(filters=10, kernel_size=10, padding="same", activation="relu", input_shape=(1, 50, 49)),
    layers.Conv1D(filters=10, kernel_size=10, padding="same", activation="relu"),
    layers.Conv1D(filters=1, kernel_size=10, padding="same", activation="relu"),
    layers.Flatten(),
    layers.LSTM(50),
    layers.Dense(1)
])

# Compile model
seq_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True))

# Train
seq_model.fit(tf_ds, epochs=100, callbacks=[checkpoint])
