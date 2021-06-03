#!python3

import tensorflow as tf

from tfdata import get_dataset
from tensorflow.keras import layers

import os 

os.environ["CUDA_VISIBLE_DEVICES"]="0"



gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# fpath = './N-CMAPSS_DS02-006.h5'
fpath = './DS02-KalmanNew.h5'

tf_ds = get_dataset(fpath, [], 1)
# tf_ds = tf_ds.take(1000)
tf_ds = tf_ds.shuffle(7)

tf_ds = tf_ds.batch(1024)


filepath = "./Model_inCNN-2LSTMs-Kalman.h5"
# layers.Conv1D(filters=50, kernel_size=10, padding="same", activation="relu", input_shape=(1,50, 18)),

phm_model = tf.keras.Sequential([
    layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, 35)),
    layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
    layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
    layers.Reshape((5,10)),
    layers.LSTM(30,return_sequences=True),
    layers.LSTM(30,return_sequences=False),
    layers.Dense(1)
])



phm_model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=0.0001, amsgrad=True))
# phm_model.load_weights(filepath)
print(phm_model.summary())

# print('\n\nmodel has been compiled\n\n')

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss',verbos=0,save_best_only=False,save_freq='epoch')

phm_model.fit(tf_ds, epochs=100,callbacks=[checkpoint])
