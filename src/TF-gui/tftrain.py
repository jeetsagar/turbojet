#!python3

import tensorflow as tf

from tfdata import get_dataset
from tensorflow.keras import layers

import os 

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"



# gpu_devices = tf.config.experimental.list_physical_devices("GPU")
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)


def trainModel(data_in,params_in):
    data_in = data_in.take(2048)
    data_in = data_in.shuffle(24)
    data_in = data_in.batch(1024)



    arch = params_in["Architecture"]
    dropout = params_in["Dropout"]
    lr = params_in["LearningRate"]
    attrs = params_in["Attrs"]
    epochs = params_in["Epochs"]


    if arch=="BaseCNN":
        if params_in["BatchNorm"]:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(50,"relu"),
                layers.Dense(1)
                ])
        else:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Flatten(),
                layers.Dense(50,"relu"),
                layers.Dense(1)
                ])

    elif arch=="CNN-LSTM":
        if params_in["BatchNorm"]:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.BatchNormalization(),
                layers.Reshape((5,10)),
                layers.LSTM(30,return_sequences=False),
                layers.Dense(50,"relu"),
                layers.Dense(1)
                ])
        else:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Reshape((5,10)),
                layers.LSTM(30,return_sequences=False),
                layers.Dense(50,"relu"),
                layers.Dense(1)
                ])

    elif arch=="CNN-2LSTM":
        if params_in["BatchNorm"]:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.BatchNormalization(),
                layers.Reshape((5,10)),
                layers.LSTM(30,return_sequences=True),
                layers.LSTM(30,return_sequences=False),
                layers.Dense(1)
                ])
        else:
            model = tf.keras.Sequential([
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu", input_shape=(1,50, attrs)),
                layers.Dropout(dropout),
                layers.Conv1D(filters=10, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Conv1D(filters=1, kernel_size=5, padding="same", activation="relu"),
                layers.Dropout(dropout),
                layers.Reshape((5,10)),
                layers.LSTM(30,return_sequences=True),
                layers.LSTM(30,return_sequences=False),
                layers.Dense(1)
                ])
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(learning_rate=lr, amsgrad=True))
    filepath = "./checkpoints/Model_in-"+arch+str(attrs)+".h5"

    losses = []

    class CustomModelCheckPoint(tf.keras.callbacks.Callback):
        def __init__(self,**kargs):
            super(CustomModelCheckPoint,self).__init__(**kargs)
            self.epoch_loss = {} # accuracy at given epoch

        def on_epoch_begin(self,epoch, logs={}):
            # Things done on beginning of epoch. 
            return

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_loss[epoch] = logs.get("loss")
            losses.append(self.epoch_loss[epoch])


    if params_in["ResumeTraining"]:
        model.load_weights(filepath)

    checkpoint2 = CustomModelCheckPoint()


    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss',verbos=0,save_best_only=True,save_freq='epoch')
    model.fit(data_in, epochs=epochs,callbacks=[checkpoint,checkpoint2])

    df_loss = pd.DataFrame()
    df_loss["Epochs"] = list(range(1,epochs+1))
    df_loss["Loss"] = losses
    df_loss.to_csv("./losses/lossTrend.csv",index=False)


