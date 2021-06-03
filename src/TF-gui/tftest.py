#!python3

import os

import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def InferModel(data_in, arch_in, mode):
    data_in = data_in.batch(2048)
    model = tf.keras.models.load_model("./checkpoints/Model_in-" + arch_in + ".h5")

    y_out = model.predict(data_in, verbose=1)

    if mode == 1:
        fname = 'y_predtrain.out'
        np.savetxt("./output/" + fname, np.array(y_out), delimiter=',')
    else:
        fname = 'y_predtest.out'
        np.savetxt("./output/" + fname, np.array(y_out), delimiter=',')
