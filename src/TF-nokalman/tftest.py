#!python3

import tensorflow as tf

from tfdata import get_dataset
from tensorflow.keras import layers
import h5py
import os 
import numpy as np 

os.environ["CUDA_VISIBLE_DEVICES"]="0"



gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# fpath = './N-CMAPSS_DS02-006.h5'
fpath = './DS02-KalmanNew.h5'
# fpath = './DS02-KalmanNew.h5'

# exit()
tf_ds = get_dataset(fpath, [],0)
# tf_ds.take(1024)
tf_ds = tf_ds.batch(1024)
# phm_model = tf.keras.Sequential([
#     layers.Conv1D(filters=20, kernel_size=10, padding="same", activation="relu", input_shape=(1,50, 18)),
#     layers.Conv1D(filters=20, kernel_size=10, padding="same", activation="relu"),
#     layers.Conv1D(filters=1, kernel_size=10, padding="same", activation="relu"),
#     layers.Flatten(),
#     layers.Dense(50,"relu"),
#     layers.Dense(1,"relu")
# ])


# phm_model = tf.keras.models.load_model("./Model_inCNNLSTM-Kalman.h5")
phm_model = tf.keras.models.load_model("./Model_inCNN2LSTMs-KalmanMoreFilters.h5")


y_out = phm_model.predict(tf_ds,verbose=1)


print(y_out)


fname = 'y_predtest2LSTMs-KalmanMoreFilters.out'

np.savetxt(fname, np.array(y_out), delimiter=',')


# exit()

y_true = np.loadtxt("./y_true.out")
# y_true = y_true#[:1024]



def rmse(pred,true):
	return np.sqrt(np.mean((pred-true)**2))

def nasafn(pred,true):
	sum_in = 0
	for i in range(len(pred)):
		if pred[i]<true[i]:
			sum_in += np.exp((1/13)*(np.abs(pred[i]-true[i])))
		else:
			sum_in += np.exp((1/10)*(np.abs(pred[i]-true[i])))
	return sum_in

y_out = np.loadtxt(fname)
y_out[y_out<0] = 0


print("RMSE : "+str(rmse(y_out,y_true)))
print("NASAsfn : "+str(nasafn(y_out,y_true)))




# exit()
# fname = './DS02-Kalman.h5'
# with h5py.File(fname, 'r') as hdf:
# 	A_test = np.array(hdf.get('A_test'))
# 	Y_test = np.array(hdf.get('Y_test'))

# unit_array = np.array(A_test[:, 0], dtype=np.int32)
# existing_units = list(np.unique(unit_array))

# y_true = []

# for unit in existing_units:
# 	unit_ind = (unit_array == unit)
# 	y_in = Y_test[unit_ind]
# 	y_true.append(y_in[:-50])

# y_true = [item for sublist in y_true for item in sublist]
# # print(len(y_true))
# y_true = np.array(y_true)
# # exit()

# def rmse(pred,true):
# 	return np.sqrt(np.mean((pred-true)**2))

# def nasafn(pred,true):
# 	sum_in = 0
# 	for i in range(len(pred)):
# 		if pred[i]<true[i]:
# 			sum_in += np.exp((1/13)*(np.abs(pred[i]-true[i])))
# 		else:
# 			sum_in += np.exp((1/10)*(np.abs(pred[i]-true[i])))
# 	return sum_in

# y_out[y_out<0] = 0
# print("RMSE : "+str(rmse(y_out,y_true)))
# print("NASAsfn : "+str(nasafn(y_out,y_true)))

# np.savetxt('y_truetestKalman.out', y_true, delimiter=',')

# print('\n\nmodel has been compiled\n\n')

