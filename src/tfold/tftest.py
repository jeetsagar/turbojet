#!python3

'''
Testing code file

'''
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

#Read the dataset
fpath = './N-CMAPSS_DS02-006.h5'

tf_ds = get_dataset(fpath, [],0)
tf_ds = tf_ds.batch(1024)

#Load the model (h5 not present in repo)
phm_model = tf.keras.models.load_model("./Model_ink5.h5")

#Predict
y_out = phm_model.predict(tf_ds,verbose=1)

#Get groundtruth
fname = './N-CMAPSS_DS02-006.h5'
with h5py.File(fname, 'r') as hdf:
	A_test = np.array(hdf.get('A_test'))
	Y_test = np.array(hdf.get('Y_test'))

unit_array = np.array(A_test[:, 0], dtype=np.int32)
existing_units = list(np.unique(unit_array))

y_true = []

for unit in existing_units:
	unit_ind = (unit_array == unit)
	y_in = Y_test[unit_ind]
	y_true.append(y_in[:-50])

y_true = [item for sublist in y_true for item in sublist]
y_true = np.array(y_true)


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

#Evaluate RMSE and scoring function based on predictions and groundtruth given abobe
y_out[y_out<0] = 0
print("RMSE : "+str(rmse(y_out,y_true)))
print("NASAsfn : "+str(nasafn(y_out,y_true)))

#Save predictions and groundtruths for further use
np.savetxt('y_truetestk5-moreEpochs.out', y_true, delimiter=',')
np.savetxt('y_predtestk5-moreEpochs.out', np.array(y_out), delimiter=',')

# print('\n\nmodel has been compiled\n\n')

