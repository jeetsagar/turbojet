"""
This file applies Unscented Kalman filter (UKF) to each unit serpately and saves multiple h5 files
"""
import time

import filterpy
import h5py
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from pandas import DataFrame
from sklearn.neural_network import MLPRegressor

# Set-up - Define file location Read dataset
filename = './N-CMAPSS_DS02-006.h5'

# Time tracking, Operation time (min):  0.003
t = time.process_time()

# Load data
with h5py.File(filename, 'r') as hdf:
    # Development set
    W_dev = np.array(hdf.get('W_dev'))  # W
    X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
    X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
    T_dev = np.array(hdf.get('T_dev'))  # T
    Y_dev = np.array(hdf.get('Y_dev'))  # RUL
    A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

    # Test set
    W_test = np.array(hdf.get('W_test'))  # W
    X_s_test = np.array(hdf.get('X_s_test'))  # X_s
    X_v_test = np.array(hdf.get('X_v_test'))  # X_v
    T_test = np.array(hdf.get('T_test'))  # T
    Y_test = np.array(hdf.get('Y_test'))  # RUL
    A_test = np.array(hdf.get('A_test'))  # Auxiliary

    # Varnams
    W_var = np.array(hdf.get('W_var'))
    X_s_var = np.array(hdf.get('X_s_var'))
    X_v_var = np.array(hdf.get('X_v_var'))
    T_var = np.array(hdf.get('T_var'))
    A_var = np.array(hdf.get('A_var'))

    # from np.array to list dtype U4/U5
    W_var = list(np.array(W_var, dtype='U20'))
    X_s_var = list(np.array(X_s_var, dtype='U20'))
    X_v_var = list(np.array(X_v_var, dtype='U20'))
    T_var = list(np.array(T_var, dtype='U20'))
    A_var = list(np.array(A_var, dtype='U20'))

W = np.concatenate((W_dev, W_test), axis=0)
X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
T = np.concatenate((T_dev, T_test), axis=0)
Y = np.concatenate((Y_dev, Y_test), axis=0)
A = np.concatenate((A_dev, A_test), axis=0)

print('')
print("Operation time (min): ", (time.process_time() - t) / 60)
print('')
print("W shape: " + str(W.shape))
print("X_s shape: " + str(X_s.shape))
print("X_v shape: " + str(X_v.shape))
print("T shape: " + str(T.shape))
print("A shape: " + str(A.shape))

main_arr = np.concatenate((W, X_s, X_v, T[:, [-1, -2, -4]], A), axis=1)
colvars = W_var + X_s_var + X_v_var + [T_var[-1], T_var[-2], T_var[-4]] + A_var

units = [2, 5, 10, 16, 18, 20, 11, 14, 15]

df_in = DataFrame(data=main_arr, columns=colvars)

# Process each unit
for u in units:
    print("---------Unit :" + str(u) + "-------------")

    df = df_in[df_in["unit"] == u]

    W_in = np.array(df[W_var][1:][:])
    X_in = np.array(df[X_s_var + X_v_var][:-1][:])  # prepare system model approximation data
    theta_in = np.array(df[[T_var[-1], T_var[-2], T_var[-4]]][1:][:])

    X_train = np.concatenate((W_in, X_in, theta_in), axis=1)
    # print(np.shape(X_train))
    Y_train = np.array(df[X_s_var + X_v_var][1:][:])
    # print(np.shape(Y_train))

    df_Y = DataFrame(data=np.concatenate((Y, A), axis=1), columns=["RUL"] + A_var)
    df_Y = df_Y[df_Y["unit"] == u]["RUL"]
    Y_rul = np.array(df_Y)

    samples = 10000

    inds = list(range(0, len(Y_train), int(len(Y_train) / samples)))

    X_in = X_train[inds][:]
    Y_in = Y_train[inds][:]

    regr = MLPRegressor(hidden_layer_sizes=(70, 50), activation='relu', batch_size=256, random_state=1,
                        max_iter=500).fit(X_in, Y_in)  # learn system model for UKF


    def fx(x, dt):  # Define UKF utility transition functions
        dt = 1
        x_pred = regr.predict(x.reshape(1, -1))
        w_arr = list(x[:4])
        xs = list(x_pred[0])
        ts = list(x[-3:])
        return np.asarray(w_arr + xs + ts)


    def hx(x):
        return x


    points = filterpy.kalman.MerweScaledSigmaPoints(35, alpha=.1, beta=2., kappa=-1)

    kf = UnscentedKalmanFilter(dim_x=35, dim_z=35, dt=1, fx=fx, hx=hx,
                               points=points)  # use ukf for 35 tracking attributes

    kf.x = np.array(df.iloc[0][:-4])
    kf.P *= 0.2
    z_std = 0.01
    kf.R = np.diag([z_std ** 2] * 35)  # UKF hyperparameters
    kf.Q = np.diag([z_std ** 2] * 35)

    X_kalman = []

    for i in range(len(df)):  # DO UKF prediction and update
        print(i)
        z = np.array(df.iloc[i][:-4])
        kf.predict()
        kf.update(z)
        X_kalman.append(np.concatenate((df.iloc[i][:4], np.array(kf.x)[4:], df.iloc[i][-4:])))

    savefile = "DS02-KalmanUnit" + str(u) + ".h5"

    f = h5py.File(savefile, 'w')

    X_kalman = np.array(X_kalman)

    f["W_dev"] = (X_kalman[:, range(4)])  # Save UKF results unit-wise in one h5 file
    f["X_s_dev"] = (X_kalman[:, range(4, 4 + len(X_s_var))])
    f["X_v_dev"] = (X_kalman[:, range(len(X_s_var) + 4, len(X_s_var) + 4 + len(X_v_var))])
    f["T_dev"] = (X_kalman[:, range(len(X_s_var) + 4 + len(X_v_var), len(X_s_var) + 4 + len(X_v_var) + 3)])
    f["A_dev"] = (X_kalman[:, [-4, -3, -2, -1]])
    f["Y_dev"] = Y_rul

    f["W_var"] = [x.encode('utf-8') for x in W_var]
    f["X_s_var"] = [x.encode('utf-8') for x in X_s_var]
    f["X_v_var"] = [x.encode('utf-8') for x in X_v_var]
    f["T_var"] = [x.encode('utf-8') for x in T_var]
    f["A_var"] = [x.encode('utf-8') for x in A_var]
    f.close()
