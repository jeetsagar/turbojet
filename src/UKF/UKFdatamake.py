"""
Utility file to merge multiple h5 unit-wise files into 1 h5 dataset file.
"""
import time

import h5py
import numpy as np

filename = './N-CMAPSS_DS02-006.h5'

t = time.process_time()

with h5py.File(filename, 'r') as hdf:
    # Development set
    X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s

    X_s_test = np.array(hdf.get('X_s_test'))  # X_s

    X_s_var = np.array(hdf.get('X_s_var'))

    X_s_var = list(np.array(X_s_var, dtype='U20'))

X_s = X_s_dev
X_sTest = X_s_test

units = [2, 5, 10, 16, 18, 20, 11, 14, 15]

Ws = []
X_sks = []
X_vks = []
Tks = []
Ys = []
As = []

WsTest = []
X_sksTest = []
X_vksTest = []
TksTest = []
YsTest = []
AsTest = []

# Read each unit h5 file (After kalman) and put in array
for u in units:
    filename = "DS02-KalmanUnit" + str(u) + ".h5"
    with h5py.File(filename, 'r') as hdf:
        if u in [2, 5, 10, 16, 18, 20]:
            W_dev = np.array(hdf.get('W_dev'))  # W
            X_sk_dev = np.array(hdf.get('X_s_dev'))  # X_s
            X_vk_dev = np.array(hdf.get('X_v_dev'))  # X_v
            Tk_dev = np.array(hdf.get('T_dev'))  # T
            Y_dev = np.array(hdf.get('Y_dev'))  # RUL
            A_dev = np.array(hdf.get('A_dev'))  # Auxiliary
        if u in [11, 14, 15]:
            W_test = np.array(hdf.get('W_dev'))  # W
            X_sk_test = np.array(hdf.get('X_s_dev'))  # X_s
            X_vk_test = np.array(hdf.get('X_v_dev'))  # X_v
            Tk_test = np.array(hdf.get('T_dev'))  # T
            Y_test = np.array(hdf.get('Y_dev'))  # RUL
            A_test = np.array(hdf.get('A_dev'))  # Auxiliary

        W_var = np.array(hdf.get('W_var'))
        X_sk_var = np.array(hdf.get('X_s_var'))
        X_vk_var = np.array(hdf.get('X_v_var'))
        Tk_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        W_var = list(np.array(W_var, dtype='U20'))
        X_sk_var = list(np.array(X_sk_var, dtype='U20'))
        X_vk_var = list(np.array(X_vk_var, dtype='U20'))
        Tk_var = list(np.array(Tk_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))
    if u in [2, 5, 10, 16, 18, 20]:
        Ws.append(W_dev)
        X_sks.append(X_sk_dev)
        X_vks.append(X_vk_dev)
        Tks.append(Tk_dev)
        Ys.append(Y_dev)
        As.append(A_dev)
    if u in [11, 14, 15]:
        WsTest.append(W_test)
        X_sksTest.append(X_sk_test)
        X_vksTest.append(X_vk_test)
        TksTest.append(Tk_test)
        YsTest.append(Y_test)
        AsTest.append(A_test)

# Concat the arrays as one combined dataset array
W_main = np.concatenate(Ws, axis=0)
X_sk_main = np.concatenate(X_sks, axis=0)
X_vk_main = np.concatenate(X_vks, axis=0)
Tk_main = np.concatenate(Tks, axis=0)
Y_main = np.concatenate(Ys, axis=0)
A_main = np.concatenate(As, axis=0)

W_mainTest = np.concatenate(WsTest, axis=0)
X_sk_mainTest = np.concatenate(X_sksTest, axis=0)
X_vk_mainTest = np.concatenate(X_vksTest, axis=0)
Tk_mainTest = np.concatenate(TksTest, axis=0)
Y_mainTest = np.concatenate(YsTest, axis=0)
A_mainTest = np.concatenate(AsTest, axis=0)

print("W shape: " + str(W_main.shape))
print("X_sk shape: " + str(X_sk_main.shape))
print("X_s_shape: " + str(X_s.shape))
print("X_vk shape: " + str(X_vk_main.shape))
print("Tk shape: " + str(Tk_main.shape))
print("A shape: " + str(A_main.shape))
print("Y shape: " + str(Y_main.shape))

print("WTest shape: " + str(W_mainTest.shape))
print("X_skTest shape: " + str(X_sk_mainTest.shape))
print("X_sTest_shape: " + str(X_sTest.shape))
print("X_vkTest shape: " + str(X_vk_mainTest.shape))
print("TkTest shape: " + str(Tk_mainTest.shape))
print("ATest shape: " + str(A_mainTest.shape))
print("YTest shape: " + str(Y_mainTest.shape))

# Save in one main h5 file
savefile = 'DS02-Kalman.h5'
f = h5py.File(savefile, 'w')

f["W_dev"] = W_main
f["X_s_dev"] = X_s
f["X_sk_dev"] = X_sk_main
f["X_vk_dev"] = X_vk_main
f["Tk_dev"] = Tk_main
f["A_dev"] = A_main
f["Y_dev"] = Y_main

f["W_test"] = W_mainTest
f["X_s_test"] = X_sTest
f["X_sk_test"] = X_sk_mainTest
f["X_vk_test"] = X_vk_mainTest
f["Tk_test"] = Tk_mainTest
f["A_test"] = A_mainTest
f["Y_test"] = Y_mainTest

f["W_var"] = [x.encode('utf-8') for x in W_var]
f["X_s_var"] = [x.encode('utf-8') for x in X_sk_var]
f["X_sk_var"] = [(x + "k").encode('utf-8') for x in X_sk_var]
f["X_vk_var"] = [(x + "k").encode('utf-8') for x in X_vk_var]
f["Tk_var"] = [(x + "k").encode('utf-8') for x in Tk_var]
f["A_var"] = [x.encode('utf-8') for x in A_var]
f.close()
