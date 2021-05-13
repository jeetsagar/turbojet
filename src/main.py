#!python3

import time
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = '../../data_set/N-CMAPSS_DS01-005.h5'

hdf = h5py.File(filename, 'r')
print(list(hdf.keys()))
hdf.close()

if __name__ == '__main__':
    pass

