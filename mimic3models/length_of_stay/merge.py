import gc
import gzip
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

path = Path('./train.h5')

with h5py.File(path, "w") as f:
    arr = f.create_dataset('X', (0, 2803, 76), maxshape=(None, 2803, 76), compression="gzip")
    timestep = f.create_dataset('timestep', (0, ), maxshape=(None, ), compression="gzip")
    target = f.create_dataset('los', (0, ), maxshape=(None, ), compression="gzip")

    for i in range(197):
        print(f'current in chunk: {i}.')
        with gzip.open(f'./temp_train/data_list_{i}.pickle', 'rb') as f:
                data = pickle.load(f, encoding='latin1')

        Xs, ts, ys, names = data

        length_of_chunk = len(Xs)

        X_arr = np.zeros((length_of_chunk, 2803, 76))
        for r_index, v in enumerate(Xs):
            length = v.shape[0]
            X_arr[r_index, -length:, :] = v
        t_arr = np.array(ts)
        y_arr = np.array(ys)

        arr.resize(arr.shape[0]+length_of_chunk, axis=0)
        timestep.resize(arr.shape[0]+length_of_chunk, axis=0)
        target.resize(arr.shape[0]+length_of_chunk, axis=0)

        arr[-length_of_chunk:, :, :] = X_arr
        timestep[-length_of_chunk:] = t_arr
        target[-length_of_chunk:] = y_arr

        del X_arr, t_arr, y_arr, data
        gc.collect()


path = Path('./test.h5')

with h5py.File(path, "w") as f:
    arr = f.create_dataset('X', (0, 2803, 76), maxshape=(None, 2803, 76), compression="gzip")
    timestep = f.create_dataset('timestep', (0, ), maxshape=(None, ), compression="gzip")
    target = f.create_dataset('los', (0, ), maxshape=(None, ), compression="gzip")

    for i in range(37):
        print(f'current in chunk: {i}.')
        with gzip.open(f'./temp_test/data_list_{i}.pickle', 'rb') as f:
                data = pickle.load(f, encoding='latin1')

        Xs, ts, ys, names = data

        length_of_chunk = len(Xs)

        # from test, the longest is actually 1993
        X_arr = np.zeros((length_of_chunk, 2803, 76))
        for r_index, v in enumerate(Xs):
            length = v.shape[0]
            X_arr[r_index, -length:, :] = v
        t_arr = np.array(ts)
        y_arr = np.array(ys)

        arr.resize(arr.shape[0]+length_of_chunk, axis=0)
        timestep.resize(arr.shape[0]+length_of_chunk, axis=0)
        target.resize(arr.shape[0]+length_of_chunk, axis=0)

        arr[-length_of_chunk:, :, :] = X_arr
        timestep[-length_of_chunk:] = t_arr
        target[-length_of_chunk:] = y_arr

        del X_arr, t_arr, y_arr, data
        gc.collect()
