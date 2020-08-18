#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/17/20 7:38 PM
# @Author  : Zhiliang Wu

# To run the python file in the directory, add the following
'''
import sys;
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend([PATH_OF_mimic3-benchmarks])
'''

import gc
import gzip
import pickle

import numpy as np
from mimic3models.length_of_stay import utils
from mimic3models import common_utils
from mimic3benchmark.readers import LengthOfStayReader
from mimic3models.preprocessing import Discretizer, Normalizer

train_reader = LengthOfStayReader(dataset_dir=PATH_OF_TRAIN,
                                  listfile=PATH_OF_CSV_IN_TRAIN)


discretizer = Discretizer(timestep=1,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')

cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)

normalizer_state =os.path.join(os.path.dirname(__file__), 'los_ts1.0.input_str:previous.start_time:zero.n5e4.normalizer')

normalizer.load_params(normalizer_state)



# save all samples in train/test folder

'''
total_samples = train_reader.get_number_of_examples()

batch_size = 15000
n_loops = int(np.ceil(total_samples / batch_size))


for i in range(n_loops):
    print(i)
    ret = common_utils.read_chunk(train_reader, batch_size)
    Xs = ret["X"]
    ts = ret["t"]
    ys = ret["y"]
    names = ret["name"]

    Xs = utils.preprocess_chunk(Xs, ts, discretizer, normalizer)
    data_list = [Xs, ts, ys, names]

    with gzip.open(f'./temp/data_list_{i}.pickle', 'wb') as f:
        pickle.dump(data_list, f)
        print('Saved!')

    del Xs, ts, ys, names, data_list
    gc.collect()
'''

# save a specific pre-process sample
'''
index = 2925433

ret = train_reader.read_example(index)

# the following preprocess_chunk requires [] structure
Xs = [ret["X"]]
ts = [ret["t"]]
ys = [ret["y"]]
names = [ret["name"]]

Xs = utils.preprocess_chunk(Xs, ts, discretizer, normalizer)
data_list = [Xs, ts, ys, names]

with gzip.open(f'./temp_train/data_list_196.pickle', 'wb') as f:
    pickle.dump(data_list, f)
    print('Saved!')
'''
