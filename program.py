import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import SignalUtils as su
from os import listdir
from NN import NN
import random
import matplotlib.pyplot as plt

paths_mapping = [("D:/dataset/combine/", 1), ("D:/dataset/other/", 0), ("D:/dataset/voice/", 1)]
files_mapping = []
for path_mapping in paths_mapping:
    files_mapping.extend([(path_mapping[0] + file, path_mapping[1]) for file in listdir(path_mapping[0])])
random.shuffle(files_mapping)
test_index = int(0.6 * len(files_mapping))
train_samples = files_mapping[0:test_index]
test_samples = files_mapping[test_index:len(files_mapping)]

nn = NN(tsteps=161, batch_size=32, epochs=1)
nn.create_feed_forward()
for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    for i in range(nn.epochs):
        print('Epoch', i, '/', nn.epochs)
        y = np.ones(X.shape[0]) * sample[1]
        print(X.shape)
        print(y.shape)

        nn.fit(X, y)

