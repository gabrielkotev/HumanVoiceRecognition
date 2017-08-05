import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import SignalUtils as su
from os import listdir
from NN import NN
import matplotlib.pyplot as plt

paths_mapping = [("D:/dataset/combine/", 1), ("D:/dataset/other/", 0), ("D:/dataset/voice", 1)]

nn = NN(tsteps=161, batch_size=32, epochs=1)
nn.create_feed_forward()
for path_mapping in paths_mapping:
    for path in listdir(path_mapping[0]):
        X = su.spectrogram_from_file(filename=path_mapping[0] + path, max_freq=8000)
        for i in range(nn.epochs):
            print('Epoch', i, '/', nn.epochs)
            y = np.ones(X.shape[0]) * path_mapping[1]
            print(X.shape)
            print(y.shape)

            nn.fit(X, y)

