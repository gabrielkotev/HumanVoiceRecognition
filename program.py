import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import SignalUtils as su
from os import listdir
from NN import NN
import random
import matplotlib.pyplot as plt

# combine the files to have with noise and without
#su.combine_waves()

paths_mapping = [("D:/dataset/combine/", 1), ("D:/dataset/other/", 0), ("D:/dataset/voice/", 1)]
files_mapping = []
for path_mapping in paths_mapping:
    files_mapping.extend([(path_mapping[0] + file, path_mapping[1]) for file in listdir(path_mapping[0])])
random.shuffle(files_mapping)
test_index = int(0.6 * len(files_mapping))
train_samples = files_mapping[0:test_index]
test_samples = files_mapping[test_index:len(files_mapping)]

nn = NN(tsteps=69, batch_size=32, epochs=1)
nn.create_feed_forward()
for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = nn.prepare_feedforward_data(X)
    print(X.shape)
    for i in range(nn.epochs):
        print('Epoch', i, '/', nn.epochs)
        y = np.ones(X.shape[0]) * sample[1]
        print(X.shape)
        print(y.shape)

        nn.fit(X, y)

scores = []
for sample in test_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = nn.prepare_feedforward_data(X)
    for i in range(nn.epochs):
        print('Epoch', i, '/', nn.epochs)
        y = np.ones(X.shape[0]) * sample[1]
        print(X.shape)
        print(y.shape)

        scores.append(nn.model.evaluate(X, y))

scores = np.asarray(scores, dtype=float)
score = np.mean(scores)
print(score * 100)