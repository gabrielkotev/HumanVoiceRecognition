import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import as_strided
import SignalUtils as su
from os import listdir
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.core import Masking
from keras import optimizers
from keras.callbacks import EarlyStopping

train = [f for f in range(0,100)]
train = np.asarray(train).reshape(100, 1, 1) / 101
test = [f for f in range(1,101)]
test = np.asarray(test).reshape(100, 1, 1) / 101


model = Sequential()
model.add(LSTM(1, input_shape=(1, 1), return_sequences=True, batch_size=25, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(train, test, batch_size=200, epochs=1000, verbose=2, validation_split=0.3)