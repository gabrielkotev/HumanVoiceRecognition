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

# combine the files to have with noise and without
#su.combine_waves()

batch_size = 32
look_back = 5
epochs = 200

train_samples, test_samples = su.get_samples()

callback = [EarlyStopping(monitor='val_loss', patience=5, mode='auto')]

model = Sequential()
model.add(LSTM(160, input_shape=(1, 161), return_sequences=True, batch_size=batch_size, stateful=False, activation='elu'))
model.add(LSTM(80, return_sequences=True, batch_size=batch_size, stateful=False, activation='elu'))
model.add(LSTM(160, batch_size=batch_size, stateful=False, activation='elu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = su.prepare_lstm_data(X, batch_size=batch_size)
    y = np.ones(X.shape[0]) * sample[1]
    model.fit(X, y, batch_size=batch_size, shuffle=False, epochs=epochs, verbose=2, validation_split=0.2, callbacks=callback)
    #model.reset_states()