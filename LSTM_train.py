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

batch_size = 1
look_back = 5
epochs = 200

train_samples, test_samples = su.get_samples()

callback = [EarlyStopping(monitor='val_loss', patience=5, mode='auto')]

model = Sequential()
model.add(LSTM(1, batch_input_shape=(None, 1, 161), return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, return_sequences=True, stateful=False, activation='sigmoid'))
model.add(LSTM(1, stateful=False, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

data = []
for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    #X = su.prepare_lstm_data(X, batch_size=batch_size)
    y = np.ones(X.shape[0]) * sample[1]
    for x in X:
        data.extend([xx for xx in x])
        print(len(data))
    #model.reset_states()

print(len(data))
data = np.asarray(data, dtype=float)
print(data.shape)
data = data.reshape(int(data.shape[0]) / 161, 161)
X = su.prepare_lstm_data(data, batch_size=batch_size)
y = np.ones(X.shape[2]) * sample[1]
model.fit(X, y, batch_size=batch_size, shuffle=False, epochs=epochs, verbose=2, validation_split=0.2, callbacks=callback)