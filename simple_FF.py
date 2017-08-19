import numpy as np
from scipy.io import wavfile
import SignalUtils as su
from os import listdir
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

train = [f for f in range(0,100)]
train = np.asarray(train).reshape(100, 1) / 101
test = [f for f in range(1,101)]
test = np.asarray(test).reshape(100, 1) / 101

model = Sequential()
model.add(Dense(1, input_dim=1, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(train, test, batch_size=25, epochs=100, verbose=2, validation_split=0.3)

prediction = np.ones(1) * 3

print(model.predict(prediction) * 101)