# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 22:40:25 2017

@author: Gabriel
"""

from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, Flatten
import numpy as np

model.summary()

input = np.ones(161 * 5).reshape(1,161, 5)

model = Sequential()
model.add(Conv1D(16, 2, input_shape=(161, 5)))
model.add(MaxPool1D())
model.add(Conv1D(16, 2))
model.add(MaxPool1D())
model.add(Flatten())
model.compile('adam', 'mean_squared_error')

prediction = model.predict(input, batch_size=1)

prediction.shape