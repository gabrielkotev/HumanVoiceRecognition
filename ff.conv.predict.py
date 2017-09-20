import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import struct
import SignalUtils as su
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPool1D, Flatten, Dropout
        
# constants
CHUNK = 16000             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 16000                 # samples per second
look_back = 5

sc = MinMaxScaler()
sc.fit_transform([[-32.22741,1], [10.3154, 1]])

batch_size=16
model_file = "d:/dataset/conv_model.h5"

model = Sequential()
model.add(Conv1D(16, 2, input_shape=(161, 5)))
model.add(MaxPool1D())
model.add(Conv1D(16, 2))
model.add(MaxPool1D())
model.add(Flatten())
model.add(Dense(256, activation='sigmoid', input_dim=look_back*161))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(model_file)

p = pyaudio.PyAudio()

attempts = 60
for i in range(attempts):
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )
    
    # binary data
    data = stream.read(CHUNK)  
    
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
        
    # create np array and offset by 128
    data_np = np.array(data_int, dtype='b')[::2] + 128
    
    X = su.spectrogram_from_data(data_np, RATE, max_freq=8000)
    
    data=[]
    X = su.prepare_simple_feedforward_data(X, look_back=look_back)
    for x in X:
        data.extend([xx for xx in x])
        
    X = np.asarray(data, dtype=float)
    # reshape for fit transform
    X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161 * look_back)
    X = sc.fit_transform(X)
    
    X = X.flatten()
    X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161, look_back)
    X = X.astype('float32')
    
    y = model.predict(X, batch_size=batch_size)
    y = (y > 0.85)
    for index in range(y.shape[0]):
        if y[index-1] and y[index] and y[index+1]:
            print('has human voice')

#stream.write(data)

plt.plot(data_np)
plt.show()