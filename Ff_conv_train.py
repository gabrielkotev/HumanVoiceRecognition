import numpy as np
from scipy.io import wavfile
import SignalUtils as su
from os import listdir
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPool1D, Flatten, Dropout
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import config
import matplotlib.pyplot as plt

paths_mapping = config.get_mapping_paths()
files_mapping = []
for path_mapping in paths_mapping:
    files_mapping.extend([(path_mapping[0] + file, path_mapping[1]) for file in listdir(path_mapping[0])])
random.shuffle(files_mapping)
test_index = int(0.6 * len(files_mapping))
train_samples = files_mapping[0:test_index]
test_samples = files_mapping[test_index:len(files_mapping)]

look_back = 5

sc = MinMaxScaler()

y=[]
data=[]
for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = su.prepare_simple_feedforward_data(X, look_back=look_back)
    for x in X:
        data.extend([xx for xx in x])

    y.extend([y for y in np.ones(X.shape[0], dtype=float) * sample[1]])

y = np.asarray(y, dtype=float)
y = y.reshape(y.shape[0], 1)
y = y.astype('float32')

X = np.asarray(data, dtype=float)
# reshape for fit transform
X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161 * look_back)
print(X.min(), x.max())
X = sc.fit_transform(X)

X = X.flatten()
X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161, look_back)
X = X.astype('float32')

batch_size=16
epochs = 20
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

#callback = [EarlyStopping(monitor='loss', patience=10, mode='auto')]
fit_history = model.fit(X, y, batch_size=batch_size, epochs=epochs)

model.save_weights(model_file)


#evaluation of the model
y_test=[]
data=[]
for sample in test_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = su.prepare_simple_feedforward_data(X, look_back=look_back)
    for x in X:
        data.extend([xx for xx in x])

    y_test.extend([y for y in np.ones(X.shape[0], dtype=float) * sample[1]])

y_test = np.asarray(y_test, dtype=float)
y_test = y_test.reshape(y_test.shape[0], 1)

X_test = np.asarray(data, dtype=float)
X_test = X_test.reshape(int(X_test.shape[0]) / (161 *  look_back), 161 * look_back)
X_test = sc.transform(X_test)
X_test = X_test.flatten()
X_test = X_test.reshape(int(X_test.shape[0]) / (161 *  look_back), 161,  look_back)

print(X_test.min())
print(X_test.max())

evaluation_history = model.evaluate(X_test, y_test)
print(evaluation_history[1])
plt.plot(fit_history.history['acc'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.show()