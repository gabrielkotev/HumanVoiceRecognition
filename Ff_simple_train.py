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
from sklearn.preprocessing import MinMaxScaler
import config
import matplotlib.pyplot as plt

def build_logistic_regression_model(look_back):
    model = Sequential()
    model.add(Dense(120, input_dim=look_back*161))
    model.add(Dense(60, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(120, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
        print(len(data))

    y.extend([y for y in np.ones(X.shape[0], dtype=float) * sample[1]])

y = np.asarray(y, dtype=float)
y = y.reshape(y.shape[0], 1)

X = np.asarray(data, dtype=float)
X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161 *  look_back)
X = sc.fit_transform(X)

batch_size=32
epochs = 20
model_file = "d:/dataset/simple_model.h5"

model = Sequential()
model.add(Dense(120, input_dim=look_back*161))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#callback = [EarlyStopping(monitor='loss', patience=10, mode='auto')]
fit_history = model.fit(X, y, batch_size=batch_size, epochs=epochs)

model.save_weights(model_file)


#evaluation of the model
y=[]
data=[]
for sample in test_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = su.prepare_simple_feedforward_data(X, look_back=look_back)
    for x in X:
        data.extend([xx for xx in x])
        print(len(data))

    y.extend([y for y in np.ones(X.shape[0], dtype=float) * sample[1]])

y = np.asarray(y, dtype=float)
y = y.reshape(y.shape[0], 1)

X = np.asarray(data, dtype=float)
X = X.reshape(int(X.shape[0]) / (161 *  look_back), 161 *  look_back)
X = sc.transform(X)

print(X.min())
print(X.max())

evaluation_history = model.evaluate(X, y)
print(evaluation_history[1])
plt.plot(fit_history.history['acc'])
plt.plot(fit_history.history['val_acc'])
plt.legend('Training', 'Validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.show()