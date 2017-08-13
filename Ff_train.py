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

batch_size=32
look_back = 5
epochs = 200
model_file = "d:/dataset/model.h5"

callback = [EarlyStopping(monitor='val_loss', patience=5, mode='auto')]

model = Sequential()
model.add(Dense(100, input_dim=look_back*23))
model.add(Dense(60, activation='elu'))
model.add(Dense(60, activation='softsign'))
model.add(Dense(120, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

try:
    model.load_weights(model_file)
except:
    print('could not load the saved model file')

for sample in train_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = nn.prepare_feedforward_data(X, look_back=look_back)
    y = np.ones(X.shape[0]) * sample[1]
    #model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2, callbacks=callback)
    for i in range(batch_size):
        model.train_on_batch(X, y)

model.save_weights(model_file)

    #train_sample = random.choice(test_samples)
   # X_test = su.spectrogram_from_file(filename=train_sample[0], max_freq=8000)
   # if X is None:
    #    continue;
   # X_test = nn.prepare_feedforward_data(X_test, look_back=look_back)
    #y_test = np.ones(X_test.shape[0]) * sample[1]
    #print("prediction after batch train ", nn.model.predict(X_test, batch_size=1, verbose=2))
    #print('evaluation after batch: ', nn.evaluate(X, y))
    #print('evaluation of test after batch: ', nn.evaluate(X_test, y_test))

train_sample = random.choice(test_samples)
X_test = su.spectrogram_from_file(filename=train_sample[0], max_freq=8000)
X_test = su.prepare_feedforward_data(X_test, look_back = 5)
y_test = np.ones(X_test.shape[0]) * sample[1]
weights = nn.model.get_weights()
predict_nn = NN(tsteps=look_back * 23, batch_size=1, epochs=1)
predict_nn.create_feed_forward()
predict_nn.model.set_weights(weights)
print("prediction ", nn.model.predict(X_test, batch_size=1, verbose=2))

scores = []
for sample in test_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = nn.prepare_feedforward_data(X, look_back = 5)
    for i in range(nn.epochs):
        y = np.ones(X.shape[0]) * sample[1]

        scores.append(nn.model.evaluate(X, y))

scores = np.asarray(scores, dtype=float)
score = np.mean(scores)
print(score * 100)