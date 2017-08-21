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

trains_samples, test_samples = su.get_samples()

batch_size=32
look_back = 5
epochs = 200
model_file = "d:/dataset/arrange_model.h5"

callback = [EarlyStopping(monitor='val_loss', patience=5, mode='auto')]

model = Sequential()
model.add(Dense(100, input_dim=look_back*23))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(60, activation='sigmoid'))
model.add(Dense(120, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.load_weights(model_file)

predictions = []
for sample in test_samples:
    X = su.spectrogram_from_file(filename=sample[0], max_freq=8000)
    if X is None:
        continue;
    X = su.prepare_feedforward_data(X, look_back=look_back)[0].reshape(1, look_back*23)
    y = sample[1]
    prediction = model.predict(X, batch_size=1, verbose=2)
    if prediction > 0.5:
        prediction = 1
    else:
        prediction = 0
    predictions.append(prediction == y)

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