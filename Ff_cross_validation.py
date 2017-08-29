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
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
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

def build_logistic_regression_model():
    model = Sequential()
    model.add(Dense(120, input_dim=look_back*161))
    model.add(Dense(60, activation='sigmoid'))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dense(120, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

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
epochs = 200
model_file = "d:/dataset/simple_model.h5"

#model = Sequential()
#model.add(Dense(120, input_dim=look_back*161))
#model.add(Dense(60, activation='sigmoid'))
#model.add(Dense(30, activation='sigmoid'))
#model.add(Dense(120, activation='sigmoid'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#callback = [EarlyStopping(monitor='val_loss', patience=10, mode='auto')]
#fit_history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.3, callbacks=callback)

#model.save_weights(model_file)

model = KerasClassifier(build_fn=build_logistic_regression_model, epochs=epochs)
cv = KFold(5, shuffle=True)
scores = cross_val_score(model, X, y, cv=cv)

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