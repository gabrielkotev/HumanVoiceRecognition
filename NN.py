from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import optimizers
import numpy as np

class NN:
    def __init__(self, tsteps=1, batch_size=25, epochs=25):
        self.tsteps = tsteps
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_feedforward_data(self, X):
        prepared_data = []
        for x in X:
            mean = 0
            i = 0
            for value in x:
                if i == 7:
                    prepared_data.append(mean / 7)
                    mean = 0
                    i = 0
                mean += value
                i += 1

        print(len(prepared_data))
        new_data = []
        step = int(X[0].size / 7)
        start_point = 0
        end_point = step * 3
        while(end_point < len(prepared_data)):
            new_data.extend([s for s in prepared_data[start_point:end_point]])
            print(len(new_data))
            start_point += step
            end_point += step

        print(len(new_data))

        return np.reshape(new_data, (int(len(new_data) / (step * 3)), step * 3))


    def create_feed_forward(self):
        model = Sequential()
        model.add(Dense(self.tsteps, input_dim=self.tsteps))
        model.add(Dense(50, activation='sigmoid'))
        model.add(Dense(1, activation='softmax'))
        sgd = optimizers.Adam(lr=0.005)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        self.model = model

    def create_lstm(self):
        model = Sequential()
        model.add(LSTM(50,
                       input_shape=(self.tsteps, 1),
                       batch_size=self.batch_size,
                       return_sequences=True,
                       stateful=True))
        model.add(LSTM(50,
                       return_sequences=False,
                       stateful=True))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model

    def fit(self, X, y):

        self.model.fit(X, y, batch_size=self.batch_size, shuffle=True, verbose=2)
        #self.model.reset_states()

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)