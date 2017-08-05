from keras.models import Sequential
from keras.layers import Dense, LSTM

class NN:
    def __init__(self, tsteps=1, batch_size=25, epochs=25):
        self.tsteps = tsteps
        self.batch_size = batch_size
        self.epochs = epochs

    def create_feed_forward(self):
        model = Sequential()
        model.add(Dense(82, input_dim=self.tsteps))
        model.add(Dense(120))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='rmsprop')
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
        model.compile(loss='mse', optimizer='rmsprop')
        self.model = model

    def fit(self, X, y):

        self.model.fit(X, y, shuffle=False)
        #self.model.reset_states()

    def predict(self, X):
        return self.model.predict(X, batch_size=self.batch_size)