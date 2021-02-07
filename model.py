from tensorflow.python.keras.layers import BatchNormalization, Activation, LSTM, Dropout, Dense, AveragePooling1D
from tensorflow.python.keras.models import Sequential


class model:

    # training the model
    def train_model(self, train_input, arousal, valence):
        batch_size = 100
        x_size = 300
        y_size = 200

        modV = Sequential()
        modV.add(BatchNormalization(input_shape=(y_size, x_size)))
        modV.add(AveragePooling1D())
        modV.add(LSTM(255))
        modV.add(Dense(256))
        modV.add(Activation('relu'))
        modV.add(BatchNormalization())
        modV.add(Dropout(0.5))
        modV.add(Dense(256))
        modV.add(Activation('relu'))
        modV.add(BatchNormalization())
        modV.add(Dropout(0.5))
        modV.add(Dense(1, activation='tanh'))

        modA = Sequential()
        modA.add(BatchNormalization(input_shape=(y_size, x_size)))
        modA.add(AveragePooling1D())
        modA.add(LSTM(255))
        modA.add(Dense(256))
        modA.add(Activation('relu'))
        modA.add(BatchNormalization())
        modA.add(Dropout(0.5))
        modA.add(Dense(256))
        modA.add(Activation('relu'))
        modA.add(BatchNormalization())
        modA.add(Dropout(0.5))
        modA.add(Dense(1, activation='tanh'))

        # training the model
        modV.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
        modV.fit(train_input, valence, batch_size=250, epochs=100)

        modA.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
        modA.fit(train_input, arousal, batch_size=250, epochs=100)
