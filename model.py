import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import image
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, LSTM, Dropout, Dense, \
    AveragePooling1D
from tensorflow.python.keras.models import Sequential

batch_size = 100
x_size = 200
y_size = 300

model = Sequential()
model.add(BatchNormalization(input_shape=(y_size, x_size)))
model.add(AveragePooling1D())
model.add(LSTM(255))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model_valence = model.add(Dense(1, activation='tanh'))
model_arousal = model.add(Dense(1, activation='tanh'))


# training the model
def train_model(train_input, valence, arousal):
    model_valence.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
    model_valence.fit(train_input, valence, batch_size=batch_size, epochs=100)
    model_arousal.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
    model_arousal.fit(train_input, arousal, batch_size=batch_size, epochs=100)
