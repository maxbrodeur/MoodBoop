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

data_image_training = pd.read_csv('train.csv')
batch_size = 100
x_size = 200
y_size = 300


def access_image(name):
    # change directory
    pic = image.imread(name)
    pic /= 255
    return pic

# TODO
def get_images_array(array):
    out = np.empty([len(array), x_size, y_size])
    for j in range(len(array)):
        out[j] = access_image(array[j])
    return out

# TODO
def get_emotion_array():

    # get the input image array
    images = data_image_training['image']
    train_input = get_images_array(images)

    
    # get an array of output emotion
    valence = data_image_training['valence']
    arousal = data_image_training['arousal']


    model = Sequential()
    model.add(BatchNormalization(input_shape=(x_size, y_size)))
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
    model_valence.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
    model_valence.fit(train_input, valence, batch_size=batch_size, epochs=100)
    model_arousal.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
    model_arousal.fit(train_input, arousal, batch_size=batch_size, epochs=100)