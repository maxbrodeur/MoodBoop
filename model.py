import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import image
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, LSTM, Dropout, Dense
from tensorflow.python.keras.models import Sequential

DUPL_COEF = 3
data_image_training = pd.read_csv('train.csv')
batch_size = 50
image_shape = 150


def duplicate_matrix(array):
    out = np.empty([len(array), len(array[0]), len(array[0][0]), DUPL_COEF])
    for a, b in enumerate(out):
        b[:, :, 0] = array[a]
        b[:, :, 1] = array[a]
        b[:, :, 2] = array[a]
    return out


def access_image(name):
    # change directory
    pic = image.imread(name)
    pic /= 255
    return pic


def get_images_array(array):
    out = np.empty([len(array), image_shape, image_shape])
    for j in range(len(array)):
        out[j] = access_image(array[j])
    return out


# get the input image array
images = data_image_training['image']
images_array = get_images_array(images)
train_input = duplicate_matrix(images_array)

# get an array of output emotion
valence = data_image_training['valence']
arousal = data_image_training['arousal']


model = Sequential()
model.add(Conv2D(input_shape=(batch_size, image_shape, image_shape, 3), activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(BatchNormalization)
model.add(LSTM(64))
model.add(BatchNormalization)
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization)
model.add(Dropout(0.5))
model_valence = model.add(Dense(1, activation='tanh'))
model_arousal = model.add(Dense(1, activation='sigmoid'))


# training the model
model_valence.compile(optimizer='adam', loss='MSE', metrics=["accuracy"])
model_valence.fit(train_input, valence, batch_size=batch_size, epochs=100)
model_arousal.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
model_arousal.fit(train_input, arousal, batch_size=batch_size, epochs=100)


