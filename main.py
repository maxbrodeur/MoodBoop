import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras import Input, Model, layers
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v1 import Adamax

ARRAY_COLUMNS = 48
DUPL_COEF = 3
NUM_EMOTIONS = 10
NUM_TRAIN_IMG = 28709
data_image_training = pd.read_csv('train.csv')
data_image_testing = pd.read_csv('test.csv')
df = pd.read_csv('label4_newFer.csv')
data_expected_output = df.to_numpy()

def duplicate_matrix(array):
    duplicated_matrix = np.empty([DUPL_COEF, ARRAY_COLUMNS, ARRAY_COLUMNS])
    for i in range(2):
        duplicated_matrix[i] = array
    return duplicated_matrix


def transform_data(pixels):
    length = len(pixels)
    array = np.empty([length, ARRAY_COLUMNS, ARRAY_COLUMNS])
    for i in range(length - 1):
        img = pixels[i].split(" ")
        matrix = np.array(img, 'float32')
        matrix /= 255
        array[i] = np.reshape(matrix, (-1, ARRAY_COLUMNS))
    return array


def vgg16_output(vgg, array):
    length = int(len(array))
    vgg_input = np.empty([length, ARRAY_COLUMNS, ARRAY_COLUMNS, DUPL_COEF])
    for a, b in enumerate(vgg_input):
        b[:, :, 0] = array[a]
        b[:, :, 1] = array[a]
        b[:, :, 2] = array[a]
    prediction = vgg.predict(vgg_input)
    del vgg_input
    output = np.empty([length, 512])
    for i in range(length - 1):
        output[i] = prediction[i]
    return output


def get_train(array):
    return array[0:(NUM_TRAIN_IMG - 1)]


def get_test(array):
    return array[NUM_TRAIN_IMG:]


# get input data for training and tests
input_train_array = transform_data(data_image_training['pixels'])
input_train = get_train(input_train_array)
input_test = transform_data(data_image_testing['pixels'])
a = len(input_test)

# get output data for training and tests
output_train = get_train(data_expected_output)
output_test = get_test(data_expected_output)
b = len(output_test)

# vgg16
vgg16 = VGG16(include_top=False, input_shape=(ARRAY_COLUMNS, ARRAY_COLUMNS, DUPL_COEF), pooling='avg',
              weights='imagenet')

# basic model
model = keras.Sequential(
    [layers.Dense(256, input_shape=(512,), activation='relu'),
     layers.Dense(256, input_shape=(256,), activation='relu'),
     layers.Dropout(0.5),
     layers.Dense(128, input_shape=(256,), activation='relu'),
     layers.Dense(NUM_EMOTIONS, activation='softmax')
     ]
)

# training the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
model.fit(vgg16_output(vgg16, input_train), output_train, batch_size=100, epochs=1000)

# testing the model
model_test_results = model.evaluate(vgg16_output(vgg16, input_test), output_test, batch_size=100)

print("model test result:".format(model_test_results))
# merging both models
inputs = Input(shape=(48, 48, 3))
vg_output = vgg16(inputs)
model_predictions = model(vg_output)
final_model = Model(input=inputs, output=model_predictions)

# testing final model
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

# testing the model
final_model_test_results = final_model.evaluate(input_test, output_test, batch_size=100)

print(" final model test result:".format(model_test_results))
