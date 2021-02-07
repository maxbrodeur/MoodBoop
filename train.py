import os

import cv2
import numpy as np

# salut karl
from PIL import Image

from MoodBoop.model import model

path = "/Users/maxbrodeur/Documents/Training/NewData"
#path = "/Users/maxbrodeur/Documents/NewData"

train_data = []
valence = []
arousal = []

for d in os.listdir(path):
    if d != ".DS_Store":
        absDir = path + "/" + d
        f = open(absDir + "/" + d + ".txt", "r")
        f.readline()

        count = 1

        for x in f:

            # matching image
            framePath = absDir + "/FRAMES/" + d + "#" + str(count) + ".jpg"
            if os.path.isfile(framePath):
                # valence and arousal values
                values = x.split(",")
                values = (np.array(values, "float32"))
                valence.append(values[0])
                arousal.append(values[1])

                # add image to training data
                image = cv2.imread(framePath)
                matrix = np.array(image)
                train_data.append(matrix)

            count = count + 1

# print("valence: " + str(len(valence)))
# print("arousal: " + str(len(arousal)))
# print("Numimages: " + str(len(train_data)))
# print("Width: " + str(len(train_data[0])) + " Height: " + str(len(train_data[0][0])))

data = np.empty([len(train_data), 200, 300])
train = np.array(train_data)
data[:, :, :] = train[:, :, :, 0]

v = np.empty([len(valence)])
for i in range(len(valence)):
    v[i] = valence[i]

a = np.empty([len(valence)])
for j in range(len(valence)):
    a[j] = arousal[j]

m = model()
m.train_model(data, a, v)
