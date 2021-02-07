import os
import cv2
import numpy as np

# salut karl

# path = "/Users/maxbrodeur/Documents/Training/NewData"


train_data = []
arousal = []
valence = []

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
                values = x.split(",")
                matrix = np.array(values, 'float32')
                # valence and arousal values
                valence.append(matrix[0])
                arousal.append(matrix[1])

                # add image to training data
                image = cv2.imread(framePath)
                train_data.append(image)

            count = count + 1

# print("valence: " + str(len(valence)))
# print("arousal: " + str(len(arousal)))
# print("Numimages: " + str(len(train_data)))
# print("Width: " + str(len(train_data[0])) + " Height: " + str(len(train_data[0][0])))
