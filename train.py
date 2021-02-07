import os

import cv2
import numpy as np

path = "/Users/maxbrodeur/Documents/Training/NewData"

train_data = []
arousal = []
valence = []

for dir in os.listdir(path):
    absDir = path+"/"+dir
    f = open(absDir + ".text", "r")
    f.readLine()
    count = 1
    for x in f:
        # matching image
        framePath = absDir + "/FRAMES" + dir + "#" + str(count) + ".jpg"
        if os.isfile(framePath):
            values = x.split(",")
            matrix = np.array(values, 'float32')
            matrix /= 255
            # valence and arousal values
            valence.append(matrix[0])
            arousal.append(matrix[1])

            #add image to training data
            image = cv2.imread(framePath)
            train_data.append(image)
            
        count = count + 1









