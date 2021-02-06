import os
import cv2

crops = []
cascPath = "haarcascade_frontalface_default.xml"

i=0
# go through data directory
for imagePath in os.listdir("data"):

    if imagePath.endswith(".jpg"):
        # ai stuff
        faceCascade = cv2.CascadeClassifier(cascPath)

        image = cv2.imread("data/" + imagePath)

        # black and white
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(48, 48),
        )

        # crop images to 48x48
        for (x, y, w, h) in faces:
            crop = gray[y:y + h, x:x + h]
            crop = cv2.resize(crop, (48, 48))
            crops.append(crop)

# clear faces directory
for face in os.listdir("faces"):
    os.unlink("faces/" + face)

# add 48x48 black and white faces to faces directory
i = 0
for im in crops:
    i = i + 1
    cv2.imwrite("faces/face" + str(i) + ".jpg", im)
