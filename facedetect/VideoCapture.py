from time import time
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
img_counter = 0
pastTime=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_counter= 0
    # Display the resulting frame
    cv2.imshow('frame',frame)
    k=cv2.waitKey(1)
    newTime = int (time() % 60)
    if  newTime!=pastTime:
        # SPACE pressed
        pastTime=newTime
        img_name = "data\opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif k & 0xFF == ord('q'):
        break

# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()