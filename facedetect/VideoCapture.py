from time import time
import numpy as np
import cv2

class VideoCapture:

    def capture(self):
        cap = cv2.VideoCapture(0)
        img_counter = 0
        pastTime = 0
        img = np.empty([10])
        while img_counter!=10:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_counter = 0
            # Display the resulting frame
            cv2.imshow('frame', frame)
            k = cv2.waitKey(1)
            newTime = int(time() % 60)
            if newTime != pastTime:
                # SPACE pressed
                pastTime = newTime
                img[img_counter] = frame
                img_counter += 1

        # When everything done, release the capture

        # cap.release()
        # cv2.destroyAllWindows()
        return img
