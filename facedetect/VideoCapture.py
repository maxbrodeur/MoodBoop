from time import time
import numpy as np
import cv2

class VideoCapture:

    def imageprocess(self, frames):
        new = []
        for i in range(len(frames)):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (300, 200))
            new.append(gray)
        return new

    def capture(self, cap):
        img_counter = 0
        pastTime = 0
        img = []
        while img_counter!=15:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_counter = 0
            # Display the resulting frame
            cv2.imshow('1', frame)
            k = cv2.waitKey(1)
            newTime = int(time() % 60)
            if newTime != pastTime:
                # SPACE pressed
                pastTime = newTime
                img.append(frame)
                img_counter += 1

        # When everything done, release the capture

        # cap.release()
        # cv2.destroyAllWindows()
        ans = self.imageprocess(img)

        return ans
