import os

from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2


class Vid2dat(object):
    # run through vid and get frames
    def vid2frames(self, videopath, numframes):
        video = VideoFileClip(videopath)
        length = video.duration
        rate = length / (numframes - 1)
        img = []
        video = cv2.VideoCapture(videopath)
        sec = 0
        done, image = video.read()
        while done:
            image = self.resizeANDgray(image, 300, 200)
            img.append(image)
            done, image = self.frameAt(video, sec)
            sec = sec + rate
            if sec > length: break
        return img

    # trim vid from t1 to t2
    def vid2trim(self, videoPath, t1, t2):
        ffmpeg_extract_subclip(videoPath, t1, t2, targetname="trim_" + videoPath)
        return "trim_" + videoPath

    # get frame at second
    def frameAt(self, video, sec):
        video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        return video.read()

    def readVid(self, videopath):
        video = cv2.VideoCapture(videopath)
        done, image = video.read()
        while done:
            cv2.imshow('frame', image)
            cv2.waitKey(1)
            done, image = video.read()

    def resizeANDgray(self, frame, h, w):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (h, w))

    def vid2frames2directory(self, videopath, newdir):
        video = VideoFileClip(videopath)
        length = video.duration
        numframes = length * 2 + 1
        rate = length / (numframes - 1)
        img = []
        video = cv2.VideoCapture(videopath)
        sec = 0
        done, image = video.read()
        count = 1
        for num in range(numframes):
            image = self.resizeANDgray(image, 300, 200)
            img.append(image)

            newDir = newdir+"/"+videopath[:4]
            os.mkdir(newdir+"/"+videopath[:4])
            cv2.imwrite(newDir + "/" + videopath[:-4] + "#" + str(count) + ".jpg", frame)
            count = count + 1

            done, image = self.frameAt(video, sec)
            sec = sec + rate
        return img
