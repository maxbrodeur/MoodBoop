import os
from shutil import copyfile
import cv2

from MoodBoop.videodata.Vid2dat import Vid2dat

v2d = Vid2dat()

path = "/Users/maxbrodeur/Documents/Training"




def prepare(pathVids, pathAnnotations, newPath):

    for videoPath in os.listdir(pathVids):

        # path names
        txtPath = videoPath[:-4] + ".txt"
        absPath = pathAnnotations + "/" + txtPath
        absPathVid = pathVids + "/" + videoPath
        print("Analyzing " + videoPath + " and " + txtPath + "...")

        if os.path.isfile(absPath):

            f = open(absPath, "r")

            # first line is valence and arousal
            f.readline()
            # only analyzing videos with one person (one arousal one valence)
            if len(f.readline().split(',')) < 3:
                count = 1
                for x in f:
                    count = count + 1

                print(str(count) + " frames...")

                # Video to frames
                frames = v2d.vid2frames(absPathVid, count)

                newDir = newPath + "/" + videoPath[:-4]
                os.mkdir(newDir)
                newDir2 = newDir + "/FRAMES"
                os.mkdir(newDir2)

                print("filling " + newDir2 + " with " + str(len(frames)) + "images ...")

                count = 1
                for frame in frames:
                    cv2.imshow("frame", frame)
                    cv2.imwrite(newDir2 + "/" + videoPath[:-4] + "#" + str(count) + ".jpg", frame)
                    count = count + 1

                copyfile(absPath, newDir + "/" + txtPath)


prepare(path + "/Video_Train_Set", path + "/Annotations_Train_Set", path + "/NewData")
# prepare(path + "/VIDTEST", path + "/ANTEST", path + "/NEWDATATEST")
