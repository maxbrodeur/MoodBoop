import matplotlib.pyplot as plt
import cv2

plt.style.use('seaborn-whitegrid')
import numpy as np


def get_graph(x, y):
    x *= 150
    x += 150
    y *= -125
    y += 125
    # image = Image.open('image.jpg')
    # image.show()
    im = cv2.imread('VA_Circle.jpg')
    im_resized = cv2.resize(im, (300, 250), interpolation=cv2.INTER_LINEAR)
    plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
    # fig = plt.figure()
    # fig.patch.set_visible(False)
    plt.axis('off')
    plt.title("Emotio-meter")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.plot(x, y, 'D', color='red')
    plt.show()