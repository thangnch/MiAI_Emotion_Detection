import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pyglet
import pygame

# Khai bao cac class
class_names = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
prev_class = None

# Load model
model = load_model("data/mymodel.h5")

#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)




# Khoi tao camera
cam=cv2.VideoCapture(0);



while(True):

    # Doc anh tu cam
    ret,img=cam.read();

    # Lat anh
    img = cv2.flip(img, 1)

    centerH = img.shape[0] // 2;
    centerW = img.shape[1] // 2;
    sizeboxW = 400;
    sizeboxH = 600;

    # Chuyen ve anh xam
    crop=img[centerH - sizeboxH // 2:centerH + sizeboxH // 2,centerW - sizeboxW // 2:centerW + sizeboxW // 2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop,(48,48))

    # Ve khung hinh chu nhat

    cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                  (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 1)

    # Chuan hoa du lieu
    x = image.img_to_array(crop)
    x = np.expand_dims(x, axis=0)
    x /= 255

    # Dua vao model
    custom = model.predict(x)

    # Hien thi thong tin
    best_class = class_names[np.argmax(custom)]
    cv2.putText(img, "Emotion: " + str(best_class), (30,  30), fontface, fontscale, fontcolor, 2)


    # Play sound
    if (best_class=="happy") and (prev_class!=best_class):
        pygame.mixer.init()
        pygame.mixer.music.load("happy_n.wav")
        pygame.mixer.music.play()

    prev_class=best_class

    cv2.imshow('Face',img)
    # Neu nhan q la thoat
    if cv2.waitKey(1)==ord('q'):
        break;
cam.release()
cv2.destroyAllWindows()

