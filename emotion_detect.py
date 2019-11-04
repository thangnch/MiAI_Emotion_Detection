import tensorflow as tf

import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import sys
import PIL


# ------------------------------
# Load weight da train
model = load_model("data/mymodel.h5")

# ------------------------------
# Khai bao 7 class emotion
class_names = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load anh
img = image.load_img(sys.argv[1], grayscale=True, target_size=(48, 48))

# Chuan hoa anh
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

# Dua vao model du doan
custom = model.predict(x)

# In emotion ra man hinh
print("Emotion predict=", class_names[np.argmax(custom)])

# ------------------------------