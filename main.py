# CNN example: https://www.kaggle.com/ruchibahl18/cats-vs-dogs-basic-cnn-tutorial/notebook

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


##### Pulls pokemon images from directory and puts their image
##### arrays into variable 'X'. 'y' is the name of the pokemon
##### image that is called in the for loop as 'p'.
##### Bulbsaur=0, Charmander=1, Squirtle=2
def load_test_data(path):
    count = 0
    for p in os.listdir(path):
        img = cv2.imread(os.path.join(path,p))
        X.append(img)

        if count < 86:
            y.append(0)
        elif count >= 86 and count < 205:
            y.append(1)
        else:
            y.append(2)
        count += 1
    return X, y

def create_model():
    in_shape = (70, 70, 3)
    model = Sequential()
    model.add( Conv2D(32, (3,3), activation='relu', input_shape=in_shape) )
    model.add( Conv2D(64, (3,3), activation='relu') )
    model.add( Conv2D(128, (3,3), activation='relu') )

    model.add( MaxPooling2D((2,2)) )
    model.add( Dropout(0.25) )
    model.add( Flatten() )

    model.add( Dense(128, activation='relu') )
    model.add( Dense(64, activation='relu') )
    model.add( Dense(3, activation='softmax') )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  shuffle=True,
                  metrics=['accuracy'])
    return model

##### Resize all images to 70x70 and save them
##### to the 'resized' folder
PATH = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/pokemon/'
destination = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/resized/'
if os.listdir(PATH) == 0:
    print('Resized is empty')
    for p in os.listdir(PATH):
        path_and_fname = PATH + p
        dest_and_fname = destination + p
        f_path = os.path.join(PATH,p)
        img = cv2.imread(f_path)
        small = cv2.resize(img, dsize=(70,70))
        cv2.imwrite(dest_and_fname, small)
else:
    print('Images have already been resized')


X = []
y = []
resized_path = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/resized/'
X, y = load_test_data(resized_path)
X = np.array(X).reshape(-1, 70, 70, 3)
y = np.array(y)
y = to_categorical(y, num_classes=3)
X = X / 255.0

model = create_model()
directory = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/model/'
checkpoint = ModelCheckpoint(directory, save_best_only=True, verbose=1,
                             save_weights_only=False, save_freq='epoch')
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

model.save(directory)
