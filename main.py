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

#####################################################################

##### Pulls pokemon images from directory and puts their image
##### arrays into variable 'X'. 'y' is the name of the pokemon
##### image that is called in the for loop as 'p'.
##### Bulbsaur=0, Charmander=1, Squirtle=2
def load_test_data():
    X = []
    y = []
    bulb_train = ['resized/bulbasaur/', 'flipped/bulbasaur/']
    charm_train = ['resized/charmander/', 'flipped/charmander/']
    squirt_train = ['resized/squirtle/', 'flipped/squirtle/']

    for path in bulb_train:
        for p in os.listdir(path):
            img = cv2.imread(os.path.join(path, p))
            X.append(img)
            y.append(0)
    for path in charm_train:
        for p in os.listdir(path):
            img = cv2.imread(os.path.join(path, p))
            X.append(img)
            y.append(1)
    for path in squirt_train:
        for p in os.listdir(path):
            img = cv2.imread(os.path.join(path, p))
            X.append(img)
            y.append(2)

    return X, y

##### Resize all images to 70x70 and save them
##### to the 'resized' folder
def resize_images():
    b_path = 'pokemon/bulbasaur/'
    c_path = 'pokemon/charmander/'
    s_path = 'pokemon/squirtle/'

    b_dest = 'resized/bulbasaur/'
    c_dest = 'resized/charmander/'
    s_dest = 'resized/squirtle/'

    for p in os.listdir(b_path):
        path_and_fname = b_path + p
        dest_and_fname = b_dest + p
        f_path = os.path.join(b_path,p)
        img = cv2.imread(f_path)
        small = cv2.resize(img, dsize=(70,70))
        cv2.imwrite(dest_and_fname, small)
    for p in os.listdir(c_path):
        path_and_fname = c_path + p
        dest_and_fname = c_dest + p
        f_path = os.path.join(c_path,p)
        img = cv2.imread(f_path)
        small = cv2.resize(img, dsize=(70,70))
        cv2.imwrite(dest_and_fname, small)
    for p in os.listdir(s_path):
        path_and_fname = s_path + p
        dest_and_fname = s_dest + p
        f_path = os.path.join(s_path,p)
        img = cv2.imread(f_path)
        small = cv2.resize(img, dsize=(70,70))
        cv2.imwrite(dest_and_fname, small)
    print('Images were resized.')

def flip_images():
    b_src = 'resized/bulbasaur/'
    c_src = 'resized/charmander/'
    s_src = 'resized/squirtle/'

    b_dst = 'flipped/bulbasaur/'
    c_dst = 'flipped/charmander/'
    s_dst = 'flipped/squirtle/'

    for file in os.listdir(b_src):
        dest_fname = b_dst + file
        f_path = os.path.join(b_src, file)
        img = cv2.imread(f_path)
        flipped = cv2.flip(img, 1)
        cv2.imwrite(dest_fname, flipped)
    for file in os.listdir(c_src):
        dest_fname = c_dst + file
        f_path = os.path.join(c_src, file)
        img = cv2.imread(f_path)
        flipped = cv2.flip(img, 1)
        cv2.imwrite(dest_fname, flipped)
    for file in os.listdir(s_src):
        dest_fname = s_dst + file
        f_path = os.path.join(s_src, file)
        img = cv2.imread(f_path)
        flipped = cv2.flip(img, 1)
        cv2.imwrite(dest_fname, flipped)
    print('Images were flipped.')

def create_model():
    in_shape = (70, 70, 3)
    model = Sequential()

    model.add( Conv2D(16, (3,3), activation='relu', padding='same', input_shape=in_shape) )
    model.add( Conv2D(32, (3,3), activation='relu', padding='same') )
    model.add( Conv2D(64, (3,3), activation='relu', padding='same') )
    model.add( Conv2D(128, (3,3), activation='relu', padding='same') )

    model.add( MaxPooling2D((2,2)) )
    model.add( Dropout(0.5) )
    model.add( Flatten() )

    model.add( Dense(128, activation='relu') )
    model.add( Dense(64, activation='relu') )
    model.add( Dense(32, activation='relu') )
    model.add( Dense(3, activation='softmax') )

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  shuffle=True,
                  metrics=['accuracy'])
    return model

#####################################################################

##### Uncomment to resize or flip images as needed
# resize_images()
# flip_images()

X, y = load_test_data()
X = np.array(X).reshape(-1, 70, 70, 3)
y = np.array(y)
y = to_categorical(y, num_classes=3)
X = X / 255.0

model = create_model()

directory = 'conv-models/model3/'
checkpoint = ModelCheckpoint(directory, save_best_only=True, verbose=1,
                             save_weights_only=False, save_freq='epoch')

model.fit(X, y, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint])

model.save(directory)
