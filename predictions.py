import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def test_1(path):
    X_test = []
    fname = []
    for p in os.listdir(path):
        fname.append(p.split('.')[0])
        img_Arr = cv2.imread(os.path.join(path,p))
        new_img = cv2.resize(img_Arr, dsize=(70, 70))
        X_test.append(new_img)
    return X_test, fname

prdct_dir = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/test-images/'
test_data, fname = test_1(prdct_dir)
test_data = np.array(test_data).reshape(-1, 70, 70, 3)
fname = np.array(fname)
test_data = test_data / 255.0

model_dir = 'C:/Users/brand/Desktop/Python/CNN-Pokemon/model/'
model = tf.keras.models.load_model(model_dir)

predictions = model.predict(test_data)
predictions = predictions * 100

p_df = pd.DataFrame(predictions, columns=['Bulbsaur', 'Charmander', 'Squirtle'])
choice = p_df.idxmax(axis=1)

p_df['Prediction'] = choice
p_df.insert(0, 'Filename', fname, True)
p_df.set_index('Filename', inplace=True)

p_df.to_csv('Trial 1 Accuracy Results.csv')

print(p_df)
model.summary()
