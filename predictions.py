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

########################################################

##### Load test images and preprocess them
prdct_dir = 'test-images/'
test_data, fname = test_1(prdct_dir)
test_data = np.array(test_data).reshape(-1, 70, 70, 3)
fname = np.array(fname)
test_data = test_data / 255.0

model_dir = 'conv-models/model3/'
model = tf.keras.models.load_model(model_dir)

##### Save model summary
summary_file = 'model-summaries/model3.txt'
with open(summary_file,'w') as fh:
    model.summary(print_fn = lambda x: fh.write(x + '\n'))
print('Model summary was written to:', summary_file.split('/')[1])

##### Make predictions using loaded model
predictions = model.predict(test_data)
predictions = np.around(predictions * 100, 6)

##### Create DataFrame from prediction results
p_df = pd.DataFrame(predictions, columns=['Bulbsaur', 'Charmander', 'Squirtle'])
choice = p_df.idxmax(axis=1)
p_df['Prediction'] = choice
p_df.insert(0, 'Filename', fname, True)
p_df.set_index('Filename', inplace=True)

# p_df.to_csv('Model 3 Accuracy Results.csv')

print(p_df)
