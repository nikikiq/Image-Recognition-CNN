# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:48:52 2017

@author: Nia Chang
"""

import numpy as np
import keras as kr
import pandas as pd

## Input set up----------------------------------------------------------------
img_height = 64
img_width = 64
num_classes = 10

if kr.backend.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
     
data = np.load('data.npz')
test_x = data['test_X']
test_x = (test_x / 225.0).reshape([-1, img_height, img_width, 1])

## Load model and predict------------------------------------------------------
model = kr.models.load_model('cnn_6.h5')
model.summary()
predictions = model.predict(test_x)

pred_labels = np.array(np.argmax(predictions, axis=1),dtype = int)
indices = np.arange(1,(len(pred_labels) + 1),1,dtype = int)
df = pd.DataFrame()
df['Id'] = indices
df['Label'] = pred_labels
df.to_csv('884622_digits_rec_v6.csv', encoding='utf-8')