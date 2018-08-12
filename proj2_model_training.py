# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:36:15 2017

@author: Nia Chang
"""

import numpy as np
import keras as kr
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

## Input set up----------------------------------------------------------------
img_height = 64
img_width = 64
num_classes = 10

if kr.backend.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)
     
data = np.load('data.npz')
X = data['train_X']
X = (X/225.0).reshape([-1, img_height, img_width, 1])
y = kr.utils.np_utils.to_categorical(data['train_y'], 
                                     num_classes = num_classes)
test_x = data['test_X'].reshape([-1, img_height, img_width, 1])
print(X.shape, y.shape, test_x.shape) 

## Set up parameters-----------------------------------------------------------  
epochs = 100
batch_size = 100
dropout = 0.4
learning_rate = 1e-03

x_train, x_validation, y_train, y_validation = train_test_split(
        X, y, test_size = 0.25 )

steps = x_train.shape[0] // batch_size

print(x_train.shape, y_train.shape)

## Build the CNN model---------------------------------------------------------

model = Sequential()

# conv1*2
model.add(Conv2D(
    filters = 32, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu',
    input_shape = (64, 64, 1)))
model.add(Conv2D(
    filters = 32, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
#kr.layers.normalization.BatchNormalization(axis=-1)

#conv2*2
model.add(Conv2D(
    filters = 64, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))
model.add(Conv2D(
    filters = 64, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
#kr.layers.normalization.BatchNormalization(axis=-1)

#conv3*2
model.add(Conv2D(
    filters = 128, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))
model.add(Conv2D(
    filters = 128, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
#kr.layers.normalization.BatchNormalization(axis=-1)

#conv4*2
model.add(Conv2D(
    filters = 256, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))
model.add(Conv2D(
    filters = 256, 
    kernel_size = (3, 3),
    padding = 'Same',
    activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))
#kr.layers.normalization.BatchNormalization(axis=-1)

# conv5*2
#model.add(Conv2D(
#    filters = 512, 
#    kernel_size = (3, 3),
#    padding = 'Same',
#    activation = 'relu'))
#model.add(Conv2D(
#    filters = 512, 
#    kernel_size = (3, 3),
#    padding = 'Same',
#    activation = 'relu'))
#model.add(MaxPool2D(pool_size = (2,2)))
#kr.layers.normalization.BatchNormalization(axis=-1)

#fully-connected layer
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(dropout))
model.add(Dense(256, activation ='relu'))
model.add(Dropout(dropout))
model.add(Dense(10, activation = 'softmax'))

optimizer = kr.optimizers.RMSprop(
    lr = learning_rate, 
    rho = 0.9, 
    epsilon = 1e-08,
    decay = 0.0)
## cross entropy loss function works better than mean square, and is rather 
## useful in NN classification.
model.compile(
    optimizer = optimizer,
    loss = "categorical_crossentropy",
    metrics = ['accuracy'])

## learning rate curve
## how much the model has been improved
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

## add noises, impage preprocess ---------------------------------------------- 
datagen = ImageDataGenerator(
        featurewise_center = False, 
        samplewise_center = False,  
        featurewise_std_normalization = False,  
        samplewise_std_normalization = False,  
        zca_whitening = False, 
        rotation_range = 10, 
        zoom_range = 0.1, 
        width_shift_range = 0.1,  
        height_shift_range = 0.1, 
        horizontal_flip = False,  
        vertical_flip = False)

datagen.fit(x_train)

## model fitting---------------------------------------------------------------
fitting = model.fit_generator(
    datagen.flow(X, y, batch_size = batch_size),
    epochs = epochs,
    validation_data = (x_validation, y_validation),
    verbose = 2,
    steps_per_epoch = steps,
    callbacks = [learning_rate_reduction,])
model.save('cnn_6.h5')

## generate figures for training-----------------------------------------------
fig, ax = plt.subplots(2,1)
ax[0].plot(fitting.history['loss'], color='b', label="Training loss")
ax[0].plot(fitting.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(fitting.history['acc'], color='b', label="Training accuracy")
ax[1].plot(fitting.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('training.png')