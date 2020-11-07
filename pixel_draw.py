import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf

airplane = np.load('full_numpy_bitmap_airplane.npy')
y_airplane = np.zeros((151623,1))
air = np.concatenate((airplane, y_airplane), axis = 1)
print(air.shape)

car = np.load('full_numpy_bitmap_car.npy')
y_car = np.ones((182764,1))
road = np.concatenate((car, y_car), axis = 1)
print(road.shape)

dataset = np.concatenate((air, road), axis = 0)
print(dataset.shape)

np.random.shuffle(dataset)
print(dataset.shape)

X = dataset[:,:784]
print(X.shape)
X = X.reshape(334387,28,28,1)
print(X.shape)

Y = dataset[:,784]
print(Y.shape)

class_names = ["airplane", "car"]

i = 5
plt.imshow(X[i,:,:].reshape(28,28),'gray')
print("label is",Y[i], "class is", class_names[int(Y[i])])

xtr = X[0:300000]
xts = X[300001:]
ytr = Y[0:300000]
yts = Y[300001:]


print('xtr shape:', xtr.shape)
print('ytr shape:', ytr.shape)
print(xtr.shape[0], 'train samples')
print(xts.shape[0], 'test samples')

xtr = xtr/255
xts = xts/255

K.clear_session()
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(2, activation='softmax')  
])

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(lr=0.01),
              loss='sparse_categorical_crossentropy',
              metrics = ['acc'])

model.summary()

hist = model.fit(xtr, ytr, epochs=10, batch_size=20, shuffle = True, validation_data=(xts,yts))

import cv2 
img = cv2.imread('car.jpg',0).reshape(1,28,28,1)
ypred = model.predict(img)
print("[airplane , car]")
print("prediction:",ypred)
plt.imshow(img.reshape(28,28),'gray');

