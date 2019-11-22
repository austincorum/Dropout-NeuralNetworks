# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
# import input_data
# Import data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
import random
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # print the training data shape
# print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Set values to float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value
x_train /= 255
x_test /= 255

# Sequential model and the layers that describe the model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(200, activation=tf.nn.relu))
model.add(Dropout(0.6))
model.add(Dense(10,activation=tf.nn.softmax))

# Configure model before training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for a fixed number of epochs
history = model.fit(x=x_train, y=y_train, validation_split=0.33, epochs=1, batch_size=10)
# Returns the loss value and accuracy values for the model
model.evaluate(x_test, y_test)
# loss, accuracy = model.evaluate(x_test, y_test)

# print "test error: " + "{:.2%}".format(test_error)
plt.subplot(1, 2, 1)
plt.ylabel('Classification Error %')
plt.xlabel('Probability of retaining a unit (p)')
plt.legend(['Test Error', 'Training Error'], loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Classification Error %')
plt.xlabel('Probability of retaining a unit (p)')
plt.legend(['Test Error', 'Training Error'], loc='upper right')

plt.tight_layout()
plt.show()
