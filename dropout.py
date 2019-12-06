# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
# import input_data
# Import data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as arange
import pandas as pd
import numpy as np
import decimal
import random
import math
import csv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
# gets rid of l
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Calculating values of p and n for figure 2
def findValueN(prob):
    # pn = 256 for 2 first hidden layers
    # pn = 512 for last layer
    n1 = 256.0/prob
    n2 = 256.0/prob
    n3 = 512.0/prob
    n = [n1,n2,n3,prob]
    # return number o
    return n

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


def dropout(*layer):
    # Sequential model and the layers that describe the model
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(layer[0], activation=tf.nn.relu))
    model.add(Dense(layer[1], activation=tf.nn.relu))
    # probability to keep value is third in the list
    model.add(Dropout(layer[3]))
    model.add(Dense(layer[2], activation=tf.nn.softmax))

    # Configure model before training
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model for a fixed number of epochs
    history_dropout = model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=1, batch_size=10)

    # Training accuracy
        # Training error = 1 - Training accuracy
    accuracy = history_dropout.history['accuracy']
    train_err = 1.00 - accuracy[-1]

    # Test Accuracy
        # Test error = 1 - Test Accuracy
    val_acc = history_dropout.history['val_accuracy']
    test_err = 1.00 - accuracy[-1]

# Calculate all p values
def float_range(start, stop, step):
  while start <= stop:
    yield float(start)
    start += decimal.Decimal(step)

def runForAllValuesOfP():
    pValues = list(float_range(0, 1.0, '0.1'))
    del pValues[0]
    for i in pValues:
        layer = findValueN(i)
        dropout(*layer)

# runs for the list 'layer'
runForAllValuesOfP()
# dropout(*layer)


# print "test error: " + "{:.2%}".format(test_error)
list_x = []
list_y = []
# list_x.append(p)
# list_y.append(test_err)
plt.subplot(2, 2, 1)
plt.ylabel('Classification Error %')
plt.xlabel('Probability of retaining a unit (p)')
plt.yticks(np.arange(0, 4, 0.5))
plt.plot(list_x, list_y)
plt.legend(['Test Error', 'Training Error'], loc='upper right')

plt.subplot(2, 2, 2)
plt.ylabel('Classification Error %')
plt.xlabel('Probability of retaining a unit (p)')
plt.yticks(np.arange(0, 3.5, 0.5))
plt.legend(['Test Error', 'Training Error'], loc='upper right')

plt.tight_layout()
plt.show()
