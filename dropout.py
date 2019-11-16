# Dropout: A Simple Way to Prevent Neural Networks from Overfitting
import input_data
# Import data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LocallyConnected2D
from tensorflow.keras.utils import to_categorical


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import os

# input layer is 784 = 28*28
# 3 hidden layers are 2048
# Output layer is 10

# Parameters
learning_rate = 0.001
epochs = 2000
batch_size = 128

# number of samples to calculate validity and accuracy of test dataset
test_validity_size = 256
num_input = 784 # MNIST input dim is 28*28
num_classes = 10 # MNIST total classes (0-9 digits)

# dropout - probablility to drop a unit - from 0.1 to 1.0
dropout = 0.1

# Dimmensions for intermediary layers
# Three layers with their channel counts and fully connected layers
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer output depth
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# Probability for the neuron to be kept. Make sure to give 1 for testing
prob_keep = tf.placeholder(tf.float32)

# last layer will use softmax for better predictions
# intermediary layers will use a ReLU as stated in the paper
# add a dropout to each layer
stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
stride = 2  # output is 14x14
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
stride = 2  # output is 7x7
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

# Reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 10 * M])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4d, W5) + B5
Y_pred = tf.nn.softmax(Ylogits)

# Define the loss function
# Previously we've used our own custom cross entropy function, but TensorFlow
# has a handy function implemented in a numerically stable way, e.g.: for log(0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Y)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# Accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define the optimizer and ask it to minimize the cross entropy loss
# AdamOptimizer is more robust to "saddle points" of the gradient
# A fixed learning rate of 0.003 was too high. The decay will decrease
# the learning rate exponentially from 0.003 to 0.0001
step = tf.placeholder(tf.int32)
lrate = 0.0001 + tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
optimizer = tf.train.AdamOptimizer(lrate)
train_step = optimizer.minimize(cross_entropy)

# Initialize the variables and the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Scores data that we're going to plot later
train_scores = dict(acc=[], loss=[])
test_scores = dict(acc=[], loss=[])

# The main loop of training
# We can use a dictionary to feed the actual data into the placeholders
iterations = 10001
print_freq = iterations//10
for i in range(iterations):
    # Load batch of images and true labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    a, c = sess.run([accuracy, cross_entropy],
                    feed_dict={X: batch_X, Y: batch_Y, pkeep: 1})
    train_scores['acc'].append(a)
    train_scores['loss'].append(c)

    if i % print_freq == 0:
        print('{i:4d}: train acc {a:.4f}, train loss {c:.5f}')

    a, c = sess.run([accuracy, cross_entropy],
                    feed_dict={X: mnist.test.images, Y: mnist.test.labels,
                               pkeep: 1})
    test_scores['acc'].append(a)
    test_scores['loss'].append(c)

    # Train
    # The iteration number will be used to decay the learning rate
    # Neurons will be dropped with a probability of 25%
    sess.run(train_step, feed_dict={X: batch_X, Y: batch_Y,
                                    step: i, pkeep: 0.7})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(train_scores['acc'], lw=0.5)
ax1.plot(test_scores['acc'])
# To plot error bars
    # ax0.errorbar(x, y, yerr=error, fmt='-o')
ax1.set_title('Accuracy')

# ylim for class error % is 0.0, 3.5
ax1.set_ylim([0.94, 1.0])

fig.legend(labels=('Train Error', 'Test Error'), loc='lower left',
           ncol=2, frameon=True, fontsize='medium')
plt.show()

# Check the accuracy in the test set
a, c = sess.run([accuracy, cross_entropy],
                feed_dict={X: mnist.test.images, Y: mnist.test.labels,
                           pkeep: 1})
print('Test accuracy {a:.4f}, Test loss {c:.5f}')

# Release the resources when no longer needed
# Can use tf.Session() as context manager instead
sess.close()
