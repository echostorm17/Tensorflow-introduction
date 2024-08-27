from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Hide TensorFlows warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

rng = np.random

# Different parameters for learning
learning_rate = 0.01
training_epochs = 1000
display_step = 100

#Population, Percent with <$5000 income, Percent Unemployed, Murders per annum per 1mil population
data = np.genfromtxt('murder_rates_data.csv', delimiter=',', skip_header=1)

# Training Data
train_test_split = int(len(data)*0.7) #70% training : 30% testing
train_X = data[:, 2][:train_test_split] #Percent unemployed
train_Y = data[:, 3][:train_test_split] #Murders per 1 million population per year
n_samples = train_X.shape[0]

# Create placeholder for providing inputs
X = tf.placeholder("float")
Y = tf.placeholder("float")

# create weights and bias and initialize with random number
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

with tf.name_scope('WX_b') as scope:
    # Construct a linear model using Y=WX+b
    pred = tf.add(tf.multiply(X, W), b)

w_h = tf.summary.histogram('weights', W)
b_h = tf.summary.histogram('biases', b)

with tf.name_scope('cost_function') as scope:
    # Calculate Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    tf.summary.scalar('cost_function', cost)

with tf.name_scope('train') as scope:
    # Gradient descent to minimize mean sequare error
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

marge_summary_op = tf.summary.merge_all()

