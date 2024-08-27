from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Hide TensorFlows warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

