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

