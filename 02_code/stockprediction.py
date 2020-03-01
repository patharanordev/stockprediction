# Import
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

from neuralnetwork import *
from ploter import *



def predict(data):

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]

    print('\n [x] Current dimension of dataset : {0}X{1}'.format(n, p))
    # print('\n [x] Dimension of dataset, row (data.shape[0]) : {0}'.format(n))
    # print('\n [x] Dimension of dataset, column (data.shape[1]) : {0}'.format(p))

    # Make data a np.array
    data = data.values
    # print('\n [x] Data.values :\n')
    # print(data)

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8*n))
    print('\n [x] Train (start, end) : [{0}, {1}]'.format(train_start, train_end))

    test_start = train_end + 1
    test_end = n
    print('\n [x] Test (start, end) : [{0}, {1}]'.format(test_start, test_end))

    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    # print('\n [x] Data_train : \n')
    # print(data_train)
    # print('\n [x] Data_test : \n')
    # print(data_test)

    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    print('\n [x] Scaling data_test : \n')
    print(data_test)

    # Build X and y
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    print('\n [x] Build X_test : \n')
    print(X_test)
    print('\n [x] Build y_test : \n')
    print(y_test)

    # Setup plot
    [plt, line1, line2] = plot(y_test)

    # [RESTORE] Testing from trainned data model
    pred = net.run(out, feed_dict={X: X_test})
    line2.set_ydata(pred)

print('\nPreparing environment...')

# Test data file path
rawDataFilePath = '../rawdata/data_stocks_test.csv'

# Number of stocks in training data
n_stocks = 500 # X_train.shape[1]

# Save path 
savePath = os.getcwd()

# Specific path of session/checkpoint
savedSessionFilePath = "./session/stocks.ckpt"

print('\n [x] Test data file path : {0}'.format(rawDataFilePath))
print('\n [x] n_stocks : {0}'.format(n_stocks))
print('\n [x] Save directory of session/checkpoint : {0}'.format(savePath))
print('\n [x] Specific path of session/checkpoint : {0}'.format(savedSessionFilePath))

# Session
net = tf.InteractiveSession()

# Preparing hidden layer
[out, X, Y] = loadHiddenLayer(n_stocks)

# Init
net.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# [RESTORE] For restore model to session
saver.restore(net, savedSessionFilePath)

# Import data
data = pd.read_csv(rawDataFilePath)

# Drop date variable
data = data.drop(['DATE'], 1)
# print('\n [x] Data :\n')
# print(data)

predict(data)

input("Press Enter to continue...")