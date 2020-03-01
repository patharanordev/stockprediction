# Import
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from neuralnetwork import *

# Save path 
savePath = os.getcwd()

# Import data
data = pd.read_csv('../rawdata/data_stocks.csv')

# Drop date variable
data = data.drop(['DATE'], 1)
print('\n [x] Data :\n')
print(data)

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

print('\n [x] Dimension of dataset, row (data.shape[0]) : {0}'.format(n))
print('\n [x] Dimension of dataset, column (data.shape[1]) : {0}'.format(p))

# Make data a np.array
data = data.values
print('\n [x] Data.values :\n')
print(data)

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
print('\n [x] Train (start, end) : [{0}, {1}]'.format(train_start, train_end))

test_start = train_end + 1
test_end = n
print('\n [x] Test (start, end) : [{0}, {1}]'.format(test_start, test_end))

data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
print('\n [x] Data_train : \n')
print(data_train)
print('\n [x] Data_test : \n')
print(data_test)

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

print('\n [x] Scaling Data_train : \n')
print(data_train)
print('\n [x] Scaling data_test : \n')
print(data_test)

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

print('\n [x] Build X_train : \n')
print(X_train)
print('\n [x] Build y_train : \n')
print(y_train)
print('\n [x] Build X_test : \n')
print(X_test)
print('\n [x] Build y_test : \n')
print(y_test)

# Number of stocks in training data
n_stocks = X_train.shape[1]
print('\n [x] n_stocks : {0}'.format(n_stocks))

# Create default session on construction
net = tf.InteractiveSession()

# Load hidden layer from custom lib (neuralnetwork.py)
[out, X, Y] = loadHiddenLayer(n_stocks)

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))
print('\n [x] MSE : \n')
print(mse)

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Setup plot
plt.ion()
fig = plt.figure()

ax1 = fig.add_subplot(111)

# # Major ticks every 5, minor ticks every 1
# major_ticks = np.arange(0, 101, 5)
# minor_ticks = np.arange(0, 101, 1)

# ax1.set_xticks(major_ticks)
# ax1.set_xticks(minor_ticks, minor=True)
# ax1.set_yticks(major_ticks)
# ax1.set_yticks(minor_ticks, minor=True)

# # And a corresponding grid
# ax1.grid(which='both')

# # Or if you want different settings for the grids:
# ax1.grid(which='minor', alpha=0.2)
# ax1.grid(which='major', alpha=0.5)

line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural net
batch_size = 256
mse_train = []
mse_test = []

print('\n\n')

# [SAVE/TRAIN] Start train the data
epochs = 10
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])

            # [OPTION] Prediction on train (this scope can be comment out)
            # pred = net.run(out, feed_dict={X: X_test})
            # line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)


# [SAVE] Save session(checkpoint) of result of training data to specific path
save_path = saver.save(net, "./session/stocks.ckpt")
print("Model saved in path: %s" % save_path)

input("Press Enter to continue...")