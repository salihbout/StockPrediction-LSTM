import matplotlib.pyplot as plt
from dataUtils import getAttijariDataset, next_batch
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np


#Getting the prepared data and plot it
data = getAttijariDataset()
# data.plot()
# plt.show()

#Train and Test split
n = data.shape[0]
train_end = int(n*0.8)
train_set = data.head(train_end)
test_set = data.tail(n-train_end)

#Scalling the data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set)
test_scaled = scaler.transform(test_set)

#Constants
num_inputs = 1
num_time_steps = 12
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
num_train_iter = 4000
batch_size = 1

#Placesholders for Tensorflow
X = tf.placeholder(dtype=tf.float32, shape=[None,num_time_steps,num_inputs])
Y = tf.placeholder(dtype=tf.float32, shape=[None,num_time_steps,num_outputs])

#LSTM Cell
cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),output_size=num_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#Loss function and optimizer
loss = tf.reduce_mean(tf.square(outputs - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#Initialize the global variables
init = tf.global_variables_initializer()

#Tensorflow session saver
saver = tf.train.Saver()

with tf.Session() as sess :
    sess.run(init)
    for iter in range(num_train_iter):
        X_batch, Y_batch = next_batch(train_scaled, batch_size, num_time_steps)
        sess.run(train, feed_dict={X:X_batch, Y:Y_batch})

        if iter % 100 == 0 :
            mse = loss.eval(feed_dict={X:X_batch, Y:Y_batch})
            print(iter, "\tMSE", mse)
    saver.save(sess, './stock_pred_model')

with tf.Session() as sess :
    saver.restore(sess, "./stock_pred_model")
    seed_num = test_scaled.shape[0]
    train_seed  = list(train_scaled[-seed_num:])
    for iter in range(seed_num):
        X_batch = np.array(train_seed[-num_time_steps:]).reshape(1,num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X:X_batch})
        train_seed.append(y_pred[0, -1, 0])

print(train_seed)
results = scaler.inverse_transform(np.array(train_seed[seed_num:]).reshape(seed_num,1))
test_set['predicted'] = results

print(test_set.head())

test_set.plot()
plt.show()