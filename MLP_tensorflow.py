#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 19:30
@desc: mlp by tf
"""

import tensorflow as tf
from numpy.random import RandomState
import numpy as np

#### 1. 定义神经网络的参数，输入和输出节点。
batch_size = 8

# [x, y] x is the dimension information

W1 = tf.Variable(tf.random_normal([200, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]))

W2 = tf.Variable(tf.random_normal([100, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))

W3 = tf.Variable(tf.random_normal([10, 1], stddev=0.1))

x = tf.placeholder(tf.float32, shape=(None, 200), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

keep_prob = tf.placeholder(tf.float32)

# 定义模型结构
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
y = tf.nn.sigmoid(tf.matmul(hidden2_drop, W3))

cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


X_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data')
X_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data')

Y_train = []
with open('/Volumes/SaveMe/data/2019/mlp/train_data_label', 'r') as f:
    for line in f:
        Y_train.append([int(line.strip())])
Y_test = []
with open('/Volumes/SaveMe/data/2019/mlp/test_data_label', 'r') as f:
    for line in f:
        Y_test.append([int(line.strip())])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 30
        end = (i*batch_size) % 30 + batch_size
        sess.run(train_step, feed_dict={x: X_train[start:end], y_: Y_train[start:end], keep_prob: 1.0})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X_train, y_: Y_train, keep_prob: 1.0})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    pre_Y = sess.run(y, feed_dict={x: X_test, keep_prob: 1.0})
    for pred, real in zip(pre_Y, Y_test):
        print(pred, real)
