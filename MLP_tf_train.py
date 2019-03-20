#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:26
@desc: 
"""

import tensorflow as tf
from numpy.random import RandomState
import numpy as np
import MLP_tf_model

BATCH_SIZE = 8
data_num = 10425
STEPS = data_num // BATCH_SIZE + 1  # STEPS = number of batches
EPOCH = 50


X_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/mlp-train-ci-vec')
X_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/mlp-test-ci-vec')

Y_train = []
with open('/Volumes/SaveMe/data/2019/mlp/train-label', 'r') as f:
    for line in f:
        Y_train.append([int(line.strip())])
Y_test = []
with open('/Volumes/SaveMe/data/2019/mlp/test-label', 'r') as f:
    for line in f:
        Y_test.append([int(line.strip())])

# 问题，MLP_tf_model里的config是不是这里就没用了？
model = MLP_tf_model(0.7, 100, 50, 200, 7)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 130
        end = (i*BATCH_SIZE) % 130 + BATCH_SIZE
        # 问题 为什么这里也有keep_pro这个参数？
        # 问题 这里的model.y_指的是训练集对应的label呗
        sess.run(model.train_step, feed_dict={model.x: X_train[start:end], model.y_: Y_train[start:end], model.keep_prob: 1.0})
        if i % 1000 == 0:
            y_pred, total_cross_entropy = sess.run((model.y, model.loss), feed_dict={model.x: X_train, model.y_: Y_train, model.keep_prob: 1.0})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
            print("training y and real y difference:", Y_train[0:2], y_pred[0:2])
    # pre_Y = sess.run(y, feed_dict={x: X_test, keep_prob: 1.0})
    # for pred, real in zip(pre_Y, Y_test):
    #     print(pred, real)
