#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:25
@desc: 
"""
import tensorflow as tf
from numpy.random import RandomState
import numpy as np


class Config(object):
    # 这里定义的参数我用了大写
    DROPOUT_KEEP = 0.9
    LAYER1_DIM = 100
    LAYER2_DIM = 10
    X_DIM = 200
    Y_DIM = 7


class CitationRecNet(object):
    def __init__(self, dropout_keep, layer1_dim, layer2_dim, x_dim, y_dim):
        # input parameter
        # 问题： 等式右边的是不是上面那行init后面括号里的参数
        self.dropout_keep = dropout_keep
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        # network parameter
        # 问题：这里需要加self吗？比如下面这行
        # [x_dim, self.layer1_dim]这两个
        self.W1 = tf.Variable(tf.random_normal([x_dim, self.layer1_dim], stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([self.layer1_dim]))
        self.W2 = tf.Variable(tf.random_normal([self.layer1_dim, self.layer2_dim], stddev=0.1))
        self.b2 = tf.Variable(tf.zeros([self.layer2_dim]))
        self.W3 = tf.Variable(tf.random_normal([layer2_dim, y_dim], stddev=0.1))

        # training data: record and label
        self.x = tf.placeholder(tf.float32, shape=(None, x_dim), name='x-input')
        self.y = tf.placeholder(tf.float32, shape=(None, y_dim), name='y-input')

        # 问题： 这边是不是要这么写一下
        self.dropout_keep = tf.placeholder(tf.float32)

        # predict data: label
        self.y_ = self.MLP()
        self.loss = -tf.reduce_mean(self.y_ * tf.log(tf.clip_by_value(self.y, 1e-10, 1.0)))
        # 另外一种写法AAA
        # self.loss = tf.nn.softmax_cross_entropy_with_logits(y,y_)
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def MLP(self):
        hidden1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)
        hidden1_drop = tf.nn.dropout(hidden1, self.dropout_keep)
        hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, self.W2) + self.b2)
        hidden2_drop = tf.nn.dropout(hidden2, self.dropout_keep)
        y_ = tf.nn.softmax(tf.matmul(hidden2_drop, self.W3))
        # 针对另外一种写法AAA
        # 这边是不是改成
        # y_ = tf.matmul(hidden2_drop, self.W3)
        return y_

    def CNN(self):
        pass




