#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-10 13:25
@desc:
"""
from __future__ import print_function
# import sys
# sys.path.append("..") #if you want to import python module from other folders,
# you need to append the system path
import tensorflow as tf
from numpy.random import RandomState
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm  # for batch normalization
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
import math

class Config(object):
    def __init__(self, args):
        self.LAYER1_DIM = args.layer1_dim
        self.LAYER2_DIM = args.layer2_dim
        self.LAYER3_DIM = args.layer3_dim
        self.LAYER4_DIM = args.layer4_dim
        self.LEARNING_RATE = args.learning_rate
        self.EPOCH = args.epoch
        self.BATCH_SIZE = args.batch_size


class CitationRecNet(object):
    def __init__(self, layer1_dim, layer2_dim, layer3_dim, layer4_dim, x_dim1, x_dim2,
                 y_dim, learning_rate, data_num):
        # in order to generate same random sequences
        tf.set_random_seed(1)

        """
        input parameter
        """
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim
        self.layer3_dim = layer3_dim
        self.layer4_dim = layer4_dim
        self.x_dim1 = x_dim1
        self.x_dim2 = x_dim2
        self.y_dim = y_dim
        self.learning_rate = learning_rate
        self.data_num = data_num

        """
        input data
        """
        # training data: record and label
        self.dropout_keep = tf.placeholder(dtype=tf.float32, name='dropout_keep')
        self.xa = tf.placeholder(tf.float32, shape=(None, self.x_dim1), name='xa-input')
        self.xb = tf.placeholder(tf.float32, shape=(None, self.x_dim2), name='xb-input')
        self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y-input')
        """
        graph structure
        """
        # predict data: label
        self.y_pred = self.MLP()
        self.y_pred_softmax = tf.nn.softmax(self.y_pred)

        """
        model training 
        """
        # reduce_logsumexp
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))
        self.loss_metric = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
        #                                            decay=0.9, momentum=0.0, epsilon=1e-10, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def MLP(self):
        with tf.variable_scope("layer1"):
            self.Wk = tf.get_variable("wak", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wq = tf.get_variable("waq", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wv = tf.get_variable("wav", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
        with tf.variable_scope("layer2"):
            self.W21 = tf.get_variable("w21",
                                      initializer=tf.random_normal([self.layer1_dim, self.layer2_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.W22 = tf.get_variable("w22",
                                      initializer=tf.random_normal([self.layer1_dim, self.layer2_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b21 = tf.get_variable("b21", initializer=tf.zeros([self.layer2_dim]), dtype=tf.float32)
            self.b22 = tf.get_variable("b22", initializer=tf.zeros([self.layer2_dim]), dtype=tf.float32)

        with tf.variable_scope("layer3"):
            self.W3 = tf.get_variable("w3",
                                      initializer=tf.random_normal([self.layer2_dim, self.layer3_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b3 = tf.get_variable("b3", initializer=tf.zeros([self.layer3_dim]), dtype=tf.float32)

        with tf.variable_scope("output"):
            self.W4 = tf.get_variable("w_output",
                                      initializer=tf.truncated_normal([self.layer3_dim, self.y_dim], stddev=0.1),
                                      dtype=tf.float32)


        """
        first layer: self attention
        """
        ak = tf.matmul(self.xa, self.Wk)
        aq = tf.matmul(self.xa, self.Wq)
        av = tf.matmul(self.xa, self.Wv)
        bk = tf.matmul(self.xb, self.Wk)
        bq = tf.matmul(self.xb, self.Wq)
        bv = tf.matmul(self.xb, self.Wv)

        score_aa = tf.matmul(aq, tf.transpose(ak))
        score_ab = tf.matmul(aq, tf.transpose(bk))
        divide_s_aa = score_aa/(tf.sqrt(float(self.layer1_dim)))
        divide_s_ab = score_ab/(tf.sqrt(float(self.layer1_dim)))
        softmax_aa = tf.nn.softmax(divide_s_aa)  #70*70
        softmax_ab = tf.nn.softmax(divide_s_ab)
        az = tf.matmul(softmax_aa, av) + tf.matmul(softmax_ab, bv)

        score_ba = tf.matmul(bq, tf.transpose(ak))
        score_bb = tf.matmul(bq, tf.transpose(bk))
        divide_s_ba = score_ba/(self.layer1_dim**0.5)
        divide_s_bb = score_bb/(self.layer1_dim**0.5)
        softmax_ba = tf.nn.softmax(divide_s_ba)
        softmax_bb = tf.nn.softmax(divide_s_bb)
        bz = tf.matmul(softmax_ba, av) + tf.matmul(softmax_bb, bv)

        hidden11 = tf.nn.dropout(az+bz, self.dropout_keep)
        hidden11_drop = tf.nn.dropout(hidden11, self.dropout_keep)

        """
        second layer: MLP
        """
        hidden21 = tf.nn.relu(tf.matmul(hidden11_drop, self.W21) + self.b21)
        hidden21_drop = tf.nn.dropout(hidden21, self.dropout_keep)

        """
        third layer: concat
        """
        hidden3 = tf.nn.sigmoid(tf.matmul(hidden21_drop, self.W3) + self.b3)
        hidden3_drop = tf.nn.dropout(hidden3, self.dropout_keep)

        """
        output layer: matmul
        """
        y_pred = tf.matmul(hidden3_drop, self.W4)
        return y_pred

    def CNN(self):
        pass




