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
        # self.y_pred = tf.clip_by_value(self.MLP(), 1e-10, 1.0)
        self.y_pred_softmax = tf.nn.softmax(self.y_pred)

        """
        model training 
        """
        self.loss = tf.reduce_logsumexp(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))
        self.loss_metric = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.y_pred, labels=self.y))

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
        #                                            decay=0.9, momentum=0.0, epsilon=1e-10, name='optimizer')
        self.train_op = self.optimizer.minimize(self.loss, name='train_op')

    def MLP(self):
        with tf.variable_scope("layer1"):
            self.Wak = tf.get_variable("wak", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Waq = tf.get_variable("waq", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wav = tf.get_variable("wav", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wbk = tf.get_variable("wbk", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wbq = tf.get_variable("wbq", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.Wbv = tf.get_variable("wbv", initializer=tf.random_normal([self.x_dim1, self.layer1_dim], stddev=0.1),
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
                                      initializer=tf.random_normal([self.layer2_dim*2, self.layer3_dim], stddev=0.1),
                                      dtype=tf.float32)
            self.b3 = tf.get_variable("b3", initializer=tf.zeros([self.layer3_dim]), dtype=tf.float32)

        with tf.variable_scope("output"):
            self.W4 = tf.get_variable("w_output",
                                      initializer=tf.truncated_normal([self.layer3_dim, self.y_dim], stddev=0.1),
                                      dtype=tf.float32)


        """
        first layer: self attention
        """
        ak = tf.matul(self.xa, self.Wak)
        aq = tf.matul(self.xa, self.Waq)
        av = tf.matul(self.xa, self.Wav)
        bk = tf.matul(self.xb, self.Wbk)
        bq = tf.matul(self.xb, self.Wbq)
        bv = tf.matul(self.xb, self.Wbv)

        az = tf.matmul(tf.nn.softmax(tf.multiply(aq, tf.transpose(ak))/(self.layer1_dim**0.5)), av)
        bz = tf.matmul(tf.nn.softmax(tf.multiply(bq, tf.transpose(bk))/(self.layer1_dim**0.5)), bv)

        hidden11_drop = tf.nn.dropout(az, self.dropout_keep)
        hidden12_drop = tf.nn.dropout(bz, self.dropout_keep)

        """
        second layer: MLP
        """
        hidden21 = tf.nn.sigmoid(tf.matmul(hidden11_drop, self.W21) + self.b21)
        hidden22 = tf.nn.sigmoid(tf.matmul(hidden12_drop, self.W22) + self.b22)

        hidden21_drop = tf.nn.dropout(hidden21, self.dropout_keep)
        hidden22_drop = tf.nn.dropout(hidden22, self.dropout_keep)

        """
        third layer: concat
        """
        hidden3 = tf.nn.sigmoid(tf.matmul(tf.concat([hidden21_drop, hidden22_drop], axis=1), self.W3) + self.b3)
        hidden3_drop = tf.nn.dropout(hidden3, self.dropout_keep)

        """
        output layer: matmul
        """
        y_pred = tf.matmul(hidden3_drop, self.W4)
        return y_pred

    def CNN(self):
        pass




