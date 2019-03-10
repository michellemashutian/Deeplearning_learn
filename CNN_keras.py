#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-08 15:04
@desc: 1D convolution
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


x_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data')
y_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data_label')
x_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data')
y_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data_label')

model = Sequential()
model.add(Conv1D())

# model.add(Dense(2, activation='sigmoid'))
# label is [0 1]
model.add(Dense(1, activation='sigmoid'))
# label is [1]


model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=10)
score = model.evaluate(x_test, y_test, batch_size=10)
y_predict = model.predict(x_test)

for i in y_predict:
    print i
print score