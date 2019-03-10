#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 19:30
@desc: mlp by keras
"""

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

x_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data')
y_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data_label')
x_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data')
y_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data_label')

model = Sequential()
model.add(Dense(64, input_dim=200, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# model.add(Dense(2, activation='sigmoid'))
# label is [0 1]
model.add(Dense(1, activation='sigmoid'))
# label is [1]

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=10)

model.save('test_model.h5')
del model
model = load_model('test_model.h5')
score = model.evaluate(x_test, y_test, batch_size=10)
y_predict = model.predict(x_test)

for i in y_predict:
    print i
print score


# keras 参数设置？
