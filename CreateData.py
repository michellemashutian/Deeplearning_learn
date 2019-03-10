#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 19:55
@desc: Create Training and Testing data
"""

from random import choice

train = open('/Volumes/SaveMe/data/2019/mlp/train_data', 'w')
test = open('/Volumes/SaveMe/data/2019/mlp/test_data', 'w')
train_label = open('/Volumes/SaveMe/data/2019/mlp/train_data_label', 'w')
test_label = open('/Volumes/SaveMe/data/2019/mlp/test_data_label', 'w')


vecs = {}
vecsid = []
with open('/Volumes/SaveMe/data/2019/pubmed/vec/vec-d2v', 'r') as f:
    for line in f:
        ids = line.strip().split('\t')[0]
        vec = line.strip().split('\t')[1]
        vecs[ids] = vec
        vecsid.append(ids)

test_truth = {}
ids = set()
with open('/Volumes/SaveMe/data/2019/pubmed/truth/test-truth-1.txt', 'r') as f:
    for line in f:
        id1 = line.strip().split(' ')[0]
        id2 = line.strip().split(' ')[2]
        ids.add(id1)
        ids.add(id2)
        if id1 in test_truth:
            test_truth[id1].append(id2)
        else:
            test_truth[id1] = [id2]

train_right = []
test_right = []

train_wrong = []
test_wrong = []

count = 0
for i in test_truth['22808056']:
    count = count + 1
    if count < 31:
        train_right.append(vecs['22808056']+' '+vecs[i])
    else:
        test_right.append(vecs['22808056']+' '+vecs[i])

for i in range(0, 100):
    idd = choice(vecsid)
    if idd in test_truth['22808056']:
        continue
    else:
        train_wrong.append(vecs['22808056'] + ' ' + vecs[idd])

for i in range(0, 15):
    idd = choice(vecsid)
    if idd in test_truth['22808056']:
        continue
    else:
        test_wrong.append(vecs['22808056'] + ' ' + vecs[idd])

for x in train_right:
    train.write(x + '\n')
    train_label.write('1\n')

for x in train_wrong:
    train.write(x + '\n')
    train_label.write('0\n')

for x in test_right:
    test.write(x + '\n')
    test_label.write('1\n')

for x in test_wrong:
    test.write(x + '\n')
    test_label.write('0\n')

train.close()
test.close()
train_label.close()
test_label.close()
