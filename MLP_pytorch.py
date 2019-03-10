#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 19:30
@desc: mlp by pytorch
"""

import torch
import numpy as np
from torchvision.datasets import mnist
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataset, TensorDataset

x_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data')
y_train = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/train_data_label')
x_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data')
y_test = np.loadtxt('/Volumes/SaveMe/data/2019/mlp/test_data_label')

#
# y_train = []
# with open('/Volumes/SaveMe/data/2019/mlp/train_data_label', 'r') as f:
#     for line in f:
#         y_train.append([int(line.strip())])
# y_test = []
# with open('/Volumes/SaveMe/data/2019/mlp/test_data_label', 'r') as f:
#     for line in f:
#         y_test.append([int(line.strip())])

x_data = torch.from_numpy(x_train)
x_label = torch.from_numpy(y_train)
y_data = torch.from_numpy(x_test)
y_label = torch.from_numpy(y_test)

deal_dataset = TensorDataset(x_data, x_label)
predict_dataset = TensorDataset(y_data, y_label)

print x_data.size()

train_data = DataLoader(deal_dataset, batch_size=20, shuffle=True)
test_data = DataLoader(predict_dataset, batch_size=20, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # CNN网络就是加几个卷积层，再修改fc1的输入为16*5*5
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(200, 150)
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(x)
        return x

mlp = MLP()

#交叉熵损失
criterion = nn.CrossEntropyLoss(size_average=False)
#随机梯度下降
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.0001)

losses = []
acces = []
eval_losses = []
eval_acces = []

for e in range(20):
    train_loss = 0
    train_acc = 0
    # 开始训练
    mlp.train()
    for im, label in train_data:
        im = Variable(im.float())
        label = Variable(label.long())
        print im, label
        # 前向传播
        print '-  -  -'
        out = mlp(im)
        print out, label
        print '--------'
        loss = criterion(out, label)

        # print out, label
    #     # 反向传播
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     # 记录误差
    #     train_loss += loss.item()
    #     # 计算分类的准确率
    #     _, pred = out.max(1)
    #     num_correct = (pred == label).sum().item()
    #     acc = num_correct / im.shape[0]
    #     train_acc += acc
    #
    # losses.append(train_loss / len(train_data))
    # acces.append(train_acc / len(train_data))
    # # 在测试集上检验效果
    # eval_loss = 0
    # eval_acc = 0
    # mlp.eval()  # 将模型改为预测模式
    # for im, label in test_data:
    #     im = Variable(im.float())
    #     label = Variable(label.long())
    #     out = mlp(im)
    #     loss = criterion(out, label)
    #     # 记录误差
    #     eval_loss += loss.item()
    #     # 记录准确率
    #     _, pred = out.max(1)
    #     num_correct = (pred == label).sum().item()
    #     acc = num_correct / im.shape[0]
    #     eval_acc += acc
    #
    # eval_losses.append(eval_loss / len(test_data))
    # eval_acces.append(eval_acc / len(test_data))
    # print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
    #       .format(e, train_loss / len(train_data), train_acc / len(train_data),
    #               eval_loss / len(test_data), eval_acc / len(test_data)))
