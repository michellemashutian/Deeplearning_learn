#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mashutian
@time: 2019-03-04 23:30
@desc: read data by pytorch
"""

import torch
import torch.utils.data as data
from torch.autograd import variable
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, path_file, list_file):
        self.vecs = vecs
        self.labels =labels

    def __getitem__(self, index):
        input = []
        output = []

        input = torch.Tensor(input)
        output = torch.Tensor(output)
        return input, output


    def __len__(self):
        return len(self.vecs)

dataset = MyDataset(vecs, labels)

