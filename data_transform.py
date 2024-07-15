#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2024/1/1 5:16
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: data_transform.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
# import lib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch


class DiabetesDataset(Dataset):
    def __init__(self, data, label):
        # xy = np.loadtxt(filepath, delimiter =',', dtype =np.float32)
        #print(data.shape)
        self.len = data.shape[0]
        data = data.astype(np.float32)
        self.x_data = torch.from_numpy(data).view(-1, 12, 1024, 1)
        self.y_data = torch.from_numpy(label).view(-1).long()
        #print(self.x_data.size())
        #print(self.y_data.size())

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


def load_data(data, label, batch_size=32, drop_last=False):
    dataset = DiabetesDataset(data, label)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=drop_last)
    return data_loader
