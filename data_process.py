#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2024/1/1 2:43
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: data_process.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
import os

import numpy as np
# import lib
import pandas as pd


def data_process(data_num, data_len):
    '''
    data_num:每类的样本个数
    data_len：每个样本的长度
    '''
    n = 0
    data_, label_ = [], []
    for i in os.listdir('data/case study 2/'):
        print(i, ',为第', n, '类')
        # file=pd.read_excel('Data/N0-H10-W1-20S-2.xlsx').iloc[:,:6].values
        file = pd.read_excel('data/case study 2/' + i).iloc[:, :12].values

        for j in range(data_num):
            start = np.random.randint(0, file.shape[0] - data_len)
            end = start + data_len
            data_.append(file[start:end, :])
            label_.append(n)
        n += 1
    data_ = np.array(data_)
    label_ = np.array(label_)
    return data_, label_


num = 200
N = 1024
data_, label_ = data_process(num, N)
'''划分后数据为:1800x1024x6'''
# In[] FFT特征提取
# 参考:https://blog.csdn.net/qq_27825451/article/details/88553441
from scipy.fftpack import fft

fea = []
for i in range(data_.shape[0]):
    fea1 = np.zeros([512, 12])
    for j in range(data_.shape[2]):
        fft_y = fft(data_[i, :, j])
        abs_y = np.abs(fft_y)
        normalization_y = abs_y / N
        tz = normalization_y[range(int(N / 2))]
        fea1[:, j] = tz
    fea.append(fea1)
fea = np.array(fea)
'''划分后数据为:1800x512x6'''

# In[]保存数据
np.savez('data/case study 2/data_process.npz', data=data_, label=label_)
np.savez('data/case study 2/data_feature.npz', data=fea, label=label_)
