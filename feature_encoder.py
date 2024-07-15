#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2023/12/30 4:43
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: feature_encoder.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
# import lib

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(15, 1)),
            nn.BatchNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 1))
        )

    def _check_input(self, x):
        C, H, W = x.shape
        assert C == 1 and H == 1024, \
            'Input to network must be 1x1024, ' \
            'but got {}x{}'.format(C, H)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        features = x.view(x.size(0), -1)

        return features


if __name__ == '__main__':
    data = torch.randn(8, 12, 1024,
                       1)  # [B,L,H,W] B: batch_size=8; L: number of sensors=14; H: data_length=1024; W: data_width=1
    data_shape = data.shape
    data = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])  # [B*L,1,H,W]
    print(data.shape)
    cnn = CNN()
    out = cnn(data)
    print(out.shape)
    features = out.view(data_shape[0], data_shape[1], -1)  # [B,L,H*W/2]
    print(features.shape)
