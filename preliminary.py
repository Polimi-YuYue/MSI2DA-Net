#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2023/12/30 6:40
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: preliminary.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
# import lib
import torch
import torch.nn as nn


class pre_diagnosis(torch.nn.Module):
    # 需要注意class_num，人为修改！！！
    # 需要注意class_num，人为修改！！！
    # 需要注意class_num，人为修改！！！
    # 需要注意class_num，人为修改！！！
    # 需要注意class_num，人为修改！！！
    def __init__(self, class_num=10, in_dim=512, mean_flag=True):
        super(pre_diagnosis, self).__init__()
        self.classifier = nn.Linear(in_dim, class_num, bias=False)
        self.mean_flag = mean_flag

    def _check_input(self, x):
        F = x.shape[2]
        assert F == 512, \
            'Input feature to network must be 512, ' \
            'but got {}'.format(F)

    def forward(self, x):
        # x in [B, N, K]
        x_shape = x.shape

        if self.mean_flag:
            x = torch.mean(x, dim=-1, keepdim=False).view(x_shape[0], x_shape[1])
        else:
            x, _ = torch.max(x, dim=-1, keepdim=False)
            x = x.view(x_shape[0], x_shape[1])
        # print(x.shape)
        output = self.classifier(x)
        return output


if __name__ == "__main__":
    input = torch.randn(8, 14, 512)
    features = torch.transpose(input, 1, 2)  # 需要转置，在主程序中添加！！[B,L,H*W/2] ==> [B,H*W/2,L]
    # print(features.shape)
    pre = pre_diagnosis(class_num=10, in_dim=512)
    print(pre)
    out = pre(features)
    print(pre_diagnosis().state_dict()['classifier.weight'].data.shape)
    # print(out.shape)
