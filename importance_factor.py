#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2023/12/30 10:30
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: importance_factor.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')


def returnImportance(feature, weight_softmax, class_idx):
    """check feature_conv>0, following relu
    """
    B, F, L = feature.shape  # 由于主程序转置, [B, L, F] ==> [B, F, L]
    importance_classes = []
    for idx in class_idx:
        importance = torch.sum(
            (weight_softmax[idx]).view(1, -1, 1) * feature,
            dim=1,
            keepdim=False
        )  # importance.shape = [B, L]
        importance = torch.nn.functional.softmax(importance, dim=-1)  # importance.shape = [B, L]
        importance_classes.append(importance)

    importance_classes_tensor = torch.stack(importance_classes, dim=1)  # importance_classes_tensor = [B, C, L] C: 类别数
    importance_std = torch.std(
        importance_classes_tensor,
        dim=1,
        keepdim=False
    )  # importance_classes_tensor = [B, L]
    uniform_importance = torch.ones((L)) / float(L)  # uniform_importance.shape = [L]
    if importance_std.is_cuda:
        uniform_importance.cuda()
    for b in range(B):
        if torch.sum(importance_std[b], dim=-1, keepdim=False) > 1e-10:
            importance_std[b] = importance_std[b] / torch.sum(importance_std[b], dim=-1, keepdim=False)
        else:
            print('std sum problem!:\t', torch.sum(importance_std[b], dim=-1, keepdim=False), importance_std[b])
            importance_std[b] = uniform_importance

    return importance_classes, importance_std  # [C, B, L], [B, L]


if __name__ == "__main__":
    input = torch.randn(8, 14, 512)
    features = torch.transpose(input, 1, 2)  # 需要转置，在主程序中添加！！[B, L, H*W/2] ==> [B, H*W/2, L]
    weights = torch.randn(10, 512)
    a, b = returnImportance(feature=features, weight_softmax=weights,
                            class_idx=[i for i in range(10)])  # 注 10 为类别数也需要修改
    a = torch.tensor([item.numpy() for item in a])
    print(a.shape)
    b = torch.tensor([item.numpy() for item in b])
    print(b.shape)
