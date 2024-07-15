#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2023/12/30 4:05
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: mymodel.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.feature_encoder import CNN
from model.preliminary import pre_diagnosis
from model.gclayer import build_adjacent_Feature, build_adjacent_PriorImportance, GCL
from model.importance_factor import returnImportance
from model.gcns import SGCNs


# class_num 修改 类别数目
# node_num 修改 传感器数量
class I2SGCNs(nn.Module):
    def __init__(self, class_num=5, in_dim=512, inter_dim=512, out_dim=512, node_num=12, sigma=2, adj_ratio=0.2,
                 keep_top=0.8, drop_p=0.5, gc_bias=bool(0), mean_flag=True, self_loop_flag=True):
        super(I2SGCNs, self).__init__()
        # extractor ==> 特征提取
        # pre_diagnosis ==> First Stage: Preliminary Fault Diagnosis
        # ref_diagnosis ==> Second Stage: Refined Fault Diagnosis
        self.extractor = CNN()
        self.pre_diagnosis = pre_diagnosis(class_num, in_dim, mean_flag)
        self.ref_diagnosis = SGCNs(class_num,
                                   in_dim,
                                   inter_dim,
                                   out_dim,
                                   node_num,
                                   sigma,
                                   adj_ratio,
                                   self_loop_flag,
                                   keep_top,
                                   drop_p,
                                   gc_bias)

    def forward(self, data, class_num=5):
        data_shape = data.shape  # [B], [L], [H], [W] B ==> Batch sizes; L ==> Number of sensors; H ==> Height of signal; W ==> Width of signal.
        data = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])  # x = [(B*L), 1, H, W]
        # extractor ==> 特征提取
        print(data.shape)
        features = self.extractor(x=data)  # [(B*L), H*W/2]
        # features reshape
        features = features.view(data_shape[0], data_shape[1], -1)  # [B, L, H*W/2]
        print('fsadfdwsfdsf')
        print(features.shape)
        features = torch.transpose(features, 1, 2)  # [B, H*W/2, L]
        # pre_diagnosis ==> First Stage: Preliminary Fault Diagnosis
        output_pre = self.pre_diagnosis(features)
        print(output_pre.shape)
        _, importance_std = returnImportance(
            feature=features.clone().detach().data,
            weight_softmax=self.pre_diagnosis.state_dict()['classifier.weight'].data,
            class_idx=[i for i in range(class_num)])
        # importance_std reshape ([B,L] => [B, 1, L])
        importance_std = torch.unsqueeze(
            importance_std,
            dim=1)
        # ref_diagnosis ==> Second Stage: Refined Fault Diagnosis
        if self.training:
            output, emb1, (bag_kept, bag_drop) = self.ref_diagnosis(features, importance_std)
            return output_pre, output, emb1, (bag_kept, bag_drop)
        else:
            output, emb1 = self.ref_diagnosis(features, importance_std)
            return output_pre, output, emb1
    # output_pre: [B, C]; output: [B, C]; emb1: [B, H*W/2, L]; bag_kept: [B, H*W/2]; bag_drop: [B, H*W/2]


if __name__ == "__main__":
    x = torch.randn(8, 14, 1024, 1)
    model = I2SGCNs()
    output_pre, output, emb1, bag_kept, bag_drop = model(x)
    print(output_pre.shape)
    print(output.shape)
    print(emb1.shape)
    print(bag_kept.shape)
    print(bag_drop.shape)
