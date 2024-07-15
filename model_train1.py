#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2024/1/1 21:47
@Author: Yue Yu
@School: Politecnico di Milano
@Email: yyu41474@gmail.com
@Filmname: model_train1.py
@Software: PyCharm
@Theme: Fault Diagnosis
'''
# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from model.feature_encoder import CNN
from model.preliminary import pre_diagnosis
from model.gclayer import build_adjacent_Feature, build_adjacent_PriorImportance, GCL
from model.importance_factor import returnImportance
from model.gcns import SGCNs
from lightning_fabric.utilities.seed import seed_everything
from tqdm import trange
from model.losses import common_loss, refine_ranking_loss, drop_consistency_loss, criterion, loss_dependence_batch
from sklearn.model_selection import train_test_split
from data_transform import load_data

matplotlib.use('TkAgg')


# 用测试集评估模型的训练好坏
def eval(model_1, model_2, model_3, test_loader):
    eval_loss = 0.0
    total_acc = 0.0
    extractor.cuda()
    extractor.eval()
    aggregator.cuda()
    aggregator.eval()
    pre_diagnosis.cuda()
    pre_diagnosis.eval()

    for i, batch in enumerate(test_loader):
        #        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        data = x.to(device)
        y = y.to(device)
        data_shape = data.shape
        input_batch = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])
        with torch.no_grad():
            features = extractor(x=input_batch)  # [(B*L), 512]
            features = features.view(data_shape[0], data_shape[1], -1)  # [B, L, 512]
            features = torch.transpose(features, 1, 2)  # [B, 512, Z]
            # pre_diagnosis ==> First Stage: Preliminary Fault Diagnosis
            output_pre = pre_diagnosis(features)
            # pre_diagnosis
            loss_pre = criterion(output_pre, y) * 1
            _, importance_std = returnImportance(
                feature=features.clone().detach().data,
                weight_softmax=pre_diagnosis.state_dict()['classifier.weight'].data,
                class_idx=[i for i in range(5)])  # B*K 注：3是类别数需要更换！
            # importance_std reshape ( [B, K]=> [B, 1, K])
            importance_std = torch.unsqueeze(
                importance_std,
                dim=1)
            logits, emb1 = aggregator(
                features,
                importances_batch=importance_std)
            # dropout_consistency loss
            loss = criterion(logits, y)
            # 记录误差
            eval_loss += (loss_pre + loss).item()
            # 记录准确率
            _, preds = logits.max(1)
            num_correct = (preds == y).sum().item()
            total_acc += num_correct

    loss = eval_loss / len(test_loader)
    acc = total_acc / (len(test_loader) * batch_size)
    return loss, acc


if __name__ == "__main__":
    ################## 超参数 ##################
    num_classes = 5
    num_sensors = 12
    batch_size = 8
    learning_rate = 0.001  # 学习率
    weight_decay = 1e-6
    total_epoch = 300  # 迭代次数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(3407)
    ################## 加载数据 ##################
    data = np.load(r'E:\1. Manuscripts\paper_2\10. Latest\data\case study 1\data_process.npz')
    in_ = data['data']
    out_ = data['label']
    in_ = in_[:, np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(in_, out_, test_size=0.3, random_state=0)
    train_loader = load_data(x_train, y_train, batch_size=batch_size)
    test_loader = load_data(x_test, y_test, batch_size=batch_size)
    # target domain data
    data = np.load(r'E:\1. Manuscripts\paper_2\10. Latest\data\case study 2\data_process.npz')
    in_ = data['data']
    out_ = data['label']
    in_ = in_[:, np.newaxis]
    x_train, x_test, y_train, y_test = train_test_split(in_, out_, test_size=0.3, random_state=0)
    train_tgt_loader = load_data(x_train, y_train, batch_size=batch_size)
    test_tgt_loader = load_data(x_test, y_test, batch_size=batch_size)
    ################## 加载模型 ##################
    extractor = CNN()
    pre_diagnosis = pre_diagnosis(class_num=5, in_dim=512, mean_flag=True)
    aggregator = SGCNs(class_num=5,
                       in_dim=512,
                       inter_dim=512,
                       out_dim=512,
                       node_num=12,
                       sigma=1.5,
                       adj_ratio=0.2,
                       self_loop_flag=True,
                       keep_top=0.9,
                       drop_p=0.4,
                       gc_bias=bool(0))
    ################## 传进device ##################
    extractor.cuda()
    pre_diagnosis.cuda()
    aggregator.cuda()
    ################## 优化器 ##################
    optimizer_extractor = torch.optim.Adam(params=extractor.parameters(), lr=1e-3 * 0.1, weight_decay=1e-6)
    optimizer_aggregator_pre = torch.optim.Adam(params=pre_diagnosis.parameters(), lr=1e-3, weight_decay=1e-6)
    optimizer_aggregator = torch.optim.Adam(params=aggregator.parameters(), lr=1e-3, weight_decay=1e-6)
    ################## 训练模型 ##################
    print("training.........................")
    # 设置测试损失list,和测试acc 列表
    val_loss_list = []
    val_acc_list = []
    # 设置训练损失list
    train_loss_list = []
    train_acc_list = []
    max_acc = 0
    for epoch in trange(total_epoch, desc='Training', unit='epoch'):
        extractor = extractor.train()  # 启用dropout
        pre_diagnosis = pre_diagnosis.train()
        aggregator = aggregator.train()
        train_loss = 0
        for batch, batch1 in zip(train_loader, train_tgt_loader):
            x, y = batch
            data = x.to(device)
            y = y.to(device)
            #
            data_shape = data.shape  # [B, L, H, W]
            input_batch = data.view(data_shape[0] * data_shape[1], 1, data_shape[2], data_shape[3])
            features = extractor(x=input_batch)  # [(B*L), 512]
            # features reshape
            features = features.view(data_shape[0], data_shape[1], -1)  # [B, L, 512]
            features = torch.transpose(features, 1, 2)  # [B, 512, Z]
            # pre_diagnosis ==> First Stage: Preliminary Fault Diagnosis
            output_pre = pre_diagnosis(features)
            # pre_diagnosis
            loss_pre = criterion(output_pre, y) * 1
            _, importance_std = returnImportance(
                feature=features.clone().detach().data,
                weight_softmax=pre_diagnosis.state_dict()['classifier.weight'].data,
                class_idx=[i for i in range(5)])  # B*K 注：3是类别数需要更换！
            # importance_std reshape ( [B, K]=> [B, 1, K])
            importance_std = torch.unsqueeze(
                importance_std,
                dim=1)

            output, emb1, (bag_kept, bag_drop) = aggregator(
                features,
                importances_batch=importance_std)  # B*C*K, B*1*K

            # dropout_consistency loss
            loss_dc = drop_consistency_loss(bag_kept, bag_drop, size_average=True) * 1

            # HSIC loss
            interval = int(emb1.shape[1] / 2)
            loss_HSIC = loss_dependence_batch(emb1[:, :interval, :], emb1[:, interval:, :]) * 1
            # print(loss_HSIC)
            # ref_diagnosis ==> Second Stage: Refined Fault Diagnosis
            logit_list = [output_pre, output]
            preds = []
            for i in range(y.shape[0]):
                pred = [logit[i][y[i]] for logit in logit_list]
                preds.append(pred)
            loss_rank = refine_ranking_loss(preds, margin=0.05, size_average=True) * 1
            # CE loss
            loss = criterion(output, y)

            # domain loss
            x_tgt, y_tgt = batch1
            data_tgt = x_tgt.to(device)
            y_tgt = y_tgt.to(device)
            data_tgt_shape = data_tgt.shape  # [B, L, H, W]
            input_tgt_batch = data_tgt.view(data_tgt_shape[0] * data_tgt_shape[1], 1, data_tgt_shape[2],
                                            data_tgt_shape[3])
            tgt_features = extractor(x=input_tgt_batch)  # [(B*L), 512]
            # features reshape
            tgt_features = tgt_features.view(data_tgt_shape[0], data_tgt_shape[1], -1)  # [B, L, 512]
            tgt_features = torch.transpose(tgt_features, 1, 2)  # [B, 512, Z]
            # pre_diagnosis ==> First Stage: Preliminary Fault Diagnosis
            output_pre = pre_diagnosis(tgt_features)
            _, importance_std = returnImportance(
                feature=tgt_features.clone().detach().data,
                weight_softmax=pre_diagnosis.state_dict()['classifier.weight'].data,
                class_idx=[i for i in range(5)])  # B*K 注：3是类别数需要更换！
            # importance_std reshape ( [B, K]=> [B, 1, K])
            importance_std = torch.unsqueeze(
                importance_std,
                dim=1)

            tgt_output, tgt_emb1, (tgt_bag_kept, tgt_bag_drop) = aggregator(
                tgt_features,
                importances_batch=importance_std)  # B*C*K, B*1*K
            domain_loss = criterion(tgt_output, y_tgt)

            ################## update param ##################
            optimizer_extractor.zero_grad()
            optimizer_aggregator.zero_grad()
            optimizer_aggregator_pre.zero_grad()
            train_loss += (loss + loss_pre + loss_dc + loss_HSIC + loss_rank + domain_loss).item()
            (loss + loss_pre + loss_dc + loss_HSIC + loss_rank+domain_loss).backward()
            optimizer_extractor.step()
            optimizer_aggregator.step()
            optimizer_aggregator_pre.step()
        # 每训练一个epoch,记录一次训练损失
        train_loss = train_loss / len(train_tgt_loader)
        train_loss_list.append(train_loss)
        _, train_acc = eval(extractor, pre_diagnosis, aggregator, train_tgt_loader)
        train_acc_list.append(train_acc)
        print("train Epoch:{},loss:{},train_acc:{}".format(epoch, train_loss, train_acc))
        # 每训练一个epoch,用当前训练的模型对验证集进行测试
        eval_loss, eval_acc = eval(extractor, pre_diagnosis, aggregator, test_tgt_loader)
        # 将每一个测试集验证的结果加入列表
        val_loss_list.append(eval_loss)
        val_acc_list.append(eval_acc)
        print("val Epoch:{},eval_loss:{},eval_acc:{}".format(epoch, eval_loss, eval_acc))
        if eval_acc > max_acc:
            max_acc = eval_acc
            # 保存最优模型参数
            torch.save(extractor, 'output/model/extractor_best.pt')
            torch.save(pre_diagnosis, 'output/model/pre_diagnosis_best.pt')
            torch.save(aggregator, 'output/model/aggregator_best.pt')
    torch.save(extractor, 'output/model/extractor_last.pt')  # 保存最后一个epoch的模型
    torch.save(pre_diagnosis, 'output/model/pre_diagnosis_last.pt')  # 保存最后一个epoch的模型
    torch.save(aggregator, 'output/model/aggregator_last.pt')  # 保存最后一个epoch的模型
    np.savetxt("output/train_loss_list.txt", train_loss_list)
    np.savetxt("output/train_acc_list.txt", train_acc_list)
    np.savetxt("output/val_loss_list.txt", val_loss_list)
    np.savetxt("output/val_acc_list.txt", val_acc_list)

    with open('output/train_loss_list.txt', 'r') as f:
        train_loss_list = f.readlines()
    train_loss = [float(i.strip()) for i in train_loss_list]
    with open('output/val_loss_list.txt', 'r') as f:
        val_loss_list = f.readlines()
    val_loss = [float(i.strip()) for i in val_loss_list]

    with open('output/train_acc_list.txt', 'r') as f:
        train_acc_list = f.readlines()
    train_acc = [float(i.strip()) for i in train_acc_list]
    with open('output/val_acc_list.txt', 'r') as f:
        val_acc_list = f.readlines()
    val_acc = [float(i.strip()) for i in val_acc_list]

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend()
    plt.savefig('result/loss curve.jpg')
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend()
    plt.savefig('result/accuracy curve.jpg')
    plt.show()
