#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time: 2023/12/30 16:01
# @Author: Yue Yu
# @School: Politecnico di Milano
# @Email: yyu41474@gmail.com
# @Filmname: losses.py
# @Software: PyCharm
# @Theme: Fault Diagnosis
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import LabelSmoothingCrossEntropy


# common_loss
def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


# loss_dependence
def loss_dependence(emb1, emb2):
    """
    emb: [B, C, K]
    """
    emb_shape = emb1.shape
    R = torch.eye(emb_shape[-1]).cuda() - (1 / emb_shape[-1]) * torch.ones(emb_shape[-1], emb_shape[-1]).cuda()
    HSIC_list = []
    for b in range(emb_shape[0]):
        K1 = torch.mm(emb1[b].t(), emb1[b])  # [K, K]
        K2 = torch.mm(emb2[b].t(), emb2[b])
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        HSIC = torch.trace(torch.mm(RK1, RK2))
        HSIC_list.append(HSIC)
    loss = torch.sum(
        torch.stack(HSIC_list, dim=0),
        dim=0,
        keepdim=False) / emb_shape[0]
    return loss


# loss_dependence_batch
def loss_dependence_batch(emb1, emb2):
    """
    emb: [B, C, K]
    """
    emb_shape = emb1.shape
    R = torch.stack([
                        torch.eye(emb_shape[-1]).cuda() - (1 / emb_shape[-1]) * torch.ones(emb_shape[-1],
                                                                                           emb_shape[-1]).cuda()
                    ] * emb_shape[0],
                    dim=0)  # [B, K, K]
    K1 = torch.bmm(torch.transpose(emb1, 1, 2), emb1)  # [B, K, K]
    K2 = torch.bmm(torch.transpose(emb2, 1, 2), emb2)  # [B, K, K]
    RK1 = torch.bmm(R, K1)  # [B, K, K]
    RK2 = torch.bmm(R, K2)
    HSIC_list = []
    for b in range(emb_shape[0]):
        HSIC = torch.trace(torch.mm(RK1[b], RK2[b]))
        HSIC_list.append(HSIC)
    loss = torch.sum(
        torch.stack(HSIC_list, dim=0),
        dim=0,
        keepdim=False) / emb_shape[0]
    return loss


# refine_ranking_loss
def refine_ranking_loss(preds, margin=0.05, size_average=True):
    """
        preds:
            list of scalar Tensor.
            Each value represent the probablity of each class
                e.g) class = 3
                    preds = [logits1[class], logits2[class]]
    """
    """
    # usage:
        label_batch = Variable(label).long().cuda()
        # rank loss
        logit_list = [output_low, output_super, output_aux, output_ensemble]
        preds = []
        for i in range(label_batch.shape[0]): 
            pred = [logit[i][label_batch[i]] for logit in logit_list]
            preds.append(pred)
        loss_rank = rank_loss.pairwise_ranking_loss(preds)

    """
    if len(preds) <= 1:
        return torch.zeros(1).cuda()
    else:
        losses = []
        for pred in preds:  # batch_size
            loss = []  # preliminary, refine
            loss.append((pred[0] - pred[1] + margin).clamp(min=0))  # preliminary -> refine
            loss = torch.sum(torch.stack(loss))
            losses.append(loss)
        losses = torch.stack(losses)
        if size_average:
            losses = torch.mean(losses)
        else:
            losses = torch.sum(losses)
        return losses


# drop_consistency_loss
def drop_consistency_loss(bag1, bag2, size_average=True):
    """
    bag: [B, C]
    """
    loss = torch.sum(
        torch.abs(
            F.normalize(input=bag1, p=2, dim=-1) - F.normalize(input=bag2, p=2, dim=-1)
        ),
        dim=-1,
        keepdim=False
    )
    if size_average:
        loss = torch.mean(loss, dim=0, keepdim=False)
    else:
        loss = torch.sum(loss, dim=0, keepdim=False)
    return loss


# classification loss
criterion = LabelSmoothingCrossEntropy()
