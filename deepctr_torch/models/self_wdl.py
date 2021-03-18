#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/17 下午2:58
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : self_wdl.py.py
# Reference:
#    [1] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems[C]//Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 2016: 7-10.(https://arxiv.org/pdf/1606.07792.pdf)
import torch.nn as nn
import torch.nn.functional as F
import torch


# from .basemodel import BaseModel
# from ..inputs import combined_dnn_input
# from ..layers import DNN
class Linear(nn.Module):
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, X):
        return self.linear(X)


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))]
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X):
        for linear in self.dnn_network:
            X = linear(X)
            X = F.relu(X)

        X = self.dropout(X)
        return X


class self_WDL(nn.Module):
    def __init__(self, feature_colums, hidden_units, dnn_dropout=0.):
        super(self_WDL, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_colums
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
                                'embed_dim'])
        self.dnn_network = DNN(hidden_units, dropout=dnn_dropout)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        dense_input, sparse_inputs = X[:, :len(self.dense_feature_cols)], X[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)
        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = torch.sigmoid(0.5 * (wide_out + deep_out))

        return outputs

    # class self_WDL(nn.Module):
    """Instantiates the Wide&Deep Learning architecture.
    :param linear_feature_columns: 包含所有用于线性模型部分的迭代器
    :param dnn_feature_columns:    包含所有用于深度网络部分的迭代器
    :param dnn_hidden_units:       一个包含正整数或为空的列表，为深度神经网络隐藏层的层数和宽度
    :param l2_reg_linear: float. L2 regularizer strength applied to wide part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :return: A PyTorch model instance.

    """


if __name__ == "__main__":
    net = self_WDL([256, 128, 64, 32, 16, 8])
    print(net)
