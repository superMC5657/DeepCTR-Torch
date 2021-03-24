#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/24 上午10:56
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : self_nfm.py.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/23 下午9:03
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : self_NFM.py
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchkeras import summary, Model
class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        """
        Args:
            hidden_units: 列表，每一层的神经元个数
            dropout:
        """
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x

class self_NFM(nn.Module):
    def __init__(self, feature_columns, hidden_units, dnn_dropout=0.):
        """
        Args:
            feature_columns: 特征信息
            hidden_units: dnn隐藏层
            dnn_dropout:
        """
        super(self_NFM, self).__init__()
        self.densen_feature_cols, self.sparse_feature_cols = feature_columns
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        self.fea_num = len(self.densen_feature_cols) + self.sparse_feature_cols[0]['embed_dim']
        hidden_units.insert(0, self.fea_num)

        self.bn = nn.BatchNorm1d(self.fea_num)
        self.dnn_network = DNN(hidden_units, dnn_dropout)
        self.nn_final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :len(self.densen_feature_cols)], x[:, len(self.densen_feature_cols):]
        sparse_inputs = sparse_inputs.long() # 转成long类型才能作为nn.embedding的输入
        sparse_embeds = [self.embed_layers['embed_' + str(i)]
                         (sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds)
        sparse_embeds = sparse_embeds.permute((1, 0, 2))
        embed_cross = 1/2 * (
            torch.pow(torch.sum(sparse_embeds, dim=1), 2) - torch.sum(torch.pow(sparse_embeds, 2), dim=1)
        )
        x = torch.cat([embed_cross, dense_inputs], dim=-1)
        x = self.bn(x)
        dnn_outputs = self.nn_final_linear(self.dnn_network(x))
        outputs = F.sigmoid(dnn_outputs)
        return outputs

