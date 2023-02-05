#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/27 上午11:26
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : self_din.py.py

import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, att_hidden_units):
        super(Attention, self).__init__()
        self.attation_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(att_hidden_units[:-1], att_hidden_units[1:]))]
        )
    def forward(self, X):
