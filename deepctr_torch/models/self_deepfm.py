#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/20 上午11:58
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : self_deepfm.py.py

"""导入包"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

# pytorch
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchkeras import summary, Model
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

class FM(nn.Module):
    """
    FM part
    """
    def __init__(self, latent_dim, fea_num):
        """
        Args:
            latent_dim: 离散特征隐向量维度
            fea_num: 离散特征embeding之后的和dense拼接的总特征个数
        """
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        # 定义三个矩阵，全局偏置，一阶权重偏置，二阶交叉矩阵
        self.w0 = nn.Parameter(torch.zeros(1,))