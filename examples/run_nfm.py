#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/24 上午10:52
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : run_nfm.py.py
import torch
import datetime
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torchkeras import summary, Model
from torch.utils.data import DataLoader, Dataset, TensorDataset
from deepctr_torch.models.self_nfm import self_NFM
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def get_xy_fd():
    # 读入训练集，验证集和测试集
    file_path = "./data/"
    train = pd.read_csv(file_path + 'train_set.csv')
    val = pd.read_csv(file_path + 'val_set.csv')
    test = pd.read_csv(file_path + 'test_set.csv')

    trn_x, trn_y = train.drop(columns='Label').values, train['Label'].values
    val_x, val_y = val.drop(columns='Label').values, val['Label'].values
    test_x = test.values

    fea_col = np.load(file_path + 'fea_col.npy', allow_pickle=True)
    dl_train_dataset = TensorDataset(torch.tensor(trn_x).float(), torch.tensor(trn_y).float())
    dl_val_dataset = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float())

    #dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=32)
    #dl_val = DataLoader(dl_val_dataset, shuffle=True, batch_size=32)
    return fea_col, (trn_x, trn_y), (val_x, val_y), test_x


def data_pipeline(trn_x, trn_y, val_x, val_y, batch_size=32):
    dl_train_dataset = TensorDataset(torch.tensor(trn_x).float(), torch.tensor(trn_y).float())
    dl_val_dataset = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float())
    dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=batch_size)
    dl_val = DataLoader(dl_val_dataset, shuffle=True, batch_size=batch_size)
    return dl_train, dl_val


def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)


def train_nfm(model, train_iter, val_iter, loss_func=None, optimizer=None, metric_name=None, metric_func=None,
              num_epochs=10, log_step_freq=10):
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])
    print('start_training.........')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('========' * 8 + '%s' % nowtime)
    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(train_iter, 1):
            # 梯度清零
            optimizer.zero_grad()
            # 正向传播
            predictions = model(features);
            loss = loss_func(predictions, labels)
            try:
                metric = metric_func(predictions, labels)
            except ValueError:
                pass
            # 反向传播
            loss.backward()
            optimizer.step()

            # 打印batch日志
            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step));
        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(val_iter, 1):
            with torch.no_grad():
                predictions = model(features)
                val_loss = loss_func(predictions, labels)
                try:
                    val_metric = metric_func(predictions, labels)
                except ValueError:
                    pass

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()
        # 记录日志
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印日志
        print((
                      "\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f, val_loss=%.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('\n' + '==========' * 8 + '%s' % nowtime)

    print('Finished Training')
    print(dfhistory)
    return dfhistory


if __name__ == "__main__":
    # 生成迭代数据
    fea_col, (trn_x, trn_y), (val_x, val_y), test_x = get_xy_fd()
    hidden_units = [256, 128, 64]
    dnn_dropout = 0.
    model = self_NFM(fea_col, hidden_units, dnn_dropout)
    # summary(model, input_shape=(trn_x.shape[1],))
    # model paras
    summary(model, input_shape=(trn_x.shape[1],))
    batch_size = 32
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0001)
    train, val = data_pipeline(trn_x, trn_y, val_x, val_y)
    # train set
    dfhistory = train_nfm(model, train, val, loss_func=loss_func, optimizer=optimizer, metric_name="auc", metric_func=auc,
              num_epochs=25)


    def plot_metric(dfhistory, metric):
        train_metrics = dfhistory[metric]
        val_metrics = dfhistory['val_' + metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'b--')
        plt.plot(epochs, val_metrics, 'r:')
        plt.title('Training and validation ' + metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_" + metric, 'val_' + metric])
        plt.show()


    # 观察损失和准确率的变化
    plot_metric(dfhistory, "loss")
    plot_metric(dfhistory, "auc")