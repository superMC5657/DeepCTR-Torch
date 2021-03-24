# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


def read_data(filename, test=False):
    data = pd.read_csv(filename)

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['Label']
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2, random_state=2020)
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    model_input = {name: data[name] for name in feature_names}
    if test:
        return model_input, linear_feature_columns, dnn_feature_columns
    model_output = data[target].values
    return model_input, model_output, linear_feature_columns, dnn_feature_columns


if __name__ == "__main__":
    train_input, train_output, linear_feature_columns, dnn_feature_columns = read_data(filename='./data/train_set.csv')
    val_input, val_output, *_ = read_data(filename='./data/val_set.csv')
    test_input, test_output, *_ = read_data(filename='./data/test_set.csv', test=True)
    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary', dnn_dropout=0.2,
                   l2_reg_embedding=1e-4, device=device)

    model.compile("sgd", "binary_crossentropy", 0.01,
                  metrics=["binary_crossentropy", "auc"], )

    train_history = model.fit(train_input, train_output, batch_size=48, epochs=20, verbose=2, validation_split=0.2)
    # val_history = model.evaluate(val_input, val_output, 256)
    # print(val_history)
    val_ans = model.predict(val_input, 256)
    print("")
    print("val LogLoss", round(log_loss(val_output, val_ans), 4))
    print("val AUC", round(roc_auc_score(val_output, val_ans), 4))
