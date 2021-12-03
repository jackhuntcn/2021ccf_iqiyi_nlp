import warnings
warnings.simplefilter('ignore')

import os
import gc
import json
import glob
import pickle

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', 200)
pd.set_option('float_format', lambda x: '%.3f' % x)
from tqdm.notebook import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

import cuml
from cuml.svm import SVR
print('RAPIDS version', cuml.__version__)


feature_names = [f'emb{i}' for i in range(768)]

with open('../input/aqy-embs/folds.json') as f:
    kfolds = json.load(f)

train = pd.read_csv('../input/aqy-embs/train_768embs.csv')
test = pd.read_csv('../input/aqy-embs/test_768embs.csv')

print(train.shape, test.shape)


def run_svr(ycol, fold):
    X_train = train.iloc[kfolds[f'fold_{fold}']['train']][feature_names]
    Y_train = train.iloc[kfolds[f'fold_{fold}']['train']][ycol].astype(np.float32)

    X_valid = train.iloc[kfolds[f'fold_{fold}']['valid']][feature_names]
    Y_valid = train.iloc[kfolds[f'fold_{fold}']['valid']][ycol].astype(np.float32)
    
    clf = SVR(C=20.0)
    clf.fit(X_train, Y_train)
    
    pred_valid = clf.predict(X_valid)
    valid_rmse = mean_squared_error(Y_valid, pred_valid, squared=True)
    
    pred = clf.predict(test[feature_names])
    
    return pred, valid_rmse


for ycol in [f'label_{i}' for i in range(6)]:
    res = np.zeros((len(test), ))
    for fold in range(10):
        pred, valid_rmse = run_svr(ycol, fold)
        print(ycol, 'fold:', fold, 'valid rmse:', valid_rmse)
        res += pred / 10
    res_df = pd.DataFrame({'id': test['id'].values, ycol: res})
    res_df.to_pickle(f'test_{ycol}.pickle')

data = pd.read_pickle('test_label_0.pickle')
for i in range(1, 6):
    tmp = pd.read_pickle(f'test_label_{i}.pickle')
    data = pd.merge(data, tmp, on='id')
print(data.shape)
print(data.head())

print(data[[f'label_{i}' for i in range(6)]].describe())
data.to_csv('emb_svr_result.csv', index=False)


