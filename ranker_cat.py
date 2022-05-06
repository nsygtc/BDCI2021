import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRanker, Pool

max_len = 200

data_train = np.array(pd.read_csv('data/train_and_test/data_train.csv'))
data_test = np.array(pd.read_csv('data/train_and_test/data_test.csv'))

# 按顺序建立cid索引，每200个一组，根据下标查询标签
idx_train = data_train[:,1].reshape(-1, max_len)
idx_test = data_test[:, 1].reshape(-1, max_len)

y_train = data_train[:, -1]
y_test = data_test[:, -1]

with open('data/train_and_test/features_train.pkl', 'rb') as f:
    X_train = pkl.load(f)
with open('data/train_and_test/features_test.pkl', 'rb') as f:
    X_test = pkl.load(f)

groups_train = []
groups_test = []

for i in range(len(y_train)):
    groups_train.append(i // max_len + 1)

for i in range(len(y_test)):
    groups_test.append(i // max_len + 1)

groups_train = np.array(groups_train)
groups_test = np.array(groups_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train = Pool(
    data=X_train,
    label=y_train,
    group_id=groups_train
)

test = Pool(
    data=X_test,
    label=y_test,
    group_id=groups_test
)

# catboost的结果保存在catboost_info里
default_parameters = {
    'custom_metric': ['NDCG:top=1', 'NDCG:top=3','NDCG:top=5','NDCG:top=10','NDCG:top=100', 'NDCG:top=200']
}

model = CatBoostRanker(
    **default_parameters
)

model.fit(
    train,
    eval_set=test,
    verbose_eval=True,
    metric_period=10,
)

y_pred = model.predict(X_test).reshape(-1, 200)

for k in [1, 3, 5, 10, 100, 200]:

    y_pred = model.predict(X_test).reshape(-1, 200)
    y_s = np.argsort(y_pred, axis=-1)[:,-k:]
    y_true = y_test.reshape(-1, 200)

    tp = 0
    for i, y_true in enumerate(y_true):
        if np.all(y_true == 0): continue
        y = np.argmax(y_true)
        if y in y_s[i]:
            tp += 1

    print('k = ', k, tp / 200)

