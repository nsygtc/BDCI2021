import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import xgboost

max_len = 200

data_train = np.array(pd.read_csv('data/train_and_test/data_train.csv'))
data_test = np.array(pd.read_csv('data/train_and_test/data_test.csv'))

idx_train = data_train[:,1].reshape(-1, 200)
idx_test = data_test[:, 1].reshape(-1, 200)

y_train = data_train[:, -1]
y_test = data_test[:, -1]

with open('data/train_and_test/features_train.pkl', 'rb') as f:
    X_train = pkl.load(f)
with open('data/train_and_test/features_test.pkl', 'rb') as f:
    X_test = pkl.load(f)

groups_train = np.ones(len(y_train) // 200) * 200
groups_test = np.ones(len(y_test) // 200) * 200

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = xgboost.XGBRanker()


model.fit(
    X=X_train,
    y=y_train,
    group=groups_train,
    eval_set=[(X_test, y_test)],
    eval_group=[groups_test],
    # eval_at=[1, 3, 5, 10, 100, 200],
    eval_metric=['ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@100', 'ndcg@200'],
    verbose=10
)

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

