import numpy as np
import pandas as pd
import pickle as pkl
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from features import get_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

"""
测试LightGBM, XGBoost, CatBoost在测试集上的结果
"""
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


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


best_parameters = {'learning_rate': 0.08, 'n_estimators': 300, 'max_depth': 5}
#
model = LGBMClassifier(
    objective='binary',
    metric='auc',
    learning_rate=0.08,
    n_estimators=300,
    boosting="gbdt",
    max_depth=5,
    num_threads = 6,
    verbose=-1
)

# model = XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',
#     learning_rate=0.3,
#     n_estimators=300,
#     max_depth=3,
#     n_jobs=6
# )
# model = CatBoostClassifier(
#     # eval_metric='AUC',
#     # loss_function='Logloss',
#     # learning_rate=0.1,
#     # depth=5
# )

y_t = y_train
y_v = y_test

model.fit(X_train, y_t)

y_train_pred_idx = model.predict_proba(X_train)[:, 1].reshape(-1, 200)
y_train_pred = np.argmax(y_train_pred_idx, axis=-1) # 这里有问题

y_t = y_train.reshape(-1, 200)
y_t = np.argmax(y_t, axis=-1)
z = np.sum(y_train_pred == y_t)
acc_t = np.sum(y_train_pred == y_t) / len(y_train_pred)
print('Training accuaracy:', acc_t)

y_test_pred_idx = model.predict_proba(X_test)[:, 1].reshape(-1, 200)
y_val_pred = np.argmax(y_test_pred_idx, axis=-1)
y_v = y_v.reshape(-1, 200)
y_v = np.argmax(y_v, axis=-1)
acc_v = np.sum(y_val_pred == y_v) / len(y_val_pred)
print('Test accuaracy:', acc_v)

