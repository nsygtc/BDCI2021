import numpy as np
import pandas as pd
import pickle as pkl
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from features import get_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from catboost import CatBoostClassifier

"""
使用默认参数，获取多个分类器在验证集上的结果
"""


max_len = 200

data_train = np.array(pd.read_csv('data/train_and_test/data_train.csv'))
data_test = np.array(pd.read_csv('data/train_and_test/data_val.csv'))

# 按顺序建立cid索引，每200个一组，根据下标查询标签
idx_train = data_train[:,1].reshape(-1, max_len)
idx_test = data_test[:, 1].reshape(-1, max_len)

# 生成并保存特征
# X_train = get_features(data_train)
# X_test = get_features(data_test)
# with open('data/train_and_test/features_train.pkl', 'wb') as f:
#     pkl.dump(X_train, f)
# with open('data/train_and_test/features_val.pkl', 'wb') as f:
#     pkl.dump(X_test, f)

# 加载已保存特征，需要先生成
with open('data/train_and_test/features_train.pkl', 'rb') as f:
    X_train = pkl.load(f)
with open('data/train_and_test/features_val.pkl', 'rb') as f:
    X_test = pkl.load(f)

y_train = data_train[:, -1]
y_test = data_test[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# model = LGBMClassifier()
# model = XGBClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = ExtraTreeClassifier()
model = CatBoostClassifier()

y_t = y_train
y_v = y_test

model.fit(X_train, y_t)

y_train_pred_idx = model.predict_proba(X_train)[:, 1].reshape(-1, max_len)
y_train_pred = np.argmax(y_train_pred_idx, axis=-1) # 这里有问题

y_t = y_train.reshape(-1, max_len)
y_t = np.argmax(y_t, axis=-1)
z = np.sum(y_train_pred == y_t)
acc_t = np.sum(y_train_pred == y_t) / len(y_train_pred)
print('Training accuaracy:', acc_t)

y_test_pred_idx = model.predict_proba(X_test)[:, 1].reshape(-1, max_len)
y_val_pred = np.argmax(y_test_pred_idx, axis=-1)
y_v = y_v.reshape(-1, max_len)
y_v = np.argmax(y_v, axis=-1)
acc_v = np.sum(y_val_pred == y_v) / len(y_val_pred)
print('Test accuaracy:', acc_v)

