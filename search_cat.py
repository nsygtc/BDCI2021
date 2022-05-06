import numpy as np
import pandas as pd
import pickle as pkl
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from features import get_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

max_len = 200

data_train = np.array(pd.read_csv('data/train_and_test/data_train.csv'))
data_val = np.array(pd.read_csv('data/train_and_test/data_val.csv'))

# 按顺序建立cid索引，每200个一组，根据下标查询标签
idx_train = data_train[:,1].reshape(-1, 200)
idx_val = data_val[:,1].reshape(-1, 200)


# X_train = get_features(data_train)
# X_val = get_features(data_val)

y_train = data_train[:, -1]
y_val = data_val[:, -1]
with open('data/train_and_test/features_train.pkl', 'rb') as f:
    X_train = pkl.load(f)
with open('data/train_and_test/features_val.pkl', 'rb') as f:
    X_val = pkl.load(f)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# 参数选择

tuned_parameters = {'learning_rate': [0.3, 0.1, 0.05, 0.03, 0.01],
                     'max_depth': [3, 5, 8],
                    }

best_parameters = {'learning_rate': 0.3,  'max_depth': -1}
best_acc = 0

for lr in tuned_parameters['learning_rate']:
    for md in tuned_parameters['max_depth']:
        print()
        print('learning_rate:', lr, 'max_depth',md)

        model = CatBoostClassifier(
            eval_metric='AUC',
            loss_function='Logloss',
            learning_rate=lr,
            depth=md
        )
        y_t = y_train
        y_v = y_val

        model.fit(X_train, y_t)

        y_train_pred_idx = model.predict_proba(X_train)[:, 1].reshape(-1, 200)
        y_train_pred = np.argmax(y_train_pred_idx, axis=-1) # 这里有问题

        y_t = y_train.reshape(-1, 200)
        y_t = np.argmax(y_t, axis=-1)
        z = np.sum(y_train_pred == y_t)
        acc_t = np.sum(y_train_pred == y_t) / len(y_train_pred)
        print('Training accuaracy:', acc_t)

        y_val_pred_idx = model.predict_proba(X_val)[:, 1].reshape(-1, 200)
        y_val_pred = np.argmax(y_val_pred_idx, axis=-1)
        y_v = y_v.reshape(-1, 200)
        y_v = np.argmax(y_v, axis=-1)
        acc_v = np.sum(y_val_pred == y_v) / len(y_val_pred)
        print('Validation accuaracy:', acc_v)

        if acc_v > best_acc:
            best_acc = acc_v
            best_parameters['learning_rate'] = lr
            best_parameters['max_depth'] = md

print()
print(best_parameters)
print(best_acc)

