import numpy as np
import pandas as pd
import json
from bidict import bidict
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

"""
    分析人对应的true postive的设备时间间隔分布情况，并根据这个进行预排序，构建候选集
"""
seed = 0

areas = [0, 1, 2, 3, 4]
f_json_device = 'data/processed/train/timearray_device.json'
f_json_person = 'data/processed/train/timearray_person.json'
f_csv_label = 'data/processed/train/preprocessed_label.csv'
f_csv_person = 'data/processed/train/preprocessed_person.csv'
f_csv_device = 'data/processed/train/preprocessed_device.csv'

with open(f_json_device, 'r') as f:
    data_device = json.load(f)['persons']

with open(f_json_person, 'r') as f:
    data_person = json.load(f)['persons']

labels = pd.read_csv(f_csv_label)
csv_person = pd.read_csv(f_csv_person)
csv_device = pd.read_csv(f_csv_device)

id2label = {}
p2records = {}
c2records = {}

for i, row in labels.iterrows():
    id2label[row['Person']] = row['Device']

for i, p in enumerate(data_person):
    p2records[p['id']] = p['arrays']

for i, c in enumerate(data_device):
    c2records[c['id']] = c['arrays']

# 根据person来找device
global_min_time_gap = []
for i, p in enumerate(data_person):
    p_id = p['id']
    c_id = id2label[p_id]
    p_records = p['arrays']
    c_records = c2records[c_id]

    min_time_gap_cross_areas = 10000

    for area in areas:
        # 当前area的记录
        # print('Area :', area)
        p_record = []
        c_record = []
        for pr in p_records:
            if pr['area'] == area:
                p_record = pr['array']
                break
        for cr in c_records:
            if cr['area'] == area:
                c_record = cr['array']
                break
        # print(p_record, '->', c_record)
        # 找person - device最近的时间差
        min_time_gap = 10000
        for p_time in p_record:
            for c_time in c_record:
                if c_time < p_time:
                    min_time_gap = min(min_time_gap, p_time - c_time)

        if min_time_gap != 10000:
            min_time_gap_cross_areas = min(min_time_gap_cross_areas, min_time_gap)

    global_min_time_gap.append(min_time_gap_cross_areas)

s = sorted(global_min_time_gap, reverse=True)
dfs = pd.Series(s)
length_counts = dfs.value_counts()

plt.figure(dpi=300, figsize=(12, 4))
plt.boxplot(global_min_time_gap, vert=False)
plt.xlim((0, 100))
plt.xlabel('Delta time')
plt.savefig('length_box.png')
plt.show()

plt.figure(dpi=300)
plt.bar(length_counts.index, length_counts.values)
plt.xlim((0, 100))
plt.xlabel('Delta time')
plt.ylabel('Number')
plt.savefig('length_bar.png')
plt.show()

"""
构建候选集
"""

threshold_time = 60
max_len = 200
devices =  list(c2records.keys())

ht = 0
res_p = []
res_c = []
res_labels = []

for i, p in enumerate(tqdm(data_person)):
    p_id = p['id']

    # candidate_device = {}
    cand_cid = {}
    cand_hits = []
    cand_time = []

    target_person = csv_person[csv_person['Person'] == p_id]

    tag = csv_device['Timestamp'] < 0

    for j, p in target_person.iterrows():

        tag = tag | (csv_device['Timestamp'] <= p['Timestamp']) & (csv_device['Timestamp'] >= p['Timestamp'] - threshold_time)

    target_divice = csv_device[tag]
    # print(target_divice.shape)
    # 计算hit和时间差
    arr_ctimestap = np.array(target_divice['Timestamp'])
    arr_cid = np.array(target_divice['Device'])
    arr_ptimestep = np.array(target_person['Timestamp'])

    for j, c_id in enumerate(arr_cid):
        c_time = arr_ctimestap[j]

        tt_person = arr_ptimestep[(arr_ptimestep >= c_time) & (arr_ptimestep <= c_time + threshold_time)]
        min_time_gap = threshold_time
        for ptime in tt_person:
            min_time_gap = min(min_time_gap, ptime - c_time)

        if c_id in cand_cid.keys():
            idx = cand_cid[c_id]
            cand_hits[idx] += 1
            cand_time[idx] += min_time_gap
        else:
            idx = len(cand_hits)
            cand_cid[c_id] = idx
            cand_hits.append(1)
            cand_time.append(min_time_gap)

    for k in range(len(cand_time)):
        cand_time[k] = cand_time[k] / cand_hits[k]

    candidate_device = pd.DataFrame(data={'Hits': cand_hits, 'Time': cand_time}, index=cand_cid.keys())
    sorted_candidate_device = candidate_device.sort_values(by=['Hits', 'Time'], ascending=(False, True))
    target_cindex = np.array(sorted_candidate_device.index)[: max_len]
    target_labels = np.zeros_like(target_cindex)
    target_labels[target_cindex == id2label[p_id]] = 1

    # 检查目标ID是否在候选集中
    target_pindex = [p_id] * max_len
    res_c.extend(target_cindex.tolist())
    res_p.extend(target_pindex)
    res_labels.extend(target_labels.tolist())

    if id2label[p_id] in target_cindex:
        ht += 1

print('召回率', ht / len(data_person))

# 按照person, 划分训练集和验证集和测试集, 8:1:1
labels = np.array(labels)
y_train, y_test = train_test_split(labels, test_size=0.2, random_state=seed, shuffle=True)
y_test, y_val = train_test_split(y_test, test_size=0.5, random_state=seed, shuffle=True)

data_train = []
data_val = []
data_test = []
for i in range(len(res_p)):
    p = res_p[i]
    c = res_c[i]
    y = res_labels[i]
    if p in y_train[:, 0]:
        data_train.append([p, c, y])
    elif p in y_val[:, 0]:
        data_val.append([p, c, y])
    else:
        data_test.append([p, c, y])

data_train = np.array(data_train)
data_val = np.array(data_val)
data_test = np.array(data_test)

pd.DataFrame(columns=['pid', 'cid', 'label'], data=data_train).to_csv('data/train_and_test/data_train.csv', index=False)
pd.DataFrame(columns=['pid', 'cid', 'label'], data=data_val).to_csv('data/train_and_test/data_val.csv', index=False)
pd.DataFrame(columns=['pid', 'cid', 'label'], data=data_test).to_csv('data/train_and_test/data_test.csv', index=False)
pd.DataFrame(columns=['pid', 'cid'], data=y_train).to_csv('data/train_and_test/y_train.csv', index=False)
pd.DataFrame(columns=['pid', 'cid'], data=y_val).to_csv('data/train_and_test/y_val.csv', index=False)
pd.DataFrame(columns=['pid', 'cid'], data=y_test).to_csv('data/train_and_test/y_test.csv', index=False)

