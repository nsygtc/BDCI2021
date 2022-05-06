import json
from tqdm import tqdm
import numpy as np
import pandas as pd

time_json_device = 'data/processed/train/timearray_device.json'
time_json_person = 'data/processed/train/timearray_person.json'
fea_json_device = 'data/processed/train/features_device.json'
fea_json_people = 'data/processed/train/features_people.json'

with open(time_json_device, 'r') as f:
    data_device = json.load(f)['persons']

with open(time_json_person, 'r') as f:
    data_person = json.load(f)['persons']

with open(fea_json_device, 'r') as f:
    fea_device = json.load(f)['people']

with open(fea_json_people, 'r') as f:
    fea_person = json.load(f)['people']

p2records = {}
c2records = {}
pid2fea = {}
cid2fea = {}
pid2time = {}
cid2time = {}

for i, p in enumerate(data_person):
    p2records[p['id']] = p['arrays']

for i, c in enumerate(data_device):
    c2records[c['id']] = c['arrays']

for i, p in enumerate(fea_person):
    fea = []
    fea.extend(p['areacount'])
    fea.extend(p['areainterval'])
    fea.extend(p['peakcount'])
    fea.extend(p['peakinterval'])
    fea.extend(p['freqnext'])
    pid2fea[p['id']] = fea

for i, c in enumerate(fea_device):
    fea = []
    fea.extend(c['areacount'])
    fea.extend(c['areainterval'])
    fea.extend(c['peakcount'])
    fea.extend(c['peakinterval'])
    fea.extend(c['freqnext'])
    cid2fea[c['id']] = fea

for i, p in enumerate(fea_person):
    pid2time[p['id']] = p['arrays']

for i, c in enumerate(fea_device):
    cid2time[c['id']] = c['arrays']

def get_features_person(pid):
    """
    [平均时间, 最早时间， 最晚时间, 中位数时间]
    """
    # records = p2records[pid]
    records = pid2time[pid]
    t = []
    for i, r in enumerate(records):
        t.extend(r)

    time_global_mean = np.mean(t)
    time_global_min = np.min(t)
    time_global_max = np.max(t)
    time_global_media = np.median(t)

    return np.array([time_global_mean, time_global_min, time_global_max, time_global_media])


def get_features_device(cid):
    """
    全局[平均时间, 最早时间， 最晚时间, 中位数时间]
    """
    records = cid2time[cid]
    t = []
    for i, r in enumerate(records):
        t.extend(r)

    time_global_mean = np.mean(t)
    time_global_min = np.min(t)
    time_global_max = np.max(t)
    time_global_media = np.median(t)

    return np.array([time_global_mean, time_global_min, time_global_max, time_global_media])

def get_features_person2(pid):
    """
    第二阶段获取的特征
    """
    return np.array(pid2fea[pid])

def get_features_device2(cid):
    """
    第二阶段获取的特征
    """
    return np.array(cid2fea[cid])
def get_features_pair(pid, cid):
    """
    获取设备和人员的一些联合统计量
    """
    p_records = pid2time[pid]
    c_records = cid2time[cid]
    p_records_global = []
    c_records_global = []

    for i in range(5):
        p_records_global.extend(p_records[i])
        c_records_global.extend(c_records[i])

    # 五个区域，每个区域cid time 小于 pid 最小time前的计数，pid不存在的话记为0
    count_min = np.zeros(5)
    for i in range(5):
        if len(p_records[i]) == 0: continue
        min_time = min(p_records[i])
        for t in c_records[i]:
            if t < min_time: count_min[i] += 1
    # 五个区域，每个区域cid time 小于 pid 最大time前的计数，pid不存在的话记为0
    count_max = np.zeros(5)
    for i in range(5):
        if len(p_records[i]) == 0: continue
        max_time = max(p_records[i])
        for t in c_records[i]:
            if t < max_time: count_max[i] += 1

    # 五个区域，每个区域cid time 小于 pid 平均time前的计数，pid不存在的话记为0
    count_avg = np.zeros(5)
    for i in range(5):
        if len(p_records[i]) == 0: continue
        avg_time = np.mean(p_records[i])
        for t in c_records[i]:
            if t < avg_time: count_avg[i] += 1

    # 全局的平均, 最小，最大，中值计数
    count_global = np.zeros(4)


    g_avg_time = np.mean(p_records_global)
    for t in c_records_global:
        if t < g_avg_time: count_global[0] += 1

    g_min_time = min(p_records_global)
    for t in c_records_global:
        if t < g_min_time: count_global[1] += 1

    g_max_time = max(p_records_global)
    for t in c_records_global:
        if t < g_max_time: count_global[2] += 1

    g_mid_time = np.median(p_records_global)
    for t in c_records_global:
        if t < g_mid_time: count_global[3] += 1

    x = cat_features(count_min, count_max)
    x = cat_features(x, count_avg)
    x = cat_features(x, count_global)
    return x

def cat_features(x1, x2):
    return np.hstack((x1, x2))

def sub_features(x1, x2):
    return x1 - x2

def get_features(data):
    res = []
    print()
    print("Building features...\n")
    for i, d in enumerate(tqdm(data)):
        pid, cid, _ = d
        x_p = cat_features(get_features_person(pid), get_features_person2(pid))
        x_c = cat_features(get_features_device(cid), get_features_device2(cid))
        x = sub_features(x_p, x_c)
        x = cat_features(x, get_features_pair(pid, cid))
        res.append(x)

    return np.array(res)



if __name__ == '__main__':
    x1 = get_features_person(2)
    # x2 = get_features_person(665)
    # z = sub_features(x1, x2)
    # print(z.shape)
    get_features_pair(1066, 7097)