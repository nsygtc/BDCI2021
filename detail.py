'''
detail.py - 第二阶段特征工程

请将此文件放置在`/DATASET_ROOT/train_dataset`下，以便其正常工作。本脚本是在阶段一的基础上进行的，请在执行本脚本前先执行阶段一特征工程文件(`preprocess.py`)。

这个阶段我们总结一些深度学习是会用得上的特征向量。第一阶段最重要的时间向量特征是变长的，可能需要一些继续序列的模型才能处理，而这个阶段产生的特征向量是定长的。

这一阶段只做了训练集的内容，没有做验证集的内容，原因在于验证集没有标签信息，实际上不能用于测试；同时测试集和训练集点位的个数不同。

特征向量的每一维都有特定的语义。我们总结了如下这些特征，含识别码或人员的ID：

| 特征           | 值类型       | 维数  | CSV列起止 | JSON键名     | 说明                                                                         |
| -------------- | ------------ | ----- | --------- | ------------ | ------------------------------------------------------------ |
| ID             | 整数         | 1     | 0-0       | id           | ID不一定是连续的，请注意。                                        |
| 时间向量       | 变长整数序列 | (5*n) | -         | array        | 即第一阶段中给出的时间向量，不出现在CSV中。对每一个点位单独记录向量，故5个。 |
| 出现次数       | 整数         | 5     | 1-5       | areacount    | 在五个点位中采集到的时间点数量。                                     |
| 出现间隔       | 浮点数       | 5     | 6-10      | areainterval | 在*每个点位中*，两次出现的平均间隔。如果出现次数不足2次，记录-1。         |
| 高峰期出现次数 | 整数         | 3     | 11-13     | peakcount    | 三个高峰期（见下文）中，该人员/识别码出现的次数，每一维对应一个高峰期。     |
| 高峰期出现间隔 | 浮点数       | 3     | 14-16     | peakinterval | 三个高峰期中，该人员/识别码的平均出现间隔。出现次数不足两次的记为-1。       |
| 最常见下一跳   | {0,1,2,3,4}  | 5     | 17-21     | freqnext     | 人员/识别码在某一位点出现后，下一次出现的最常见位点。对每一个位点记录一维。 |
| 总维数        |              |   21   | 0-21     |              | 总维数的计算包含ID，不含ID则为20维。                                |

五维的特征都是按照点位0到点位4的顺序排列的。

### 关于高峰期

人员和识别码的出现分布存在三个主要的峰值区间，分布在5000s之前、15000s-17000s左右，以及37000s左右，猜测是三餐的时间。（详细的分析参见数据可视化部分）

因此，我们取以下三个时段，作为三个高峰期进行重点分析：1)[2000,5000];2)[14000,18000];3)[35000,39000]。

凡是和高峰期相关的数据，三个高峰期都按上文所述顺序，即时间先后顺序排列。

### 关于最常见下一跳

这里用于联系不同的点位。例如某个识别码的最常见下一跳特征是`[1,3,0,0,1]`，这说明此人：

- 在0号位点出现后，下一次最有可能出现的位点是1号位点；
- 在1号位点出现后，下一次最有可能出现的位点是3号位点；
- 在2号位点出现后，下一次最有可能出现的位点是0号位点；
- 在3号位点出现后，下一次最有可能出现的位点是0号位点；
- 在4号位点出现后，下一次最有可能出现的位点是1号位点。

这里的最有可能等等说法实际上指在数据集里面这样的记录占比最大。不过，目前并没有记录到这些位点的转移概率。

关于代码中使用的原数据，参见阶段一特征工程。
'''

import csv
import json
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Any, Mapping, List

devrec_name = "preprocessed_device.csv"
pplrec_name = "preprocessed_person.csv"
devarr_name = "timearray_device.json"
pplarr_name = "timearray_person.json"


def get_all_times(fname):
    lines = csv.reader(open(fname))
    next(lines)
    return sorted([int(t[1]) for t in lines])


def timegraph():
    INTERVAL = 20
    times_d = get_all_times(devrec_name)
    times_p = get_all_times(pplrec_name)
    latest = max(times_d[-1], times_p[-1])
    SLOTS = (latest // INTERVAL) + 1
    y_d = [0] * SLOTS
    y_p = [0] * SLOTS
    for t in times_d:
        y_d[t // INTERVAL] += 1
    for t in times_p:
        y_p[t // INTERVAL] += 1
    plt.figure()
    plt.subplot(211)
    plt.title("Device")
    plt.plot([i * INTERVAL for i in range(SLOTS)], y_d)
    plt.subplot(212)
    plt.title("People")
    plt.plot([i * INTERVAL for i in range(SLOTS)], y_p)
    plt.show()


def maxpos(seq):
    pos = 0
    for i in range(len(seq)):
        if seq[i] > seq[pos]:
            pos = i
    return pos


def jsonize(obj: Mapping[Any, Mapping], keyname: str = "id") -> List[Mapping]:
    lst = []
    for key in obj:
        newobj = deepcopy(obj[key])
        newobj[keyname] = key
        lst.append(newobj)
    return lst


def generate_feature_vecs():
    # 读取son
    with open(devarr_name) as fdev:
        obj = json.load(fdev)
    arrays_dev: List[Mapping] = obj["people"]
    with open(pplarr_name) as fppl:
        obj = json.load(fppl)
    arrays_ppl: List[Mapping] = obj["people"]

    # 先把人员和识别码的时间数组搬过来，之后
    # 统计人员和识别码在四个区域的出现次数
    feature_keywords = [
        "areacount",
        "areainterval",
        "peakcount",
        "peakinterval",
        "freqnext",
    ]
    features_ppl: Mapping[int, Mapping] = {}
    for item in arrays_ppl:
        pid = item["id"]
        arrays = [[], [], [], [], []]
        for arr in item["arrays"]:
            arrays[arr["area"]] = arr["array"]
        features_ppl[pid] = {
            "arrays": arrays,
            "areacount": [len(arr) for arr in arrays],
        }
    features_dev = {}
    for item in arrays_dev:
        did = item["id"]
        arrays = [[], [], [], [], []]
        for arr in item["arrays"]:
            arrays[arr["area"]] = arr["array"]
        features_dev[did] = {
            "arrays": arrays,
            "areacount": [len(arr) for arr in arrays],
        }
    ppl_list = list(features_ppl)
    dev_list = list(features_dev)

    # 统计在5个区域出现的平均时间间隔
    # 如果出现次数少于两次，填缺省值-1
    for pid in ppl_list:
        intervals = [-1] * 5
        for area, array in enumerate(features_ppl[pid]["arrays"]):
            if len(array) >= 2:
                intervals[area] = (array[-1] - array[0]) / (len(array) - 1)
        features_ppl[pid]["areainterval"] = intervals
    for did in dev_list:
        intervals = [-1] * 5
        for area, array in enumerate(features_dev[did]["arrays"]):
            if len(array) >= 2:
                intervals[area] = (array[-1] - array[0]) / (len(array) - 1)
        features_dev[did]["areainterval"] = intervals

    # 统计三个高峰时段的出现次数和平均时间间隔
    peaks = [(2000, 5000), (14000, 18000), (35000, 39000)]
    for pid in ppl_list:
        peak_showcount = []
        peak_interval = []
        for start, end in peaks:
            peak_showups = []
            for area in range(5):
                peak_showups += [
                    t for t in features_ppl[pid]["arrays"][area] if start <= t <= end
                ]
            peak_showcount.append(len(peak_showups))
            if len(peak_showups) >= 2:
                peak_interval.append(
                    (max(peak_showups) - min(peak_showups)) / (len(peak_showups) - 1)
                )
            else:
                peak_interval.append(-1)
        features_ppl[pid]["peakcount"] = peak_showcount
        features_ppl[pid]["peakinterval"] = peak_interval
    for did in dev_list:
        peak_showcount = []
        peak_interval = []
        for start, end in peaks:
            peak_showups = []
            for area in range(5):
                peak_showups += [
                    t for t in features_dev[did]["arrays"][area] if start <= t <= end
                ]
            peak_showcount.append(len(peak_showups))
            if len(peak_showups) >= 2:
                peak_interval.append(
                    (max(peak_showups) - min(peak_showups)) / (len(peak_showups) - 1)
                )
            else:
                peak_interval.append(-1)
        features_dev[did]["peakcount"] = peak_showcount
        features_dev[did]["peakinterval"] = peak_interval

    # 统计从每个区域出来后最常去的下一个区域
    for pid in ppl_list:
        freqnext = [-1] * 5
        records = []
        for area, array in enumerate(features_ppl[pid]["arrays"]):
            records += [(area, t) for t in array]
        if len(records) > 1:
            nxtcount = [[0] * 5 for _ in range(5)]
            records.sort(key=lambda x: x[1])
            arealist = [a for a, t in records]
            for i in range(1, len(arealist)):
                nxtcount[arealist[i - 1]][arealist[i]] += 1
            for i in range(5):
                freqnext[i] = maxpos(nxtcount[i])
        features_ppl[pid]["freqnext"] = freqnext
    for did in dev_list:
        freqnext = [-1] * 5
        records = []
        for area, array in enumerate(features_dev[did]["arrays"]):
            records += [(area, t) for t in array]
        if len(records) > 1:
            nxtcount = [[0] * 5 for _ in range(5)]
            records.sort(key=lambda x: x[1])
            arealist = [a for a, t in records]
            for i in range(1, len(arealist)):
                nxtcount[arealist[i - 1]][arealist[i]] += 1
            for i in range(5):
                freqnext[i] = maxpos(nxtcount[i])
        features_dev[did]["freqnext"] = freqnext

    # 写json
    output_jsonobj_ppl = jsonize(features_ppl)
    output_jsonobj_dev = jsonize(features_dev)
    with open("features_people.json", "w") as fjson:
        json.dump({"people": output_jsonobj_ppl}, fjson, indent=1)
    with open("features_device.json", "w") as fjson:
        json.dump({"people": output_jsonobj_dev}, fjson, indent=1)

    # 写csv
    # CSV部分不含时间向量，只有生成的特征向量
    with open("features_people.csv", "w") as fcsv:
        for pid in ppl_list:
            fcsv.write(str(pid))
            for kw in feature_keywords:
                for val in features_ppl[pid][kw]:
                    fcsv.write("," + str(val))
            fcsv.write("\n")
    with open("features_device.csv", "w") as fcsv:
        for did in dev_list:
            fcsv.write(str(did))
            for kw in feature_keywords:
                for val in features_dev[did][kw]:
                    fcsv.write("," + str(val))
            fcsv.write("\n")


if __name__ == "__main__":
    generate_feature_vecs()
