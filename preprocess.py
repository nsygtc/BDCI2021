'''
preprocess.py - 第一阶段特征工程

请将此文件放置在`/DATASET_ROOT/train_dataset`下，以便其正常工作

脚本产生以下这些文件：

- preprocessed_device：是原始数据c表（识别码记录表）的处理后数据。有3列，分别是点位的内部ID、时间（时间的表示方式后面会说）以及识别码的内部ID。
- preprocessed_person：与device相似，第3列表示人员的内部ID。
- preprocessed_label：这个表是原始数据label表的处理结果。两列，第一列是人员的内部ID，第二列是识别码的内部ID。这个表只在训练数据中有，测试数据中没有。
- id2areas, id2person, id2device：这几个文件记录了进行特征工程后点位、人员和识别码的内部ID与原始数据键名的对应关系。
- timearray_device，timearray_person：这两个文件分别保存转化过来的识别码和人员的时间向量（变长）。数据是三维的，因此没有用csv而是用了json。

下面是一些注意事项：

1. 内部ID都是整数，以满足矩阵化的要求。识别码和人员的内部ID虽然数据类型一样，但并不能对应上，如内部ID为1的人员，对应设备识别码内部ID不一定也为1。实际上这个对应是不可能做到的，因为识别码和人员的个数不一致，见注意事项2。
2. 人员数量只有2000个（注意到在训练数据中和测试数据中应该各有4个人没有在记录表里出现过。没有在记录表里出现过的人员没有分配内部ID），但是识别码数量却高达10000个，相当于平均5个识别码才能有1个对应到实际的人员。我在处理数据的时候将记录全部保留了，后续可能会当做负样本来处理。
3. 时间的处理。虽然数据集里面给的时刻是精确到微秒的，但是实际的精度显然只到秒一级。因此我对时间的编码也只精确到秒一级：首先遍历数据集（人员表和识别码表是分开遍历的，因为这两个时间本来也不匹配），找出数据集中出现最早的时间，之后第二次遍历之，对每一个记录求出它的时间相比之前找到的最早时间，延后的秒数。这个秒数就是时间的编码。举例来说，如果数据集中最早的时间是8:00:00，而某条记录的时间是11:45:14，那么这条记录时间字段的编码就是(11-8)\*3600+45\*60+14=13514。这里重新强调一遍两个数据表的时间编码也是不匹配的，即人员表的0时刻不等同于识别码表的0时刻，反之亦然。这么做的原因是因为原始记录中识别码出现时间和人员出现时间本身就是不匹配的，因此这里维持与原数据集的对应关系没有什么意义。

关于时间向量及其文件的说明。时间向量本身处理的不是很复杂，实际上就是把给定的人员（或识别码），在每一个点位上依次出现的时间全部收集出来，组成一个向量，这就是时间向量。这个向量显然是变长的，而不是定长的。两个json文件的文件格式相同，如下所示：

```json
{
    "people':[ // 为处理方便，不论是识别码的记录还是人员的记录，这里的键名都是people
        {
            "id":1,
            "arrays":[
                {
                    "area":0,
                    "array":[0,1,2,3]
                }
                // other areas...
            ]
        }
        //other people...
    ]
}
```

解释一下做这种简单处理的想法。首先对数据集做出一些假设：（绝大多数）在人员表中出现的记录，在识别码表中应当都能找到（考虑到识别码表比人员表大很多，反之应当不成立），只不过时间点上会差一个*固定*的时间差。假设我们能够通过其他方法获知这个时间差，那么我们实际上需要做的就是定义一个在两个时间向量（其中一个是识别码的，另一个是人员的）间的距离函数即可，识别码可以简单地归类到与其距离最短的人员上。

例如，通过对两个向量间的元素进行匹配的距离函数：在这两个向量（的元素）上建立一个映射，要求至少一个向量的元素都能映射到另一个向量的元素上，并且一个元素只能与另外的一个元素建立映射。（人话：就是找到识别码表上的记录与人员表上的记录的对应关系，每一种映射表示一种可能的对应关系）。对所有对应关系都可以求一个总距离（如L2，即所有对应元素差值的平方和），然后我们找到总距离中最小的那个，就是两个时间向量的距离。这个算法应该还能用dp的方法进行优化，此处暂时不进行深入讨论（到时候再说.jpg）。

本质上，这里是用时间向量直接代替了人员和识别码，暂时不涉及更深入的挖掘工作。

有关更详细的挖掘和特征提取，参见第二阶段特征工程的脚本`detail.py`。
'''

import csv
import json
from datetime import datetime
from bidict import bidict

device_fname = "CCF2021_run_record_c_Train.csv"
person_fname = "CCF2021_run_record_p_Train.csv"
label_fname = "CCF2021_run_label_Train.csv"

id2person: bidict[int, str] = bidict()
id2device: bidict[int, str] = bidict()
id2area: bidict[int, str] = bidict()
areaid2location: dict[int, tuple[str, str]] = {}


def produce_time_array(records: list[tuple[int, int, int]], filename):
    """传入的record一个三元tuple列表，每个tuple依次保存点位编号、时间戳和人员/识别码编号\n
    filename是保存文件的名称，请带上扩展名"""
    # 这个字典保存所有人员的时间数组，字典person2array[p][a]表示人员（识别码）p在
    # 区域a中产生的时间数组
    # 时间数组比较简单，就是所有出现时间，进行排序（排序步骤在写json的时候进行）
    person2tarray: dict[int, dict[int, list[int]]] = {}
    for area, timestamp, person in records:
        if person not in person2tarray:
            person2tarray[person] = {}
        if area not in person2tarray[person]:
            person2tarray[person][area] = []
        person2tarray[person][area].append(timestamp)
    # 下面构建json文件
    # json的文件格式大概是：
    # {"persons":[
    #   {"id":xx,"arrays":[
    #     {"area":xx,"array":[xx,xx,...]}
    #     ,...
    #   ]}
    #   ,...
    # ]}
    # 不过不会有缩进，问题不大
    dumplist = []
    for p in person2tarray:
        pobj = {"id": p, "arrays": []}
        for area in person2tarray[p]:
            pobj["arrays"].append(
                {"area": area, "array": sorted(person2tarray[p][area])}
            )
        dumplist.append(pobj)
    with open(filename, "w") as fjson:
        json.dump({"persons": dumplist}, fjson)


def read_csv(filename):
    with open(filename) as f:
        csv_content = csv.reader(f)
        columns = next(csv_content)
        return columns, [r for r in csv_content]


def dump_relation_infos():
    with open("id2person.csv", "w") as fcsv:
        fcsv.write("内部id,人员编号\n")
        for pid in id2person:
            fcsv.write(f"{pid},{id2person[pid]}\n")
    with open("id2device.csv", "w") as fcsv:
        fcsv.write("内部id,识别码\n")
        for did in id2device:
            fcsv.write(f"{did},{id2device[did]}\n")
    with open("id2areas.csv", "w") as fcsv:
        fcsv.write("内部id,点位,经度,纬度\n")
        for areaid in id2area:
            location = areaid2location[areaid]
            fcsv.write(f"{areaid},{id2area[areaid]},{location[0]},{location[1]}\n")


def process_device_tbl():
    _, rows = read_csv(device_fname)
    # 遍历数据集，首先将字符串格式的时间转换成datetime格式
    times = [datetime.fromisoformat(row[3]) for row in rows]
    earliest = min(times)
    filtered_columns = ["Area", "Timestamp", "Device"]
    noidinfo = len(id2area) == 0
    filtered_rows: list[tuple[int, int, int]] = []
    # 第二次遍历数据集，整理信息
    for i, row in enumerate(rows):
        # 点位信息，如果先前没有整理ID就要整理，同时收集经纬度信息
        if noidinfo and (row[0] not in id2area.values()):
            areaid2location[len(id2area)] = (row[1], row[2])
            id2area[len(id2area)] = row[0]
        areaid = id2area.inverse[row[0]]
        # 识别码
        if row[4] not in id2device.values():
            id2device[len(id2device)] = row[4]
        devid = id2device.inverse[row[4]]
        # 时间戳
        # 格式是当前时间较最早的时间延后的秒数
        # 毫、微秒的部分丢弃，因为数据集精度实际上没有这么高
        deltime = times[i] - earliest
        assert deltime.days == 0
        filtered_rows.append((areaid, deltime.seconds, devid))
    with open("preprocessed_device.csv", "w") as fwrite:
        fwrite.write(",".join(filtered_columns) + "\n")
        for areaid, timestamp, devid in filtered_rows:
            fwrite.write(f"{areaid},{timestamp},{devid}\n")
    produce_time_array(filtered_rows, "timearray_device.json")


def process_person_tbl():
    _, rows = read_csv(person_fname)
    # 遍历数据集，首先将字符串格式的时间转换成datetime格式
    times = [datetime.fromisoformat(row[3]) for row in rows]
    earliest = min(times)
    filtered_columns = ["Area", "Timestamp", "Person"]
    noidinfo = len(id2area) == 0
    filtered_rows: list[tuple[int, int, int]] = []
    # 第二次遍历数据集，整理信息
    for i, row in enumerate(rows):
        # 点位信息，如果先前没有整理ID就要整理，同时收集经纬度信息
        if noidinfo and (row[0] not in id2area.values()):
            areaid2location[len(id2area)] = (row[1], row[2])
            id2area[len(id2area)] = row[0]
        areaid = id2area.inverse[row[0]]
        # 人员ID
        if row[4] not in id2person.values():
            id2person[len(id2person)] = row[4]
        personid = id2person.inverse[row[4]]
        # 时间戳
        # 格式是当前时间较最早的时间延后的秒数
        # 毫、微秒的部分丢弃，因为数据集精度实际上没有这么高
        deltime = times[i] - earliest
        assert deltime.days == 0
        filtered_rows.append((areaid, deltime.seconds, personid))
    with open("preprocessed_person.csv", "w") as fwrite:
        fwrite.write(",".join(filtered_columns) + "\n")
        for areaid, timestamp, personid in filtered_rows:
            fwrite.write(f"{areaid},{timestamp},{personid}\n")
    produce_time_array(filtered_rows, "timearray_person.json")


def process_label_tbl():
    if len(id2person) == 0 or len(id2device) == 0:
        raise RuntimeError("需要先执行人员表和识别码表的转换")
    _, rows = read_csv(label_fname)
    with open("preprocessed_label.csv", "w") as fcsv:
        fcsv.write("Person,Device\n")
        for person, device in rows:
            if person not in id2person.values() or device not in id2device.values():
                continue
            fcsv.write(f"{id2person.inverse[person]},{id2device.inverse[device]}\n")


if __name__ == "__main__":
    process_device_tbl()
    process_person_tbl()
    process_label_tbl()
    dump_relation_infos()
