import pandas as pd
from pylab import mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

df_pre_c = pd.read_csv("..\\data\\preprocessed_person.csv")
print(df_pre_c.head(2))
df_time = df_pre_c.iloc[:, 1]
print(df_time.head(20))
print("---------------------------------")
'''
    对人员的检测数据集进行预处理
    并按照4分钟对时间段进行分段
'''
df_run_p = pd.read_csv("..\\data\\CCF2021_run_record_p_Train.csv")
df_run_p = df_run_p.drop(['经度', '纬度'], axis=1)
df_run_p['时间段'] = '1'
print(df_run_p.head(2))
for i in range(0, len(df_run_p)):
    tmp = df_pre_c.iloc[i, 1]
    df_run_p.iloc[i, 3] = int(tmp / 240)
print("人员处理后数据集情况：")
print(df_run_p.head(2))
print("---------------------------------")

# 此处为寻找人流量随时间的变化趋势，读取的文件应该是CCF2021_run_record_p_Train.csv
'''
    D00画图
'''
df_D00 = df_run_p
df_D00.head(1794).groupby('时间段')['人员编号'].count().plot(color='r')
plt.xlabel("时间段")
plt.ylabel("人员流量")
plt.title("D00人员流量随时间变化")
plt.show()

'''
    D01画图
'''
df_D01 = df_run_p
df_D01 = df_D01.iloc[1794:]
# print(df_D01.head(10))
df_D01.head(3737).groupby('时间段')['人员编号'].count().plot(color='b')
plt.xlabel("时间段")
plt.ylabel("人员流量")
plt.title("D01人员流量随时间变化")
plt.show()

'''
    D02画图
'''
df_D02 = df_run_p
df_D02 = df_D02.iloc[5532:]
print(df_D02.head(10))
df_D02.head(3612).groupby('时间段')['人员编号'].count().plot(color='y')
plt.xlabel("时间段")
plt.ylabel("人员流量")
plt.title("D02人员流量随时间变化")
plt.show()

'''
    D03画图
'''
df_D03 = df_run_p
df_D03 = df_D03.iloc[9144:]
print(df_D03.head(10))
df_D03.head(3696).groupby('时间段')['人员编号'].count().plot(color='#7FFF00')
plt.xlabel("时间段")
plt.ylabel("人员流量")
plt.title("D03人员流量随时间变化")
plt.show()

'''
    D04画图
'''
df_D04 = df_run_p
df_D04 = df_D04.iloc[12839:]
print(df_D04.head(10))
df_D04.groupby('时间段')['人员编号'].count().plot(color='#87CEFA')
plt.xlabel("时间段")
plt.ylabel("人员流量")
plt.title("D04人员流量随时间变化")
plt.show()

# 此处为寻找特征码随时间的变化趋势，读取的文件应该是CCF2021_run_record_p_Train.csv
'''
    对设备的检测数据集进行预处理
    并按照4分钟对时间段进行分段
'''
df_pre_device = pd.read_csv("..\\data\\preprocessed_device.csv")
df_time = df_pre_device.iloc[:, 1]
df_run_device = pd.read_csv("..\\data\\CCF2021_run_record_c_Train.csv")
df_run_device = df_run_device.drop(['经度', '纬度'], axis=1)
df_run_device['时间段'] = '1'
for i in range(0, len(df_run_device)):
    tmp = df_pre_device.iloc[i, 1]
    df_run_device.iloc[i, 3] = int(tmp / 240)
print(df_run_device.head(20))

'''
    D00画图
'''
df_D00 = df_run_device
print(df_D00.head(5))
df_D00.head(32823).groupby('时间段')['特征码'].count().plot(color='r')
plt.xlabel("时间段")
plt.ylabel("特征码检测数量")
plt.title("D00特征码检测数量随时间变化")
plt.show()

'''
    D01画图
'''
df_D01 = df_run_device
df_D01 = df_D01.iloc[32823:]
print(df_D01.head(5))
df_D01.head(63461).groupby('时间段')['特征码'].count().plot(color='b')
plt.xlabel("时间段")
plt.ylabel("特征码检测数量")
plt.title("D01特征码检测数量随时间变化")
plt.show()

'''
    D02画图
'''
df_D02 = df_run_device
df_D02 = df_D02.iloc[63461:]
print(df_D02.head(5))
df_D02.head(95647).groupby('时间段')['特征码'].count().plot(color='y')
plt.xlabel("时间段")
plt.ylabel("特征码检测数量")
plt.title("D02特征码检测数量随时间变化")
plt.show()

'''
    D03画图
'''
df_D03 = df_run_device
df_D03 = df_D03.iloc[95647:]
print(df_D03.head(5))
df_D03.head(74859).groupby('时间段')['特征码'].count().plot(color='#7FFF00')
plt.xlabel("时间段")
plt.ylabel("特征码检测数量")
plt.title("D03特征码检测数量随时间变化")
plt.show()

'''
    D04画图
'''
df_D04 = df_run_device
df_D04 = df_D04.iloc[74859:]
print(df_D04.head(5))
df_D04.groupby('时间段')['特征码'].count().plot(color='#87CEFA')
plt.xlabel("时间段")
plt.ylabel("特征码检测数量")
plt.title("D04特征码检测数量随时间变化")
plt.show()

'''
    画五个点位在空间中分布的图像
'''
p_name = ['D00', 'D01', 'D02', 'D03', 'D04']
x = [118.3765, 118.3759, 118.3671, 118.3748, 118.3737]
y = [31.2352, 31.2352, 31.2342, 31.2365, 31.2335]
plt.scatter(x, y, c="red", marker="o", alpha=1)
for i in range(5):
    plt.text(x[i], y[i], p_name[i])
plt.xlim(118.3670, 118.3770)
plt.ylim(31.2330, 31.2370)
plt.xlabel("经度")
plt.ylabel("纬度")
plt.title("空间分布")
plt.grid()
plt.show()
'''
    检查数据中的缺失值情况
'''
for i in range(0, len(df_run_p)):
    tmp = df_run_p.iloc[i, 1]
    if pd.isnull(tmp):
        print(str(df_run_p.iloc[i, 1]) + "====>" + str(df_run_p.iloc[i, 0]))
for i in range(0, len(df_run_p)):
    tmp = df_run_p.iloc[i, 2]
    if pd.isnull(tmp):
        print(str(df_run_p.iloc[i, 2]) + "====>" + str(df_run_p.iloc[i, 0]))

'''
    检查每个用户被检测到的次数
'''
df_run_person = df_run_p.sort_values(axis=0, by='人员编号', ascending=True)
print(df_run_person.head(2))
df_run_person.groupby('人员编号')['发现时间'].count().plot()
plt.xlabel("人员编号")
plt.ylabel("发现次数")
plt.title("人员发现次数分布")
plt.show()

'''
    追踪某一个用户(P1001)的轨迹
'''
P1001_track = []
for i in range(0, len(df_run_p)):
    tmp = df_run_p.iloc[i, 2]
    if tmp == 'P1001':
        P1001_track.append([df_run_p.iloc[i, 0], df_run_p.iloc[i, 1]])
print(P1001_track)

'''
    随着时间推进，每个点位的人流量变化
'''
df_run_p['时间段'] = '1'
for i in range(0, len(df_run_p)):
    tmp = df_run_p.iloc[i, 1]
df_run_time = df_run_p.sort_values(axis=0, by='发现时间', ascending=True)
df_run_time.groupby('发现时间')['时间段'].count().plot()
plt.show()

'''
    追踪某一个用户(P1001)设备(Cgzxnfek)的轨迹
'''
P1001_device_track = []
for i in range(0, len(df_run_device)):
    tmp = df_run_device.iloc[i, 2]
    if tmp == 'Cgzxnfek':
        P1001_device_track.append([df_run_device.iloc[i, 0], df_run_device.iloc[i, 1]])
print(P1001_device_track)
