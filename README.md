# BDCI2021——基于泛在感知数据的关联融合匹配模型
### 1. 问题背景及分析

   近年来随着物联网、移动通信、前端感知等技术的高速发展，针对人像、车辆、设备等数据的泛在感知采集正在得到广泛的应用。这些感知手段往往相互独立，难以在前端直接对人、车、物的数据进行有效的关联融合。但通过设置多个感知点位，经过一定时间的数据积累，可以在后台进行大数据的关联分析，<b>计算得到人、车、物之间的关联关系，实现泛在感知数据的关联融合</b>。关联计算的准确率和海量数据的计算性能，是实际工作中的主要难点。

本项目选题来源于2021 CCF BDCI 数据算法赛道-锐安科技赛题：泛在感知数据关联融合计算（https://www.datafountain.cn/competitions/539/datasets ）。

同时，本项目涉及的两个样本数据文件采用随机生成方式产生，分别模拟前端点位对<b>人像的采集数据</b>以及这些点位对<b>某硬件特征码</b>的采集数据，用于关联融合计算。硬件特征码包括但不限于手机IMSI信息、手机IMEI信息、网卡MAC地址等。另外，根据实际场景模拟，无论是人像采集数据还是硬件特征码采集数据，均存在大量的数据丢失。

### 2. 问题描述

本小组拟利用模拟采集到的人像数据与硬件特征码数据，采用统计学方法、数据挖掘、图计算等多种相关算法，例如，传统机器学习方法，learning to rank, 图神经网络等，将两种数据进行关联融合计算，提取其中的样本特征，最终为每一个测试集中的人像从测试集的硬件特征码中找到其对应的匹配度最高的硬件特征码，并将结果输出为模型推理结果文件。

### 3. 代码文件说明
- dataset：相关数据集
- visualization.py：用来对数据集进行预处理，并将处理后的数据进行可视化的工作。
