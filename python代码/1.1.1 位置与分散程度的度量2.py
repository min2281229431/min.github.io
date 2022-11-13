import numpy as np
import scipy.stats as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# =========================================================================== #
"""
多维数组或矩阵求统计量
示例：数据是从某学校抽样30个学生的身高、体重、胸围和坐高等抽样数据
"""

# 身高
x1 = np.array([148, 139, 160, 149, 159, 142, 153, 150, 151, 139,
               140, 161, 158, 140, 137, 152, 149, 145, 160, 156,
               151, 147, 157, 147, 157, 151, 144, 141, 139, 148])
# 体重
x2 = np.array([41, 34, 49, 36, 45, 31, 43, 43, 42, 31,
               29, 47, 49, 33, 31, 35, 47, 35, 47, 44,
               42, 38, 39, 30, 48, 36, 36, 30, 32, 38])
# 胸围
x3 = np.array([72, 71, 77, 67, 80, 66, 76, 77, 77, 68,
               64, 78, 78, 67, 66, 73, 82, 70, 74, 78,
               73, 73, 68, 65, 80, 74, 68, 67, 68, 70])
# 坐高
x4 = np.array([78, 76, 86, 79, 86, 76, 83, 79, 80, 74,
               74, 84, 83, 77, 73, 79, 79, 77, 87, 85,
               82, 78, 80, 75, 88, 80, 76, 76, 73, 78])
'''
数据分析、统计建模、机器学习等领域一般将数据存储为列向量，
Numpy、Scipy、Pandas、StatsModels、SkLearn、Tensorflow、Pytorch等基本上都是处理列向量。
例外：Numpy求随机向量之间的协方差矩阵时，则是按照行向量进行计算的。
'''


# 将x1,x2,x3,x4四个向量合并存储为矩阵，并转置为列向量，    .T操作符（或属性）是对矩阵进行转置。
stu_data = np.matrix([x1, x2, x3, x4]).T
print('学生的身高、体重、胸围和坐高（前五个）：\n', stu_data[0:6])

# 求均值
mean_ = np.round(stu_data.mean(0), 1)
print(mean_)                                                 # [[149.   38.7  72.2  79.4]]
stu_mean = np.round(stu_data.mean(0), 1).ravel()             # [149.   38.7  72.2  79.4]
print(stu_mean)
# stu_data.mean(0)函数的参数0表示列向量方向，如果是1则是行向量方向  np.mean(stu_data, axis=0)输入结果也是一样的
# round(data, 1)表示对数据四舍五入，1表示保留1位小数
# ravel()函数将二维矩阵展平为一维向量,数据类型转变为Numpy数组
print('\n学生的平均身高、平均体重、平均胸围和平均坐高分别为：\n %.1f, %.1f, %.1f, %.1f'
      % (stu_mean[0], stu_mean[1], stu_mean[2], stu_mean[3]))
