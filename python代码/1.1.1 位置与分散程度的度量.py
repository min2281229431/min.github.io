import numpy as np
import scipy.stats as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# =========================================================================== #
"""示例：某学校15个学生体重（单位：公斤）抽样调查数据."""

weights = np.array([75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7,
                    63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0])

# 1.均值
w_mean = np.mean(weights)
w_mean1 = weights.mean()
print('体重数据的均值为：', w_mean)
# 限定范围内的数据求均值
limitedMean = st.tmean(weights, (60, 70))
sorted_weig = sorted(weights, reverse=True)  # reverse的缺省值为False

# 2.中位数
# 对称分布比如t分布和正态分布，均值与中位数很接近：偏态分布的二者相差比较大，比如F分布
median_weig = np.median(weights)
print('体重数据的中位数为：', median_weig)

# 3.分位数
quantiles = np.quantile(weights, [0.1, 0.2, 0.4, 0.6, 0.8, 1])
print('体重的[10%,20%,40%,60%,80%,100%]分位数：\n', quantiles)

# 4.方差
v = np.var(weights)  # 有偏估计或样本方差
v_unb = st.tvar(weights)  # 无偏估计
print('体重数据方差的估计为：%0.2f,无偏估计为：%0.2f' % (v, v_unb))

# 5.标准差
s = np.std(weights)  # 有偏估计或样本标准差
s_unb = st.tstd(weights)  # 无偏估计
print('体重数据标准差的估计为：%0.2f,无偏估计为：%0.2f' % (s, s_unb))

# 6.变异系数
cv = s_unb / w_mean * 100  # 变异系数，无量纲，用百分数表示
print('体重数据的变异系数为：', np.round(cv, 2), '%')

# 7.极差与标准误
R_weights = np.max(weights) - np.min(weights)  # 极差：最大值-最小值
print('体重数据的极差：%0.2f' % R_weights)
sm_weights = st.tstd(weights) / np.sqrt(len(weights))  # 标准误：数据标准差（无偏）/数据量**0.5
print('体重数据的标准误：%0.2f' % sm_weights)
