""" """
import numpy as np
import scipy.stats as st
import pandas as pd

weights = np.array([75.0, 64.0, 47.4, 66.9, 62.2, 62.2, 58.7,
                    63.5, 66.6, 64.0, 57.0, 69.0, 56.9, 50.0, 72.0])
w_mean = np.mean(weights)
s = np.std(weights)  # 有偏估计或样本标准差
s_unb = st.tstd(weights)  # 无偏估计标准差

'''
偏度计算：
（1）偏度表示曲线是向左偏或右偏，又称为副偏态或正偏态。
（2）偏度越接近0，越符合正态分布的曲线。
（3）偏度小于0称分布具有负偏离，也称左偏态；反之就是正偏态或右偏态。

'''
###偏度计算公式
n = len(weights)
# 三阶矩，其他各阶矩的计算依次类推
u3 = np.sum((weights - w_mean) ** 3) / n

###使用使用总体标准差的无偏估计，计算的偏度是修正后偏度
skew1 = ((n ** 2) * u3) / ((n - 1) * (n - 2) * (s_unb ** 3))  # 偏度

###pandas计算是修正后偏度
pd_weights = pd.Series(weights)
skew_pandas = pd_weights.skew()
print('Pandas计算公式手工计算以及调用函数计算结果：')
print('skew1:', skew1, 'skew_pandas:', skew_pandas)

###无修正偏度的手工计算，使用样本标准差
skew2 = np.sum((weights - w_mean) ** 3) / ((s ** 3) * n)

###scipy计算公式和结果
print('\nScipy计算公式手工计算以及调用函数计算结果（无修正）：')
skew_scipy = st.skew(weights)
print('skew2:', skew2, 'skew_scipy:', skew_scipy)

'''
(1)使用Scipy的skew函数，如果将第二个参数bias设为False，计算结果就和Pandas完全相同了。
   bias参数表示是否修正，如果为False表示修正，反之则不修正。
(2)总体上感觉修正后偏度比较准确，但是很多场合仍用无修正的偏度进行统计量的计算。
(3)StatsModels的线性回归模型对残差的正态分布性（Jarque-Bera、Omnibus检验等）
   进行检验时，使用的偏度就是无修正的，包括峰度也是无修正的。
'''
skew_scipy_bias = st.skew(weights, bias=False)
print('\nScipy进行修正后的偏度：', skew_scipy_bias)
print('-'*100)

# ==================================================================================================================== #
'''
峰度的计算：
（1）峰度表示曲线是扁平态（低峰态）还是尖峰态。
（2）正常值有两种定义：Fisher定义该值为0；Pearson定义为3。
（3）按照Fisher定义，峰度=0表示正好符合正态分布的曲线；大于0表示峰比较尖，反之表示比较平。
'''
###峰度计算，StatsModels多使用无修正的峰度
# 手工实现留作练习
kurt_pandas = pd_weights.kurt()
kurt_scipy = st.kurtosis(weights, bias=False)
kurt_scipy_bias = st.kurtosis(weights, bias=True)  # True是bias的缺省值
print('\nPandas计算峰度:', kurt_pandas, '\n\nScipy计算峰度（修正后）:',
      kurt_scipy, '\n\nScipy计算峰度（无修正）:', kurt_scipy_bias)
