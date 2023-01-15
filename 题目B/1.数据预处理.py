# 预处理
import pandas as pd
import numpy as np
from scipy import signal

# 读入高成本传感器数据
high_quality_data = []
for i in range(1, 13):#13
    high_quality = pd.read_csv(f"High quality sensor 0{i}.csv")
    high_quality = high_quality.values # convert to numpy array
    high_quality_data.append(high_quality)
from scipy.signal import resample


# 读入低成本传感器 A 数据
low_cost_A_data = []
for i in range(1, 6):#6
    low_cost_A = pd.read_csv(f"low-cost sensor A 0{i}.csv")
    low_cost_A = low_cost_A.values # convert to numpy array
    low_cost_A_data.append(low_cost_A)

# 读入低成本传感器 B 数据
low_cost_B_data = []
for i in range(1, 11):#11
    low_cost_B = pd.read_excel(f"low-cost sensor B 0{i}.xls")
    low_cost_B = low_cost_B.values # convert to numpy array
    low_cost_B_data.append(low_cost_B)

# 去除趋势
high_quality_data = [signal.detrend(data) for data in high_quality_data]
low_cost_A_data = [signal.detrend(data) for data in low_cost_A_data]
low_cost_B_data = [signal.detrend(data) for data in low_cost_B_data]

# 中值滤波

low_cost_A_data = [signal.medfilt(data, kernel_size=3) for data in low_cost_A_data]
low_cost_B_data = [signal.medfilt(data,kernel_size=3) for data in low_cost_B_data]
high_quality_data=[signal.medfilt(data,kernel_size=3) for data in high_quality_data]


# 时域分析
# 时域分析
# 计算低成本传感器和高成本传感器的时域特征

# # 均值
# low_cost_mean_A = np.mean(low_cost_A_data, axis=1)
# low_cost_mean_B = np.mean(low_cost_B_data, axis=1)
# high_quality_mean = np.mean(high_quality_data, axis=1)
#
# # 峰值
# low_cost_peak_A_data = np.amax(low_cost_A_data, axis=1)
# low_cost_peak_B_data = np.amax(low_cost_B_data, axis=1)
# high_quality_peak = np.amax(high_quality_data, axis=1)



# 均值
low_cost_A_data = np.concatenate(low_cost_A_data, axis=0)
low_cost_B_data = np.concatenate(low_cost_B_data, axis=0)
high_quality_data = np.concatenate(high_quality_data, axis=0)

low_cost_mean_A = np.mean(low_cost_A_data, axis=1)
low_cost_mean_B = np.mean(low_cost_B_data, axis=1)
high_quality_mean = np.mean(high_quality_data, axis=1)
# 峰值
low_cost_peak_A_data = np.amax(low_cost_A_data, axis=1)
low_cost_peak_B_data = np.amax(low_cost_B_data, axis=1)
high_quality_peak = np.amax(high_quality_data, axis=1)



import matplotlib.pyplot as plt

#  //注意这里
# # 绘制时域特征图
# # plt.plot().title='时域特征图'
# mean

plt.figure(figsize=(144,90))
plt.title('The mean of the data of the 3 kinds of sensors',fontsize=20)
plt.plot(high_quality_mean[::100], 'b-', label='High Mean')

plt.plot(low_cost_mean_A[::100], 'y-', label='Low_A Mean')

plt.plot(low_cost_mean_B[::100], 'g-', label='Low_B Mean')

plt.xlabel('Sample')
plt.ylabel('Mean of the data of the sensors')
plt.legend()
plt.show()

# peak
plt.figure(figsize=(144,90))
plt.title('The peak of the datas of the 3 kinds of sensors',fontsize=20)
plt.xlabel('samples')
plt.ylabel('Peak of the datas of the sensors')
plt.plot(high_quality_peak[::100], 'g-', label='High Peak')
plt.plot(low_cost_peak_A_data[::100], 'y-', label='Low_A Peak')
plt.plot(low_cost_peak_B_data[::100], 'h-', label='Low_B Peak')
plt.legend()
plt.show()
# # 绘制低成本传感器 A 的均值时域特征图
# plt.plot(low_cost_mean_A[::20], label='Low-cost sensor A')
# plt.xlabel('样例')
# plt.ylabel('Amplitude')
# plt.title('Mean of Low-cost sensor A')
# plt.legend()
# plt.show()
#
# # 绘制低成本传感器 B 的均值时域特征图
# plt.plot(low_cost_mean_B[::20], label='Low-cost sensor B')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.title('Mean of Low-cost sensor B')
# plt.legend()
# plt.show()
#
# # 绘制高成本传感器的均值时域特征图
# plt.plot(high_quality_mean[::20], label='High-quality sensor')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')
# plt.title('Mean of High-quality sensor')
# plt.legend()
# plt.show()






# 频域分析

from scipy.fft import fft

# 高成本传感器功率谱密度
high_quality_fft = [fft(data) for data in high_quality_data]
high_quality_psd = [np.abs(fft)**2 for fft in high_quality_fft]

# 低成本传感器A功率谱密度
low_cost_fft_A = [fft(data) for data in low_cost_A_data]
low_cost_psd_A = [np.abs(fft)**2 for fft in low_cost_fft_A]

#低成本传感器B功率谱密度
low_cost_fft_B = [fft(data) for data in low_cost_B_data]
low_cost_psd_B = [np.abs(fft)**2 for fft in low_cost_fft_B]

#可视化比较
import matplotlib.pyplot as plt

# high_quality_psd = [signal.welch(data, fs=100) for data in high_quality_data]
# low_cost_psd_A = [signal.welch(data, fs=100) for data in low_cost_A_data]
# low_cost_psd_B = [signal.welch(data, fs=100) for data in low_cost_B_data]



plt.figure(figsize=(144,90))
plt.title('Power spectral density')
plt.plot(high_quality_psd[::100], 'y-', label='High')
plt.plot(low_cost_psd_A[::100], 'b-', label='Low A')
plt.plot(low_cost_psd_B[::100], 'g-', label='Low B')
plt.xlabel('Samples')
plt.ylabel('power spectral density of the data of sensors')
plt.legend()
plt.show()



# 4.

# 使用低成本传感器A数据与高成本传感器数据进行比对
low_cost_A_data1 = np.concatenate(low_cost_A_data, axis=0)
high_quality_data1 = np.concatenate(high_quality_data, axis=0)

# 将所有数据按时间顺序排序。
# 按时间顺序排序
high_quality_data1 = resample(high_quality_data1, len(low_cost_A_data1))
low_cost_A_data1 = low_cost_A_data1[low_cost_A_data1.argsort()]
high_quality_data1 = high_quality_data1[high_quality_data1.argsort()]

# 将高成本传感器数据的值与低成本传感器数据的值相比较，找出两者之间的差异。
# 将高成本传感器数据与低成本传感器数据按时间顺序对应

if high_quality_data1.shape[0] != low_cost_A_data1.shape[0]:
    # 如果不相等，截断或补齐数据
    if high_quality_data1.shape[0] > low_cost_A_data1.shape[0]:
        high_quality_data1 = high_quality_data1[:low_cost_A_data.shape[0]]
    else:
        low_cost_A_data1 = low_cost_A_data1[:high_quality_data.shape[0]]
# 找出差异
diff = high_quality_data1- low_cost_A_data1

# 将差异数据绘制成图像。
# 绘制差异数据图像
import matplotlib.pyplot as plt
plt.figure(figsize=(144,90))
plt.title('The diff of the low_cost_data and the high_quality_data',fontsize=20)
plt.xlabel('samples',fontsize=16)
plt.ylabel('diff',fontsize=16)
plt.plot(diff[::300])
plt.legend()
plt.show()
#
# 分析差异数据图像，找出其中的规律。
# 分析差异数据图像，找出规律
# 例如，使用统计学方法确定差异数据的均值、标准差等。
mean_diff = np.mean(diff)
std_diff = np.std(diff)
print(f"Mean difference: {mean_diff}")
print(f"Standard deviation of difference: {std_diff}")


# 零加速度补偿算法

# 一些额外的数据处理:
# high_quality_data = [data for data in high_quality_data if data.ndim > 0]
# low_cost_A_data = [data for data in low_cost_A_data if data.ndim > 0]
# low_cost_A_data = [data for data in low_cost_A_data if data.ndim > 0]
# low_cost_A_data = np.array(low_cost_A_data)
# low_cost_B_data = np.array(low_cost_B_data)
# high_quality_data = np.array(high_quality_data)

# 零加速度补偿
# 计算低成本传感器 A 和 B 的零点
# low_cost_A_data = low_cost_A_data[:, np.newaxis]
# low_cost_B_data = low_cost_B_data[:, np.newaxis]
# zero_point_A = np.mean(low_cost_A_data, axis=0)
# zero_point_B = np.mean(low_cost_B_data, axis=0)
#
# # 零点补偿
# low_cost_A_data = low_cost_A_data - zero_point_A[:, np.newaxis]
# low_cost_B_data = low_cost_B_data - zero_point_B[:, np.newaxis]
# 纠正后的均值
# low_cost_A_data = np.concatenate(low_cost_A_data, axis=0)
# low_cost_B_data = np.concatenate(low_cost_B_data, axis=0)
# high_quality_data = np.concatenate(high_quality_data, axis=0)
# 新零加速度补偿
# 零加速度补偿
# 零加速度补偿算法
def zero_acceleration_compensation(data, threshold):
    # 差分
    diff_data = np.diff(data, axis=0)
    # 零加速度补偿
    for i in range(diff_data.shape[0]):
        if abs(diff_data[i].all()) < threshold:
            diff_data[i] = 0
    # 通过差分数据还原原始数据
    compensated_data = np.cumsum(diff_data, axis=0)
    compensated_data = np.concatenate((data[0].reshape(1, -1), compensated_data), axis=0)
    return compensated_data


threshold = 0.1 # 设置阈值
low_cost_data_A2 = zero_acceleration_compensation(low_cost_A_data, threshold)
low_cost_data_B2 = zero_acceleration_compensation(low_cost_B_data, threshold)

low_cost_mean_A2 = np.mean(low_cost_data_A2, axis=1)
low_cost_mean_B2 = np.mean(low_cost_data_B2, axis=1)
high_quality_mean = np.mean(high_quality_data, axis=1)
print(low_cost_A_data.shape)
print(low_cost_B_data.shape)
print(high_quality_data.shape)
# 画修复后的diff
# 使用低成本传感器A数据与高成本传感器数据进行比对
low_cost_A_data2 = np.concatenate(low_cost_data_A2, axis=0)
high_quality_data2 = np.concatenate(high_quality_data, axis=0)

# 将所有数据按时间顺序排序。
# 按时间顺序排序
high_quality_data2 = resample(high_quality_data2, len(low_cost_A_data2))
low_cost_A_data2 = low_cost_A_data2[low_cost_A_data2.argsort()]
high_quality_data2 = high_quality_data1[high_quality_data2.argsort()]

# 将高成本传感器数据的值与低成本传感器数据的值相比较，找出两者之间的差异。
# 将高成本传感器数据与低成本传感器数据按时间顺序对应

if high_quality_data2.shape[0] != low_cost_A_data2.shape[0]:
    # 如果不相等，截断或补齐数据
    if high_quality_data2.shape[0] > low_cost_A_data2.shape[0]:
        high_quality_data2 = high_quality_data2[:low_cost_A_data.shape[0]]
    else:
        low_cost_A_data2 = low_cost_A_data2[:high_quality_data.shape[0]]
# 找出差异
diff =   low_cost_A_data2-high_quality_data2

plt.figure(figsize=(144,90))
plt.title('The diff of the low_cost_data and the high_quality_data after corrected',fontsize=20)
plt.xlabel('samples',fontsize=16)
plt.ylabel('diff',fontsize=16)
plt.plot(diff[::300])
plt.legend()
plt.show()
# # 随机森林回归
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
#
# # 将数据拼接为一个整体
# data_A = np.concatenate(low_cost_A_data)
# data_B = np.concatenate(low_cost_B_data)
# data_high = np.concatenate(high_quality_data)
# high_quality_data= [resample(data, num=len(low_cost_B_data), t=np.arange(0, len(data))/1000) for data in high_quality_data]
#
# data_high = resample(data_high, num=len(low_cost_B_data))
# # 划分训练集和测试集
#
# X_train, X_test, y_train, y_test = train_test_split(low_cost_B_data, data_high, test_size=0.2, random_state=42)
# # 定义随机森林回归模型
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # 训练模型
# rf.fit(X_train, y_train)
#
# # 预测
# y_pred = rf.predict(X_test)
#
# # 评估模型精度
# mse = mean_squared_error(y_test, y_pred)
# print("The mean squared error of the model is: ", mse)
#
# # 使用模型对低成本传感器 B 的数据进行预测
# predicted_high_quality_data = rf.predict(data_B)
#

# 纠正后的均值的图像:
# mean

plt.figure(figsize=(144,90))
plt.title('The mean of the data of the 3 kinds of sensors after corrected',fontsize=20)
plt.plot(high_quality_mean[::100], 'b-', label='High Mean')

plt.plot(low_cost_mean_A[::100], 'y-', label='Corrected Low_A Mean')

plt.plot(low_cost_mean_B[::100], 'g-', label='Corrected Low_B Mean')

plt.xlabel('Sample')
plt.ylabel('Mean of the data of the sensors after corrected')
plt.legend()
plt.show()

