# 预处理
import pandas as pd
import numpy as np
from scipy import signal

# 读入高成本传感器数据
high_quality_data = []
for i in range(1, 13):#13
    high_quality = pd.read_csv(f"High quality sensor 0{i}.csv")
    high_quality_data.append(high_quality)
# 读入低成本传感器 A 数据
low_cost_A_data = np.array([])
for i in range(1, 6):
    low_cost_A = pd.read_csv(f"low-cost sensor A 0{i}.csv")
    low_cost_A = low_cost_A.values # convert to numpy array

    low_cost_A_data = np.concatenate((low_cost_A_data, low_cost_A), axis=0)

# 读入低成本传感器 B 数据
low_cost_B_data = np.array([])
for i in range(1, 6):
    low_cost_B = pd.read_csv(f"low-cost sensor B 0{i}.csv")
    low_cost_B = low_cost_B.values # convert to numpy array
    low_cost_B_data = np.concatenate((low_cost_B_data,low_cost_B),axis=0)



# 去除趋势
high_quality_data = [signal.detrend(data) for data in high_quality_data]
low_cost_A_data = [signal.detrend(data) for data in low_cost_A_data]
low_cost_B_data = [signal.detrend(data) for data in low_cost_B_data]

# 中值滤波
low_cost_A_data = [signal.medfilt(data, kernel_size=3) for data in low_cost_A_data]
low_cost_B_data = [signal.medfilt(data,kernel_size=3) for data in low_cost_B_data]
high_quality_data=[signal.medfilt(data,kernel_size=3) for data in high_quality_data]


# 时域分析
# 计算低成本传感器和高成本传感器的时域特征

# 均值
low_cost_mean_A = np.mean(low_cost_A_data, axis=1)
low_cost_mean_B = np.mean(low_cost_B_data, axis=1)
high_quality_mean = np.mean(high_quality_data, axis=1)

# 峰值
low_cost_peak_A_data = np.amax(low_cost_A_data, axis=1)
low_cost_peak_B_data = np.amax(low_cost_B_data, axis=1)
high_quality_peak = np.amax(high_quality_data, axis=1)

# 画图
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(low_cost_mean_A, label='Low cost A mean')
plt.plot(low_cost_mean_B, label='Low cost B mean')
plt.plot(high_quality_mean, label='High quality mean')
plt.legend()
plt.title('Time domain analysis - Mean')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize=(12,8))
plt.plot(low_cost_peak_A_data, label='Low cost A peak')
plt.plot(low_cost_peak_B_data, label='Low cost B peak')
plt.plot(high_quality_peak, label='High quality peak')
plt.legend()
plt.title('Time domain analysis - Peak')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# 频域分析
# 计算低成本传感器和高成本传感器的频域特征

# 傅里叶变换
low_cost_A_freq = np.fft.fft(low_cost_A_data)
low_cost_B_freq = np.fft.fft(low_cost_B_data)
high_quality_freq = np.fft.fft(high_quality_data)

# 计算频率
low_cost_freq_A = np.fft.fftfreq(len(low_cost_A_data), d=1/50)
low_cost_freq_B = np.fft.fftfreq(len(low_cost_B_data), d=1/50)
high_quality_freq = np.fft.fftfreq(len(high_quality_data), d=1/50)

# 绘制频域分析图
import matplotlib.pyplot as plt

plt.figure()
plt.plot(low_cost_freq_A, np.abs(low_cost_freq_A), label='low-cost sensor A')
plt.plot(low_cost_freq_B, np.abs(low_cost_freq_B), label='low-cost sensor B')
plt.plot(high_quality_freq, np.abs(high_quality_freq), label='high-quality sensor')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()