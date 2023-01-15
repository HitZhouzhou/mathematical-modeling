import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

low_cost_acc_data_A = []
for i in range(1, 6):
    low_cost = pd.read_csv(f"low-cost sensor A 0{i}.csv")
    low_cost = low_cost.values
    low_cost_acc_data_A.append(low_cost)

low_cost_acc_data = np.concatenate(low_cost_acc_data_A, axis=0)

# low_cost_acc_data_B=[]
# for i in range(1, 11):
#     low_cost = pd.read_excel(f"low-cost sensor B 0{i}.xls")
#     low_cost = low_cost.values[:,:4]
#     low_cost_acc_data_B.append(low_cost)
#
# low_cost_acc_data = np.concatenate(low_cost_acc_data_A, axis=0)
# low_cost_acc_data = np.concatenate((low_cost_acc_data_A, low_cost_acc_data_B), axis=0)

# 提取时间和加速度x,y,z
time_low = low_cost_acc_data[:, 0]
acc_x_low = low_cost_acc_data[:, 1]
acc_y_low = low_cost_acc_data[:, 2]
acc_z_low = low_cost_acc_data_A[:, 3]

# 求出每个时间点的加速度x,y,z的平均值
mean_acc_x_low = []
mean_acc_y_low = []
mean_acc_z_low = []

for t in np.unique(time_low):
    mask = time_low == t
    mean_acc_x_low.append(np.mean(acc_x_low[mask]))
    mean_acc_y_low.append(np.mean(acc_y_low[mask]))
    mean_acc_z_low.append(np.mean(acc_z_low[mask]))


high_quality_data = []
for i in range(1, 13):
    high_quality = pd.read_csv(f"High quality sensor 0{i}.csv")
    high_quality = high_quality.values  # convert to numpy array
    high_quality_data.append(high_quality)

high_quality_data = np.concatenate(high_quality_data, axis=0)
time_high = high_quality_data[:, 0]
acc_x_high = high_quality_data[:, 1]
acc_y_high = high_quality_data[:, 2]
acc_z_high = high_quality_data[:, 3]
gyro_x_high = high_quality_data[:, 4]
gyro_y_high = high_quality_data[:, 5]
gyro_z_high = high_quality_data[:, 6]

mean_acc_x_high = np.array([np.mean(acc_x_high[time_high == t]) for t in np.unique(time_high)])
mean_acc_y_high = np.array([np.mean(acc_y_high[time_high == t]) for t in np.unique(time_high)])
mean_acc_z_high = np.array([np.mean(acc_z_high[time_high == t]) for t in np.unique(time_high)])
mean_gyro_x_high = np.array([np.mean(gyro_x_high[time_high == t]) for t in np.unique(time_high)])
mean_gyro_y_high = np.array([np.mean(gyro_y_high[time_high == t]) for t in np.unique(time_high)])
mean_gyro_z_high = np.array([np.mean(gyro_z_high[time_high == t]) for t in np.unique(time_high)])

# 画图
plt.figure(figsize=(144, 90))
plt.title('The acceration of x of the high data and the low data')
plt.plot(np.unique(time_high), mean_acc_x_high, label="Acceleration x of the high")
plt.plot(np.unique(time_low), mean_acc_x_low,label='Acceleration x of the low')
plt.xlabel("Time (s)")
plt.ylabel("Sensor value")
plt.legend()
plt.show()
