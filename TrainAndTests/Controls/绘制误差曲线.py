import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from _context import *
from Visualize.plot_tools import set_axes_equal
from Math_calculates.sub_of_angles import *

# 读取 CSV 文件
current_dir = os.path.join(project_root, "TrainAndTests/Controls")

file_name = "PID_static_delta_h-4000_20260319_161717.csv"

csv_path = os.path.join(current_dir, 'test_result', file_name)
df = pd.read_csv(csv_path)

# 如果 CSV 中包含 round 列，则自动过滤出第一轮的数据，防止轨迹重叠
if 'round' in df.columns:
    df = df[df['round'] == 1]

# 提取数据
time = df['time'].values

# 有期望值的项
theta = df['theta'].values
theta_req = df['theta_req'].values
psi = df['psi'].values
psi_req = df['psi_req'].values
v = df['v'].values
v_req = df['v_req'].values
h = df['h'].values
h_req = df['h_req'].values

# 没有期望值的项
phi = df['phi'].values
alpha = df['alpha'].values
beta = df['beta'].values
Ny = df['Ny'].values

# 控制量
aileron = df['aileron'].values
elevator = df['elevator'].values
rudder = df['rudder'].values
throttle = df['throttle'].values

# 角度误差量
psi_error = sub_of_degree(psi, psi_req)
theta_error = sub_of_degree(theta, theta_req)

# 转换展示范围(只管psi绝对值，误差值不转)
psi = rel2custom_degree(psi)
psi_req = rel2custom_degree(psi_req)

# 绘制
# figure(1) = plt.figure(figsize=(10, 6))
ax1 = plt.subplot(2, 2, 1)
ax1.plot(time, psi, label='psi')
ax1.plot(time, psi_req, label='psi_req')
ax1.legend()
plt.grid(True)

ax2 = plt.subplot(2, 2, 2)
ax2.plot(time, theta, label='theta')
ax2.plot(time, theta_req, label='theta_req')
ax2.legend()
plt.grid(True)

ax3 = plt.subplot(2, 2, 3)
ax3.plot(time, v, label='v')
ax3.plot(time, v_req, label='v_req')
ax3.legend()
plt.grid(True)

ax4 = plt.subplot(2, 2, 4)
ax4.plot(time, h, label='h')
ax4.plot(time, h_req, label='h_req')
ax4.legend()

plt.grid(True)
plt.show()
