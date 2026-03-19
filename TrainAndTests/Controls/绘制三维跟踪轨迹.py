import numpy as np
import matplotlib.pyplot as plt
import csv
from _context import *
from Visualize.plot_tools import set_axes_equal

import pandas as pd

# 读取 CSV 文件
current_dir = os.path.join(project_root, "TrainAndTests/Controls")

file_name = "FlightControl_parallel无课程无蒸馏_有过载限制_动态lr_wave__trajectory_20260319_113444.csv"

csv_path = os.path.join(current_dir, 'test_result', file_name)
df = pd.read_csv(csv_path)

# 如果 CSV 中包含 round 列，则自动过滤出第一轮的数据，防止轨迹重叠
if 'round' in df.columns:
    df = df[df['round'] == df['round'].iloc[0]]

# 提取数据
time = df['time'].values
uav_n = df['uav_N'].values
uav_u = df['uav_U'].values
uav_e = df['uav_E'].values
target_n = df['target_N'].values
target_u = df['target_U'].values
target_e = df['target_E'].values


# 绘制轨迹
fig = plt.figure(1, figsize=(10, 8))
ax3d = fig.add_subplot(projection='3d')

# 只绘制红色的无人机轨迹
ax3d.plot(uav_e, uav_n, uav_u, c='r', label='UAV Trajectory', linewidth=2)

# 每隔 10s 绘制一次指示线 (每 10s / dt_decide = 500 个点，但推荐基于 time 判断)
for i in range(1, len(time)):
    # 每 10s 取一个点，考虑 dt=0.02
    if (round(time[i], 2) % 30 == 0) and (time[i] != time[i-1]):
        u_p = np.array([uav_e[i], uav_n[i], uav_u[i]])
        t_p = np.array([target_e[i], target_n[i], target_u[i]])
        
        # 1. 计算轨迹切向 (这里用前后点的差表示速度方向)
        v_dir = np.array([uav_e[i]-uav_e[i-1], uav_n[i]-uav_n[i-1], uav_u[i]-uav_u[i-1]])
        v_norm = np.linalg.norm(v_dir)
        if v_norm > 0:
            v_dir = v_dir / v_norm
            tangent_end = u_p + v_dir * 10000 # 10km 长度
            ax3d.plot([u_p[0], tangent_end[0]], [u_p[1], tangent_end[1]], [u_p[2], tangent_end[2]], 
                      'y--', alpha=0.6, linewidth=1, label='Velocity Tangent' if 'Velocity Tangent' not in ax3d.get_legend_handles_labels()[1] else "")
        
        # 2. 绘制 UAV 与目标的连线
        ax3d.plot([u_p[0], t_p[0]], [u_p[1], t_p[1]], [u_p[2], t_p[2]], 
                  'b:', alpha=0.6, linewidth=1, label='To Target' if 'To Target' not in ax3d.get_legend_handles_labels()[1] else "")
        # # 画个蓝色散点标出目标位置
        # ax3d.scatter(t_p[0], t_p[1], t_p[2], c='b', s=10)

set_axes_equal(ax3d)
ax3d.set_xlabel('East (m)')
ax3d.set_ylabel('North (m)')
ax3d.set_zlabel('Up (m)')

ax3d.legend()
plt.title("UAV Trajectory & Pointing Indicators (per 10s)")
plt.show()
