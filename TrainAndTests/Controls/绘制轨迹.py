import numpy as np
import matplotlib.pyplot as plt
import csv
from _context import *


# 读取 CSV 文件
current_dir = os.path.join(project_root, "TrainAndTests/Controls")
with open(os.path.join(current_dir, 
'test_result/FlightControl_parallel无课程无蒸馏_有过载限制_动态lr_wave__trajectory_20260319_113444.csv'
), 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 跳过表头
    data = list(reader)

# 如果 CSV 中包含 round 列，则自动过滤出第一轮的数据，防止轨迹重叠
if len(data[0]) > 7:
    first_round = data[0][-1]
    data = [row for row in data if row[-1] == first_round]

# 提取数据
time = [float(row[0]) for row in data]
uav_n = [float(row[1]) for row in data]
uav_u = [float(row[2]) for row in data]
uav_e = [float(row[3]) for row in data]
target_n = [float(row[4]) for row in data]
target_u = [float(row[5]) for row in data]
target_e = [float(row[6]) for row in data]

# 设置坐标轴等比例
def set_axes_equal(ax):
    """确保3D图的坐标轴单位长度相等。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])
    # 东北天
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    # 设置等显示缩放比例
    ax.set_box_aspect([1, 1, 1])

# 绘制轨迹
fig = plt.figure(1, figsize=(10, 8))
ax3d = fig.add_subplot(projection='3d')

# 只绘制红色的无人机轨迹
ax3d.plot(uav_e, uav_n, uav_u, c='r', label='UAV Trajectory', linewidth=2)

# 每隔 10s 绘制一次指示线 (每 10s / dt_decide = 500 个点，但推荐基于 time 判断)
for i in range(1, len(time)):
    # 每 10s 取一个点，考虑 dt=0.02
    if (round(time[i], 2) % 10 == 0) and (time[i] != time[i-1]):
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
        # 画个蓝色散点标出目标位置
        ax3d.scatter(t_p[0], t_p[1], t_p[2], c='b', s=10)

set_axes_equal(ax3d)
ax3d.legend()
plt.title("UAV Trajectory & Pointing Indicators (per 10s)")
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(uav_e, uav_n, uav_u, label='UAV')
# plt.plot(target_e, target_n, target_u, label='Target')
# plt.xlabel('East')
# plt.ylabel('Up')
# plt.zlabel('North')
# plt.title('Flight Trajectory')
# plt.legend()
# plt.grid(True)
# plt.show()