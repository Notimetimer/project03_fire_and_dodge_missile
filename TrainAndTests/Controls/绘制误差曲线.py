import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from _context import *
from Visualize.plot_tools import set_axes_equal
from Math_calculates.sub_of_angles import *

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 CSV 文件路径
current_dir = os.path.join(project_root, "TrainAndTests/Controls")
test_res_dir = os.path.join(project_root, "logs", "control_test_results")

file_name = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave_.csv"

"FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave__steady"
"FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave_"

file_name2 = "PID_wave_.csv"

"PID_wave__steady"
"PID_wave_"

def load_processed_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告：文件不存在 {file_path}")
        return None
    df = pd.read_csv(file_path)
    if 'round' in df.columns:
        df = df[df['round'] == 1]
    return df

# 加载两个文件的数据
df1 = load_processed_data(os.path.join(test_res_dir, file_name))
df2 = load_processed_data(os.path.join(test_res_dir, file_name2))

# --- 辅助绘图函数 ---
def plot_comparison(ax, df_list, col_name, labels, is_error=False, req_col=None):
    colors = ['r', 'b'] # 红的是 RL，蓝的是 PID
    linestyles = ['-', '--']
    
    for i, df in enumerate(df_list):
        if df is None: continue
        t = df['time'].values
        val = df[col_name].values
        
        if is_error and req_col:
            req = df[req_col].values
            err = sub_of_degree(val, req)
            ax.plot(t, err, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]} Error')
        else:
            ax.plot(t, val, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]} {col_name}')
            if req_col:
                # 绘制各自的指令值作为参考，使用相同颜色但不同线型
                ax.plot(t, df[req_col].values, color=colors[i], linestyle=':', alpha=0.5, label=f'{labels[i]} Target')

# --- Figure 1: 核心跟踪性能对比 ---
plt.figure(figsize=(15, 10))

# 1. 航向角误差对比
ax1 = plt.subplot(3, 1, 1)
plot_comparison(ax1, [df1, df2], 'psi', ['PPO', 'PID'], is_error=True, req_col='psi_req')
ax1.set_title("航向角误差 (Heading Error) 对比")
ax1.set_ylabel(r"$\varepsilon_{\psi}$ (°)")
ax1.legend(); ax1.grid(True)

# 2. 俯仰角对比
ax2 = plt.subplot(3, 1, 2)
plot_comparison(ax2, [df1, df2], 'theta', ['PPO', 'PID'], req_col='theta_req')
ax2.set_title("俯仰角 (Pitch) 跟踪对比")
ax2.set_ylabel(r"$\theta$ (°)")
ax2.legend(); ax2.grid(True)

# 3. 速度对比
ax3 = plt.subplot(3, 1, 3)
plot_comparison(ax3, [df1, df2], 'v', ['PPO', 'PID'], req_col='v_req')
ax3.set_title("速度 (Velocity) 跟踪对比")
ax3.set_ylabel("v (m/s)")
ax3.legend(); ax3.grid(True)

plt.suptitle(f"控制器核心性能对比\nRL: {file_name}\nPID: {file_name2}", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Figure 2: 飞行包线与控制量分析 ---
plt.figure(figsize=(15, 12))

# 1. Alpha 与 Ny 对比 (左轴 Alpha, 右轴 Ny)
ax2_1 = plt.subplot(2, 2, 1)
ax2_1_r = ax2_1.twinx()
ax2_1.plot(df1['time'], df1['alpha'], 'r-', label='RL Alpha')
ax2_1.plot(df2['time'], df2['alpha'], 'b--', label='PID Alpha')
ax2_1_r.plot(df1['time'], df1['Ny'], 'r:', alpha=0.6, label='RL Ny')
ax2_1_r.plot(df2['time'], df2['Ny'], 'b:', alpha=0.6, label='PID Ny')
ax2_1.set_title("迎角 (Alpha) 与 法向过载 (Ny)")
ax2_1.set_ylabel("Alpha (°)"); ax2_1_r.set_ylabel("Ny (g)")
ax2_1.legend(loc='upper left'); ax2_1_r.legend(loc='upper right'); ax2_1.grid(True)

# 2. Phi 与 高度 对比 (左轴 Phi, 右轴 h)
ax2_2 = plt.subplot(2, 2, 2)
ax2_2_r = ax2_2.twinx()
ax2_2.plot(df1['time'], df1['phi'], 'r-', label='RL Phi')
ax2_2.plot(df2['time'], df2['phi'], 'b--', label='PID Phi')
ax2_2_r.plot(df1['time'], df1['h'], 'r:', alpha=0.6, label='RL Height')
ax2_2_r.plot(df2['time'], df2['h'], 'b:', alpha=0.6, label='PID Height')
ax2_2.set_title("滚转角 (Phi) 与 高度 (Height)")
ax2_2.set_ylabel("Phi (°)"); ax2_2_r.set_ylabel("Height (m)")
ax2_2.legend(loc='upper left'); ax2_2_r.legend(loc='upper right'); ax2_2.grid(True)

# 3. RL 控制量
ax2_3 = plt.subplot(2, 2, 3)
ctrl_cols = ['aileron', 'elevator', 'rudder', 'throttle']
for col in ctrl_cols:
    ax2_3.plot(df1['time'], df1[col], label=col)
ax2_3.set_title("RL 控制器指令")
ax2_3.set_ylabel("Normalized Command")
ax2_3.legend(); ax2_3.grid(True)

# 4. PID 控制量
ax2_4 = plt.subplot(2, 2, 4)
for col in ctrl_cols:
    ax2_4.plot(df2['time'], df2[col], label=col)
ax2_4.set_title("PID 控制器指令")
ax2_4.set_ylabel("Normalized Command")
ax2_4.legend(); ax2_4.grid(True)

plt.suptitle(f"飞行包线与控制特性分析\nRL: {file_name}\nPID: {file_name2}", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
