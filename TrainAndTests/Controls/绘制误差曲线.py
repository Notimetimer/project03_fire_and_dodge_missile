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
file_name2 = "PID_wave_.csv"

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

# 创建画布
plt.figure(figsize=(15, 10))

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
            if req_col and i == 1: # 绘制最后一次加载的指令值作为参考
                ax.plot(t, df[req_col].values, 'k:', alpha=0.5, label='Command Line')

# 1. 航向角误差对比
ax1 = plt.subplot(3, 1, 1)
plot_comparison(ax1, [df1, df2], 'psi', ['RL', 'PID'], is_error=True, req_col='psi_req')
ax1.set_title("航向角误差 (Heading Error) 对比")
ax1.set_ylabel(r"$\varepsilon_{\psi}$ (°)")
ax1.legend(); ax1.grid(True)

# 2. 俯仰角对比
ax2 = plt.subplot(3, 1, 2)
plot_comparison(ax2, [df1, df2], 'theta', ['RL', 'PID'], req_col='theta_req')
ax2.set_title("俯仰角 (Pitch) 跟踪对比")
ax2.set_ylabel(r"$\theta$ (°)")
ax2.legend(); ax2.grid(True)

# 3. 速度对比
ax3 = plt.subplot(3, 1, 3)
plot_comparison(ax3, [df1, df2], 'v', ['RL', 'PID'], req_col='v_req')
ax3.set_title("速度 (Velocity) 跟踪对比")
ax3.set_ylabel("v (m/s)")
ax3.legend(); ax3.grid(True)

# 4. 高度对比
# ax4 = plt.subplot(2, 2, 4)
# plot_comparison(ax4, [df1, df2], 'h', ['RL', 'PID'], req_col='h_req')
# ax4.set_title("高度 (Altitude) 跟踪对比")
# ax4.set_ylabel("Height (m)")
# ax4.legend(); ax4.grid(True)

plt.suptitle(f"控制器性能对比\nRL: {file_name}\nPID: {file_name2}", fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
