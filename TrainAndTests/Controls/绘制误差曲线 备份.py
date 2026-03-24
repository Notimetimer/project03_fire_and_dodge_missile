import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from _context import *
from Visualize.plot_tools import set_axes_equal
from Math_calculates.sub_of_angles import *

# 设置字体以支持中文及调整字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 14        # 坐标轴标签 (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 14       # x轴刻度
plt.rcParams['ytick.labelsize'] = 14       # y轴刻度
plt.rcParams['legend.fontsize'] = 14       # 图例
plt.rcParams['axes.titlesize'] = 14        # 子图标题 (ax.set_title)
plt.rcParams['figure.titlesize'] = 15      # 总标题 (plt.suptitle)

# 读取 CSV 文件路径
current_dir = os.path.join(project_root, "TrainAndTests/Controls")
test_res_dir = os.path.join(project_root, "logs", "control_test_results")

file_name = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave_.csv"

"FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave__steady"
"FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_wave_"

file_name2 = "PID_wave_.csv"

"PID_wave__steady"
"PID_wave_"

"FlightControl_parallel无课程无蒸馏_有过载限制_动态lr_wave_"

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
            ax.plot(t, err, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]}')
        else:
            ax.plot(t, val, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]}') #  {col_name}')
            if req_col:
                # 绘制各自的指令值作为参考，使用相同颜色但不同线型
                ax.plot(t, df[req_col].values, color=colors[i], linestyle=':', alpha=0.8, label=f'{labels[i]} Target')

# --- Figure 1: 核心跟踪性能对比 ---
plt.figure(figsize=(15, 12))

# 1. 航向角误差对比
ax1 = plt.subplot(2, 2, 1)
plot_comparison(ax1, [df1, df2], 'psi', ['PPO', 'PID'], is_error=True, req_col='psi_req')
# ax1.set_title("航向角误差 (Heading Error) 对比")
ax1.set_ylabel("heading error/ degree")
# ax1.set_ylabel(r"$\varepsilon_{\psi}$ (°)")
ax1.legend(); ax1.grid(True)

# 2. 俯仰角对比
ax2 = plt.subplot(2, 2, 2)
plot_comparison(ax2, [df1, df2], 'theta', ['PPO', 'PID'], req_col='theta_req')
# ax2.set_title("俯仰角 (Pitch) 跟踪对比")
ax2.set_ylabel("pitch angle/ degree")
# ax2.set_ylabel(r"$\theta$ (°)")
ax2.legend(); ax2.grid(True)

# 3. 速度对比
ax3 = plt.subplot(2, 2, 3)
plot_comparison(ax3, [df1, df2], 'v', ['PPO', 'PID'], req_col='v_req')
# ax3.set_title("速度 (Velocity) 跟踪对比")
ax3.set_ylabel("speed/ m/s")
ax3.legend(); ax3.grid(True)

plt.suptitle(f"控制器核心性能对比\nRL: {file_name}\nPID: {file_name2}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- Figure 2: 飞行包线与控制量分析 ---
plt.figure(figsize=(15, 12))

# 1. Alpha 与 Ny 对比 (左轴 Alpha, 右轴 Ny)
ax2_1 = plt.subplot(2, 2, 1)
ax2_1_r = ax2_1.twinx()
ax2_1.plot(df1['time'], df1['alpha'], 'r-', label='PPO')
ax2_1.plot(df2['time'], df2['alpha'], 'b--', label='PID')
ax2_1_r.plot(df1['time'], df1['Ny'], 'r:', alpha=0.6, label='PPO')
ax2_1_r.plot(df2['time'], df2['Ny'], 'b:', alpha=0.6, label='PID')
# ax2_1.set_title("迎角 (Alpha) 与 法向过载 (Ny)")
ax2_1.set_ylabel("Attack Angle/ degree"); ax2_1_r.set_ylabel("Overload/ g")
ax2_1.legend(loc='upper left'); ax2_1_r.legend(loc='upper right'); ax2_1.grid(True)

# 2. Phi 与 高度 对比 (左轴 Phi, 右轴 h)
ax2_2 = plt.subplot(2, 2, 2)
ax2_2_r = ax2_2.twinx()
ax2_2.plot(df1['time'], df1['phi'], 'r-', label='PPO')
ax2_2.plot(df2['time'], df2['phi'], 'b--', label='PID')
ax2_2_r.plot(df1['time'], df1['h'], 'r:', alpha=0.6, label='PPO')
ax2_2_r.plot(df2['time'], df2['h'], 'b:', alpha=0.6, label='PID')
# ax2_2.set_title("滚转角 (Phi) 与 高度 (Height)")
ax2_2.set_ylabel("Phi (°)"); ax2_2_r.set_ylabel("Height (m)")
ax2_2.legend(loc='upper left'); ax2_2_r.legend(loc='upper right'); ax2_2.grid(True)

# 3. RL 控制量
ax2_3 = plt.subplot(2, 2, 3)
ctrl_cols = ['aileron', 'elevator', 'rudder', 'throttle']
for col in ctrl_cols:
    smoothed_val = df1[col].ewm(span=10).mean()
    ax2_3.plot(df1['time'], smoothed_val, label=col)
# ax2_3.set_title("RL 控制器指令")
ax2_3.set_ylabel("Normalized Command")
ax2_3.legend(); ax2_3.grid(True)

# 4. PID 控制量
ax2_4 = plt.subplot(2, 2, 4)
for col in ctrl_cols:
    smoothed_val = df2[col].ewm(span=10).mean()
    ax2_4.plot(df2['time'], smoothed_val, label=col)
# ax2_4.set_title("PID 控制器指令")
ax2_4.set_ylabel("Normalized Command")
ax2_4.legend(); ax2_4.grid(True)

plt.suptitle(f"飞行包线与控制特性分析\nRL: {file_name}\nPID: {file_name2}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 统计性能指标打印 ---
def print_stats(df, label):
    if df is None:
        print(f"\n警告：未能加载 {label} 的数据，跳过统计。")
        return
    
    # 计算各项误差的绝对值
    psi_err = np.abs(sub_of_degree(df['psi'].values, df['psi_req'].values))
    theta_err = np.abs(sub_of_degree(df['theta'].values, df['theta_req'].values))
    v_err = np.abs(df['v'].values - df['v_req'].values)
    
    # 计算统计量
    avg_psi_err = np.nanmean(psi_err)
    avg_theta_err = np.nanmean(theta_err)
    avg_v_err = np.nanmean(v_err)
    
    max_ny = df['Ny'].max()
    min_ny = df['Ny'].min()
    max_alpha = df['alpha'].max()
    min_alpha = df['alpha'].min()
    
    print(f"\n[{label}] 性能统计指标:")
    print(f"  算术平均航向误差 : {avg_psi_err:.4f} °")
    print(f"  算术平均俯仰误差 : {avg_theta_err:.4f} °")
    print(f"  算术平均速度误差 : {avg_v_err:.4f} m/s")
    print(f"  最大过载 (Max Ny): {max_ny:.4f} g")
    print(f"  最小过载 (Min Ny): {min_ny:.4f} g")
    print(f"  最大迎角 (Max Alpha): {max_alpha:.4f} °")
    print(f"  最小迎角 (Min Alpha): {min_alpha:.4f} °")

print("\n" + "="*50)
print("             控制器性能统计汇总")
print("="*50)
print_stats(df1, "PPO")
print_stats(df2, "PID")
print("="*50 + "\n")

plt.show()
