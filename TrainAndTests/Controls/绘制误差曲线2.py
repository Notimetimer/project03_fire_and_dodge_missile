import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from _context import *
from Visualize.plot_tools import set_axes_equal
from Math_calculates.sub_of_angles import *
import matplotlib.ticker as ticker

# 设置字体以支持中文及调整字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 14        # 坐标轴标签 (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 14       # x轴刻度
plt.rcParams['ytick.labelsize'] = 14       # y轴刻度
plt.rcParams['legend.fontsize'] = 12       # 图例
plt.rcParams['axes.titlesize'] = 14        # 子图标题 (ax.set_title)
plt.rcParams['figure.titlesize'] = 15      # 总标题 (plt.suptitle)
plt.rcParams['legend.framealpha'] = 0.5    # 图例背景透明度 (0为完全透明)

# 读取 CSV 文件路径
current_dir = os.path.join(project_root, "TrainAndTests/Controls")
test_res_dir = os.path.join(project_root, "logs", "control_test_results")

# 文件名定义
file_ppo_norm = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_violent_wave_.csv"
file_ppo_steady = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_mild_wave_.csv"
file_ppo_splits = "FlightControl_parallel目标会动_高度可超调_有过载限制_动态lr_SplitS_.csv"
file_pid_norm = "PID_violent_wave_.csv"
file_pid_steady = "PID_mild_wave_.csv"
file_pid_splits = "PID_SplitS_.csv"
file_original_ppo_norm = "FlightControl_parallel目标会动_高度可超调_自动调方差_violent_wave_.csv"
file_original_ppo_steady = "FlightControl_parallel目标会动_高度可超调_自动调方差_mild_wave_.csv"
file_original_ppo_splits = "FlightControl_parallel目标会动_高度可超调_自动调方差_SplitS_.csv"


def load_processed_data(file_path):
    if not os.path.exists(file_path):
        print(f"警告：文件不存在 {file_path}")
        return None
    df = pd.read_csv(file_path)
    if 'round' in df.columns:
        df = df[df['round'] == 1]
    return df

# 加载工况数据
df_ppo_norm = load_processed_data(os.path.join(test_res_dir, file_ppo_norm))
df_ppo_steady = load_processed_data(os.path.join(test_res_dir, file_ppo_steady))
df_ppo_splits = load_processed_data(os.path.join(test_res_dir, file_ppo_splits))
df_pid_norm = load_processed_data(os.path.join(test_res_dir, file_pid_norm))
df_pid_steady = load_processed_data(os.path.join(test_res_dir, file_pid_steady))
df_pid_splits = load_processed_data(os.path.join(test_res_dir, file_pid_splits))
df_original_ppo_norm = load_processed_data(os.path.join(test_res_dir, file_original_ppo_norm))
df_original_ppo_steady = load_processed_data(os.path.join(test_res_dir, file_original_ppo_steady))
df_original_ppo_splits = load_processed_data(os.path.join(test_res_dir, file_original_ppo_splits))

# --- 辅助函数：角度展开 ---
def unwrap_angles(angles):
    """
    使角度序列连续，避免 0/360 突跳
    """
    unwrapped = np.zeros_like(angles)
    unwrapped[0] = angles[0]
    for i in range(1, len(angles)):
        diff = sub_of_degree(angles[i], angles[i-1])
        unwrapped[i] = unwrapped[i-1] + diff
    return unwrapped

# --- 辅助绘图函数 ---
def plot_comparison(ax, df_list, col_name, labels, is_error=False, req_col=None, unwrap=False):
    colors = ['r', 'b', 'lightgreen'] # 红的是 PPO，蓝的是 PID，浅绿的是 Original PPO
    linestyles = ['-', '--', '-']
    alphas = [0.67, 0.67, 0.7]  # Original PPO 使用稍低的透明度
    
    # 先绘制 PPO 绿色曲线 (i=2) 在最底层，再绘制 EDC-PPO 和 PID 覆盖在上层
    draw_order = [2, 0, 1]
    for i in draw_order:
        df = df_list[i]
        if df is None: continue
        t = df['time'].values
        val = df[col_name].values
        
        if is_error and req_col:
            req = df[req_col].values
            err = sub_of_degree(val, req)
            ax.plot(t, err, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]}', alpha=alphas[i])
        else:
            if unwrap:
                val_plot = unwrap_angles(val)
                ax.plot(t, val_plot, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]}', alpha=alphas[i])
                if req_col:
                    req_plot = unwrap_angles(df[req_col].values)
                    ax.plot(t, req_plot, color=colors[i], linestyle=':', alpha=0.5, lw=1.5)
            else:
                ax.plot(t, val, color=colors[i], linestyle=linestyles[i], label=f'{labels[i]}', alpha=alphas[i])
                if req_col:
                    ax.plot(t, df[req_col].values, color=colors[i], linestyle=':', alpha=0.5, lw=1.5)

# --- Figure 1: 核心跟踪性能对比 (Mild vs Violent vs Split-S) ---
fig1 = plt.figure(figsize=(12, 6))

# 1. 航向角对比 (展开值)
ax1_l = plt.subplot(3, 3, 1)
plot_comparison(ax1_l, [df_ppo_steady, df_pid_steady, df_original_ppo_steady], 'psi', ['EDC-PPO', 'PID', 'PPO'], is_error=False, req_col='psi_req', unwrap=True)
ax1_l.set_ylabel("航向角(°)"); ax1_l.legend(loc='lower right', draggable=True); ax1_l.grid(True)
ax1_l.yaxis.set_major_locator(ticker.MultipleLocator(45))

ax1_m = plt.subplot(3, 3, 2)
plot_comparison(ax1_m, [df_ppo_norm, df_pid_norm, df_original_ppo_norm], 'psi', ['EDC-PPO', 'PID', 'PPO'], is_error=False, req_col='psi_req', unwrap=True)
ax1_m.set_ylabel("航向角(°)"); ax1_m.legend(loc='lower right', draggable=True); ax1_m.grid(True)
ax1_m.yaxis.set_major_locator(ticker.MultipleLocator(45))

ax1_r = plt.subplot(3, 3, 3)
plot_comparison(ax1_r, [df_ppo_splits, df_pid_splits, df_original_ppo_splits], 'psi', ['EDC-PPO', 'PID', 'PPO'], is_error=False, req_col='psi_req', unwrap=True)
ax1_r.set_ylabel("航向角(°)"); ax1_r.legend(loc='lower right', draggable=True); ax1_r.grid(True)
ax1_r.yaxis.set_major_locator(ticker.MultipleLocator(45))

# 2. 俯仰角对比
ax2_l = plt.subplot(3, 3, 4)
plot_comparison(ax2_l, [df_ppo_steady, df_pid_steady, df_original_ppo_steady], 'theta', ['EDC-PPO', 'PID', 'PPO'], req_col='theta_req')
ax2_l.set_ylabel("俯仰角 (°)"); ax2_l.legend(loc='lower right', draggable=True); ax2_l.grid(True)
# ax2_l.yaxis.set_major_locator(ticker.MultipleLocator(45))

ax2_m = plt.subplot(3, 3, 5)
plot_comparison(ax2_m, [df_ppo_norm, df_pid_norm, df_original_ppo_norm], 'theta', ['EDC-PPO', 'PID', 'PPO'], req_col='theta_req')
ax2_m.set_ylabel("俯仰角 (°)"); ax2_m.legend(loc='lower right', draggable=True); ax2_m.grid(True)
# ax2_l.yaxis.set_major_locator(ticker.MultipleLocator(45))

ax2_r = plt.subplot(3, 3, 6)
plot_comparison(ax2_r, [df_ppo_splits, df_pid_splits, df_original_ppo_splits], 'theta', ['EDC-PPO', 'PID', 'PPO'], req_col='theta_req')
ax2_r.set_ylabel("俯仰角 (°)"); ax2_r.legend(loc='lower right', draggable=True); ax2_r.grid(True)
# ax2_l.yaxis.set_major_locator(ticker.MultipleLocator(45))

# 3. 速度对比
ax3_l = plt.subplot(3, 3, 7)
plot_comparison(ax3_l, [df_ppo_steady, df_pid_steady, df_original_ppo_steady], 'v', ['EDC-PPO', 'PID', 'PPO'], req_col='v_req')
ax3_l.set_ylabel("空速 (m/s)"); ax3_l.set_xlabel("时间 (s)"); ax3_l.legend(loc='lower right', draggable=True); ax3_l.grid(True)

ax3_m = plt.subplot(3, 3, 8)
plot_comparison(ax3_m, [df_ppo_norm, df_pid_norm, df_original_ppo_norm], 'v', ['EDC-PPO', 'PID', 'PPO'], req_col='v_req')
ax3_m.set_ylabel("空速 (m/s)"); ax3_m.set_xlabel("时间 (s)"); ax3_m.legend(loc='lower right', draggable=True); ax3_m.grid(True)

ax3_r = plt.subplot(3, 3, 9)
plot_comparison(ax3_r, [df_ppo_splits, df_pid_splits, df_original_ppo_splits], 'v', ['EDC-PPO', 'PID', 'PPO'], req_col='v_req')
ax3_r.set_ylabel("空速 (m/s)"); ax3_r.set_xlabel("时间 (s)"); ax3_r.legend(loc='lower right', draggable=True); ax3_r.grid(True)

# plt.suptitle(f"控制器核心性能对比 (Mild vs Violent vs Split-S)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# # --- Figure 2: 飞行包线与控制量分析 (使用 Normal 工况) ---
# plt.figure(figsize=(10, 8))

# # 1. Alpha 与 Ny 对比
# ax2_1 = plt.subplot(2, 2, 1)
# ax2_1_r = ax2_1.twinx()
# ax2_1.plot(df_ppo_norm['time'], df_ppo_norm['alpha'], 'r-', label='EDC-PPO')
# ax2_1.plot(df_pid_norm['time'], df_pid_norm['alpha'], 'b--', label='PID')
# ax2_1.plot(df_original_ppo_norm['time'], df_original_ppo_norm['alpha'], 'lightgreen', linestyle='-.', label='PPO', alpha=0.6)
# ax2_1_r.plot(df_ppo_norm['time'], df_ppo_norm['Ny'], 'r:', alpha=0.8)
# ax2_1_r.plot(df_pid_norm['time'], df_pid_norm['Ny'], 'b:', alpha=0.8)
# ax2_1_r.plot(df_original_ppo_norm['time'], df_original_ppo_norm['Ny'], 'lightgreen', linestyle=':', alpha=0.6)
# ax2_1.set_ylabel("Attack Angle / degree"); ax2_1_r.set_ylabel("Overload / g")
# ax2_1.set_xlabel("时间 (s)")
# ax2_1.legend(loc='upper left'); ax2_1_r.legend(loc='upper right'); ax2_1.grid(True)

# # 2. Phi 与 高度 对比
# ax2_2 = plt.subplot(2, 2, 2)
# ax2_2_r = ax2_2.twinx()
# ax2_2.plot(df_ppo_norm['time'], df_ppo_norm['phi'], 'r-', label='EDC-PPO')
# ax2_2.plot(df_pid_norm['time'], df_pid_norm['phi'], 'b--', label='PID')
# ax2_2.plot(df_original_ppo_norm['time'], df_original_ppo_norm['phi'], 'lightgreen', linestyle='-.', label='PPO', alpha=0.6)
# ax2_2_r.plot(df_ppo_norm['time'], df_ppo_norm['h'], 'r:', alpha=0.8)
# ax2_2_r.plot(df_pid_norm['time'], df_pid_norm['h'], 'b:', alpha=0.8)
# ax2_2_r.plot(df_original_ppo_norm['time'], df_original_ppo_norm['h'], 'lightgreen', linestyle=':', alpha=0.6)
# ax2_2.set_ylabel("Phi / degree"); ax2_2_r.set_ylabel("Height / m")
# ax2_2.set_xlabel("时间 (s)")
# ax2_2.legend(loc='upper left'); ax2_2_r.legend(loc='upper right'); ax2_2.grid(True)

# # 3. PPO 控制量
# ax2_3 = plt.subplot(2, 2, 3)
# ctrl_cols = ['aileron', 'elevator', 'rudder', 'throttle']
# for col in ctrl_cols:
#     smoothed_val = df_ppo_norm[col].ewm(span=10).mean()
#     ax2_3.plot(df_ppo_norm['time'], smoothed_val, label=col)
# ax2_3.set_title("PPO Command")
# ax2_3.set_ylabel("Normalized Command"); ax2_3.set_xlabel("时间 (s)")
# ax2_3.legend(); ax2_3.grid(True)

# # 4. PID 控制量
# ax2_4 = plt.subplot(2, 2, 4)
# for col in ctrl_cols:
#     smoothed_val = df_pid_norm[col].ewm(span=10).mean()
#     ax2_4.plot(df_pid_norm['time'], smoothed_val, label=col)
# ax2_4.set_title("PID Command")
# ax2_4.set_ylabel("Normalized Command"); ax2_4.set_xlabel("时间 (s)")
# ax2_4.legend(); ax2_4.grid(True)

# # 5. Original PPO 控制量
# plt.figure(figsize=(10, 3))
# ax2_5 = plt.subplot(1, 3, 1)
# for col in ctrl_cols:
#     smoothed_val = df_original_ppo_norm[col].ewm(span=10).mean()
#     ax2_5.plot(df_original_ppo_norm['time'], smoothed_val, label=col, color='lightgreen', alpha=0.6)
# ax2_5.set_title("Original PPO Command")
# ax2_5.set_ylabel("Normalized Command"); ax2_5.set_xlabel("时间 (s)")
# ax2_5.legend(); ax2_5.grid(True)

# plt.suptitle(f"飞行包线与控制特性分析 (Normal Case)")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 统计性能指标打印 ---
def print_stats(df, label):
    if df is None:
        print(f"  [警告] 未能加载 {label} 的数据。")
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
    
    print(f"  [{label}] 性能统计指标:")
    print(f"    算术平均航向误差 : {avg_psi_err:.4f} °")
    print(f"    算术平均俯仰误差 : {avg_theta_err:.4f} °")
    print(f"    算术平均速度误差 : {avg_v_err:.4f} m/s")
    print(f"    最大过载 (Max Ny): {max_ny:.4f} g")
    print(f"    最小过载 (Min Ny): {min_ny:.4f} g")
    print(f"    最大迎角 (Max Alpha): {max_alpha:.4f} °")
    print(f"    最小迎角 (Min Alpha): {min_alpha:.4f} °")

print("\n" + "="*60)
print("             控制器多任务性能统计汇总")
print("="*60)

scenarios = [
    ("平缓波动正弦信号", df_ppo_steady, df_pid_steady, df_original_ppo_steady),
    ("剧烈波动正弦信号", df_ppo_norm, df_pid_norm, df_original_ppo_norm),
    ("破S机动", df_ppo_splits, df_pid_splits, df_original_ppo_splits)
]

for name, d_ppo, d_pid, d_original_ppo in scenarios:
    print(f"\n>>> 工况: {name}")
    print_stats(d_ppo, "PPO")
    print_stats(d_pid, "PID")
    print_stats(d_original_ppo, "Original PPO")

print("\n" + "="*60 + "\n")

plt.show()
