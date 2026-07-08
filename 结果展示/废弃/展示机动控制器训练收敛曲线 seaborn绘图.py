# python库
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 强制交互式后端，支持图例拖动
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# 自定义库
from _context import *
from Visualize.plot_training_curve_from_csv import plot_training_results
from Algorithms.rl_utils import moving_average # 使用我们新版本的MA

# =============================================================================
# 全局样式配置 (Seaborn 主旋律) darkgrid 或者 whitegrid
# =============================================================================
# 配置 Seaborn 主题，使用自定义浅色背景，并设置中文字体
sns.set_theme(style="whitegrid", font="SimHei", rc={
    "axes.unicode_minus": False,
    # "axes.facecolor": "#f0f0f0",      # 自定义浅灰背景
    # "figure.facecolor": "#f0f0f0",
    # "axes.edgecolor": "#cccccc",
    # "grid.color": "#d0d0d0"
})

# =============================================================================
# 全局视觉参数 (在此调节线粗细和深浅)
# =============================================================================
LW_SMOOTH = 1.2   # 平滑主趋势线粗
LW_RAW = 0.7      # 原始噪声背景线细
ALPHA_RAW = 0.1   # 原始噪声背景透明度 (0.1 风格)

# 数据路径
returns_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining.csv")
survive_rates_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining.csv")
psi_error_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining.csv")
theta_error_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining.csv")
v_error_path = os.path.join(project_root, "logs", "EMAVErrorOfControllerTraining.csv")

returns_auto_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining_AutoStd.csv")
survive_rates_auto_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining_AutoStd.csv")
psi_error_auto_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining_AutoStd.csv")
theta_error_auto_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining_AutoStd.csv")
v_error_auto_path = os.path.join(project_root, "logs", "EMAVErrorOfControllerTraining_AutoStd.csv")

ao_error_path = os.path.join(project_root, "logs", "AvgAO.csv")
ao_error_auto_path = os.path.join(project_root, "logs", "AvgAO_AutoStd.csv")

# =============================================================================
# PID 基准性能常量 (来自于最近的测试报告)
# =============================================================================
PID_AVG_REWARD = 355.5
PID_AVG_SURVIVE = 1.0
PID_AVG_V_ERR = 183.5
PID_AVG_PSI_ERR = 5.038
PID_AVG_THETA_ERR = 3.37
PID_AVG_AO = 5.619

# 数据预处理通用函数
def prepare_metric_df(path, label, scale=1.0, smooth_p=35):
    df = pd.read_csv(path)
    df['Raw'] = df['Value'].values * scale
    df['Smooth'] = moving_average(df['Value'].values, smooth_p) * scale
    df['Metric'] = label 
    return df

# =============================================================================
# Figure 1：奖励函数 & 成功率
# =============================================================================
fig1 = plt.figure(figsize=(8, 6), dpi=100)
ax1_l = fig1.add_subplot(1, 1, 1)

# 准备数据
df_reward = prepare_metric_df(returns_path, "Episode Reward", smooth_p=35)
df_survive = prepare_metric_df(survive_rates_path, "Survive Rate", smooth_p=20)
df_reward_auto = prepare_metric_df(returns_auto_path, "Episode Reward", smooth_p=35)
df_survive_auto = prepare_metric_df(survive_rates_auto_path, "Survive Rate", smooth_p=20)

# 反转生存率 → 超出性能约束比率
df_survive['Raw'] = 1 - df_survive['Raw'].values
df_survive['Smooth'] = 1 - df_survive['Smooth'].values
df_survive_auto['Raw'] = 1 - df_survive_auto['Raw'].values
df_survive_auto['Smooth'] = 1 - df_survive_auto['Smooth'].values

# 绘制左轴 (Reward)
color_reward = "tab:red"
sns.lineplot(data=df_reward, x='Step', y='Raw', ax=ax1_l, color=color_reward, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward, x='Step', y='Smooth', ax=ax1_l, color=color_reward, linewidth=LW_SMOOTH, label='EDC-PPO 回报')

# 绘制左轴 (Reward AutoStd)
color_reward_auto = "tab:orange"
sns.lineplot(data=df_reward_auto, x='Step', y='Raw', ax=ax1_l, color=color_reward_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward_auto, x='Step', y='Smooth', ax=ax1_l, color=color_reward_auto, linewidth=LW_SMOOTH, label='PPO 回报')

# 绘制 PID 基准奖励虚线 (暖色调，与左轴匹配)
ax1_l.axhline(PID_AVG_REWARD, color='indianred', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 回报')

# 绘制右轴 (Survive Rate)
ax1_r = ax1_l.twinx()
color_survive = "tab:blue"
sns.lineplot(data=df_survive, x='Step', y='Raw', ax=ax1_r, color=color_survive, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive, x='Step', y='Smooth', ax=ax1_r, color=color_survive, linewidth=LW_SMOOTH, label='EDC-PPO 超出性能约束比率')

# 绘制右轴 (Survive Rate AutoStd)
color_survive_auto = "tab:green"
sns.lineplot(data=df_survive_auto, x='Step', y='Raw', ax=ax1_r, color=color_survive_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive_auto, x='Step', y='Smooth', ax=ax1_r, color=color_survive_auto, linewidth=LW_SMOOTH*0.9, label='PPO 超出性能约束比率')

# 绘制 PID 基准超出性能约束比率虚线 (冷色调，与右轴匹配)
pid_exceed = 1 - PID_AVG_SURVIVE
ax1_r.axhline(pid_exceed, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 超出性能约束比率')

# 在虚线末端标注 PID 数值
xmax1 = ax1_l.get_xlim()[1]
ax1_l.text(xmax1*0.98, PID_AVG_REWARD*-0.8, f'PID: {PID_AVG_REWARD}',
           color='indianred', fontsize=9, fontweight='bold', va='bottom', ha='right')
ax1_r.text(xmax1*0.98, pid_exceed + 0.02, f'PID: {pid_exceed:.2f}',
           color='steelblue', fontsize=9, fontweight='bold', va='bottom', ha='right')

# 合并图例
ax1_r.legend_.remove() if ax1_r.get_legend() else None
h1, l1 = ax1_l.get_legend_handles_labels()
h2, l2 = ax1_r.get_legend_handles_labels()
leg1 = ax1_l.legend(handles=h1+h2, labels=l1+l2, loc='center right', frameon=True, fontsize=9)

# 装饰 (参考 Figure 1 风格)
# 调深 Y 轴颜色 (加深版彩色)
color_reward_dark = "tab:red"
color_survive_dark = "tab:blue"

ax1_l.set_ylabel("平均累积回报", color=color_reward_dark, fontweight='bold', fontsize=12) # Return
ax1_r.set_ylabel("超出性能约束比率", color=color_survive_dark, fontweight='bold', fontsize=12)
ax1_l.set_xlabel("Steps", fontweight='bold')
ax1_l.tick_params(axis='y', labelcolor=color_reward_dark, labelsize=10, width=1.5)
ax1_r.tick_params(axis='y', labelcolor=color_survive_dark, labelsize=10, width=1.5)

# 强制所有坐标轴刻度数字加粗，并保持轴对应的颜色 (加深版)
for label in ax1_l.get_yticklabels(): 
    label.set_fontweight('bold')
    label.set_color(color_reward_dark)
for label in ax1_r.get_yticklabels(): 
    label.set_fontweight('bold')
    label.set_color(color_survive_dark)
for label in ax1_l.get_xticklabels(): 
    label.set_fontweight('bold')

# 保持网格在最底层
ax1_l.set_axisbelow(True)
ax1_r.set_axisbelow(True)

# 隐藏左侧 y 轴网格，显示 x 轴和右侧 y 轴网格，淡化网格线
ax1_l.grid(False, axis='y')
ax1_l.grid(True, axis='x', alpha=0.3, color='lightgray')
ax1_r.grid(True, alpha=0.3, color='lightgray')
sns.despine(ax=ax1_l, right=False)

# 布局收尾 (Figure 1)
fig1.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

# =============================================================================
# Figure 2：指令跟踪误差 (AO) 追踪 — 半对数坐标
# =============================================================================
fig2 = plt.figure(figsize=(8, 6), dpi=100)
ax2 = fig2.add_subplot(1, 1, 1)

# 读取 AO 数据
df_ao = prepare_metric_df(ao_error_path, "指令跟踪误差 (EDC-PPO)", smooth_p=20)
df_ao_auto = prepare_metric_df(ao_error_auto_path, "指令跟踪误差 (PPO)", smooth_p=20)

# 绘制 AO 曲线 - 暖色系
color_ao = "tab:red"
sns.lineplot(data=df_ao, x='Step', y='Raw', ax=ax2, color=color_ao, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_ao, x='Step', y='Smooth', ax=ax2, color=color_ao, linewidth=LW_SMOOTH, label='EDC-PPO 指令跟踪误差')

color_ao_auto = "tab:orange"
sns.lineplot(data=df_ao_auto, x='Step', y='Raw', ax=ax2, color=color_ao_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_ao_auto, x='Step', y='Smooth', ax=ax2, color=color_ao_auto, linewidth=LW_SMOOTH, label='PPO 指令跟踪误差')

# 绘制 PID 基准线 (虚线)
ax2.axhline(PID_AVG_AO, color='indianred', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 指令跟踪误差')

# 在虚线末端标注 PID 数值
xmax2 = ax2.get_xlim()[1]
ax2.text(xmax2*0.98, PID_AVG_AO*1.15, f'PID: {PID_AVG_AO}°',
         color='indianred', fontsize=9, fontweight='bold', va='bottom', ha='right')

# 图例
handles, labels = ax2.get_legend_handles_labels()
filtered_handles, filtered_labels = [], []
for h, l in zip(handles, labels):
    if l not in ['Metric', 'Raw']:
        filtered_handles.append(h)
        filtered_labels.append(l)
ax2.legend(handles=filtered_handles, labels=filtered_labels, loc='upper right', frameon=True, fontsize=9)

# 装饰 — 半对数坐标
import matplotlib.ticker as ticker
ax2.set_yscale('log')
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1, 2, 5], numticks=5))
formatter = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
ax2.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=20))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

ax2.set_ylabel("指令跟踪误差(°)", fontweight='bold', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.5)
ax2.set_xlabel("Steps", fontweight='bold')

for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2.get_xticklabels():
    label.set_fontweight('bold')

ax2.set_axisbelow(True)
ax2.grid(True, which='both', axis='y', alpha=0.3, color='lightgray')
ax2.grid(True, axis='x', alpha=0.5, color='lightgray')
sns.despine(ax=ax2, right=True)

fig2.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

plt.show()
