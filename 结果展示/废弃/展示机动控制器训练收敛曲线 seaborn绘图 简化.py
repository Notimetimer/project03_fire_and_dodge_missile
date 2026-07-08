# python库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# 自定义库
from _context import *
from Visualize.plot_training_curve_from_csv import plot_training_results
from Algorithms.rl_utils import moving_average # 使用我们新版本的MA

# =============================================================================
# 全局样式配置 (Seaborn 主旋律)
# =============================================================================
# 配置 Seaborn 主题，使用 darkgrid 风格，并设置中文字体
sns.set_theme(style="darkgrid", font="SimHei", rc={"axes.unicode_minus": False})

# =============================================================================
# 全局视觉参数 (在此调节线粗细和深浅)
# =============================================================================
LW_SMOOTH = 1.5   # 平滑主趋势线粗
LW_RAW = 0.5      # 原始噪声背景线细
ALPHA_RAW = 0.1   # 原始噪声背景透明度 (0.1 风格)

# 数据路径
returns_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining.csv")
survive_rates_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining.csv")
returns_auto_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining_AutoStd.csv")
survive_rates_auto_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining_AutoStd.csv")
psi_error_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining.csv")
theta_error_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining.csv")
v_error_path = os.path.join(project_root, "logs", "EMAVErrorOfControllerTraining.csv")
v_error_auto_path = os.path.join(project_root, "logs", "EMAVErrorOfControllerTraining_AutoStd.csv")

ao_error_path = os.path.join(project_root, "logs", "AvgAO.csv")
ao_error_auto_path = os.path.join(project_root, "logs", "AvgAO_AutoStd.csv")

# =============================================================================
# PID 基准性能常量 (来自于最近的测试报告)
# =============================================================================
PID_AVG_REWARD = -111.73
PID_AVG_V_ERR = 180.758
PID_AVG_PSI_ERR = 4.886
PID_AVG_THETA_ERR = 2.625
PID_AVG_AO = 5.619

# 数据预处理通用函数
def prepare_metric_df(path, label, scale=1.0, smooth_p=35):
    df = pd.read_csv(path)
    df['Raw'] = df['Value'].values * scale
    df['Smooth'] = moving_average(df['Value'].values, smooth_p) * scale
    df['Metric'] = label 
    return df

# =============================================================================
# Figure 1：奖励函数 & 生存率
# =============================================================================
fig1 = plt.figure(figsize=(8, 6), dpi=100)
ax1_l = fig1.add_subplot(1, 1, 1)

# 准备数据
df_reward = prepare_metric_df(returns_path, "Episode Reward", smooth_p=35)
df_survive = prepare_metric_df(survive_rates_path, "Survive Rate", smooth_p=20)
df_reward_auto = prepare_metric_df(returns_auto_path, "Episode Reward", smooth_p=35)
df_survive_auto = prepare_metric_df(survive_rates_auto_path, "Survive Rate", smooth_p=20)

# 绘制左轴 (Reward)
color_reward = "tab:red"
sns.lineplot(data=df_reward, x='Step', y='Raw', ax=ax1_l, color=color_reward, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward, x='Step', y='Smooth', ax=ax1_l, color=color_reward, linewidth=LW_SMOOTH, label='Episode Reward (Limited Std)')

# 绘制左轴 (Reward AutoStd)
color_reward_auto = "tab:orange"
sns.lineplot(data=df_reward_auto, x='Step', y='Raw', ax=ax1_l, color=color_reward_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward_auto, x='Step', y='Smooth', ax=ax1_l, color=color_reward_auto, linewidth=LW_SMOOTH, label='Episode Reward')

# 绘制右轴 (Survive Rate)
ax1_r = ax1_l.twinx()
color_survive = "tab:blue"
sns.lineplot(data=df_survive, x='Step', y='Raw', ax=ax1_r, color=color_survive, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive, x='Step', y='Smooth', ax=ax1_r, color=color_survive, linewidth=LW_SMOOTH, label='Survive Rate (Limited Std)')

# 绘制右轴 (Survive Rate AutoStd)
color_survive_auto = "tab:green"
sns.lineplot(data=df_survive_auto, x='Step', y='Raw', ax=ax1_r, color=color_survive_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive_auto, x='Step', y='Smooth', ax=ax1_r, color=color_survive_auto, linewidth=LW_SMOOTH, label='Survive Rate')

# 合并图例
ax1_r.legend_.remove() if ax1_r.get_legend() else None 
h1, l1 = ax1_l.get_legend_handles_labels()
h2, l2 = ax1_r.get_legend_handles_labels()
ax1_l.legend(handles=h1+h2, labels=l2+l1, loc='lower right', frameon=True, fontsize=9)

# 装饰 (参考 Figure 1 风格)
# 调深 Y 轴颜色 (加深版彩色)
color_reward_dark = "tab:red"
color_survive_dark = "tab:blue"

ax1_l.set_ylabel("Episode Reward", color=color_reward_dark, fontweight='bold', fontsize=12)
ax1_r.set_ylabel("Survive Rate", color=color_survive_dark, fontweight='bold', fontsize=12)
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
# Figure 2：Pointing Error (AO) & V Error 追踪
# =============================================================================
fig2 = plt.figure(figsize=(8, 6), dpi=100)
ax2 = fig2.add_subplot(1, 1, 1)

# 读取 AO 数据
df_ao = prepare_metric_df(ao_error_path, "Pointing Error (Limited Std)", smooth_p=20)
df_ao_auto = prepare_metric_df(ao_error_auto_path, "Pointing Error", smooth_p=20)

# 绘制左轴 (AO Error) - 暖色系
color_ao = "tab:red"
sns.lineplot(data=df_ao, x='Step', y='Raw', ax=ax2, color=color_ao, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_ao, x='Step', y='Smooth', ax=ax2, color=color_ao, linewidth=LW_SMOOTH, label='Pointing Error (Limited Std)')

color_ao_auto = "tab:orange"
sns.lineplot(data=df_ao_auto, x='Step', y='Raw', ax=ax2, color=color_ao_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_ao_auto, x='Step', y='Smooth', ax=ax2, color=color_ao_auto, linewidth=LW_SMOOTH, label='Pointing Error')

# 绘制左轴 PID 基准线 (虚线)
ax2.axhline(PID_AVG_AO, color='tab:red', linestyle='--', linewidth=1.5, alpha=1.0, label='PID Pointing Error')

# 创建右轴 (V Error) - 冷色系
ax2_r = ax2.twinx()
df_v = prepare_metric_df(v_error_path, "V Error", smooth_p=35)
df_v_auto = prepare_metric_df(v_error_auto_path, "V Error", smooth_p=35)

color_v = "tab:blue"
sns.lineplot(data=df_v, x='Step', y='Raw', ax=ax2_r, color=color_v, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_v, x='Step', y='Smooth', ax=ax2_r, color=color_v, linewidth=LW_SMOOTH, label='V Error (Limited Std)')

color_v_auto = "tab:cyan"
sns.lineplot(data=df_v_auto, x='Step', y='Raw', ax=ax2_r, color=color_v_auto, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_v_auto, x='Step', y='Smooth', ax=ax2_r, color=color_v_auto, linewidth=LW_SMOOTH, label='V Error')

# 绘制右轴 PID 基准线 (虚线)
ax2_r.axhline(PID_AVG_V_ERR, color=color_v, linestyle='--', linewidth=1.5, alpha=1.0, label='PID V Error')

# 合并图例
ax2_r.legend_.remove() if ax2_r.get_legend() else None
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2_r.get_legend_handles_labels()
ax2.legend(handles=h1+h2, labels=l2+l1, loc='upper right', frameon=True, fontsize=9)

# 装饰
color_ao_dark = "tab:red"
color_v_dark = "tab:blue"

ax2.set_ylabel("Pointing Error (°)", color=color_ao_dark, fontweight='bold', fontsize=12)
ax2_r.set_ylabel("V Error (m/s)", color=color_v_dark, fontweight='bold', fontsize=12)
ax2.set_xlabel("Steps", fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_ao_dark, labelsize=10, width=1.5)
ax2_r.tick_params(axis='y', labelcolor=color_v_dark, labelsize=10, width=1.5)

for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color(color_ao_dark)
for label in ax2_r.get_yticklabels():
    label.set_fontweight('bold')
    label.set_color(color_v_dark)
for label in ax2.get_xticklabels():
    label.set_fontweight('bold')

ax2.set_axisbelow(True)
ax2_r.set_axisbelow(True)
ax2.grid(False, axis='y')
ax2.grid(True, axis='x', alpha=0.3, color='lightgray')
ax2_r.grid(True, alpha=0.3, color='lightgray')
sns.despine(ax=ax2, right=False)

fig2.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

plt.show()
