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
# 配置 Seaborn 主题，让颜色和网格更漂亮
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 再次确认中文支持
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 全局视觉参数 (在此调节线粗细和深浅)
# =============================================================================
LW_SMOOTH = 1.5   # 平滑主趋势线粗
LW_RAW = 0.5      # 原始噪声背景线细
ALPHA_RAW = 0.25   # 原始噪声背景透明度 (0~1)

# 数据路径
returns_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining.csv")
survive_rates_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining.csv")
psi_error_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining.csv")
theta_error_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining.csv")
v_error_path = os.path.join(project_root, "logs", "EMAVErrorOfControllerTraining.csv")

# =============================================================================
# PID 基准性能常量 (来自于最近的测试报告)
# =============================================================================
PID_AVG_REWARD = -51.72
PID_AVG_V_ERR = 175.492
PID_AVG_PSI_ERR = 5.517
PID_AVG_THETA_ERR = 3.765

# 开启画布 (2x1 竖向排布)
fig = plt.figure(figsize=(8, 12), dpi=100)

# 读取并预处理数据的通用函数
def prepare_metric_df(path, label, scale=1.0, smooth_p=35):
    df = pd.read_csv(path)
    df['Raw'] = df['Value'].values * scale
    df['Smooth'] = moving_average(df['Value'].values, smooth_p) * scale
    df['Metric'] = label 
    return df

# =============================================================================
# 第一行：奖励函数 & 生存率
# =============================================================================
ax1_l = fig.add_subplot(2, 1, 1)

# 准备数据
df_reward = prepare_metric_df(returns_path, "Episode Reward", smooth_p=35)
df_survive = prepare_metric_df(survive_rates_path, "Survive Rate", smooth_p=20)

# 绘制左轴 (Reward)
color_reward = sns.color_palette("muted")[3]
sns.lineplot(data=df_reward, x='Step', y='Raw', ax=ax1_l, color=color_reward, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward, x='Step', y='Smooth', ax=ax1_l, color=color_reward, linewidth=LW_SMOOTH, label='Episode Reward')

# 绘制右轴 (Survive Rate)
ax1_r = ax1_l.twinx()
color_survive = sns.color_palette("muted")[0]
sns.lineplot(data=df_survive, x='Step', y='Raw', ax=ax1_r, color=color_survive, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive, x='Step', y='Smooth', ax=ax1_r, color=color_survive, linewidth=LW_SMOOTH, label='Survive Rate')

# 合并图例
ax1_r.legend_.remove() if ax1_r.get_legend() else None 
h1, l1 = ax1_l.get_legend_handles_labels()
h2, l2 = ax1_r.get_legend_handles_labels()
ax1_l.legend(handles=h1+h2, labels=l1+l2, loc='lower right', frameon=True)

# 装饰
ax1_l.set_ylabel("Episode Reward")
ax1_r.set_ylabel("Survive Rate")
ax1_l.set_xlabel("Steps")
ax1_l.grid(True, which='both', linestyle='--', alpha=0.9, color='lightgray')
ax1_r.grid(False) 
sns.despine(ax=ax1_l, right=False)

# =============================================================================
# 第二行：航向误差 & 俯仰误差 (极致数据驱动绘图)
# =============================================================================
ax2 = fig.add_subplot(2, 1, 2)

# 读取并预处理误差数据到一个 DataFrame
def prepare_metric_df(path, label, scale=1.0):
    df = pd.read_csv(path)
    df['Raw'] = df['Value'].values * scale
    df['Smooth'] = moving_average(df['Value'].values, 35) * scale
    df['Metric'] = label # 类别标签
    return df

df_psi = prepare_metric_df(psi_error_path, "PPO Psi Error")
df_theta = prepare_metric_df(theta_error_path, "PPO Theta Error")
df_v = prepare_metric_df(v_error_path, "PPO V Error", scale=1.0) # 速度不缩放，放到右轴

df_merged = pd.concat([df_psi, df_theta]) # 拼成长表，仅包含角度误差

# 一键启动 Seaborn 原生绘图 (主轴 ax2，对数坐标，画 Psi 和 Theta)
colors = [sns.color_palette("muted")[1], sns.color_palette("muted")[4]] # 橙色, 紫色
sns.lineplot(data=df_merged, x='Step', y='Raw', hue='Metric', ax=ax2, linewidth=LW_RAW, alpha=ALPHA_RAW, palette=colors, legend=False)
sns.lineplot(data=df_merged, x='Step', y='Smooth', hue='Metric', ax=ax2, linewidth=LW_SMOOTH, palette=colors)

# 创建双 Y 轴 (线性坐标，画 V Error)
ax2_r = ax2.twinx()
color_v = sns.color_palette("muted")[2] # 绿色
sns.lineplot(data=df_v, x='Step', y='Raw', ax=ax2_r, color=color_v, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_v, x='Step', y='Smooth', ax=ax2_r, color=color_v, linewidth=LW_SMOOTH, label='PPO V Error')

# 绘制 PID 基准误差 (加深颜色，使用点线)
ax2.axhline(PID_AVG_PSI_ERR, color=colors[0], linestyle='--', linewidth=1.5, alpha=1.0, label='PID Psi Error')
ax2.axhline(PID_AVG_THETA_ERR, color=colors[1], linestyle='--', linewidth=1.5, alpha=1.0, label='PID Theta Error')
ax2_r.axhline(PID_AVG_V_ERR, color=color_v, linestyle='--', linewidth=1.5, alpha=1.0, label='PID V Error')

# 处理并合并两轴图例
ax2_r.legend_.remove() if ax2_r.get_legend() else None 
handles1, labels1 = ax2.get_legend_handles_labels()
handles2, labels2 = ax2_r.get_legend_handles_labels()
# 去除冗余的组名 'Metric' 以及可能的 'Raw' 标签
filtered_handles, filtered_labels = [], []
for h, l in zip(handles1 + handles2, labels1 + labels2):
    if l not in ['Metric', 'Raw']: 
        filtered_handles.append(h)
        filtered_labels.append(l)
ax2.legend(handles=filtered_handles, labels=filtered_labels, loc='upper right', frameon=True)

# 装饰底轴风格
import matplotlib.ticker as ticker
ax2.set_yscale('log')
# 调整 Y 轴范围：通过调低 Y 轴下限（而不是拉高上限）来从物理高度上推开重合的线
# ax2.set_ylim(0.1, 50)     # 左轴：角度误差范围
ax2_r.set_ylim(0, 350) # 右轴：速度误差范围，下限扩充以提供视觉余量
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1, 2, 5], numticks=5))
formatter = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
ax2.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=20))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

ax2.set_ylabel("Error Degree (Log Scale)")
ax2_r.set_ylabel("V Error (m/s)")
ax2.set_xlabel("Steps")
ax2.grid(True, which='both', linestyle='--', alpha=0.9, color='lightgray')
ax2_r.grid(False) # 关键：关闭右边辅助轴的网格，避免和左边的对数网格冲突
sns.despine(ax=ax2, right=False)

# =============================================================================
# 布局细节优化：强行指定子图间距
# =============================================================================
# hspace=0.25 提供了巨大的垂直留白，完美避开 xlabel 与下面图表的冲突
fig.subplots_adjust(hspace=0.25, top=0.9, bottom=0.1, left=0.2, right=0.8)

plt.show()
