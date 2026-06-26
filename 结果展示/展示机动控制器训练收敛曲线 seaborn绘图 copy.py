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

# =============================================================================
# PID 基准性能常量 (来自于最近的测试报告)
# =============================================================================
PID_AVG_REWARD = 476.33
PID_AVG_SURVIVE = 1.0
PID_AVG_V_ERR = 34.906
PID_AVG_PSI_ERR = 5.038
PID_AVG_THETA_ERR = 2.785

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

# 反转生存率 → 失败率
df_survive['Raw'] = 1 - df_survive['Raw'].values
df_survive['Smooth'] = 1 - df_survive['Smooth'].values
df_survive_auto['Raw'] = 1 - df_survive_auto['Raw'].values
df_survive_auto['Smooth'] = 1 - df_survive_auto['Smooth'].values

# 绘制左轴 (Reward)
color_reward = "tab:red"
sns.lineplot(data=df_reward, x='Step', y='Raw', ax=ax1_l, color=color_reward, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_reward, x='Step', y='Smooth', ax=ax1_l, color=color_reward, linewidth=LW_SMOOTH, label='PPO 累积回报')


# 绘制 PID 基准奖励虚线 (暖色调，与左轴匹配)
ax1_l.axhline(PID_AVG_REWARD, color='indianred', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 累积回报')

# 绘制右轴 (Survive Rate)
ax1_r = ax1_l.twinx()
color_survive = "tab:blue"
sns.lineplot(data=df_survive, x='Step', y='Raw', ax=ax1_r, color=color_survive, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_survive, x='Step', y='Smooth', ax=ax1_r, color=color_survive, linewidth=LW_SMOOTH, label='PPO 失败率')


# 绘制 PID 基准失败率虚线 (冷色调，与右轴匹配)
pid_exceed = 1 - PID_AVG_SURVIVE
ax1_r.axhline(pid_exceed, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 失败率')

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

ax1_l.set_ylabel("平均累积回报", color='black', fontweight='bold', fontsize=12) # Return
ax1_r.set_ylabel("失败率", color='black', fontweight='bold', fontsize=12)
ax1_l.set_xlabel("Steps", fontweight='bold')
ax1_l.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.5)
ax1_r.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.5)

# 强制所有坐标轴刻度数字加粗
for label in ax1_l.get_yticklabels(): 
    label.set_fontweight('bold')
for label in ax1_r.get_yticklabels(): 
    label.set_fontweight('bold')
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
# Figure 2：误差追踪 (Psi & Theta & V)
# =============================================================================
fig2 = plt.figure(figsize=(8, 6), dpi=100)
ax2 = fig2.add_subplot(1, 1, 1)

# 读取并预处理误差数据到一个 DataFrame
def prepare_metric_df_simple(path, label, scale=1.0):
    df = pd.read_csv(path)
    df['Raw'] = df['Value'].values * scale
    df['Smooth'] = moving_average(df['Value'].values, 35) * scale
    df['Metric'] = label # 类别标签
    return df

df_psi = prepare_metric_df_simple(psi_error_path, "EDC-PPO 航向角误差")
df_theta = prepare_metric_df_simple(theta_error_path, "EDC-PPO 俯仰角误差")
df_psi_auto = prepare_metric_df_simple(psi_error_auto_path, "PPO 航向角误差")
df_theta_auto = prepare_metric_df_simple(theta_error_auto_path, "PPO 俯仰角误差")
df_v = prepare_metric_df_simple(v_error_path, "PPO 速度误差(m/s)", scale=1.0) # 速度不缩放，放到右轴

colors_main = ["tab:red", "tab:blue"]    # 与 Figure1 左/右轴主色一致
color_v = "tab:green"  # 速度误差颜色

# PPO 航向角
sns.lineplot(data=df_psi, x='Step', y='Raw', ax=ax2, color=colors_main[0], linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_psi, x='Step', y='Smooth', ax=ax2, color=colors_main[0], linewidth=LW_SMOOTH, label='PPO 航向角误差')
# PPO 俯仰角
sns.lineplot(data=df_theta, x='Step', y='Raw', ax=ax2, color=colors_main[1], linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_theta, x='Step', y='Smooth', ax=ax2, color=colors_main[1], linewidth=LW_SMOOTH, label='PPO 俯仰角误差')

# 添加右轴显示速度误差
ax2_r = ax2.twinx()
sns.lineplot(data=df_v, x='Step', y='Raw', ax=ax2_r, color=color_v, linewidth=LW_RAW, alpha=ALPHA_RAW, legend=False)
sns.lineplot(data=df_v, x='Step', y='Smooth', ax=ax2_r, color=color_v, linewidth=LW_SMOOTH, label='PPO 速度误差')


# PID 基准误差虚线 (与 Figure1 PID 颜色一致)
ax2.axhline(PID_AVG_PSI_ERR, color='indianred', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 航向角误差')
ax2.axhline(PID_AVG_THETA_ERR, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 俯仰角误差')
ax2_r.axhline(PID_AVG_V_ERR, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.8, label='PID 速度误差')

# 在虚线末端标注 PID 数值
xmax2 = ax2.get_xlim()[1]
ax2.text(xmax2*0.98, PID_AVG_PSI_ERR*1.15, f'PID: {PID_AVG_PSI_ERR}°',
         color=colors_main[0], fontsize=9, fontweight='bold', va='bottom', ha='right')
ax2.text(xmax2*0.98, PID_AVG_THETA_ERR*0.7, f'PID: {PID_AVG_THETA_ERR}°',
         color=colors_main[1], fontsize=9, fontweight='bold', va='bottom', ha='right')
ax2_r.text(xmax2*0.98, PID_AVG_V_ERR*0.7, f'PID: {PID_AVG_V_ERR} m/s',
         color=color_v, fontsize=9, fontweight='bold', va='bottom', ha='right')

# 合并图例
ax2_r.legend_.remove() if ax2_r.get_legend() else None
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2_r.get_legend_handles_labels()
leg2 = ax2.legend(handles=h1+h2, labels=l1+l2, loc='upper right', frameon=True, fontsize=9)

# 装饰底轴风格
import matplotlib.ticker as ticker
ax2.set_yscale('log')
# ax2_r.set_ylim(0, 350) 
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1, 2, 5], numticks=5))
formatter = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
ax2.yaxis.set_major_formatter(formatter)
ax2.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=20))
ax2.yaxis.set_minor_formatter(ticker.NullFormatter())

# 装饰底轴风格
ax2.set_ylabel("角度误差(°)", fontweight='bold', fontsize=12, color='black')
ax2_r.set_ylabel("速度误差(m/s)", fontweight='bold', fontsize=12, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.5)
ax2_r.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.5)
ax2.set_xlabel("Steps", fontweight='bold')

# 确保所有坐标轴数字也加粗
for label in ax2.get_yticklabels():
    label.set_fontweight('bold')
for label in ax2_r.get_yticklabels(): 
    label.set_fontweight('bold')
for label in ax2.get_xticklabels():
    label.set_fontweight('bold')

# 保持网格在最底层
ax2.set_axisbelow(True)
ax2_r.set_axisbelow(True)

# 针对对数坐标轴启用淡化的网格线
ax2.grid(True, which='both', axis='y', alpha=0.3, color='lightgray')
ax2.grid(True, axis='x', alpha=0.5, color='lightgray')
ax2_r.grid(True, axis='y', alpha=0.3, color='lightgray')
sns.despine(ax=ax2, right=False)

# 使用 tight_layout 并像决策曲线图一样留出顶部空白，防止标题被遮挡
fig2.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])

plt.show()
