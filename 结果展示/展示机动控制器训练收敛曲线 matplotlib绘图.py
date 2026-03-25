# python库
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 自定义库
from _context import *
from Visualize.plot_training_curve_from_csv_old import plot_training_results

# 配置 Seaborn 主题，让颜色和网格更漂亮
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 再次确认中文支持

# 数据路径
returns_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining.csv")
survive_rates_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining.csv")
psi_error_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining.csv")
theta_error_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining.csv")

# 开启画布 (2x1 竖向排布)
fig = plt.figure(figsize=(10, 18), dpi=100)

# =============================================================================
# 第一行：奖励函数 & 生存率
# =============================================================================
ax1_l = fig.add_subplot(2, 1, 1)
plot_training_results(returns_path, ax=ax1_l, smooth_type='ma', smooth_param=35, 
                      ylabel="Episode Reward", xlabel='Steps', 
                      color=sns.color_palette("muted")[3], # 漂亮的深红色
                      legend='Episode Reward', title=None, 
                      y_scale_type="linear", y_log_subs=[1, 2, 5], numticks=3,
                      show=False)

ax1_r = ax1_l.twinx()
plot_training_results(survive_rates_path, ax=ax1_r, smooth_type='ma', smooth_param=20, 
                      ylabel="Survive Rate", 
                      color=sns.color_palette("muted")[0], # 漂亮的蓝色
                      legend='Survive Rate', title=None,
                      show=False)

# 合并图例并加深网格
lines1, labels1 = ax1_l.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1_l.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True)
ax1_l.grid(True, which='both', linestyle='--', alpha=0.9, color='lightgray')
ax1_r.grid(False) 

# =============================================================================
# 第二行：航向误差 & 俯仰误差 (共用同一个 Y 轴)
# =============================================================================
ax2 = fig.add_subplot(2, 1, 2)
# 绘制航向误差
plot_training_results(psi_error_path, ax=ax2, smooth_type='ma', smooth_param=35, 
                      ylabel="Error Degree ", xlabel='Steps', 
                      color=sns.color_palette("muted")[1], # 漂亮的橙色
                      legend='Psi Error', title=None,
                      y_scale_type="log", y_log_subs=[1, 2, 5], numticks=5,
                      show=False)

# 叠加俯仰误差
plot_training_results(theta_error_path, ax=ax2, smooth_type='ma', smooth_param=35, 
                      color=sns.color_palette("muted")[4], # 漂亮的紫色
                      legend='Theta Error', title=None,
                      y_scale_type="log", y_log_subs=[1, 2, 5], numticks=5,
                      show=False)

ax2.legend(loc='upper right', frameon=True)
ax2.grid(True, which='both', linestyle='--', alpha=0.9, color='lightgray')

# =============================================================================
# 布局细节优化：强行指定子图间距
# =============================================================================
# hspace=0.45 提供了巨大的垂直留白，完美避开 xlabel 与下面图表的冲突
fig.subplots_adjust(hspace=0.25, top=0.9, bottom=0.1, left=0.2, right=0.8)

plt.show()
