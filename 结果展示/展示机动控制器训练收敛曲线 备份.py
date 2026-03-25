# python库
import numpy as np
import matplotlib.pyplot as plt
import os
# 自定义库
from _context import *
from Visualize.plot_training_curve_from_csv import plot_training_results

# 数据路径
returns_path = os.path.join(project_root, "logs", "ReturnOfControllerTraining.csv")
survive_rates_path = os.path.join(project_root, "logs", "SurviveRateOfControllerTraining.csv")
psi_error_path = os.path.join(project_root, "logs", "EMAPsiErrorOfControllerTraining.csv")
theta_error_path = os.path.join(project_root, "logs", "EMAThetaErrorOfControllerTraining.csv")

# 开启画布 (2x1 竖向排布，增加高度以防遮挡)
fig = plt.figure(figsize=(12, 18), dpi=100)

# =============================================================================
# 第一行：奖励函数 & 生存率 (左轴对数，右轴线性)
# =============================================================================
ax1_l = fig.add_subplot(2, 1, 1)
# 奖励曲线 (左轴)
plot_training_results(returns_path, ax=ax1_l, smooth_type='ma', smooth_param=35, 
                      ylabel="Cumulative Return (Log)", xlabel='Epoch', 
                      color='firebrick', legend='Return', title=None, 
                      y_scale_type="log", y_log_subs=[1, 2, 5], numticks=3,
                      show=False)

ax1_r = ax1_l.twinx()
# 生存率曲线 (右轴 - 0~1)
plot_training_results(survive_rates_path, ax=ax1_r, smooth_type='ma', smooth_param=20, 
                      ylabel="Survive Rate", color='royalblue', 
                      legend='Survive Rate', title=None,
                      show=False)

# 合并图例
lines1, labels1 = ax1_l.get_legend_handles_labels()
lines2, labels2 = ax1_r.get_legend_handles_labels()
ax1_l.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
ax1_l.grid(True, which='both', linestyle='--', alpha=0.8, color='gray')
ax1_r.grid(False) 

# =============================================================================
# 第二行：航向误差 & 俯仰误差 (合并在同一个对数轴，便于对比)
# =============================================================================
ax2 = fig.add_subplot(2, 1, 2)
# 绘制航向误差
plot_training_results(psi_error_path, ax=ax2, smooth_type='ma', smooth_param=35, 
                      ylabel="Error Degree (Log)", xlabel='Epoch', 
                      color='darkorange', legend='Psi Error', title=None,
                      y_scale_type="log", y_log_subs=[1, 2, 5], numticks=5,
                      show=False)

# 在同一个轴 ax2 上叠加俯仰误差
plot_training_results(theta_error_path, ax=ax2, smooth_type='ma', smooth_param=35, 
                      color='purple', legend='Theta Error', title=None,
                      y_scale_type="log", y_log_subs=[1, 2, 5], numticks=5,
                      show=False)

# 统一管理 ax2 的图例
ax2.legend(loc='upper right')
ax2.grid(True, which='both', linestyle='--', alpha=0.8, color='gray')

# 细节：增加 pad 防止 xlabel 和 科学计数法标签遮挡
plt.tight_layout(pad=4.0) 
plt.show()
