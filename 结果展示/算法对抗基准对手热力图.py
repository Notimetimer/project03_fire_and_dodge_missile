"""
绘制各算法对基准对手的对抗得分热力图。
颜色范围：白色(低分) -> 深蓝色(高分)，每个单元格显示数值。
版式：宋体 6号(7.5pt)，DPI 600，宽度 5cm。
图例栏高度与矩阵区域严格一致。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# --- 全局字体：宋体 6号(7.5pt) ---
global_font_size = 10.5  # 六号 = 7.5pt
plt.rcParams['font.family'] = 'SimSun'        # 宋体
plt.rcParams['font.size'] = global_font_size               
plt.rcParams['axes.unicode_minus'] = False

# --- 数据 ---
algorithms = ['IL-PPO', 'IL-SL-PPO', 'IL-SLA-PPO', 'IL-SE-SAC']
opponents = ['对手1', '对手2', '对手3', '对手4']

score_matrix = np.array([
    [0.85, 0.96, 0.98, 0.57],   # IL-PPO
    [0.64, 0.93, 0.90, 0.53],   # IL-SL-PPO
    [0.87, 0.92, 0.95, 0.73],   # IL-SLA-PPO
    [0.72, 0.65, 0.72, 0.50],   # IL-SE-SAC
])

# --- 绘图：宽度 5cm ---
cm = 1 / 2.54                       # 1cm = 1/2.54 inch
fig_width = 10 * cm
fig_height = fig_width              # 略低于宽，紧凑布局
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Blues 配色：白色 -> 深蓝色；先不画 colorbar，后续手动对齐高度
im = sns.heatmap(
    score_matrix,
    annot=True,
    fmt='.2f',                      # 保留前导零：0.85 而非 .85
    cmap='Blues',
    vmin=0.0,
    vmax=1.0,
    xticklabels=opponents,
    yticklabels=algorithms,
    square=True,
    linewidths=0.4,
    linecolor='white',
    annot_kws={'size': global_font_size},        # 单元格数字统一
    cbar=False,
    ax=ax,
)

# 数值在深色背景上用白色，浅色背景上用黑色，提升可读性
for text in im.texts:
    val = float(text.get_text())
    text.set_color('white' if val >= 0.7 else 'black')

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
# 左侧算法名称倾斜显示，避免名称挤在一起
plt.setp(ax.get_yticklabels(), rotation=45, ha='right', va='center')
ax.set_title('各算法对基准对手的对抗水平', pad=14)
ax.set_xlabel('基准对手', labelpad=5)
ax.set_ylabel('算法')

# --- 手动创建 colorbar，使其高度与矩阵(ax)严格一致 ---
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
mappable = ax.collections[0]
cbar = fig.colorbar(mappable, cax=cax)
cbar.set_label('对抗得分', labelpad=4)
cbar.ax.tick_params(labelsize=global_font_size)   # colorbar 刻度同样 6号

plt.tight_layout()

# 保存图片：DPI 600
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '算法对抗基准对手热力图.png')
plt.savefig(out_path, dpi=400, bbox_inches='tight')
print(f"热力图已保存至: {out_path}")

plt.show()
