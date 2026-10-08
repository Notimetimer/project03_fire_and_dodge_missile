"""
绘制各算法对基准对手的对抗得分热力图。
颜色范围：白色(低分) -> 深蓝色(高分)，每个单元格显示数值。
版式：宋体 6号(7.5pt)，DPI 600，宽度 5cm。
图例栏高度与矩阵区域严格一致。
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# --- 全局字体：宋体 6号(7.5pt) ---
global_font_size = 10.5  # 六号 = 7.5pt
plt.rcParams['font.family'] = 'SimSun'        # 宋体
plt.rcParams['font.size'] = global_font_size               
plt.rcParams['axes.unicode_minus'] = False

# --- 数据 ---
algorithms = ['IL-PPO', 'PPO']
opponents = ['对手1', '对手2', '对手3', '对手4']

# 取每个 rule*_score 列的最后 N_TAIL 个数值求平均
N_TAIL = 19

CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_png2')

# 算法 -> 对应实验文件名前缀
# 注意：PPO0.3_flymask_v0h0 是 PPO0.3_flymask_v0h0_fireSL 的子串，
# 因此必须用 "前缀 + '-run-'" 做精确匹配，避免把 fireSL 实验归到非 fireSL 组
ALGO_FILE_PREFIXES = {
    'IL-PPO':     'test_norandom_vs_rules_PPO0.3_flymask_v0h0-run-',
    'PPO': 'test_norandom_vs_rules_NoIL_flymask_v0h0-run-20260909-131801'
}

RULE_COLS = ['rule0_score', 'rule1_score', 'rule2_score', 'rule3_score']

# 对每个算法：
#   1. 匹配该算法下的所有 CSV（同算法多次实验）
#   2. 每个文件取 rule0~rule3_score 列的最后 N_TAIL 行，在这 N_TAIL 个数值内求平均（得到该文件的 4 个均值）
#   3. 再在同算法的多个实验文件之间求平均，得到该算法对 4 个对手的最终得分
score_matrix = np.zeros((len(algorithms), len(opponents)))
for i, algo in enumerate(algorithms):
    prefix = ALGO_FILE_PREFIXES[algo]
    files = sorted(glob.glob(os.path.join(CSV_DIR, prefix + '*.csv')))
    per_file_means = []
    for f in files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        tail = df[RULE_COLS].tail(N_TAIL)
        per_file_means.append(tail.mean().values)
    score_matrix[i] = np.mean(per_file_means, axis=0)

print("各算法对基准对手的对抗得分矩阵（行=算法，列=对手1~对手4）:")
print(np.round(score_matrix, 4))

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
