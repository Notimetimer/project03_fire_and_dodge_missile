"""
绘制最终策略间博弈矩阵 heatmap。

数据来源：
    结果展示/outputs/combat_matrix.csv  （由 最终智能体博弈矩阵 并行 原目录.py 生成）

绘图风格参考：绘制各算法vs规则胜率生存率.py
    - 字体：英文 Times New Roman，中文宋体
    - 图尺寸：7cm × 5cm
    - 配色：蓝色渐变 colormap（白 → 深蓝），与项目整体风格一致
    - 统一坐标轴/刻度字号
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# 本文件在 结果展示/废弃/ 下，父目录 结果展示/ 中含有 _context.py 和
# 绘制各算法vs规则胜率生存率.py，把父目录加入 sys.path 以便复用其样式常量。
_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from _context import *  # 提供 project_root
import 绘制各算法vs规则胜率生存率 as style  # 复用图尺寸/保存路径/rcParams 等样式常量

# --- 字体与字号（与 算法对抗基准对手结果矩阵.py 一致） ---
global_font_size = 10.5
plt.rcParams['font.family'] = 'SimSun'        # 宋体
plt.rcParams['font.size'] = global_font_size
plt.rcParams['axes.unicode_minus'] = False


# ======================== CSV 配置 ========================
CSV_DIR = os.path.join(project_root, "结果展示", "outputs")

# 可同时绘制多个矩阵（如半程 / 全程），每个 (文件名, 标题)
CSV_FILES = [
    ("combat_matrix.csv", "全程训练"),
    # ("combat_matrix_half.csv", "半程训练"),
]


# ======================== 配色（heatmap 专用，与 算法对抗基准对手结果矩阵.py 一致） ========================
# 白色 → 深蓝 的顺序色图，用于胜率 [0, 1]，由 sns.heatmap 的 cmap='Blues' 指定


def draw_heatmap(ax, csv_name, title, show_y=True):
    """在指定 ax 上绘制单个博弈矩阵 heatmap（绘图方式与 算法对抗基准对手结果矩阵.py 一致）。"""
    csv_path = os.path.join(CSV_DIR, csv_name)
    if not os.path.exists(csv_path):
        ax.set_title(f"[缺失] {csv_name}")
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)
    results = df.values
    labels = [str(col).replace('_', '-') for col in df.columns.tolist()]

    # seaborn heatmap：白色格子分隔线，Blues 配色，[0,1] 范围，不自动画 colorbar
    im = sns.heatmap(
        results,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels if show_y else False,
        square=True,
        linewidths=0.4,
        linecolor='white',
        annot_kws={'size': global_font_size},
        cbar=False,
        ax=ax,
    )

    # 去掉 heatmap 上的十字网格线
    ax.grid(False)

    # 数值在深色背景上用白色，浅色背景上用黑色，提升可读性
    for text in im.texts:
        val = float(text.get_text())
        text.set_color('white' if val >= 0.7 else 'black')

    # X 轴刻度与标签移到矩阵上方
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='left', va='bottom',
             rotation_mode='anchor', fontsize=global_font_size)
    if show_y:
        plt.setp(ax.get_yticklabels(), rotation=20, ha='right', va='center',
                 rotation_mode='anchor', fontsize=global_font_size)

    ax.set_title(title, fontsize=global_font_size, pad=4)


def add_colorbar(fig, ax):
    """给单个 ax 添加 colorbar，高度与矩阵严格对齐（与 算法对抗基准对手结果矩阵.py 一致）。"""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label('对抗得分', fontsize=global_font_size, labelpad=4)
    cbar.ax.tick_params(labelsize=global_font_size)


def main():
    # 颜色范围固定 [0, 1]，由 sns.heatmap 的 vmin/vmax 指定
    n = len(CSV_FILES)
    # 画布尺寸：每个矩阵 10cm × 10cm（与 算法对抗基准对手结果矩阵.py 一致），
    # 保证单元格足够大以容纳 10.5pt 字号的数字和刻度标签
    cm = 1 / 2.54
    fig_size = 10 * cm
    fig, axes = plt.subplots(1, n, figsize=(fig_size * n, fig_size),
                             squeeze=False)
    axes = axes[0]

    for ax, (csv_name, title) in zip(axes, CSV_FILES):
        draw_heatmap(ax, csv_name, title=None, show_y=(ax == axes[0]))
        # 每个子图单独 colorbar，高度与矩阵区域严格一致
        add_colorbar(fig, ax)

    fig.tight_layout(pad=1.0)

    # 保存
    out_pdf = os.path.join(style.SAVE_BASE_DIR, "draw_pdf", "combat_matrix.pdf")
    out_svg = os.path.join(style.SAVE_BASE_DIR, "draw_svg", "combat_matrix.svg")
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(out_svg), exist_ok=True)
    fig.savefig(out_pdf, format='pdf', dpi=style.dpi)
    fig.savefig(out_svg, format='svg', dpi=style.dpi)
    print(f"已保存: {out_pdf} / {out_svg}")

    plt.show()


if __name__ == "__main__":
    main()
