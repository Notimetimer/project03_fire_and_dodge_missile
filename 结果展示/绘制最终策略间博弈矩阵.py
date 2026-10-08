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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 本文件在 结果展示/废弃/ 下，父目录 结果展示/ 中含有 _context.py 和
# 绘制各算法vs规则胜率生存率.py，把父目录加入 sys.path 以便复用其样式常量。
_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from _context import *  # 提供 project_root
import 绘制各算法vs规则胜率生存率 as style  # 复用字号/图尺寸/rcParams 等样式常量


# ======================== CSV 配置 ========================
CSV_DIR = os.path.join(project_root, "结果展示", "outputs")

# 可同时绘制多个矩阵（如半程 / 全程），每个 (文件名, 标题)
CSV_FILES = [
    ("combat_matrix.csv", "全程训练"),
    # ("combat_matrix_half.csv", "半程训练"),
]


# ======================== 配色（heatmap 专用，与项目整体蓝色风格一致） ========================
# 白色 → 深蓝 的顺序色图，用于胜率 [0, 1]
END_COLOR = (0.06, 0.1, 0.38)
CMAP = LinearSegmentedColormap.from_list("custom_blue", [(1.0, 1.0, 1.0), END_COLOR], N=256)


def build_norm(values):
    """根据所有矩阵的实际取值范围构建 TwoSlopeNorm，使 0.5 居中。"""
    if not values:
        return TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    pad = 0.15 * max(v_max - v_min, 1e-6)
    vmin = max(0.0, v_min - pad)
    vmax = min(1.0, v_max + pad)
    return TwoSlopeNorm(vmin=vmin, vcenter=(vmin + vmax) / 2, vmax=vmax)


def draw_heatmap(ax, csv_name, title, show_y=True):
    """在指定 ax 上绘制单个博弈矩阵 heatmap。"""
    csv_path = os.path.join(CSV_DIR, csv_name)
    if not os.path.exists(csv_path):
        ax.set_title(f"[缺失] {csv_name}")
        print(f"[跳过] 找不到文件: {csv_path}")
        return

    df = pd.read_csv(csv_path, index_col=0)
    results = df.values
    labels = [str(col).replace('_', '-') for col in df.columns.tolist()]

    im = ax.imshow(results, cmap=CMAP, norm=NORM, aspect='auto')

    # 去掉十字网格线：rcParams 默认 axes.grid=True，在 heatmap 上会画出
    # 交叉网格线，这里显式关闭。
    ax.grid(False)

    # 修复：鼠标悬停时 matplotlib 默认的 format_cursor_data 会对 inf 值
    # 调用 math.log10 导致 OverflowError。这里覆盖为安全格式化，跳过
    # _g_sig_digits 的危险计算。
    def _safe_cursor_data(data):
        try:
            v = float(data)
            if np.isfinite(v):
                return f"{v:.2f}"
        except (TypeError, ValueError):
            pass
        return ""

    im.format_cursor_data = _safe_cursor_data

    # 标注数值
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            val = results[i, j]
            # 深色背景用白字，浅色背景用黑字
            text_color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=5, color=text_color)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', va='top',
                       rotation_mode='anchor', fontsize=5)
    if show_y:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, rotation=30, ha='right', va='center',
                           rotation_mode='anchor', fontsize=5)
    else:
        ax.set_yticks([])

    ax.set_title(title, fontsize=style.label_fontsize, pad=4)
    ax.set_xlabel("对手 / 列", fontsize=6)
    if show_y:
        ax.set_ylabel("评估方 / 行", fontsize=6)

    # 外框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.linewidth)
        spine.set_color('0.35')


def add_colorbar(fig, ax):
    """给 figure 添加共享 colorbar。"""
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('平均对抗得分', fontsize=6)


def main():
    # 收集所有矩阵的取值，统一 colorbar 范围
    all_values = []
    for csv_name, _ in CSV_FILES:
        csv_path = os.path.join(CSV_DIR, csv_name)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            all_values.extend(df.values.flatten().tolist())
    global NORM
    NORM = build_norm(all_values)

    n = len(CSV_FILES)
    # 画布缩小到原来的 1/4（边长各 /2），并留出图与窗口之间的边距
    HEATMAP_W_IN = (12 / 2) / style.CM_PER_INCH
    HEATMAP_H_IN = (10 / 2) / style.CM_PER_INCH
    fig, axes = plt.subplots(1, n, figsize=(HEATMAP_W_IN * n, HEATMAP_H_IN),
                             squeeze=False)
    axes = axes[0]

    for ax, (csv_name, title) in zip(axes, CSV_FILES):
        draw_heatmap(ax, csv_name, title=None, show_y=(ax == axes[0]))

    # 先收紧布局，再加 colorbar（colorbar 与 tight_layout 顺序敏感）
    # pad 控制子图与 figure 边框（即图与窗口）之间的留白
    fig.tight_layout(pad=1.5)
    add_colorbar(fig, axes)

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
