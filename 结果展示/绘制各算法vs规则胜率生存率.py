"""
绘制 4 个算法（各 3 次重复实验）在训练过程中对 4 个规则对手的平均指标曲线。

数据来源：
    结果展示/exp_png2/test_norandom_vs_rules_<实验目录名>.csv
每个 CSV 包含 25/50 个训练进度点，列含 avg_score / avg_win / avg_lose /
avg_draw / avg_perish（对 4 规则的平均）。

处理流程：
    1. 同一算法的 3 个重复 CSV，把指标列线性插值到统一横轴（0 ~ 2e6 步）；
    2. 3 条曲线堆叠，求 mean / min / max；
    3. 每个指标一张独立 figure：mean 画实线，min~max 画浅色阴影。

指标：
    - avg_score      平均得分
    - avg_win        平均胜率
    - avg_lose       平均负率
    - avg_draw       平均平率
    - survive        平均生存率 = avg_win + avg_draw - avg_perish

绘图风格参考：对比开火有监督补偿项.py
    - 字体：英文 Times New Roman，中文宋体
    - 图尺寸：7cm × 5cm
    - tab10 配色，第5色亮紫、第8色纯黑
    - 参考线 0 / 0.5 / 1.0
    - 横轴训练步数（科学计数法），纵轴 [-0.05, 1.05]
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _context import *  # 包含 project_root


# ======================== 算法分组配置 ========================
CSV_DIR = os.path.join(project_root, "结果展示", "exp_png2")

# (算法显示名, [3 次重复实验的目录名])
# 每组 3 个目录名即 DIR_NAME_LIST 中同算法的三次重复
# 统一在后面给前缀 test_norandom_vs_rules_

ALGORITHM_GROUPS = [
    ("IL-PPO", [
            "PPO0.3_flymask_v0h0-run-20260921-194617",
            "PPO0.3_flymask_v0h0-run-20260928-111836",
            "PPO0.3_flymask_v0h0-run-20261001-152601",
        ]),
    ("IL-SE-SAC", [
        "SAC0.3_flymask_v1h1-run-20260923-213238",
        "SAC0.3_flymask_v1h1-run-20260928-093645",
        "SAC0.3_flymask_v1h1-run-20260929-194715",
    ]),
    ("IL-SL-PPO", [
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260921-122428",
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260928-233900",
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260930-093137",
    ]),
    ("IL-SLA-PPO", [
        "PPO0.3_flymask_v0h0_fireSL-run-20260924-145554",
        "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654",
        "PPO0.3_flymask_v0h0_fireSL-run-20260930-125149",
    ]),    
]

# 指标定义：(CSV 列名, 显示名)；survive 由 win+draw-perish 计算，列名占位
METRICS = [
    ("avg_score", "与基准对手对抗平均得分"),
    ("avg_win",   "与基准对手对抗平均胜率"),
    ("avg_lose",  "与基准对手对抗平均负率"),
    ("avg_draw",  "与基准对手对抗平均平率"),
    ("survive",   "与基准对手对抗平均生存率"),
]

# 统一横轴
X_MIN = 0.0
X_MAX = 2e6
NUM_POINTS = 100


# ======================== 全局视觉参数（与参考文件一致） ========================
label_fontsize = 7.5       # 六号 = 7.5 pt
tick_fontsize = 6.5
legend_fontsize = 5.5
linewidth = 0.6            # 曲线线宽
refer_linewidth = 0.55     # 参考线宽
legend_linewidth = 1.0     # 图例线条粗细
fill_alpha = 0.10          # min-max 阴影透明度
dpi = 200

COLOR_CYCLE = 5
linestyles = ['-', ':', '--']
smooth_window = 5          # 居中滑动窗口平滑窗口大小；<=1 则不平滑

CM_PER_INCH = 2.54
FIG_WIDTH_CM = 7
FIG_HEIGHT_CM = 5
FIG_WIDTH_IN = FIG_WIDTH_CM / CM_PER_INCH
FIG_HEIGHT_IN = FIG_HEIGHT_CM / CM_PER_INCH

plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'axes.axisbelow': True,
    'figure.dpi': dpi,
    'grid.linewidth': 0.3,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '0.35',
})

SAVE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "各算法vs规则胜率生存率")


def smooth_curve(data, window_size):
    """居中滑动窗口平滑，保持无相位延迟。"""
    if window_size <= 1 or data is None or len(data) <= 1:
        return np.asarray(data)
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= len(data):
        window_size = len(data) - (1 if len(data) % 2 == 0 else 0)
    if window_size <= 1:
        return np.asarray(data)
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()


def load_experiment_csv(dir_name):
    """根据实验目录名读取对应 CSV，返回 DataFrame（已按 step 升序）。"""
    csv_path = os.path.join(CSV_DIR, f"test_norandom_vs_rules_{dir_name}.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('step').reset_index(drop=True)
    return df


def compute_metric_series(df, metric_key):
    """
    从 DataFrame 提取指标序列。
    survive = avg_win + avg_draw - avg_perish
    """
    if metric_key == 'survive':
        return df['avg_win'].to_numpy() + df['avg_draw'].to_numpy() - df['avg_perish'].to_numpy()
    return df[metric_key].to_numpy()


def interpolate_to_grid(steps, values, x_target):
    """把 (steps, values) 线性插值到统一 x_target 网格上。"""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[valid], values[valid]
    if len(steps) < 2:
        return np.full_like(x_target, np.nan)
    sort_idx = np.argsort(steps)
    steps, values = steps[sort_idx], values[sort_idx]
    steps, unique_idx = np.unique(steps, return_index=True)
    values = values[unique_idx]
    return np.interp(x_target, steps, values, left=values[0], right=values[-1])


def load_group_stats(algo_label, dir_names, x_target):
    """
    对单个算法的多次重复实验，把每个指标插值到统一横轴后，
    返回 { metric_key: {'mean','min','max','runs'} }。
    """
    # 先加载所有重复的 DataFrame
    dfs = []
    for d in dir_names:
        try:
            df = load_experiment_csv(d)
            dfs.append(df)
            print(f"  已加载: {d}  ({len(df)} 个进度点)")
        except Exception as e:
            print(f"  [跳过] {d}: {e}")

    if not dfs:
        return None

    stats = {}
    for metric_key, _ in METRICS:
        curves = []
        for df in dfs:
            vals = compute_metric_series(df, metric_key)
            interp = interpolate_to_grid(df['step'].to_numpy(), vals, x_target)
            curves.append(interp)
        stacked = np.vstack(curves)  # (num_runs, num_points)
        stats[metric_key] = {
            'mean': np.mean(stacked, axis=0),
            'min': np.min(stacked, axis=0),
            'max': np.max(stacked, axis=0),
            'runs': len(curves),
        }
    return stats


def style_axis(ax, ylabel_text, metric_key):
    """统一坐标轴样式：参考线、标签、刻度、科学计数法、坐标范围。

    所有指标都画 0 / 0.5 / 1.0 参考线；
    avg_score 不限制 0~1，由数据自动决定 y 轴范围；
    其余指标（win/lose/draw/survive）固定 [-0.05, 1.05]。
    """
    for hval in [0, 0.5, 1.0]:
        ax.axhline(hval, color='gray', linestyle='-', linewidth=refer_linewidth,
                   alpha=0.8, zorder=1)
    if metric_key != 'avg_score':
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('训练步数', fontweight='bold', fontsize=label_fontsize)
    ax.set_ylabel(ylabel_text, fontweight='bold', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(tick_fontsize)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.4)


def add_legend(ax):
    leg = ax.legend(
        loc="lower right", ncol=2, fontsize=legend_fontsize,
        framealpha=0.35, borderpad=0.25, labelspacing=0.2,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.6,
    )
    if leg is not None:
        leg.set_draggable(True)
        for legobj in leg.get_lines():
            legobj.set_linewidth(legend_linewidth)
            legobj.set_alpha(1.0)


def save_figure(fig, save_name):
    if not SAVE_BASE_DIR:
        return
    pdf_dir = os.path.join(SAVE_BASE_DIR, "draw_pdf")
    svg_dir = os.path.join(SAVE_BASE_DIR, "draw_svg")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{save_name}.pdf")
    svg_path = os.path.join(svg_dir, f"{save_name}.svg")
    fig.savefig(pdf_path, format='pdf', dpi=dpi)
    fig.savefig(svg_path, format='svg', dpi=dpi)
    print(f"已保存: {pdf_path} / {svg_path}")


def main():
    x_target = np.linspace(X_MIN, X_MAX, NUM_POINTS)

    # 读取所有算法组的统计数据
    # group_results: [(algo_label, stats_dict)]
    group_results = []
    for algo_label, dir_names in ALGORITHM_GROUPS:
        print(f"\n算法: {algo_label}")
        stats = load_group_stats(algo_label, dir_names, x_target)
        if stats is not None:
            group_results.append((algo_label, stats))

    if not group_results:
        print("没有可绘制的数据，退出。")
        return

    # 配色：matplotlib tab10（与参考文件 sns.color_palette('tab10') 等价）
    cmap = plt.get_cmap('tab10')
    n = len(group_results)
    palette = [cmap(i) for i in range(max(10, n))]
    if len(palette) >= 8:
        palette[4] = (0.8, 0.0, 0.8, 1.0)  # 亮紫洋红
        palette[7] = (0.0, 0.0, 0.0, 1.0)  # 纯黑

    # 每个指标一张独立 figure
    for metric_key, ylabel_text in METRICS:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

        for j, (algo_label, stats) in enumerate(group_results):
            entry = stats.get(metric_key)
            if entry is None:
                continue
            mean_y = smooth_curve(entry['mean'], smooth_window)
            min_y = smooth_curve(entry['min'], smooth_window)
            max_y = smooth_curve(entry['max'], smooth_window)

            color = palette[j % COLOR_CYCLE]
            ls = linestyles[(j // COLOR_CYCLE) % len(linestyles)]

            ax.plot(x_target, mean_y, label=algo_label, color=color,
                    linewidth=linewidth, linestyle=ls, alpha=1.0, zorder=3)
            ax.fill_between(x_target, min_y, max_y, color=color,
                            alpha=fill_alpha, edgecolor='none', linewidth=0, zorder=2)

        style_axis(ax, ylabel_text, metric_key)
        add_legend(ax)
        fig.tight_layout(pad=0.2)
        save_figure(fig, metric_key)
        print(f"绘制: {ylabel_text}")

    plt.show()


if __name__ == "__main__":
    main()
