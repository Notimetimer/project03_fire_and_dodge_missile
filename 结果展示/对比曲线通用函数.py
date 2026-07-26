import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _context import *
from Algorithms.rl_utils import moving_average, ema
import seaborn as sns

# --- 环境与绘图配置 ---
sns.set_theme(style="whitegrid", font="SimHei", rc={"axes.unicode_minus": False})

# TensorBoard 默认配色序列
TENSORBOARD_COLORS = [
    '#12b5cb', '#e52592', '#f9ab00', '#9334e6',
    '#7cb342', '#e8710a', '#425066', '#353acd',
    '#278e00',
]


def plot_single_curve(data_dir, curves, xlabel="Step", ylabel="",
                      ref_lines=None, ax=None, smooth_type="MA", smooth_window=31,
                      draw_original=1.0, sign=1, linewidth=1.2):
    """
    绘制单个 subplot 风格的曲线图。

    Args:
        data_dir (str): CSV 文件所在目录。
        smooth_window: MA 时为窗口大小；EMA 时作为平滑指数 epsilon 传入 ema()。
        smooth_type (str): "MA" 使用移动平均，"EMA" 使用指数滑动平均。
        draw_original (float): 是否在平滑曲线背后绘制未平滑的原始曲线；
                               0 表示不绘制，大于 0 时作为透明度倍率（基准 alpha=0.06）。
        linewidth (float): 曲线线宽，默认 1.2。
        sign (int or float): 对 CSV 中 Value 列的乘数，1 为原值，-1 为取负。
        curves (list): 每条曲线配置，形如
            [
                {'Data_source': 'SLWSPFSP0.2-run-20260622-185856.csv', 'label': None, 'linestyle': '-'},
                {'Data_source': 'SLWSPFSP0.3-run-20260618-221044.csv', 'label': '自定义名', 'linestyle': '--'},
            ]
            Data_source 必填；label 为 None 或缺省时自动截取 Data_source 中 "-run" 之前的部分。
        xlabel (str): x 轴标签。
        ylabel (str): y 轴标签。
        ref_lines (list): 参考线数值列表，默认 [0, 0.5, 1.0]。
        ax (matplotlib.axes.Axes): 可选外部传入的 ax，否则新建一个 figure。

    Returns:
        fig, ax
    """
    sign = float(sign)
    assert abs(sign) == 1.0, "sign 只能为 1 或 -1"
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        created_fig = True
    else:
        fig = ax.figure

    if ref_lines is None:
        ref_lines = []

    y_min_total, y_max_total = float('inf'), float('-inf')
    has_plotted = False

    for i, cfg in enumerate(curves):
        csv_name = cfg['Data_source']
        csv_path = os.path.join(data_dir, csv_name)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"读取 {csv_path} 失败: {e}")
            continue

        # 自动截取 "-run" 之前的部分作为 label
        raw_name = csv_name.rsplit('.', 1)[0]
        default_label = raw_name.split('-run')[0] if '-run' in raw_name else raw_name
        label = cfg.get('label', default_label)
        if label is None:
            label = default_label

        color = cfg.get('color', TENSORBOARD_COLORS[i % len(TENSORBOARD_COLORS)])
        linestyle = cfg.get('linestyle', '-')

        if 'Step' not in df.columns or 'Value' not in df.columns:
            print(f"{csv_path} 缺少 Step 或 Value 列")
            continue

        raw = df['Value'] * sign
        if smooth_type == "EMA":
            smooth = ema(raw, smooth_window)
        else:
            smooth = moving_average(raw, smooth_window)

        y_min_total = min(y_min_total, np.nanmin(smooth))
        y_max_total = max(y_max_total, np.nanmax(smooth))

        # 同颜色极浅原始曲线（不加入图例），深度由 draw_original 倍率控制
        if draw_original:
            sns.lineplot(x=df['Step'], y=raw, ax=ax, color=color,
                         alpha=0.06 * float(draw_original), linewidth=linewidth, linestyle=linestyle)

        # 同颜色加粗平滑曲线，用于图例
        sns.lineplot(x=df['Step'], y=smooth, ax=ax, label=label,
                     color=color, linewidth=linewidth, linestyle=linestyle)
        has_plotted = True

    # 参考线
    for hval in ref_lines:
        ax.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=1.0, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 设置 x 轴刻度为整百万：范围<=10M 用 1M，否则用 2M
    if has_plotted:
        x_min, x_max = ax.get_xlim()
        x_range = x_max - x_min
        tick_step = 1e6 if x_range <= 10e6 else 2e6
        start = np.ceil(x_min / tick_step) * tick_step
        end = np.floor(x_max / tick_step) * tick_step + 0.5 * tick_step
        ax.set_xticks(np.arange(start, end, tick_step))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    if has_plotted and y_min_total < y_max_total:
        pad = (y_max_total - y_min_total) * 0.05
        ax.set_ylim(y_min_total - pad, y_max_total + pad)

    if has_plotted:
        leg = ax.legend(loc="upper left")
        leg.set_draggable(True)

    if created_fig:
        fig.tight_layout(pad=1.0)

    return fig, ax


if __name__ == "__main__":

    draw_type = [
            # 1
            {"dir_name": "对基准对手score",   "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 1,   "ylabel": "对基准策略平均Score", "linewidth": 1.2},
            # 2
            {"dir_name": "奖励函数",          "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 0.5, "ylabel": "累积奖励值",     "linewidth": 1.2},
            # 3
            {"dir_name": "策略熵",            "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 2,   "ylabel": "策略熵",           "linewidth": 1.2},
            # 4
            {"dir_name": "双杀率",            "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 0.5, "ylabel": r"与基准策略双杀率", "linewidth": 1.2},
            # {"dir_name": "ATA30",            "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 0.5, "ylabel": r"$ATA_{guidance}$", "linewidth": 1.2},
            # 5
            # {"dir_name": "actor_loss",        "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 0.5, "ylabel": r"Actor loss", "linewidth": 1.2},
            # {"dir_name": "delta_theta30",    "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": -1, "draw_original": 1,   "ylabel": r"$\theta_{guidance}$", "linewidth": 1.2},
            # # 6
            # {"dir_name": "delta_psi_threat", "smooth_type": "MA",    "SMOOTH_WINDOW": 41, "sign": 1,  "draw_original": 1,   "ylabel": r"$\left|\Delta \psi_{RWR}\right|$", "linewidth": 1.2},
        ]

    BASE_DIR = r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\训练曲线csv"

    # 不指定 color 时，按顺序自动从 TENSORBOARD_COLORS 取色
    CURVES = [
        {'Data_source': 'SLWS-PFSP-run-20260618-221044.csv', 'label': "SLWS-PFSP", 'linestyle': '-'},
        {'Data_source': 'HLWS-PFSP-run-20260616-130304.csv', 'label': "HLWS-PFSP", 'linestyle': '-'},
        {'Data_source': 'PFSP-run-20260615-234324.csv', 'label': 'PFSP', 'linestyle': '-'},
        # {'Data_source': 'SLWS-PFSP(A3C)-run-20260630-220403.csv', 'label': 'SLWS-A3C', 'linestyle': '-'},
        {'Data_source': 'FixedOpp-run-20260614-163906.csv', 'label': 'FixedOpp', 'linestyle': '-'},
        {'Data_source': 'SLWS-FixedOpp-run-20260712-113316.csv', 'label': 'SLWS-FixedOpp', 'linestyle': '-'},
    ]

    for figure_draw in draw_type:
        DATA_DIRECTORY = os.path.join(BASE_DIR, figure_draw["dir_name"])
        print(f"\n正在绘制: {figure_draw['dir_name']} -> {figure_draw['ylabel']}")
        plot_single_curve(
            DATA_DIRECTORY, CURVES,
            xlabel="Step",
            ylabel=figure_draw["ylabel"],
            smooth_type=figure_draw["smooth_type"],
            smooth_window=figure_draw["SMOOTH_WINDOW"],
            draw_original=figure_draw["draw_original"],
            sign=figure_draw["sign"],
            linewidth=figure_draw["linewidth"]
        )

    plt.show()
