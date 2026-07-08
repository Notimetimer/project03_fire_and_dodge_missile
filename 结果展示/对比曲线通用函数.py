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
                      ref_lines=None, ax=None, smooth_type="MA", smooth_window=31):
    """
    绘制单个 subplot 风格的曲线图。

    Args:
        data_dir (str): CSV 文件所在目录。
        smooth_window: MA 时为窗口大小；EMA 时作为平滑指数 epsilon 传入 ema()。
        smooth_type (str): "MA" 使用移动平均，"EMA" 使用指数滑动平均。
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

        raw = df['Value']
        if smooth_type == "EMA":
            smooth = ema(raw, smooth_window)
        else:
            smooth = moving_average(raw, smooth_window)

        y_min_total = min(y_min_total, np.nanmin(smooth))
        y_max_total = max(y_max_total, np.nanmax(smooth))

        # 同颜色极浅原始曲线（不加入图例）
        sns.lineplot(x=df['Step'], y=raw, ax=ax, color=color,
                     alpha=0.06, linewidth=1.6, linestyle=linestyle)

        # 同颜色加粗平滑曲线，用于图例
        lw = 1.8 if linestyle == ':' else 1.6
        sns.lineplot(x=df['Step'], y=smooth, ax=ax, label=label,
                     color=color, linewidth=lw, linestyle=linestyle)
        has_plotted = True

    # 参考线
    for hval in ref_lines:
        ax.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=1.0, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    # OriginalData_dir = os.path.join(project_root, "logs", "OriginalData")
    DATA_DIRECTORY = r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\训练曲线csv\奖励函数"
    # score
    r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\训练曲线csv\对基准对手score"
    r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\训练曲线csv\奖励函数"


    
    # os.path.join(project_root, "logs", "Data")

    SMOOTH_WINDOW = 0.97 # 41
    X_LABEL = "Step"
    Y_LABEL = "平均对抗得分"

    # 不指定 color 时，按顺序自动从 TENSORBOARD_COLORS 取色
    CURVES = [
        {'Data_source': 'SLWSPFSP0.2-run-20260622-185856.csv', 'label': "SLWS-PPO-0.2", 'linestyle': '-'},
        {'Data_source': 'SLWSPFSP0.3-run-20260618-221044.csv', 'label': "SLWS-PPO-0.3", 'linestyle': '-'},
        {'Data_source': 'SLWSA3C0.3-run-20260630-220403.csv', 'label': "SLWS-A3C-0.3", 'linestyle': '-'},
        {'Data_source': 'HLWSPFSP-run-20260616-130304.csv', 'label': None, 'HLWS-PPO': '-'},
        {'Data_source': 'CSPFSP-run-20260615-234324.csv', 'label': None, 'PPO': '-'},
    ]

    fig, ax = plot_single_curve(DATA_DIRECTORY, CURVES,
                                xlabel=X_LABEL, ylabel=Y_LABEL, smooth_type="EMA", smooth_window=SMOOTH_WINDOW)
    plt.show()
