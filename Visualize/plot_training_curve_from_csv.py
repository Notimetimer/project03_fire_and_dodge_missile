from _context import *
from Algorithms.rl_utils import moving_average, ema
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置字体以支持中文及调整字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 14        # 坐标轴标签 (xlabel, ylabel)
plt.rcParams['xtick.labelsize'] = 14       # x轴刻度
plt.rcParams['ytick.labelsize'] = 14       # y轴刻度
plt.rcParams['legend.fontsize'] = 14       # 图例
plt.rcParams['axes.titlesize'] = 14        # 子图标题 (ax.set_title)
plt.rcParams['figure.titlesize'] = 15      # 总标题 (plt.suptitle)

def plot_training_results(csv_path, ax=None, smooth_type='ema', smooth_param=0.01, 
                          legend=None, ylabel='Value', xlabel='Step', title=None, show=True):
    """
    绘制训练曲线
    csv_path: CSV文件路径
    ax: 传入的 Matplotlib Axes 对象，如果为 None 则新建 Figure
    smooth_type: 'ma' (Moving Average) 或 'ema' (Exponential Moving Average)
    smooth_param: ma时为窗口大小(int)，ema时为平滑因子(float, 0-1)
    show: 是否执行 plt.legend() 和 plt.show()
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return None

    # 读取数据
    df = pd.read_csv(csv_path)
    steps = df['Step'].values
    values = df['Value'].values
    
    # 应用平滑
    if smooth_type == 'ma':
        window = int(smooth_param)
        smoothed_values = moving_average(values, window_size=window)
        v_min = pd.Series(values).rolling(window=window, center=True, min_periods=1).min().values
        v_max = pd.Series(values).rolling(window=window, center=True, min_periods=1).max().values
    else:
        smoothed_values = ema(values, epsilon=smooth_param)
        window = max(2, len(values) // 20)
        v_min = pd.Series(values).rolling(window=window, center=True, min_periods=1).min().values
        v_max = pd.Series(values).rolling(window=window, center=True, min_periods=1).max().values

    # 确定绘图对象
    if ax is None:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 绘制扫掠区域 (阴影部分)
    ax.fill_between(steps, v_min, v_max, color='firebrick', alpha=0.15, label='Min-Max Range')
    
    # 绘制平滑曲线
    ax.plot(steps, smoothed_values, color='firebrick', linewidth=2.5, label=legend if legend else 'Smoothed Trend')
    
    # 设置标签与字体
    ax.set_xlabel(xlabel if xlabel else "Step")
    ax.set_ylabel(ylabel if ylabel else "Value")
    if title:
        ax.set_title(title, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if show:
        ax.legend(frameon=True, framealpha=0.6)
        plt.tight_layout()
        plt.show()
        
    return ax

if __name__ == "__main__":
    # 示例 1: 独立绘图
    csv_file = r"D:\360极速浏览器X下载\IL_and_PFSP_分阶段_混规则对手_挑战_并行 A3C-run-20260305-090818.csv"
    # if os.path.exists(csv_file):
    #     plot_training_results(csv_file, smooth_type='ema', smooth_param=0.01, title="Standalone Plot")

    # 示例 2: 作为组合子图调用
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_training_results(csv_file, ax=ax1, title="Subplot A", show=False)
    plot_training_results(csv_file, ax=ax2, title="Subplot B", show=False)
    plt.show()
