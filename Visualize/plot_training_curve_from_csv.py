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

import matplotlib.ticker as ticker

def plot_training_results(csv_path, ax=None, smooth_type='ma', smooth_param=35, 
                          legend=None, ylabel=None, xlabel=None, title=None, 
                          y_scale_type='linear', y_log_subs=None, numticks=15,
                          y_major_step=None, y_minor_step=None,
                          color='firebrick', show=True):
    """
    绘制训练曲线
    y_scale_type: 'linear' 或 'log'
    y_major_step: 线性坐标时的标数间隔（主刻度步长）；对数坐标下目前暂不生效
    y_minor_step: 线性坐标时的细分网格间隔（次刻度步长）
    y_log_subs: 对数坐标的子刻度分布
    numticks: 对数/线性轴建议的最大标签数量
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
        # 计算扫掠区域 (局部窗口内的极值)
        v_min = pd.Series(values).rolling(window=window, center=True, min_periods=1).min().values
        v_max = pd.Series(values).rolling(window=window, center=True, min_periods=1).max().values
    else:
        # EMA 模式：将输入的 smooth_param (等效窗口大小 N) 转换为 epsilon (2/(N+1))
        # 这样设置 smooth_param=35 时，EMA 和 MA 的延迟感是相近的
        eps = 2.0 / (smooth_param + 1.0)
        smoothed_values = ema(values, epsilon=eps)
        # EMA的背景窗口也随参数动态调整
        window = int(smooth_param)
        v_min = pd.Series(values).rolling(window=window, center=True, min_periods=1).min().values
        v_max = pd.Series(values).rolling(window=window, center=True, min_periods=1).max().values

    # 确定绘图对象
    if ax is None:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # 绘制原始数据 (背景噪声，作为半透明细线)
    ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.5, label='Raw' if show else None)
    
    # 绘制平滑曲线 (主趋势)
    ax.plot(steps, smoothed_values, color=color, linewidth=2.5, label=legend if legend else 'Smoothed Trend')
    
    # 设置标签与字体
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight='bold')
    
    # --- 统一坐标轴刻度配置 ---
    if y_scale_type == 'log':
        ax.set_yscale('log')
        # 1. 主刻度（标数字）：由 y_log_subs 和 numticks 动态控制
        subs = y_log_subs if y_log_subs is not None else np.arange(1, 10).astype(float)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=subs, numticks=numticks))
        
        # 2. 格式化器： labelOnlyBase=False 让 2, 5 这种子位置也能显示数字
        formatter = ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        ax.yaxis.set_major_formatter(formatter)
        
        # 3. 次刻度（只画网格背景）：固定全显示 1-9
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10), numticks=20))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter()) # 辅助网格线不带数字标签
        
    elif y_scale_type == 'linear':
        ax.set_yscale('linear')
        # 4. 线性步长：支持按“每隔 10 个标一个数”这种逻辑
        if y_major_step is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major_step))
        if y_minor_step is not None:
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor_step))
    
    # 统一打开次刻度网格（which='both'），加深线条颜色和不透明度
    ax.grid(True, which='both', linestyle='--', alpha=0.8, color='gray')
    
    if show:
        ax.legend(frameon=True, framealpha=0.6)
        plt.tight_layout()
        plt.show()
        
    return ax

if __name__ == "__main__":
    # 示例 1: 独立绘图
    csv_file1 = os.path.join(project_root, "logs", "预训练actorloss.csv")
    csv_file2 = os.path.join(project_root, "logs", "预训练criticloss.csv")
    # r"D:\360极速浏览器X下载\\FlightControl_parallel无课程无蒸馏.csv"

    # if os.path.exists(csv_file):
    #     plot_training_results(csv_file, smooth_type='ema', smooth_param=0.01, title="Standalone Plot")

    # 示例 2: 作为组合子图调用
    fig, (ax1, ax2) = plt.subplots(1, 2) #, figsize=(15, 6))
    plot_training_results(csv_file1, ax=ax1, smooth_type='ma', smooth_param=35, title="ActorLoss", xlabel='epoch', y_scale_type='linear', show=False)
    plot_training_results(csv_file2, ax=ax2, smooth_type='ma', smooth_param=35, title="CriticLoss", xlabel='epoch', y_scale_type='log', show=False)
    plt.show()

