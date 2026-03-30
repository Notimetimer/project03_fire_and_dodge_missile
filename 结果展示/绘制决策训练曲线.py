import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from _context import * # 包含 project_root
from Algorithms.rl_utils import moving_average
import seaborn as sns

# --- 1. 环境与绘图配置 ---
# 使用 seaborn 风格，并设置中文字体
sns.set_theme(style="darkgrid", font="SimHei", rc={"axes.unicode_minus": False})

OriginalData_dir = os.path.join(project_root, "logs", "OriginalData")
Data_dir = os.path.join(project_root, "logs", "Data")

def plot_training_curves(data_dir, exp_name, name_list=None, name_list_show=None, 
                         win_rate_cols=None, display_titles=None, linestyles=None):
    """
    绘制指定实验名称的训练曲线
    data_dir: csv文件所在的目录
    exp_name: 实验名称 (即文件名中第一个 "_" 前面的部分，如 "ILHOPFSP")
    """
    # 构建文件路径
    main_file = os.path.join(data_dir, f"{exp_name}_main.csv")
    win_rate_file = os.path.join(data_dir, f"{exp_name}_win_rate.csv")
    
    # ---------------- 1. 读取数据 ----------------
    try:
        df_main = pd.read_csv(main_file)
    except Exception as e:
        print(f"读取 {main_file} 失败: {e}")
        df_main = None
        
    try:
        df_win = pd.read_csv(win_rate_file)
    except Exception as e:
        print(f"读取 {win_rate_file} 失败: {e}")
        df_win = None

    # ---------------- 2. 绘制第一个 Figure: Main 和 PoolSize ----------------
    # 根据已成功读取的数据量决定子图的行数
    fig1_rows = 0
    if df_main is not None: fig1_rows += 1
    
    if fig1_rows > 0:
        fig1, axes1 = plt.subplots(fig1_rows, 1, figsize=(8, 4 * fig1_rows))
        # fig1.suptitle(f"{exp_name} - 基础训练指标")
        
        # 统一将 axes1 转为列表方便通过索引访问
        if fig1_rows == 1:
            axes1 = [axes1]
            
        plot_idx = 0
        
        # Subplot: Reward 和 Score
        if df_main is not None and not df_main.empty:
            ax1 = axes1[plot_idx]
            
            Reward = df_main['Reward']
            Score = df_main['Score']
            
            # 平滑曲线
            Reward_smoothed = moving_average(Reward, 21)
            Score_smoothed = moving_average(Score, 21)

            # Reward放左侧纵坐标
            sns.lineplot(x=df_main['Step'], y=Reward, ax=ax1, color='tab:blue', alpha=0.1) # 0.05
            sns.lineplot(x=df_main['Step'], y=Reward_smoothed, ax=ax1, label='Episode Reward', color='tab:blue', linewidth=1.5) # 2
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Episode Reward', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            # 隐藏左侧y轴的网格线，但保留x轴网格线
            ax1.grid(False, axis='y')
            ax1.grid(True, axis='x')
            
            # Score放右侧纵坐标 (改名为 Avg Score)
            ax1_twin = ax1.twinx()
            sns.lineplot(x=df_main['Step'], y=Score, ax=ax1_twin, color='tab:orange', alpha=0.1) # 0.05
            sns.lineplot(x=df_main['Step'], y=Score_smoothed, ax=ax1_twin, label='Avg Score', color='tab:orange', linewidth=1.5) # 2
            ax1_twin.set_ylabel('Avg Score', color='tab:orange')
            ax1_twin.tick_params(axis='y', labelcolor='tab:orange')
            # 开启右侧y轴的网格线
            ax1_twin.grid(True)
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            if ax1_twin.get_legend() is not None:
                ax1_twin.get_legend().remove()
            
            ax1.set_title('Episode Reward & Avg Score')
            plot_idx += 1
            
        fig1.tight_layout()

    # ---------------- 3. 绘制第二个 Figure: Win Rate ----------------
    if name_list is not None and name_list_show is not None:
        # 如果未传入，则使用默认的列名、标题和线型
        if win_rate_cols is None:
            win_rate_cols = ['VsRule0', 'VsRule1', 'VsRule2', 'VsRule3', 'VsRule4']
        if display_titles is None:
            display_titles = ['Rule1', 'Rule2', 'Rule3', 'Rule4', 'Rule5']
        if linestyles is None:
            # 索引对应：0, 1, 2(点线), 3, 4(点划线), 5(点划线)
            linestyles = ['-', '-', ':', '-', '-.', '-.']
        
        n_cols = len(win_rate_cols)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 5.5))
        fig2.suptitle("Win Rate Against All Rules", fontsize=16, y=0.81)
        
        if n_cols == 1:
            axes2 = [axes2]
            
        palette = sns.color_palette("tab10", len(name_list))
            
        for i, (col, display_title) in enumerate(zip(win_rate_cols, display_titles)):
            ax = axes2[i]
            
            # 由于可能出现数据缺失的情况，我们需要记录有没有画线
            has_plotted = False
            for j, (name, show_name) in enumerate(zip(name_list, name_list_show)):
                win_file = os.path.join(data_dir, f"{name}_win_rate.csv")
                try:
                    df = pd.read_csv(win_file)
                    if col in df.columns:
                        raw_data = df[col]
                        smoothed_data = moving_average(raw_data, 101)
                        
                        # 获取当前实验对应的线型
                        ls = linestyles[j % len(linestyles)]
                        # 根据线型动态调整粗细：点线必须加粗到1.7，否则用1.4
                        lw = 1.7 if ls == ':' else 1.4
                        
                        sns.lineplot(x=df['Step'], y=raw_data, ax=ax, color=palette[j], alpha=0.04) # 调低背景线透明度
                        sns.lineplot(x=df['Step'], y=smoothed_data, ax=ax, label=str(show_name), 
                                     color=palette[j], linewidth=lw, linestyle=ls)
                        has_plotted = True
                except Exception as e:
                    pass
                
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Step')
            ax.set_ylabel('')
            # 显示对应映射的新名称，而不是原CSV列名
            ax.set_title(display_title)
            # 只有在真正画了线的情况下才调用 legend，避免 No artists 警告
            if has_plotted:
                ax.legend(loc="lower right") 

        # pad=1.0 减小多余空白，避免挤压图表高度
        # 将 w_pad 从 1.0 调低到 0.4，让子图在横向上贴得更近
        fig2.tight_layout(pad=1.0, w_pad=0.4, rect=[0, 0, 1, 0.86])

    elif df_win is not None and not df_win.empty:
        # Fallback for single experiment
        win_rate_cols = [col for col in df_win.columns if col.startswith('VsRule')]
        if not win_rate_cols:
            win_rate_cols = ['VsRule1', 'VsRule2', 'VsRule3', 'VsRule4', 'VsRule5']
        
        n_cols = len(win_rate_cols)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 5.5))
        fig2.suptitle("Win Rate Against All Rules", fontsize=16, y=0.81)
        
        if n_cols == 1:
            axes2 = [axes2]
            
        for i, col in enumerate(win_rate_cols):
            ax = axes2[i]
            if col in df_win.columns:
                raw_data = df_win[col]
                smoothed_data = moving_average(raw_data, 101)
                
                sns.lineplot(x=df_win['Step'], y=raw_data, ax=ax, color='tab:red', alpha=0.06) # alpha=0.05
                sns.lineplot(x=df_win['Step'], y=smoothed_data, ax=ax, label=col, color='tab:red', linewidth=1.6)
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.text(0.5, 0.5, f"{col} Data Missing", ha='center', va='center')
                
            ax.set_xlabel('Step')
            ax.set_ylabel('')
            ax.set_title(col)
            
        fig2.tight_layout(pad=1.0, w_pad=0.4, rect=[0, 0, 1, 0.86])

    plt.show()

if __name__ == "__main__":
    # 保留您原有的目录设定
    DATA_DIRECTORY = Data_dir # r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\Data"
    EXPERIMENT_NAME = "ILHOPFSP"

    name_list = [
        "ILHOPFSP",
        "ILPFSP",
        "HOPFSP",
        "ILDFSP",
        "A3C",
        "Rule4"
    ]
    name_list_show = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    
    win_rate_cols = ['VsRule0', 'VsRule1', 'VsRule2', 'VsRule3', 'VsRule4']
    display_titles = ['Rule1', 'Rule2', 'Rule3', 'Rule4', 'Rule5']
    linestyles = ['-', '-', '--', '-', '--', ':']
    
    plot_training_curves(DATA_DIRECTORY, EXPERIMENT_NAME, name_list, name_list_show, 
                         win_rate_cols, display_titles, linestyles)
