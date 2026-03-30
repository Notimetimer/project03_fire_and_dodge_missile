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

def plot_training_curves(data_dir, exp_name):
    """
    绘制指定实验名称的训练曲线
    data_dir: csv文件所在的目录
    exp_name: 实验名称 (即文件名中第一个 "_" 前面的部分，如 "ILHOPFSP")
    """
    # 构建文件路径
    main_file = os.path.join(data_dir, f"{exp_name}_main.csv")
    pool_size_file = os.path.join(data_dir, f"{exp_name}_pool_size.csv")
    win_rate_file = os.path.join(data_dir, f"{exp_name}_win_rate.csv")
    
    # ---------------- 1. 读取数据 ----------------
    try:
        df_main = pd.read_csv(main_file)
    except Exception as e:
        print(f"读取 {main_file} 失败: {e}")
        df_main = None
        
    try:
        df_pool = pd.read_csv(pool_size_file)
    except Exception as e:
        print(f"读取 {pool_size_file} 失败: {e}")
        df_pool = None
        
    try:
        df_win = pd.read_csv(win_rate_file)
    except Exception as e:
        print(f"读取 {win_rate_file} 失败: {e}")
        df_win = None

    # ---------------- 2. 绘制第一个 Figure: Main 和 PoolSize ----------------
    # 根据已成功读取的数据量决定子图的行数
    fig1_rows = 0
    if df_main is not None: fig1_rows += 1
    if df_pool is not None: fig1_rows += 1
    
    if fig1_rows > 0:
        fig1, axes1 = plt.subplots(fig1_rows, 1, figsize=(8, 4 * fig1_rows))
        fig1.suptitle(f"{exp_name} - 基础训练指标")
        
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
            Reward_smoothed = moving_average(Reward, 100)
            Score_smoothed = moving_average(Score, 100)

            # Reward放左侧纵坐标
            sns.lineplot(x=df_main['Step'], y=Reward, ax=ax1, label='Reward (Raw)', color='tab:blue', alpha=0.3)
            sns.lineplot(x=df_main['Step'], y=Reward_smoothed, ax=ax1, label='Reward (Smoothed)', color='tab:blue', linewidth=2)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Reward', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Score放右侧纵坐标
            ax1_twin = ax1.twinx()
            sns.lineplot(x=df_main['Step'], y=Score, ax=ax1_twin, label='Score (Raw)', color='tab:orange', alpha=0.3)
            sns.lineplot(x=df_main['Step'], y=Score_smoothed, ax=ax1_twin, label='Score (Smoothed)', color='tab:orange', linewidth=2)
            ax1_twin.set_ylabel('Score', color='tab:orange')
            ax1_twin.tick_params(axis='y', labelcolor='tab:orange')
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            if ax1_twin.get_legend() is not None:
                ax1_twin.get_legend().remove()
            
            ax1.set_title('Reward & Score')
            plot_idx += 1
        
        # Subplot: PoolSize 放下面
        if df_pool is not None and not df_pool.empty:
            ax2 = axes1[plot_idx]
            sns.lineplot(x=df_pool['Step'], y=df_pool['PoolSize'], ax=ax2, label='PoolSize', color='tab:green', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Pool Size')
            ax2.set_title('Pool Size')
            ax2.legend(loc='upper left')
            
        fig1.tight_layout()

    # ---------------- 3. 绘制第二个 Figure: Win Rate ----------------
    if df_win is not None and not df_win.empty:
        # 假设 win_rate 文件里面必定包含的列，如无则自适应
        win_rate_cols = [col for col in df_win.columns if col.startswith('VsRule')]
        if not win_rate_cols:
            # 防止列名不完全匹配时的备选方案
            win_rate_cols = ['VsRule0', 'VsRule1', 'VsRule2', 'VsRule3', 'VsRule4']
        
        n_cols = len(win_rate_cols)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=True)
        fig2.suptitle(f"{exp_name} - 胜率分析")
        
        if n_cols == 1:
            axes2 = [axes2]
            
        for i, col in enumerate(win_rate_cols):
            ax = axes2[i]
            if col in df_win.columns:
                raw_data = df_win[col]
                smoothed_data = moving_average(raw_data, 100)
                
                sns.lineplot(x=df_win['Step'], y=raw_data, ax=ax, label=f"{col} (Raw)", color='tab:red', alpha=0.3)
                sns.lineplot(x=df_win['Step'], y=smoothed_data, ax=ax, label=f"{col} (Smoothed)", color='tab:red', linewidth=2)
                ax.set_ylim(-0.05, 1.05)
            else:
                ax.text(0.5, 0.5, f"{col} Data Missing", ha='center', va='center')
                
            ax.set_xlabel('Step')
            if i == 0:
                ax.set_ylabel('Win Rate')
            ax.set_title(col)
            ax.legend(loc="lower right")

        fig2.tight_layout()

    plt.show()

if __name__ == "__main__":
    # 保留您原有的目录设定
    DATA_DIRECTORY = Data_dir # r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\Data"
    EXPERIMENT_NAME = "ILHOPFSP"
    
    plot_training_curves(DATA_DIRECTORY, EXPERIMENT_NAME)
