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

def plot_training_curves(data_dir, exp_name, win_rate_cols=None, display_titles=None):
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
        # 固定 16:10 左右的比例 (8, 5)
        fig1, axes1 = plt.subplots(fig1_rows, 1, figsize=(8, 5 * fig1_rows))
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

            # 定义标注用的深色，曲线用回原色
            color_reward_line = 'tab:blue'
            color_score_line = 'tab:orange'
            color_reward_label = 'black'
            color_score_label = 'black'

            # Reward放左侧纵坐标
            sns.lineplot(x=df_main['Step'], y=Reward, ax=ax1, color=color_reward_line, alpha=0.1)
            sns.lineplot(x=df_main['Step'], y=Reward_smoothed, ax=ax1, label='累积回报', color=color_reward_line, linewidth=1.5)
            ax1.set_xlabel('Step', fontweight='heavy')
            ax1.set_ylabel('累积回报', color='black', fontweight='heavy', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='black')
            # 解决网格遮挡曲线的问题，并将网格置于底层
            ax1.set_axisbelow(True)
            ax1.grid(False, axis='y')
            ax1.grid(True, axis='x', alpha=0.4)
            
            # Score放右侧纵坐标 (改名为 对抗得分)
            ax1_twin = ax1.twinx()
            sns.lineplot(x=df_main['Step'], y=Score, ax=ax1_twin, color=color_score_line, alpha=0.1)
            sns.lineplot(x=df_main['Step'], y=Score_smoothed, ax=ax1_twin, label='对抗得分', color=color_score_line, linewidth=1.5)
            ax1_twin.set_ylabel('对抗得分', color='black', fontweight='heavy', fontsize=12)
            ax1_twin.tick_params(axis='y', labelcolor='black')
            # 开启右侧y轴的网格线，同样置于底层
            ax1_twin.set_axisbelow(True)
            ax1_twin.grid(True, alpha=0.4)
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            if ax1_twin.get_legend() is not None:
                ax1_twin.get_legend().remove()

            # 加粗左右y轴刻度标签
            for label in ax1.get_yticklabels():
                label.set_fontweight('heavy')
            for label in ax1_twin.get_yticklabels():
                label.set_fontweight('heavy')
            
            # ax1.set_title('累积回报 & Avg score', fontweight='heavy')
            plot_idx += 1
            
        # 增加 pad 并控制 rect 以确保左右两边和窗口边缘有足够的留白
        fig1.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])


    # ---------------- 3. 绘制第二个 Figure: 相对基准得分曲线 ----------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("tab10", 5)
    y_min2, y_max2 = float('inf'), float('-inf')
    has_plotted2 = False

    logs_dir = os.path.join(project_root, "logs")
    for i in range(1, 6):
        csv_file = os.path.join(logs_dir, f"{i}.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                if 'Step' in df.columns and 'Value' in df.columns:
                    raw = df['Value']
                    smooth = moving_average(raw, 71)

                    sns.lineplot(x=df['Step'], y=raw, ax=ax2, color=colors[i-1], alpha=0.05)
                    sns.lineplot(x=df['Step'], y=smooth, ax=ax2, label=f'相对基准{i}得分', color=colors[i-1], linewidth=1.6)

                    y_min2 = min(y_min2, np.nanmin(smooth))
                    y_max2 = max(y_max2, np.nanmax(smooth))
                    has_plotted2 = True
            except Exception as e:
                print(f"读取 {csv_file} 失败: {e}")

    ax2.set_xlabel('Step', fontweight='heavy')
    ax2.set_ylabel('与基准策略对抗得分', fontweight='heavy')
    # 加粗y轴刻度标签
    for label in ax2.get_yticklabels():
        label.set_fontweight('heavy')
    for label in ax2.get_xticklabels():
        label.set_fontweight('heavy')
    # ax2.set_title('相对基准得分曲线', fontweight='heavy')

    if has_plotted2 and y_min2 < y_max2:
        pad = (y_max2 - y_min2) * 0.05
        ax2.set_ylim(y_min2 - pad, y_max2 + pad)
    elif has_plotted2:
        ax2.set_ylim(y_min2 - 0.05, y_min2 + 0.05)

    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.4)
    fig2.tight_layout(pad=3.0)

    # ---------------- 4. 绘制第四个 Figure: Imitation Learning 收敛曲线 ----------------
    il_curve_file = os.path.join(data_dir, "ILCurve.csv")
    if os.path.exists(il_curve_file):
        try:
            df_il = pd.read_csv(il_curve_file)
            if 'Step' in df_il.columns and 'Value' in df_il.columns:
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                
                raw_y = df_il['Value']
                # 使用滑动平均平滑数据
                smooth_loss = moving_average(raw_y, 15)
                
                # # 绘制浅色原始曲线
                # sns.lineplot(x=df_il['Step'], y=raw_y, ax=ax4, color='tab:green', alpha=0.2)
                # # 绘制深色平滑曲线
                # sns.lineplot(x=df_il['Step'], y=smooth_loss, ax=ax4, color='darkgreen', linewidth=1.8)

                sns.lineplot(x=df_il['Step'], y=raw_y, ax=ax4, color='darkgreen', linewidth=1.8)

                # 标注最低点和最高点的纵轴坐标
                min_idx = np.nanargmin(raw_y)
                max_idx = np.nanargmax(raw_y)
                min_x = df_il['Step'].iloc[min_idx]
                min_y = raw_y.iloc[min_idx]
                max_x = df_il['Step'].iloc[max_idx]
                max_y = raw_y.iloc[max_idx]

                ax4.text(min_x, min_y, f'{min_y:.3f}', color='darkgreen', fontsize=9,
                         ha='center', va='bottom', fontweight='heavy')
                ax4.text(max_x, max_y, f'{max_y:.3f}', color='darkgreen', fontsize=9,
                         ha='center', va='top', fontweight='heavy')

                ax4.set_xlabel("Epoch", fontweight='heavy')
                ax4.set_ylabel("分类准确率", fontweight='heavy')
                # 加粗刻度标签
                for label in ax4.get_yticklabels():
                    label.set_fontweight('heavy')
                for label in ax4.get_xticklabels():
                    label.set_fontweight('heavy')
                
                # 解决网格遮挡曲线的问题
                ax4.set_axisbelow(True)
                ax4.grid(True, alpha=0.4)
                
                fig4.tight_layout(pad=3.0)
        except Exception as e:
            print(f"读取或绘制 {il_curve_file} 失败: {e}")

    plt.show()

if __name__ == "__main__":
    # 保留您原有的目录设定
    DATA_DIRECTORY = Data_dir # r"d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\Data"
    EXPERIMENT_NAME = "SLWS-PFSP"
    
    win_rate_cols = ['0', '1', '2'] 
    display_titles = ['Agent Vs Rule1', 'Agent Vs Rule2', 'Agent Vs Rule3'] 
    linestyles = ['-', '-', '--', '-', '-.', '--', '-', ':'] # 使5(dash-dot)和8(dotted)线型完全不同
    
    plot_training_curves(DATA_DIRECTORY, EXPERIMENT_NAME, win_rate_cols=win_rate_cols, display_titles=display_titles)
