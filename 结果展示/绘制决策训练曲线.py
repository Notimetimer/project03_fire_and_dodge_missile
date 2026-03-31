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
            color_reward_label = 'navy'
            color_score_label = 'darkorange'

            # Reward放左侧纵坐标
            sns.lineplot(x=df_main['Step'], y=Reward, ax=ax1, color=color_reward_line, alpha=0.1)
            sns.lineplot(x=df_main['Step'], y=Reward_smoothed, ax=ax1, label='Episode Reward', color=color_reward_line, linewidth=1.5)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Episode Reward', color=color_reward_label, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color_reward_label)
            # 解决网格遮挡曲线的问题，并将网格置于底层
            ax1.set_axisbelow(True)
            ax1.grid(False, axis='y')
            ax1.grid(True, axis='x', alpha=0.4)
            
            # Score放右侧纵坐标 (改名为 Avg Win Rate)
            ax1_twin = ax1.twinx()
            sns.lineplot(x=df_main['Step'], y=Score, ax=ax1_twin, color=color_score_line, alpha=0.1)
            sns.lineplot(x=df_main['Step'], y=Score_smoothed, ax=ax1_twin, label='Avg Win Rate', color=color_score_line, linewidth=1.5)
            ax1_twin.set_ylabel('Avg Win Rate', color=color_score_label, fontweight='bold')
            ax1_twin.tick_params(axis='y', labelcolor=color_score_label)
            # 开启右侧y轴的网格线，同样置于底层
            ax1_twin.set_axisbelow(True)
            ax1_twin.grid(True, alpha=0.4)
            
            # 合并图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
            if ax1_twin.get_legend() is not None:
                ax1_twin.get_legend().remove()
            
            ax1.set_title('Episode Reward & Avg Win Rate', fontweight='bold')
            plot_idx += 1
            
        # 增加 pad 并控制 rect 以确保左右两边和窗口边缘有足够的留白
        fig1.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])

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

    # ---------------- 4. 绘制第三个 Figure: Actor Loss (ILCurve) ----------------
    il_curve_file = os.path.join(data_dir, "ILCurve.csv")
    if os.path.exists(il_curve_file):
        try:
            df_il = pd.read_csv(il_curve_file)
            if 'Epoch' in df_il.columns and 'ActorLoss' in df_il.columns:
                fig3, ax3 = plt.subplots(figsize=(8, 4))
                loss_data = df_il['ActorLoss']
                
                # Figure 3 不需要平滑，直接绘制原始数据
                sns.lineplot(x=df_il['Epoch'], y=loss_data, ax=ax3, color='tab:purple', linewidth=1.5)
                
                ax3.set_title('Imitation Learning Actor Loss')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Actor Loss')
                # ax3.legend(loc='upper right') # 只有一条线时也可以不用 legend
                # 增加 pad 并控制 rect 以确保左右两边和窗口边缘有足够的留白
                fig3.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])
        except Exception as e:
            print(f"读取或绘制 {il_curve_file} 失败: {e}")

    # ---------------- 5. 绘制第四个 Figure: Win Rate of Algorithm 1 Against All Rules ----------------
    if name_list is not None and len(name_list) > 0:
        first_algo_name = name_list[0]
        first_algo_win_file = os.path.join(data_dir, f"{first_algo_name}_win_rate.csv")
        if os.path.exists(first_algo_win_file):
            try:
                df_win1 = pd.read_csv(first_algo_win_file)
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                
                # 为不同的 Rule 设置调色盘
                palette4 = sns.color_palette("tab10", len(win_rate_cols))
                has_plotted4 = False
                
                for k, (col, display_title) in enumerate(zip(win_rate_cols, display_titles)):
                    if col in df_win1.columns:
                        raw_data = df_win1[col]
                        # 降低平滑度，单独对 Figure 4 使用较弱的平滑 (比如 31 或 41)
                        smoothed_data = moving_average(raw_data, 41)
                        
                        sns.lineplot(x=df_win1['Step'], y=raw_data, ax=ax4, color=palette4[k], alpha=0.1)
                        sns.lineplot(x=df_win1['Step'], y=smoothed_data, ax=ax4, label=display_title, 
                                     color=palette4[k], linewidth=1.5)
                        has_plotted4 = True
                
                # 画出 0, 0.5 和 1 的黑色虚线
                ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                ax4.axhline(0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
                ax4.axhline(1, color='black', linestyle='--', linewidth=1.5, alpha=0.6)

                if has_plotted4:
                    ax4.set_ylim(-0.05, 1.05)
                    ax4.set_xlabel('Step', fontweight='bold')
                    ax4.set_ylabel('Win Rate', fontweight='bold')
                    # ax4.set_title(f"Win Rate of {first_algo_name} Against All Rules", fontweight='bold')
                    ax4.legend(loc='lower right')
                    fig4.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])
            except Exception as e:
                print(f"读取或绘制 {first_algo_win_file} (Figure 4) 失败: {e}")
    # ---------------- 6. 绘制第五个 Figure: Elo Rank & Elite Opponent Pool Size ----------------
    if name_list is not None and len(name_list) > 0:
        first_algo_name = name_list[0]
        rank_file = os.path.join(data_dir, f"{first_algo_name}_Elo_Rank.csv")
        pool_file = os.path.join(data_dir, f"{first_algo_name}_pool_size.csv")
        
        has_rank = os.path.exists(rank_file)
        has_pool = os.path.exists(pool_file)
        
        if has_rank or has_pool:
            try:
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                color_rank = 'tab:green'
                color_pool = 'tab:red'
                
                # 绘制 Elo Rank (左边Y轴)
                if has_rank:
                    df_rank = pd.read_csv(rank_file)
                    if 'Step' in df_rank.columns and 'Rank' in df_rank.columns:
                        rank_data = df_rank['Rank']
                        rank_smoothed = moving_average(rank_data, 21)
                        
                        sns.lineplot(x=df_rank['Step'], y=rank_data, ax=ax5, color=color_rank, alpha=0.1)
                        sns.lineplot(x=df_rank['Step'], y=rank_smoothed, ax=ax5, label='Elo Rank', color=color_rank, linewidth=1.5)
                        
                        ax5.set_xlabel('Step', fontweight='bold')
                        ax5.set_ylabel('Elo Rank', color=color_rank, fontweight='bold')
                        ax5.tick_params(axis='y', labelcolor=color_rank)
                        for label in ax5.get_yticklabels():
                            label.set_fontweight('bold')
                            label.set_color(color_rank)
                        for label in ax5.get_xticklabels():
                            label.set_fontweight('bold')
                            
                        ax5.set_axisbelow(True)
                        ax5.grid(False, axis='y')
                        ax5.grid(True, axis='x', alpha=0.4)
                
                # 绘制 Elite Opponent Pool Size (右边Y轴)
                if has_pool:
                    ax5_twin = ax5.twinx() if has_rank else ax5
                    df_pool = pd.read_csv(pool_file)
                    
                    if 'Step' in df_pool.columns and 'PoolSize' in df_pool.columns:
                        pool_data = df_pool['PoolSize']
                        pool_smoothed = moving_average(pool_data, 21)
                        
                        sns.lineplot(x=df_pool['Step'], y=pool_data, ax=ax5_twin, color=color_pool, alpha=0.1)
                        sns.lineplot(x=df_pool['Step'], y=pool_smoothed, ax=ax5_twin, label='Elite Opponent Pool Size', color=color_pool, linewidth=1.5)
                        
                        ax5_twin.set_ylabel('Elite Opponent Pool Size', color=color_pool, fontweight='bold')
                        ax5_twin.tick_params(axis='y', labelcolor=color_pool)
                        for label in ax5_twin.get_yticklabels():
                            label.set_fontweight('bold')
                            label.set_color(color_pool)
                            
                        ax5_twin.set_axisbelow(True)
                        ax5_twin.grid(True, alpha=0.4)
                        
                        # 合并图例 (如果有两轴的话)
                        if has_rank:
                            handles5, labels5 = ax5.get_legend_handles_labels()
                            handles5_twin, labels5_twin = ax5_twin.get_legend_handles_labels()
                            ax5.legend(handles5 + handles5_twin, labels5 + labels5_twin, loc='center right')
                            if ax5_twin.get_legend() is not None:
                                ax5_twin.get_legend().remove()
                        else:
                            ax5_twin.legend(loc='center right')
                
                ax5.set_title('Elo Rank & Elite Opponent Pool Size', fontweight='bold')
                fig5.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])
            
            except Exception as e:
                print(f"读取或绘制 Figure 5 (Rank & PoolSize) 失败: {e}")

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
    display_titles = ['vs Rule1', 'vs Rule2', 'vs Rule3', 'vs Rule4', 'vs Rule5']
    linestyles = ['-', '-', '--', '-', '--', ':']
    
    plot_training_curves(DATA_DIRECTORY, EXPERIMENT_NAME, name_list, name_list_show, 
                         win_rate_cols, display_titles, linestyles)
