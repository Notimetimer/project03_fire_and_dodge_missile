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
            
            # Score放右侧纵坐标 (改名为 Episode Score)
            ax1_twin = ax1.twinx()
            sns.lineplot(x=df_main['Step'], y=Score, ax=ax1_twin, color=color_score_line, alpha=0.1)
            sns.lineplot(x=df_main['Step'], y=Score_smoothed, ax=ax1_twin, label='Episode Score', color=color_score_line, linewidth=1.5)
            ax1_twin.set_ylabel('Episode Score', color=color_score_label, fontweight='bold')
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
            
            ax1.set_title('Episode Reward & Avg score', fontweight='bold')
            plot_idx += 1
            
        # 增加 pad 并控制 rect 以确保左右两边和窗口边缘有足够的留白
        fig1.tight_layout(pad=3.0, rect=[0.05, 0.05, 0.95, 0.95])

    # ---------------- 3. 绘制第二个 Figure: Score ----------------
    if name_list is not None and name_list_show is not None:
        # 如果未传入，则使用默认的列名、标题和线型
        if win_rate_cols is None:
            win_rate_cols = ['VsRule0', 'VsRule1', 'VsRule2'] # , 'VsRule3', 'VsRule4']
        if display_titles is None:
            display_titles = ['Agent Vs Rule1', 'Agent Vs Rule2', 'Agent Vs Rule3'] # , 'Rule4', 'Rule5']
        if linestyles is None:
            # 索引对应：0, 1, 2(点线), 3, 4(点划线), 5(点划线)
            linestyles = ['-', '-', ':', '-', '-.', '-.']
        
        n_cols = len(win_rate_cols)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.5))
        fig2.suptitle("Score vs. All Rules", fontsize=16, y=0.81)
        
        if n_cols == 1:
            axes2 = [axes2]
            
        palette = list(sns.color_palette("tab10", max(10, len(name_list))))
        # 强行定制显眼的颜色区分：5号加深，8号设为纯黑
        if len(palette) >= 8:
            palette[4] = (0.8, 0.0, 0.8) # 5号: 亮紫洋红
            palette[7] = (0.0, 0.0, 0.0) # 8号: 纯黑
            
        for i, (col, display_title) in enumerate(zip(win_rate_cols, display_titles)):
            ax = axes2[i]
            
            # 记录当前子图所有平滑曲线的最小值和最大值
            y_min_total, y_max_total = float('inf'), float('-inf')
            has_plotted = False
            for j, (name, show_name) in enumerate(zip(name_list, name_list_show)):
                win_file = os.path.join(data_dir, f"{show_name}.csv") # READ {show_name}.csv (which is 1.csv to 8.csv)
                try:
                    df = pd.read_csv(win_file)
                    if col in df.columns:
                        raw_data = df[col]
                        smoothed_data = moving_average(raw_data, 101)
                        
                        y_min_total = min(y_min_total, np.nanmin(smoothed_data))
                        y_max_total = max(y_max_total, np.nanmax(smoothed_data))

                        # 获取当前实验对应的线型
                        ls = linestyles[j % len(linestyles)]
                        # 根据线型动态调整粗细：点线加粗到1.36，否则用1.12 (约为原先0.8倍)
                        lw = 1.36 if ls == ':' else 1.12
                        
                        # sns.lineplot(x=df['Step'], y=raw_data, ax=ax, color=palette[j], alpha=0.04) # 移除原始数据背景
                        sns.lineplot(x=df['Step'], y=smoothed_data, ax=ax, label=str(show_name), 
                                     color=palette[j], linewidth=lw, linestyle=ls)
                        has_plotted = True
                except Exception as e:
                    pass
                
            if has_plotted and y_min_total < y_max_total:
                pad = (y_max_total - y_min_total) * 0.05
                ax.set_ylim(y_min_total - pad, y_max_total + pad)
            elif has_plotted:
                ax.set_ylim(y_min_total - 0.05, y_min_total + 0.05)
            else:
                ax.set_ylim(-0.05, 1.05)

            ax.set_xlabel('Step')
            ax.set_ylabel('')
            # 显示对应映射的新名称，而不是原CSV列名
            ax.set_title(display_title)
            
            # 添加参考线 (无图例)
            for hval in [0, 0.5, 1.0]:
                ax.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=1.0, zorder=5)
                
            # 只有在真正画了线的情况下才调用 legend，避免 No artists 警告
            if has_plotted:
                ax.legend(loc="lower right", fontsize='small', ncol=2) 

        # pad=1.0 减小多余空白，避免挤压图表高度
        # 将 w_pad 从 1.0 调低到 0.4，让子图在横向上贴得更近
        fig2.tight_layout(pad=1.0, w_pad=0.4, rect=[0, 0, 1, 0.86])

    elif df_win is not None and not df_win.empty:
        # Fallback for single experiment
        win_rate_cols = [col for col in df_win.columns if col.startswith('VsRule')]
        if not win_rate_cols:
            win_rate_cols = ['Agent Vs Rule1', 'Agent Vs Rule2', 'Agent Vs Rule3'] 
        
        n_cols = len(win_rate_cols)
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.0))
        fig2.suptitle("Score vs. All Rules", fontsize=16, y=0.81)
        
        if n_cols == 1:
            axes2 = [axes2]
            
        for i, col in enumerate(win_rate_cols):
            ax = axes2[i]
            if col in df_win.columns:
                raw_data = df_win[col]
                smoothed_data = moving_average(raw_data, 101)
                
                sns.lineplot(x=df_win['Step'], y=raw_data, ax=ax, color='tab:red', alpha=0.06) # alpha=0.05
                sns.lineplot(x=df_win['Step'], y=smoothed_data, ax=ax, label=col, color='tab:red', linewidth=1.6)
                
                y_min, y_max = np.nanmin(smoothed_data), np.nanmax(smoothed_data)
                if y_min < y_max:
                    pad = (y_max - y_min) * 0.05
                    ax.set_ylim(y_min - pad, y_max + pad)
                else:
                    ax.set_ylim(y_min - 0.05, y_min + 0.05)
            else:
                ax.text(0.5, 0.5, f"{col} Data Missing", ha='center', va='center')
                
            ax.set_xlabel('Step')
            ax.set_ylabel('')
            ax.set_title(col)
            
            # 添加参考线 (无图例)
            for hval in [0, 0.5, 1.0]:
                ax.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=1.0, zorder=5)
            
        fig2.tight_layout(pad=1.0, w_pad=0.4, rect=[0, 0, 1, 0.86])

    # ---------------- 4. 绘制第三个 Figure: 实验 1 的 Score 曲线变化 ----------------
    csv_1_file = os.path.join(data_dir, "1.csv")
    if os.path.exists(csv_1_file):
        try:
            df1 = pd.read_csv(csv_1_file)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            
            colors_rules = sns.color_palette("tab10", len(win_rate_cols))
            
            y_min3, y_max3 = 1.0, 0.0
            has_plotted3 = False
            
            for i, (col, display_title) in enumerate(zip(win_rate_cols, display_titles)):
                if col in df1.columns:
                    raw = df1[col]
                    smooth = moving_average(raw, 51)
                    
                    sns.lineplot(x=df1['Step'], y=raw, ax=ax3, color=colors_rules[i], alpha=0.1) # 恢复原始数据背景
                    sns.lineplot(x=df1['Step'], y=smooth, ax=ax3, label=display_title, color=colors_rules[i], linewidth=1.6)
                    
                    y_min3 = min(y_min3, np.nanmin(smooth))
                    y_max3 = max(y_max3, np.nanmax(smooth))
                    has_plotted3 = True
            
            ax3.set_xlabel("Step", fontweight='bold')
            ax3.set_ylabel("Test Score", fontweight='bold')

            # 添加参考线 (无图例)
            for hval in [0, 0.5, 1.0]:
                ax3.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=1.0, zorder=5)
            
            if has_plotted3:
                ax3.legend(loc='lower right')
                p_range3 = y_max3 - y_min3
                if p_range3 > 0:
                    ax3.set_ylim(y_min3 - 0.05 * p_range3, y_max3 + 0.05 * p_range3)
                else:
                    ax3.set_ylim(y_min3 - 0.05, y_min3 + 0.05)
            
            fig3.tight_layout(pad=3.0)
        except Exception as e:
            print(f"读取或绘制 {csv_1_file} 失败: {e}")


    # ---------------- 5. 绘制第四个 Figure: Imitation Learning 收敛曲线 ----------------
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
                
                ax4.set_xlabel("Epoch", fontweight='bold')
                ax4.set_ylabel("分类准确率", fontweight='bold')
                
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
    EXPERIMENT_NAME = "ILHOPFSP"

    name_list = [
        "CSVersusRules-run-20260614-163906",
        "CSPFSP-run-20260615-234324",
        "HLWSPFSP-run-20260616-130304",
        "SLWSPFSP0.3-run-20260618-221044",
        "SLWSPFSP0.5-run-20260620-211720",
        "SLWSPFSP0.2-run-20260622-185856",
    ]
    # [
    #     "IL_and_MixedPFSP_分阶段_挑战_并行_分层2s",
    #     "IL_and_MixedPFSP_低门槛_挑战_并行_分层2s",
    #     "IL_and_MixedPFSP_高门槛_挑战_并行_分层2s",
    #     "IL_and_PFSP_挑战_并行_分层2s",
    #     "纯Rule4训练_分层_挑战2s",
    #     "NoILPFSP_分阶段_混规则对手_挑战_并行_分层2s",
    #     "NoILPFSP_分阶段_挑战_并行_分层2s",
    #     "IL_and_deltaFSP_挑战_并行_分层2s",
    # ]
    name_list_show = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    
    win_rate_cols = ['0', '1', '2'] 
    display_titles = ['Agent Vs Rule1', 'Agent Vs Rule2', 'Agent Vs Rule3'] 
    linestyles = ['-', '-', '--', '-', '-.', '--', '-', ':'] # 使5(dash-dot)和8(dotted)线型完全不同
    
    plot_training_curves(DATA_DIRECTORY, EXPERIMENT_NAME, name_list, name_list_show, 
                         win_rate_cols, display_titles, linestyles)
