import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os
from _context import * # 包含 project_root

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_combat_matrix(csv_path, team_labels=None, 
                       title=None,
                       xlabel="Opponent Team (Column)",
                       ylabel="Evaluated Team (Row)",
                       cbar_label="Win Rate",
                       color_theme='blue',
                       show=True):
    """
    读取 CSV 并绘制博弈矩阵热力图。
    :param csv_path: CSV 文件路径
    :param team_labels: (Optional) 团队显示名称列表。如果不传，则使用 CSV 的列名。
    :param title: 图表标题
    :param xlabel: X轴标签
    :param ylabel: Y轴标签
    :param cbar_label: 颜色条标签
    :param color_theme: 'blue' (白蓝紫) 或 'red' (白红)
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found at {csv_path}")
        return

    # 读取 CSV (假设带有 header 和 index)
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return

    results_matrix = df.values
    
    # 确定标签
    if team_labels is not None:
        if len(team_labels) != len(df.columns):
             print(f"[Warning] Provided team_labels length ({len(team_labels)}) "
                   f"does not match matrix dimension ({len(df.columns)}). Using CSV headers.")
             labels = df.columns.tolist()
        else:
            labels = team_labels
    else:
        labels = df.columns.tolist()

    num_teams = len(labels)
    
    # [修改] 判断标签总长是否过长，如果所有标签总长度超过一定限制(比如40)，将展示改为数字代号
    total_len = sum(len(str(lbl)) for lbl in labels)
    use_numbered_legend = total_len > 40
    
    if use_numbered_legend:
        display_labels = [str(i+1) for i in range(num_teams)]
    else:
        display_labels = labels

    # 4. 绘图部分
    # 动态调整图片大小
    fig_size = max(8, num_teams + 2)
    plt.figure(figsize=(fig_size + 2, fig_size))
    
    # [修改] 使用从白到深蓝的颜色映射
    end_color = (0.06, 0.1, 0.38)  # 深蓝色 （0.6， 0.05， 0.05)深红色
    colors = [(1.0, 1.0, 1.0), end_color] # 白色到深蓝色
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)
    
    # 增加鲁棒性：根据当前数据的最小值和最大值自动调整色彩深度
    v_min_actual, v_max_actual = results_matrix.min(), results_matrix.max()
    v_range = v_max_actual - v_min_actual if v_max_actual > v_min_actual else 1.0
    
    # 增加 15% 的余量
    padding = 0.15 * v_range
    vmin = max(0.0, v_min_actual - padding)
    vmax = min(1.0, v_max_actual + padding)

    # 保持以 0.5 (中立评分) 为色彩过渡的中点，或者使用数据的真实中点
    vcenter = (vmin + vmax) / 2
    
    if vmin >= vmax:
        norm = None
    else:
        # 动态归一化并保留余地
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    # 绘制热力图并自定义 Colorbar 字体
    cbar_kws = {"label": cbar_label, "shrink": 0.8}
    
    ax = sns.heatmap(
        results_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        # vmin=vmin,
        # vmax=vmax,
        # xticklabels=display_labels,
        # yticklabels=display_labels,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 14},
        cbar_kws=cbar_kws
    )

    # 给矩阵外围加边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 获取并放大 Colorbar 标签及刻度字号
    cbar = ax.collections[0].colorbar
    cbar.set_label(cbar_label, size=14)
    cbar.ax.tick_params(labelsize=12)
    
    # 给颜色表(Colorbar)外围加边框
    if cbar.outline is not None:
        cbar.outline.set_visible(True)
        cbar.outline.set_linewidth(1.5)
        cbar.outline.set_edgecolor('black')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # 调整刻度标签字号：上边缘X轴倾斜15度，左边缘Y轴倾斜15度
    plt.xticks(fontsize=12, rotation=15, ha='left')
    plt.yticks(fontsize=12, rotation=15, ha='right')

    # 使用传入的自定义文本
    if title:
        plt.title(title, fontsize=18, pad=30)
    plt.xlabel(xlabel, fontsize=16, labelpad=20)
    plt.ylabel(ylabel, fontsize=16)

    # 只需要原始边距，去掉下方文字说明
    plt.subplots_adjust(top=0.78, bottom=0.12, left=0.2, right=0.95)
    if show:
        plt.show()

def draw_row_means_bar_chart(csv_path, team_labels=None, 
                             xlabel="Avg Score", 
                             title=None,
                             show=True):
    """
    绘制矩阵每行的均值及其标准差的横向条形图。
    """
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path, index_col=0)
    means = df.mean(axis=1)
    stds = df.std(axis=1)
    
    if team_labels is not None:
        labels = team_labels
    else:
        labels = df.index.tolist()

    # 绘制
    # 动态调整高度
    fig_height = max(5, len(labels) * 0.6)
    plt.figure(figsize=(10, fig_height))
    
    y_pos = range(len(labels))
    
    # 使用 seaborn 调色板
    colors = sns.color_palette("Blues_d", len(labels))
    colors = colors[::-1] # 反转颜色，让均值高的颜色更深
    
    # 重新按均值排序以获得更好的视觉效果？或者保持原序？
    # 用户通常希望保持矩阵行序，或者按性能排序。这里保持矩阵行序。
    plt.barh(y_pos, means, xerr=stds, align='center', alpha=0.85, 
             color=colors, edgecolor='black', linewidth=1.2, capsize=8)
    
    plt.yticks(y_pos, labels, fontsize=12)
    plt.xticks(fontsize=12)
    plt.gca().invert_yaxis()  # 反转 Y 轴，使第一行在顶部
    
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.title("Performance Average Across All Opponents", fontsize=16, fontweight='bold', pad=20)
        
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # 在条形图末尾标注数值
    for i, v in enumerate(means):
        plt.text(v + stds[i] + 0.01, i, f"{v:.2f}", color='black', va='center', fontweight='bold')

    plt.tight_layout()
    if show:
        plt.show()

if __name__ == "__main__":
    # 绘制实验内自博弈进度对比（历史切片博弈矩阵）
    csv_path = os.path.join(project_root, "结果展示", "outputs", "history_combat_matrix.csv")
    
    if os.path.exists(csv_path):
        print(f"正在读取并绘制: {csv_path}")
        draw_combat_matrix(
            csv_path = csv_path, 
            team_labels = ['1/4', '2/4', '3/4', '4/4'], # [修正] 对齐 4x4 维度
            title=None,
            xlabel="Opponent / Column",
            ylabel="Evaluated / Row",
            cbar_label="Score Rate",
        )
    else:
        print(f"找不到文件: {csv_path}，请先运行生成矩阵的脚本。")