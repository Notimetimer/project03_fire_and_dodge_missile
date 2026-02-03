import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_combat_matrix(csv_path, team_labels=None):
    """
    读取 CSV 并绘制博弈矩阵热力图。
    :param csv_path: CSV 文件路径
    :param team_labels: (Optional) 团队显示名称列表。如果不传，则使用 CSV 的列名。
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

    # 4. 绘图部分（单色：白 -> 紫）
    # 动态调整图片大小
    fig_size = max(8, num_teams + 2)
    plt.figure(figsize=(fig_size + 2, fig_size))
    
    # 手动指定色彩空间向量（RGB 0-1）
    colors = [
        (1.0, 1.0, 1.0),
        (0.6, 0.05, 0.05),
    ]
    
    cmap = LinearSegmentedColormap.from_list("white_purple", colors, N=256)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    ax = sns.heatmap(
        results_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Red Team Win Rate", "shrink": 0.8}
    )

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.title("Cross-Algorithm Final Agents Combat Matrix", fontsize=14, pad=40)
    plt.xlabel("Blue Team (Opponent / Column)", fontsize=12, labelpad=15)
    plt.ylabel("Red Team (Evaluated / Row)", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 简单测试逻辑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_path = os.path.join(current_dir, "outputs", "combat_matrix.csv")
    print(f"Testing draw on: {test_path}")
    draw_combat_matrix(test_path)