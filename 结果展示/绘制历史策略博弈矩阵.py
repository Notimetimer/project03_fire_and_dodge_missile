from read_n_draw_inter_experiment_tests import *

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
        cbar_label="Avg Score",
    )
else:
    print(f"找不到文件: {csv_path}，请先运行生成矩阵的脚本。")