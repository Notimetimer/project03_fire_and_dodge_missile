from read_n_draw_inter_experiment_tests import *

mission_names = [
        'IL_and_MixedPFSP',
        'IL_and_PFSP',
        'MixedPFSP',
        'IL_and_deltaFSP',
        'IL_and_MixedPFSP_A3C',
        '纯Rule4训练',
    ]

csv_path = os.path.join(project_root, "结果展示", "outputs", "combat_matrix.csv")

# 读取实际 CSV 并动态适配标签
if os.path.exists(csv_path):
    import pandas as pd
    df_temp = pd.read_csv(csv_path, index_col=0)
    num_columns = len(df_temp.columns)
    
    # 自动对齐标签长度（优先匹配当前的 CSV 维度）
    team_labels = mission_names[:num_columns]
    
    print(f"正在读取并绘制: {csv_path} (矩阵维度: {num_columns}x{num_columns})")
    draw_combat_matrix(
        csv_path = csv_path, 
        team_labels = team_labels, 
        title=None,
        xlabel="Opponent / Column",
        ylabel="Evaluated / Row",
        cbar_label="Win Rate",
    )
else:
    print(f"找不到文件: {csv_path}，请先运行生成矩阵的脚本。")