from read_n_draw_inter_experiment_tests import *

mission_names = [
        1, # 'IL_and_MixedPFSP_分阶段_挑战_并行_分层2s',
        2, # 'IL_and_MixedPFSP_低门槛_挑战_并行_分层2s',
        3, # 'IL_and_MixedPFSP_高门槛_挑战_并行_分层2s',
        4, # 'IL_and_PFSP_挑战_并行_分层2s',
        5, # '纯Rule4训练_分层_挑战2s',
        6, # 'NoILPFSP_分阶段_混规则对手_挑战_并行_分层2s',
        7, # NoILPFSP_分阶段_挑战_并行_分层2s
        8, # IL_and_deltaFSP_挑战_并行_分层2s
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
        cbar_label="Pairwise Avg Score",
        show=False  # 设置为 False，不立即阻塞
    )

    # --- Figure 2: Row Means Bar Chart ---
    print(f"正在读取并绘制: {csv_path} (各行均值与标准差)")
    draw_row_means_bar_chart(
        csv_path = csv_path,
        team_labels = team_labels,
        xlabel="Cross-Opponent Mean Score",
        title="Main Agent Mean Performance Against All Opponents",
        show=False  # 设置为 False，不立即阻塞
    )

    # 最后统一显示所有窗口
    plt.show()
else:
    print(f"找不到文件: {csv_path}，请先运行生成矩阵的脚本。")