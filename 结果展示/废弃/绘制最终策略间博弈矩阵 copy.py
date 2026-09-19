from read_n_draw_inter_experiment_tests import *

mission_names = None
    # [
    #     1, # 'IL_and_MixedPFSP_分阶段_挑战_并行_分层2s',
    #     2, # 'IL_and_MixedPFSP_低门槛_挑战_并行_分层2s',
    #     3, # 'IL_and_MixedPFSP_高门槛_挑战_并行_分层2s',
    #     4, # 'IL_and_PFSP_挑战_并行_分层2s',
    #     5, # '纯Rule4训练_分层_挑战2s',
    #     6, # 'NoILPFSP_分阶段_混规则对手_挑战_并行_分层2s',
    #     7, # NoILPFSP_分阶段_挑战_并行_分层2s
    #     8, # IL_and_deltaFSP_挑战_并行_分层2s
    # ]

csv_files = [
    ("combat_matrix_backup.csv", "Full Progress"),
    ("combat_matrix_half_backup.csv", "Half Progress"),
]

for csv_name, label in csv_files:
    csv_path = os.path.join(project_root, "结果展示", "outputs", csv_name)
    if not os.path.exists(csv_path):
        print(f"[SKIP] 找不到文件: {csv_path}")
        continue

    import pandas as pd
    df_temp = pd.read_csv(csv_path, index_col=0)
    num_columns = len(df_temp.columns)
    if mission_names:
        team_labels = mission_names[:num_columns]
    else:
        team_labels = None

    print(f"\n正在读取并绘制 [{label}]: {csv_path} (矩阵维度: {num_columns}x{num_columns})")
    draw_combat_matrix(
        csv_path = csv_path,
        team_labels = team_labels,
        title=f"Combat Matrix ({label})",
        cbar_label="Pairwise Avg Score",
        show=False  # 设置为 False，不立即阻塞
    )

    # # 行均值柱状图
    # print(f"正在读取并绘制 [{label}] 行均值图")
    # draw_row_means_bar_chart(
    #     csv_path = csv_path,
    #     team_labels = team_labels,
    #     xlabel="Cross-Opponent Mean Score",
    #     title=f"Mean Performance Against All Opponents ({label})",
    #     show=False  # 设置为 False，不立即阻塞
    # )

# 最后统一显示所有窗口
plt.show()