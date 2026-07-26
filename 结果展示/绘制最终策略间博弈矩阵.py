from read_n_draw_inter_experiment_tests import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns

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
    ("combat_matrix_half.csv", "Half Progress"),
    ("combat_matrix.csv", "Full Progress"),
]

# ---------- 公共数据与配色 ----------
all_values = []
for csv_name, _ in csv_files:
    csv_path = os.path.join(project_root, "结果展示", "outputs", csv_name)
    if not os.path.exists(csv_path):
        print(f"[SKIP] 找不到文件: {csv_path}")
        continue
    df_temp = pd.read_csv(csv_path, index_col=0)
    all_values.extend(df_temp.values.flatten().tolist())

v_min_actual = min(all_values) if all_values else 0.0
v_max_actual = max(all_values) if all_values else 1.0
v_range = v_max_actual - v_min_actual if v_max_actual > v_min_actual else 1.0
padding = 0.15 * v_range
vmin = max(0.0, v_min_actual - padding)
vmax = min(1.0, v_max_actual + padding)
vcenter = (vmin + vmax) / 2
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

end_color = (0.06, 0.1, 0.38)
cmap = LinearSegmentedColormap.from_list("custom_blue", [(1.0, 1.0, 1.0), end_color], N=256)


# ---------- 绘制单张 heatmap 的公共函数 ----------
def draw_heatmap(ax, csv_name, label, show_y=True):
    csv_path = os.path.join(project_root, "结果展示", "outputs", csv_name)
    if not os.path.exists(csv_path):
        ax.set_title(f"[MISSING] {csv_name}")
        return
    df = pd.read_csv(csv_path, index_col=0)
    results = df.values
    labels = [str(col).replace('_', '-') for col in df.columns.tolist()]

    sns.heatmap(
        results,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        norm=norm,
        xticklabels=labels,
        yticklabels=labels if show_y else False,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 12},
        cbar=False,
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')

    ax.xaxis.tick_top()
    ax.set_title(label, fontsize=16, pad=15)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha='left',
        va='bottom',
        rotation_mode='anchor',
        fontsize=11,
    )
    if show_y:
        ax.set_yticklabels(
            labels,
            rotation=20,
            ha='right',
            va='center',
            rotation_mode='anchor',
            fontsize=11,
        )
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("")


def add_colorbar(fig):
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.15, right=0.85)


# ---------- Figure 1：合并绘制 ----------
fig1, axes = plt.subplots(1, 2, figsize=(16, 7.2), gridspec_kw={'wspace': 0.05})
for ax, (csv_name, label) in zip(axes, csv_files):
    draw_heatmap(ax, csv_name, label, show_y=(ax == axes[0]))
add_colorbar(fig1)


# ---------- Figure 2 / 3：分别单独绘制 ----------
for csv_name, label in csv_files:
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    draw_heatmap(ax, csv_name, label, show_y=True)
    add_colorbar(fig)

plt.show()