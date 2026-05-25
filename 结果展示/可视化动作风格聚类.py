
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from _context import *

from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path


# ==========================================
# 0. 全版本通杀 threadpoolctl 补丁 (守卫3.8环境)
# ==========================================
try:
    import threadpoolctl
    threadpoolctl.threadpool_info = lambda *args, **kwargs: []
    class DummyContextManager:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    threadpoolctl.threadpool_limits = DummyContextManager
    if hasattr(threadpoolctl, 'ThreadpoolController'):
        threadpoolctl.ThreadpoolController.info = lambda self, *args, **kwargs: []
        threadpoolctl.ThreadpoolController.limit = lambda self, *args, **kwargs: DummyContextManager()
except Exception:
    pass

def run_offline_tactical_analysis(json_path="Elite_Fire_Stats.json", n_clusters=4, weights=None):
    # 1. 严格读取项目文件
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到特征统计文件: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        
    # 过滤可能存在字典底部的统计控制键（如 __LAST_UPDATE_STEP__ 等）
    valid_keys = [k for k in data_dict.keys() if not k.startswith("__")]
    
    # 提取纯弧度动作向量
    X_raw = np.array([data_dict[k] for k in valid_keys], dtype=np.float32)
    
    if len(X_raw) < n_clusters:
        print(f"[Warning] 样本量 ({len(X_raw)}) 小于设定的流派数 ({n_clusters})，自动重设簇数。")
        n_clusters = len(X_raw)

    # 2. 依据你给出的4角度参数架构进行基础归一化（Max Base Scaling）
    # 映射关系: 
    # [开火俯仰角(±pi/2), 开火后30s内Avg_ATA(0~pi), 告警后Avg_delta_psi(0~pi), 开火后30s内Avg_delta_theta(±pi/2), 开火后30s内Avg_delta_psi(0~pi)]
    X_norm = np.column_stack((
        X_raw[:, 0] / (np.pi / 2),  # fire_theta
        X_raw[:, 1] / np.pi,        # Avg_ATA
        X_raw[:, 2] / np.pi,        # delta_psi_threat (防守核心)
        X_raw[:, 3] / (np.pi / 2),  # delta_theta
        X_raw[:, 4] / np.pi         # delta_psi
    ))

    # 3. 应用维度相对重要性乘子
    strat_weights = weights
    X_weighted = X_norm * strat_weights

    # 4. KMeans 线性空间流派自动切分
    # 预防去重向量少于簇数抛出异常
    n_unique = len(np.unique(X_weighted, axis=0))
    actual_clusters = min(n_clusters, n_unique)
    
    if actual_clusters > 1:
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_weighted)
    else:
        labels = np.zeros(len(X_raw), dtype=int)

    # 5. 画布架构部署 (20x12 复合看板，使用GridSpec实现左右不同行间距)
    fig = plt.figure(figsize=(20, 12))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1, 1, 1, 1, 1])
    
    feature_names = [
        'Fire Pitch (deg)', 
        'Avg ATA 30s (deg)', 
        'Warning Delta Psi Threat (deg)', 
        'Delta Pitch 30s (deg)', 
        'Delta Psi 30s (deg)'
    ]

    # --- 读取 Elo 评分 ---
    elo_path = os.path.join(os.path.dirname(json_path), "elo_ratings.json")
    elo_scores = None
    if os.path.exists(elo_path):
        with open(elo_path, 'r', encoding='utf-8') as f:
            elo_dict = json.load(f)
        elo_scores = np.array([elo_dict.get(k, np.nan) for k in valid_keys], dtype=np.float64)

    # --- 左半壁：5个原始维度的一维数轴离散分布图（回归数轴形式，不显示Elo） ---
    for idx in range(5):
        ax = fig.add_subplot(gs[idx, 0])
        x_vals = X_raw[:, idx] / np.pi * 180
        y_jitter = np.random.normal(0, 0.04, size=len(X_raw)) if len(X_raw) > 1 else np.zeros(len(X_raw))
        ax.scatter(x_vals, y_jitter, c=labels, cmap='Set1', s=60, alpha=0.8, edgecolors='k')
        ax.set_ylim(-0.5, 0.5)
        ax.get_yaxis().set_visible(False)
        ax.set_title(f'1D Physical Dimension: {feature_names[idx]}', fontsize=11, weight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)

    # --- 右上：3D PCA 全空间拓扑投影 ---
    if X_weighted.shape[1] >= 3 and len(X_raw) >= 3:
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_weighted)
        var_3d = np.sum(pca_3d.explained_variance_ratio_) * 100

        ax_3d = fig.add_subplot(gs[0:2, 1], projection='3d')
        ax_3d.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=labels, cmap='Set1', s=70, alpha=0.8, edgecolors='k')
        ax_3d.set_title(f'3D PCA Space Projection (Retains {var_3d:.1f}% Variance)', fontsize=11, weight='bold')
        ax_3d.set_xlabel('PC1')
        ax_3d.set_ylabel('PC2')
        ax_3d.set_zlabel('PC3')

    # --- 右中：各聚类平均Elo条形图（X轴类别，Y轴Elo，标注样本数） ---
    if elo_scores is not None and actual_clusters > 1:
        ax_bar = fig.add_subplot(gs[2, 1])
        unique_labels = np.unique(labels)
        avg_elos = []
        counts = []
        for lbl in unique_labels:
            mask = labels == lbl
            avg_elos.append(np.nanmean(elo_scores[mask]))
            counts.append(np.sum(mask))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        bars = ax_bar.bar(unique_labels, avg_elos, color=colors, alpha=0.8, edgecolor='k')
        ax_bar.set_xlabel('Cluster ID', fontsize=10)
        ax_bar.set_ylabel('Average Elo Rating', fontsize=10)
        ax_bar.set_title('Average Elo by Cluster', fontsize=11, weight='bold')
        ax_bar.grid(True, linestyle='--', alpha=0.4, axis='y')
        
        # 在条形上方标注样本数
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax_bar.annotate(f'n={count}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)

    # --- 右下：2D PCA 平面投影 ---
    if X_weighted.shape[1] >= 2 and len(X_raw) >= 2:
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_weighted)
        var_2d = np.sum(pca_2d.explained_variance_ratio_) * 100

        ax_2d = fig.add_subplot(gs[3:5, 1])
        ax_2d.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='Set1', s=70, alpha=0.8, edgecolors='k')
        ax_2d.set_title(f'2D PCA Space Projection (Retains {var_2d:.1f}% Variance)', fontsize=11, weight='bold')
        ax_2d.set_xlabel('PC1')
        ax_2d.set_ylabel('PC2')
        ax_2d.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Joint Tactical Sub-Space & PCA Dimensionality Reduction看板", fontsize=16, weight='bold')
    plt.tight_layout()
    
    # 导出为高清晰度物理图像
    output_img = "joint_tactical_analysis.png"
    plt.show()
    # plt.savefig(output_img, dpi=150)
    # plt.close()
    print(f"[Success] 离线战术看板分析完成！保存路径: {output_img}")

if __name__ == "__main__":
    # 长名称优先（指定日期和时间）
    dir_name = None # "IL_and_MixedPFSP_分阶段_挑战_并行_分层2s-run-20260408-175230"
   
    # 短名称次之（自动找最新实验结果）
    experiment_name = 'IL_and_Mixed经典PFSP_多技术流派_并行_分层_rule3_0.1'
    # --- 查找模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    
    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, experiment_name)
    print(f"Log directory: {latest_log_dir}")
    json_path = os.path.join(latest_log_dir, "Elite_Fire_Stats.json")
    strat_weights = np.array([
        1.0,  # 开火俯仰角 (平射 vs 高抛)
        1.0,  # Avg ATA     (是否维持中制导)
        1.0,  # delta_psi_threat (置尾逃逸 vs 三九线缠斗，重度提权)
        1.0,  # delta_theta (常规机动幅度)
        1.0   # delta_psi   (常规机动幅度)
    ])
    run_offline_tactical_analysis(json_path, n_clusters=5, weights=strat_weights)