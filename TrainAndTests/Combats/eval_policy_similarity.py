import sys
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

# 把当前路径加入 sys.path 防止 import 错误
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cur_dir))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

from CombatPPOWithIL3_parallel_hierarch import load_il_and_transitions
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid

# ==============================================================================
# 请在这里填入你正在进行训练或需要分析的 Log 文件夹名称（或绝对路径） ! 
# 默认会去 ../../logs/combat/ 下面找。
# 例如: r"经典PFSP无规则对手训练-run-202X..." 或绝对路径
# ==============================================================================
TARGET_LOG_DIR = None 

def get_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2 + 1e-8)

def extract_features(actor_net, states_tensor, device):
    """
    给定一批探测状态输入，提取两类特征：
    1. probs: 原始概率分布 (受熵影响大)
    2. logits: 中心化对数概率 (反映纯战术意图，受熵影响小)
    """
    with torch.no_grad():
        states_tensor = states_tensor.to(device)
        output = actor_net(states_tensor)
        
        raw_probs_list = []
        centered_logits_list = []
        
        # 处理 Categorical 动作
        if 'cat' in output:
            for probs in output['cat']:
                if probs.dim() == 1:
                    probs = probs.unsqueeze(1)
                raw_probs_list.append(probs)
                
                # 转换为中心化 Logits: log(p) - mean(log(p))
                log_p = torch.log(probs + 1e-10)
                centered_logits = log_p - log_p.mean(dim=-1, keepdim=True)
                centered_logits_list.append(centered_logits)
                
        # 处理 Bernoulli 动作
        if 'bern' in output:
            bern_logits = output['bern']
            if bern_logits.dim() == 1:
                bern_logits = bern_logits.unsqueeze(1)
            
            # 转为概率 [p, 1-p] 以对齐多离散维度特征
            p = torch.sigmoid(bern_logits)
            raw_probs_list.append(torch.cat([p, 1-p], dim=-1))
            
            # 同样中心化 Bernoulli 的 Logits
            # sigmoid 对应的 logits 就是 bern_logits, 但我们需要 [logit_p, logit_(1-p)]
            # 简化处理：直接用 logit_p
            centered_logits_list.append(bern_logits - bern_logits.mean(dim=-1, keepdim=True))
        
        # 拼接特征
        feat_probs = torch.cat(raw_probs_list, dim=-1).flatten().cpu().numpy()
        feat_logits = torch.cat(centered_logits_list, dim=-1).flatten().cpu().numpy()
        
        return feat_probs, feat_logits

def main(TARGET_LOG_DIR):
    log_dir = TARGET_LOG_DIR
    if len(sys.argv) > 1:
        log_dir = sys.argv[1]

    if log_dir is None or "请填入" in log_dir:
        print("错误: 请手动修改代码中的 TARGET_LOG_DIR，或通过命令行参数传入你想测试的日志目录。")
        return
        
    # 如果给的是相对名字，就自动拼接到 logs/combat 下
    if not os.path.isabs(log_dir):
        project_root = os.path.dirname(os.path.dirname(cur_dir))
        log_dir = os.path.join(project_root, "logs", "combat", log_dir)
        
    if not os.path.exists(log_dir):
        print(f"错误: 目录 {log_dir} 不存在。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")

    # ==================================
    # 1. 构建环境和网络尺寸变量
    # ==================================
    class DummyArgs:
        max_episode_len = 600
        R_cage = 69e3 # 45
    dummy_env = ChooseStrategyEnv(DummyArgs())
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    hidden_dim = [128, 128, 128] # 具体隐藏层应当对齐你训练脚本配置
    del dummy_env
    
    # ==================================
    # 2. 读取 Elite_Elo_ratings 和最新策略
    # ==================================
    elo_path = os.path.join(log_dir, "elo_ratings.json") # elite_elo_ratings
    if not os.path.exists(elo_path):
        print(f"日志目录下未能找到 file: {elo_path}")
        return
    
    with open(elo_path, 'r', encoding='utf-8') as f:
        elite_dict = json.load(f)
    
    # 筛选只含神经网络的模型
    agent_keys = [k for k in elite_dict.keys() if k.startswith("actor_rein")]
    if not agent_keys:
        print("Elite池中目前为空或没有 actor_rein 关键字。")
        return

    def extract_num(k):
        try:
            return int(re.search(r'actor_rein(\d+)', k).group(1))
        except:
            return -1
        
    all_models = [f for f in os.listdir(log_dir) if f.startswith("actor_rein") and f.endswith(".pt")]
    if not all_models:
        print("指定目录下找不到 .pt 模型。")
        return
        
    latest_model_name = max(all_models, key=extract_num).replace(".pt", "")
    print(f"读取到的最强/最新基准 (Learner): {latest_model_name}")

    if latest_model_name not in agent_keys:
        agent_keys.append(latest_model_name)
        
    # 按检查点发布序号升序来排（代表不同世代）
    agent_keys.sort(key=extract_num)

    # ==================================
    # 3. 制备探测状态（快照池）
    # ==================================
    print("正在加载 IL 参考状态作为探测数据集 (Probe Set)...")
    il_data, _ = load_il_and_transitions(
        os.path.join(cur_dir, "IL"),
        "il_transitions_combat_LR.pkl",
        "transition_dict_combat_LR.pkl"
    )
    all_states = np.array(il_data['states'], dtype=np.float32)
    
    # 我们打500个均匀散布的状态切片做动作指纹，这对算力压力极小，区分度足够好
    sample_size = min(500, len(all_states))
    idx = np.random.RandomState(42).choice(len(all_states), sample_size, replace=False)
    probe_states = torch.tensor(all_states[idx], dtype=torch.float32)
    print(f"快照制备完成: 取样 {sample_size} 个状态。")

    # ==================================
    # 4. 收集策略指纹 
    # ==================================
    net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    feature_dict = {}
    
    t_start = time.time()
    print("开始遍历精英池智能体并提取输出指纹特征...")
    
    for key in agent_keys:
        model_path = os.path.join(log_dir, f"{key}.pt")
        if not os.path.exists(model_path):
            continue
            
        try:
            state_dict = torch.load(model_path, map_location=device)
            # 因为有时保存的参数带着 Wrapper ("net.xx")
            if any(k.startswith('net.') for k in state_dict.keys()):
                new_state_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith('net.')}
                net.load_state_dict(new_state_dict, strict=False)
            else:
                net.load_state_dict(state_dict, strict=False)
                
            net.eval()
            vec_prob, vec_logit = extract_features(net, probe_states, device)
            feature_dict[key] = {'prob': vec_prob, 'logit': vec_logit}
        except Exception as e:
            print(f"解析并提取 {key} 的特征失败: {e}")

    if latest_model_name not in feature_dict:
        print("提取基准最新策略失败！计算终止。")
        return
        
    learner_prob = feature_dict[latest_model_name]['prob']
    learner_logit = feature_dict[latest_model_name]['logit']
    
    # ==================================
    # 5. 余弦相似度计算与数据制备 
    # ==================================
    sim_scores_prob = []
    sim_scores_logit = []
    elo_scores = []
    plot_labels = []
    
    for key in agent_keys:
        if key in feature_dict:
            # 概率空间相似度
            sim_p = get_cosine_similarity(learner_prob, feature_dict[key]['prob'])
            sim_scores_prob.append(sim_p)
            
            # Logit 空间相似度
            sim_l = get_cosine_similarity(learner_logit, feature_dict[key]['logit'])
            sim_scores_logit.append(sim_l)
            
            # 从 elite_dict 提取 Elo
            elo = elite_dict.get(key, 1200)
            elo_scores.append(elo)
            plot_labels.append(extract_num(key))
            
    t_end = time.time()
    print(f"★ 策略相似度和 Elo 数据准备完成！总计算时间: {t_end - t_start:.2f} 秒。")

    # ==================================
    # 6. 计算 PFSP_challenge 匹配概率 (用于 Figure 2)
    # ==================================
    sigma = 400  # 对其并行脚本常用的默认值
    elos_np = np.array(elo_scores, dtype=np.float64)
    target_elo = elo_scores[-1] # 以最新 Learner 的 Elo 为基准
    
    # 遵循 hierarch 脚本的逻辑: actual_target = min(max(elos), target_elo + 300)
    actual_target = min(np.max(elos_np), float(target_elo) + 300)
    diffs = elos_np - actual_target
    raw_scores = np.exp(-0.5 * (diffs / sigma)**2)
    match_probs = (raw_scores / (raw_scores.sum() + 1e-12)) * 100 # 转化为百分比
    
    # ==================================
    # 7. 绘图 
    # ==================================
    if plot_labels:
        color_sim = 'tab:blue'
        
        # --- Figure 1: Elo vs Prob Similarity ---
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        color_elo = 'tab:red'
        ax1.set_xlabel("Agent Index")
        ax1.set_ylabel("Elo", color=color_elo, fontweight='bold')
        ax1.plot(plot_labels, elo_scores, marker='s', linestyle='--', color=color_elo, alpha=0.5)
        ax1.tick_params(axis='y', labelcolor=color_elo)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel("Prob Similarity", color=color_sim, fontweight='bold')
        ax2.plot(plot_labels, sim_scores_prob, marker='o', color=color_sim)
        ax2.tick_params(axis='y', labelcolor=color_sim)
        plt.title("Fig 1: Elo & Behavioral Similarity (Prob Space)")
        fig1.tight_layout()

        # --- Figure 2: Match Prob vs Prob Similarity ---
        fig2, ax1_f2 = plt.subplots(figsize=(10, 5))
        color_prob = 'tab:green'
        ax1_f2.set_ylabel("Match Prob (%)", color=color_prob, fontweight='bold')
        ax1_f2.plot(plot_labels, match_probs, marker='s', color=color_prob, alpha=0.7)
        ax1_f2.fill_between(plot_labels, match_probs, color=color_prob, alpha=0.1)
        
        prob_min, prob_max = np.min(match_probs), np.max(match_probs)
        padding = (prob_max - prob_min) * 0.1 if prob_max > prob_min else 0.1
        ax1_f2.set_ylim(prob_min - padding, prob_max + padding)
        
        ax2_f2 = ax1_f2.twinx()
        ax2_f2.set_ylabel("Prob Similarity", color=color_sim, fontweight='bold')
        ax2_f2.plot(plot_labels, sim_scores_prob, marker='o', color=color_sim)
        plt.title("Fig 2: Match Prob & behavioral Similarity")
        fig2.tight_layout()

        # --- Figure 3: Logit Space Similarity (Strategic Intent) ---
        fig3, ax1_f3 = plt.subplots(figsize=(10, 5))
        ax1_f3.set_xlabel("Agent Index")
        ax1_f3.set_ylabel("Intent Similarity (Logit Space)", color='tab:purple', fontweight='bold')
        ax1_f3.plot(plot_labels, sim_scores_logit, marker='D', linestyle='-', color='tab:purple', label="Logit Similarity")
        
        # 叠加一个 Prob Similarity 做对比
        ax1_f3.plot(plot_labels, sim_scores_prob, marker='.', linestyle=':', color='gray', alpha=0.5, label="Prob Similarity (for Ref)")
        
        ax1_f3.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        ax1_f3.set_ylim(min(min(sim_scores_logit), min(sim_scores_prob)) - 0.05, 1.05)
        ax1_f3.grid(True, alpha=0.3)
        ax1_f3.legend()
        plt.title("Fig 3: Strategic Intent Similarity (Logit Space vs Prob Space)")
        fig3.tight_layout()

        plt.show()

if __name__ == "__main__":
    TARGET_LOG_DIR = r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\IL_and_PFSP_分阶段_混规则对手_挑战_并行_分层_A3C-run-20260329-174051"
    main(TARGET_LOG_DIR)
