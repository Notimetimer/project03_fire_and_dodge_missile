import os
import sys
import numpy as np
import torch
import argparse
import glob
import re
from math import pi
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import product

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules_new_hierarchical import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper # 纯MLP
from VsBaseline_while_training_hierarch import test_worker

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2  # 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# def create_initial_state():
#     """创建固定的初始状态"""
#     blue_height, red_height = 8000, 8000
#     red_psi, blue_psi = -pi / 2, pi / 2
#     red_N, red_E = 0, 55e3  # 55e3
#     blue_N, blue_E = red_N, -red_E # -45e3
#     DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
#     DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
#     return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- 3. 主程序 ---
if __name__ == "__main__":

    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = None
    # dir_name = "IL_and_Mixed经典PFSP_挑战_并行_分层_rule3_无mask-run-20260531-233800"
   

    # 次要
    experiment_name = 'IL_and_Mixed经典PFSP_挑战_并行_分层_rule3_无mask'
    
    'IL_and_Mixed经典PFSP_挑战_并行_分层_rule3_0.3'
    
    'IL_and_Mixed经典PFSP_挑战_并行_分层_训练满熵项'
    'IL_and_Pure经典PFSP_挑战_并行_分层_训练满熵项'
    'IL_and_Mixed经典PFSP_挑战_并行_分层_训练满熵项方边界'
    'NoILPFSP_分阶段_混规则对手_挑战_并行_训练满熵项'
    'NoILand_PurePFSP_分阶段_混规则对手_挑战_并行_训练满熵项'
    'NoILPFSP_分阶段_混规则对手_密集奖励函数调试'

    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    parser.add_argument("--mission-name", type=str, default=experiment_name, help="Mission name to find the log directory.")
    args = parser.parse_args()    

    args.agent_id = None # 40
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3) # 55e3
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    # 构建场地边界
    vertices = None # 默认圆形边界
    # 南北长54km，东西宽100km的长方形边界
    # vertices = [[29.9e3, 50e3], [-29.9e3, 50e3], [-29.9e3, -50e3], [29.9e3, -50e3]]
    env = ChooseStrategyEnv(env_args, tacview_show=0, vertices=vertices)
    env.dt_move = 0.04 # 25
    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    

    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, args.mission_name)
    
    # 如果要硬编码为本地绝对路径，使用原始字符串并检查存在性
    # hardcoded = r'D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\RL_combat_PFSP-run-20251215-175820'
    # if os.path.exists(hardcoded):
    #     latest_log_dir = hardcoded
    
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{args.mission_name}' in '{logs_root_dir}'")
    
    agent_path = find_latest_agent_path(latest_log_dir, args.agent_id)
    if not agent_path:
        raise FileNotFoundError(f"No agent file found in '{latest_log_dir}' (ID: {args.agent_id or 'latest'})")

    print()
    print(f"Found log directory: {latest_log_dir}")
    print(f"Loading agent weights from: {agent_path}")
    print()

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    # 注意：测试时只需要 Actor Wrapper，不需要完整的 PPO agent
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() # **非常重要**：设置为评估模式


    # if env.tacview_show:
    #     env.tacview = Tacview()
    #     env.tacview.handshake()
    #     env.visualize_cage()

    # --- 温度参数网格搜索 ---
    rule_opponents = [1, 2, 3]
    num_runs = 5  # 每组参数重复对局次数

    temp_bern_values = np.round(np.arange(0.1, 2.0, 0.3), 6).tolist()  # [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
    temp_cat_values  = np.round(np.arange(0.2, 1.1, 0.2), 6).tolist()  # [0.2, 0.4, 0.6, 0.8, 1.0]

    grid_params = list(product(temp_bern_values, temp_cat_values))
    total_tasks = len(grid_params) * len(rule_opponents)
    print(f"\n温度参数网格搜索: {len(temp_bern_values)} x {len(temp_cat_values)} = {len(grid_params)} 组参数")
    print(f"对手序列: {rule_opponents}，每组总任务数: {total_tasks}")
    print(f"temp_bern: {temp_bern_values}")
    print(f"temp_cat:  {temp_cat_values}")

    current_weights = {k: v.cpu().clone() for k, v in actor_wrapper.state_dict().items()}

    # 结果矩阵: shape (len_bern, len_cat)
    win_rate_grid   = np.zeros((len(temp_bern_values), len(temp_cat_values)))
    return_grid     = np.zeros((len(temp_bern_values), len(temp_cat_values)))

    ctx = mp.get_context('spawn')
    num_workers = min(mp.cpu_count(), total_tasks)
    print(f"启动进程池，worker 数: {num_workers}")

    with ctx.Pool(processes=num_workers) as pool:
        # 为每组 (temp_bern, temp_cat) 并行提交所有 rule 任务
        for i_b, t_bern in enumerate(temp_bern_values):
            for i_c, t_cat in enumerate(temp_cat_values):
                temperature = {'cat': t_cat, 'bern': t_bern}
                tasks = []
                for r_idx in rule_opponents:
                    obj = pool.apply_async(
                        test_worker,
                        kwds={
                            'model_state_dict': current_weights,
                            'rule_num': r_idx,
                            'env_args': env_args,
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dims_dict': action_dims_dict,
                            'dt_maneuver_val': dt_maneuver,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'no_out': 0,
                            'deterministic': True,
                            'restrict_fire': True,
                            'vertices': vertices,
                            'Temperature': temperature,
                        }
                    )
                    tasks.append(obj)

                results = [t.get() for t in tasks]
                # result 结构: (rule_num, avg_score, avg_return, wins, loses, draws)
                # test_worker 内部已按 num_runs 对同对手不同轮次做了平均
                # 这里再对不同对手之间做第二次平均
                rule_scores = {}
                for rule_num, score, result2, wins, loses, draws in results:
                    rule_scores[rule_num] = {'winrate': score, 'return': result2}

                avg_win = np.mean([v['winrate'] for v in rule_scores.values()])
                avg_ret = np.mean([v['return']  for v in rule_scores.values()])
                win_rate_grid[i_b, i_c] = avg_win
                return_grid[i_b, i_c]   = avg_ret

                print(f"  t_bern={t_bern:.1f}  t_cat={t_cat:.1f}  "
                      f"avg_winrate={avg_win:.3f}  avg_return={avg_ret:.3f}")
                for r_num in sorted(rule_scores.keys()):
                    print(f"      vs Rule_{r_num}: winrate={rule_scores[r_num]['winrate']:.3f}  return={rule_scores[r_num]['return']:.3f}")

    print("\n所有网格搜索任务完成，开始绘图...")

    # --- 绘制热度图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bern_labels = [f"{v:.1f}" for v in temp_bern_values]
    cat_labels  = [f"{v:.1f}" for v in temp_cat_values]

    for ax, data, title, cmap in zip(
        axes,
        [win_rate_grid, return_grid],
        ["平均胜率 (Win Rate)", "平均回报 (Episode Return)"],
        ["RdYlGn", "RdYlBu"]
    ):
        im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower')
        ax.set_xticks(range(len(temp_cat_values)))
        ax.set_xticklabels(cat_labels)
        ax.set_yticks(range(len(temp_bern_values)))
        ax.set_yticklabels(bern_labels)
        ax.set_xlabel("temperature_cat")
        ax.set_ylabel("temperature_bern")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        # 在每个格子上标注数值
        for iy in range(data.shape[0]):
            for ix in range(data.shape[1]):
                ax.text(ix, iy, f"{data[iy, ix]:.3f}",
                        ha='center', va='center', fontsize=7,
                        color='black')

    fig.suptitle(f"温度系数网格搜索  vs Rule{rule_opponents}  (num_runs={num_runs}/rule)", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(project_root, "logs",
        f"temp_grid_search_{time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(save_path, dpi=150)
    print(f"热度图已保存至: {save_path}")
    plt.show()
    print("\nAll tests completed.")

