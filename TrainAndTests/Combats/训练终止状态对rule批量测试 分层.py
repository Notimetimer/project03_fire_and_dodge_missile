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
import csv
import torch.multiprocessing as mp  # 使用 torch 的多进程模块

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules_new_hierarchical import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper # 纯MLP
from VsBaseline_while_training_hierarch import test_worker

# --- 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def sub_of_radian(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def create_initial_state_worker(randomized=1):
    blue_height = 9000
    red_height = 9000
    # 初始航向随机化
    red_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-np.pi/3, np.pi/3) * randomized)
    blue_psi = sub_of_radian(np.pi/2 + np.random.uniform(-np.pi/3, np.pi/3) * randomized)
    init_North = np.random.uniform(-30e3, 30e3) * randomized
    red_N = init_North
    red_E = 55e3 # 45e3
    blue_N = init_North
    blue_E = -45e3

    red_birth_state = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    blue_birth_state = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return red_birth_state, blue_birth_state

# --- 3. 主程序 ---
if __name__ == "__main__":

    # experiment_name = "只模仿学习" # "只模仿学习"
    experiment_name = "只模仿学习-run-20260413-100737"
    
    "只模仿学习-run-20260413-100737"
    "无学习-run-20260413-100719"

    parser = argparse.ArgumentParser("RL/IL Combat Test - Evaluation")
    parser.add_argument("--agent-id", type=int, default=0, help="Specific agent ID to test (0 for actor_rein0).")
    parser.add_argument("--mission-name", type=str, default=experiment_name, help="Mission name to find the log directory.")
    parser.add_argument("--num-matches", type=int, default=50, help="Number of matches per rule.")
    args = parser.parse_args()    

    args.agent_id = 0 # 强制加载模仿学习完毕后的第一个参数 (actor_rein0.pt)
    
    # # --- 环境和模型参数 (必须与训练时一致) ---
    # env_args = argparse.Namespace(max_episode_len=15*60, R_cage=63.0e3) # 训练时默认是 45e3
    args.max_episode_len = 12*60
    args.R_cage=45e3

    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    env = ChooseStrategyEnv(args, tacview_show=False) # 取消可视化
    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    
    latest_log_dir = os.path.join(logs_root_dir, args.mission_name)

    # latest_log_dir = get_latest_log_dir(logs_root_dir, args.mission_name)
    
    # if not latest_log_dir:
    #     raise FileNotFoundError(f"No log directory found for mission '{args.mission_name}' in '{logs_root_dir}'")
    
    import json
    # 如果目录里面没有elo_ratings.json，按当前的方式，取actor_rein后接最大数字的来测试，否则取actor_rein开头且elo值最大的来测试。
    elo_path = os.path.join(latest_log_dir, 'elo_ratings.json')
    if os.path.exists(elo_path):
        with open(elo_path, 'r', encoding='utf-8') as f:
            elo_ratings = json.load(f)
        # 寻找具有最高 elo 分数且以 'actor_rein' 开头的名称
        best_agent = None
        best_elo = -float('inf')
        for k, v in elo_ratings.items():
            if k.startswith('actor_rein') and v > best_elo:
                best_elo = v
                best_agent = k
        if best_agent is None:
            raise FileNotFoundError(f"elo_ratings.json found, but no 'actor_rein' keys inside.")
        agent_path = os.path.join(latest_log_dir, f"{best_agent}.pt")
        print(f"Loading agent '{best_agent}' based on highest Elo score: {best_elo}")
    else:
        # 取actor_rein后接最大数字的
        agent_files = glob.glob(os.path.join(latest_log_dir, 'actor_rein*.pt'))
        if not agent_files:
            raise FileNotFoundError(f"No agent file found in '{latest_log_dir}'")
        max_idx = -1
        agent_path = None
        for file in agent_files:
            match = re.search(r'actor_rein(\d+)\.pt', file)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
                    agent_path = file
        if not agent_path:
            agent_path = find_latest_agent_path(latest_log_dir, args.agent_id)
        print(f"Loading latest parameter index from fallback logic.")

    print(f"\nFound log directory: {latest_log_dir}")
    print(f"Loading agent weights from: {agent_path}\n")

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() # 评估模式

    env.tacview_show = 0
    env.shielded = 1
    env.no_out = 0 # 防止出界，测试专用
    
    # --- 循环测试 ---
    rule_opponents = [0, 1, 2, 3, 4]
    
    results_summary = {}
    
    print(f"=== Starting Evaluation ({args.num_matches} matches per rule) ===")
    
    try:
        # 进程通信设置
        mp.set_start_method('spawn', force=True)
        # --- A. 启动并行测试进程池 (Async Test Pool) ---
        # 一次最多同时并行processes个工作进程，每个工作进程处理完maxtasksperchild个任务后，会重新开启新的工作进程以阻止内存泄漏
        # processes 收费站窗口数，maxtasksperchild 窗口处理完多少个任务后换班
        test_pool = mp.Pool(processes=10, maxtasksperchild=10) 

        # 1. 深度拷贝当前 Actor 权重到 CPU 内存 (注意：需要使用 wrapper 的 state_dict 以包含 net. 前缀)
        current_weights = {k: v.cpu().clone() for k, v in actor_wrapper.state_dict().items()}

        # 2. 分发测试任务并【立即阻塞等待】
        # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
        num_runs = args.num_matches
        test_tasks = []
        # 发送 5 * num_runs 个任务到进程池，但是强制要求每个任务内部只跑 1 场
        for r_idx in [0, 1, 2, 3, 4]*num_runs:
            obj = test_pool.apply_async(
                test_worker, 
                args=(current_weights, r_idx, args, 
                        state_dim, hidden_dim, action_dims_dict, 
                        dt_maneuver, 'cpu', 1, action_cycle_multiplier) # <--- 这里改成 1
            )
            test_tasks.append(obj)
        # 等待所有测试进程结束
        test_results = [t.get() for t in test_tasks]
        
        # 显式关闭进程池，防止主进程结束时出现 AttributeError
        test_pool.close()
        test_pool.join()

        rule_score_sum = {rule_num: 0 for rule_num in rule_opponents}
        rule_wins = {rule_num: 0 for rule_num in rule_opponents}
        rule_loses = {rule_num: 0 for rule_num in rule_opponents}
        rule_draws = {rule_num: 0 for rule_num in rule_opponents}

        for i in test_results:
            rule_num, score, result2, wins, loses, draws = i
            rule_score_sum[rule_num] += score
            rule_wins[rule_num] += wins
            rule_loses[rule_num] += loses
            rule_draws[rule_num] += draws
        
        # 计算平均分并填充到 summary 中
        rule_score_mean = {rule_num: score/num_runs for rule_num, score in rule_score_sum.items()}
        results_summary = rule_score_mean
        
        # 同时记录胜平负比例
        results_detail = {
            r: (rule_wins[r]/num_runs, rule_loses[r]/num_runs, rule_draws[r]/num_runs)
            for r in rule_opponents
        }

        print("test_results_mean_score", rule_score_mean)

        for r_num, score in rule_score_mean.items():
            w, l, d = results_detail[r_num]
            print(f"  [Test Result] Rule_{r_num}: Score={score:.3f} | Win={w:.3f}, Lose={l:.3f}, Draw={d:.3f}")


    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        if 'test_pool' in locals():
            test_pool.terminate()
            test_pool.join()
    
    print("\nAll tests completed. Generating CSV report...")
    
    # --- 写入 CSV ---
    csv_filename = os.path.join(latest_log_dir, f"IL_Evaluation_Results.csv") # _{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Opponent", "Score", "Win_Ratio", "Lose_Ratio", "Draw_Ratio"])
        for k, v in results_summary.items():
            w, l, d = results_detail[k]
            writer.writerow([k, f"{v:.4f}", f"{w:.4f}", f"{l:.4f}", f"{d:.4f}"])

    print(f"Report saved to: {csv_filename}")
