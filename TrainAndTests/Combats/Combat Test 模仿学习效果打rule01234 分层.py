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

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules_new_hierarchical import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper # 纯MLP

# --- 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 30
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
    red_E = 45e3
    blue_N = init_North
    blue_E = -45e3

    red_birth_state = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    blue_birth_state = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return red_birth_state, blue_birth_state

# --- 3. 主程序 ---
if __name__ == "__main__":

    experiment_name = '只模仿学习'

    parser = argparse.ArgumentParser("RL/IL Combat Test - Evaluation")
    parser.add_argument("--agent-id", type=int, default=0, help="Specific agent ID to test (0 for actor_rein0).")
    parser.add_argument("--mission-name", type=str, default=experiment_name, help="Mission name to find the log directory.")
    parser.add_argument("--num-matches", type=int, default=40, help="Number of matches per rule.")
    args = parser.parse_args()    

    args.agent_id = 0 # 强制加载模仿学习完毕后的第一个参数 (actor_rein0.pt)
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=12*60, R_cage=45e3) # 训练时默认是 45e3
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    env = ChooseStrategyEnv(env_args, tacview_show=False) # 取消可视化
    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    
    latest_log_dir = get_latest_log_dir(logs_root_dir, args.mission_name)
    
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{args.mission_name}' in '{logs_root_dir}'")
    
    agent_path = find_latest_agent_path(latest_log_dir, args.agent_id)
    if not agent_path:
        raise FileNotFoundError(f"No agent file found in '{latest_log_dir}' (ID: {args.agent_id})")

    print(f"\nFound log directory: {latest_log_dir}")
    print(f"Loading agent weights from: {agent_path}\n")

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() # 评估模式

    env.tacview_show = 0
    env.shielded = 1
    env.no_out = 1 # 防止出界，测试专用
    
    # --- 循环测试 ---
    rule_opponents = [0, 1, 2, 3, 4]
    
    results_summary = {}
    
    print(f"=== Starting Evaluation ({args.num_matches} matches per rule) ===")
    
    try:
        for rule_num in rule_opponents:
            print(f"\n--- Testing vs Rule_{rule_num} ---")
            
            match_scores = []
            
            for match_idx in range(args.num_matches):
                # 重置环境（加入随机化）
                rb, bb = create_initial_state_worker(randomized=1)
                # 可选配环境大小随机化，测试时也可以加上
                env.R_cage = np.random.uniform(30e3, 45e3) 
                env.reset(red_birth_state=rb, blue_birth_state=bb, ego_side='r')

                done = False
                last_r_action_label = 0
                last_b_action_label = 0
                r_action_label = 0
                b_action_label = 0
                r_fire = False

                # 回合仿真循环
                total_steps = round(env_args.max_episode_len / dt_maneuver)
                for count in range(total_steps):
                    if not env.running or done:
                        break

                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)

                    # 决策
                    if count % action_cycle_multiplier == 0:
                        # --- 红方 (RL 智能体) ---
                        with torch.no_grad():
                            r_action_exec, _, _, r_action_check = actor_wrapper.get_action(
                                r_obs, explore={'cont':0, 'cat':0, 'bern':1}, check_obs=r_check_obs, bern_threshold=0.38
                                )
                            
                        r_action_label = r_action_exec['cat'][0]
                        r_fire = r_action_exec['bern'][0]
                        last_r_action_label = r_action_label

                        if r_fire:
                            launch_missile_immediately(env, 'r', tabu=1)

                        # --- 蓝方 (规则智能体) ---
                        b_state_check = env.unscale_state(b_check_obs)
                        b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
                        last_b_action_label = b_action_label
                        if b_fire:
                            launch_missile_immediately(env, 'b', tabu=1)

                    # 执行机动并步进
                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    
                    # 统计结束条件
                    done, _, _, _ = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)

                # 单盘统计
                score = 0.5
                res_str = "Draw"
                if env.win:
                    score = 1.0
                    res_str = "Win"
                elif env.lose:
                    score = 0.0
                    res_str = "Lose"
                
                match_scores.append(score)
                print(f"Match {match_idx+1}/{args.num_matches} -> {res_str}")

            avg_win_rate = np.mean(match_scores)
            results_summary[f"Rule_{rule_num}"] = avg_win_rate
            print(f"==> Rule_{rule_num} Total Win Rate: {avg_win_rate * 100:.2f}%")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    
    print("\nAll tests completed. Generating CSV report...")
    
    # --- 写入 CSV ---
    csv_filename = os.path.join(latest_log_dir, f"IL_Evaluation_Results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Opponent", "Win_Rate"])
        for k, v in results_summary.items():
            writer.writerow([k, f"{v:.4f}"])
            
    print(f"Report saved to: {csv_filename}")
