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
import matplotlib.pyplot as plt

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

from BasicRules_new_hierarchical import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper # 纯MLP

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def create_initial_state():
    """创建固定的初始状态"""
    blue_height, red_height = 8000, 8000
    red_psi, blue_psi = -pi / 2 + random.uniform(-pi/4, pi/4), \
        pi / 2 + random.uniform(-pi/4, pi/4)
    red_N, red_E = random.uniform(-20,20)*1e3, 45e3
    blue_N, blue_E = random.uniform(-20,20)*1e3, -45e3
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]), 'psi': red_psi}
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]), 'psi': blue_psi}
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

# --- 3. 主程序 ---
if __name__ == "__main__":

    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = "IL_and_Mixed经典PFSP_挑战_并行_分层_训练满熵项-run-20260515-104131"
   

    # 次要
    experiment_name = "IL_and_Pure经典PFSP_挑战_并行_分层_训练满熵项-run-20260516-170432"

    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    parser.add_argument("--mission-name", type=str, default=experiment_name, help="Mission name to find the log directory.")
    args = parser.parse_args()    

    red_agent_id = None # 700
    blue_agent_id = None # 200
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=55.00e3) # 55e3
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    # 构建场地边界
    vertices = None # 默认圆形边界
    # 南北长54km，东西宽100km的长方形边界
    # vertices = [[29.9e3, 50e3], [-29.9e3, 50e3], [-29.9e3, -50e3], [29.9e3, -50e3]]
    env = ChooseStrategyEnv(env_args, tacview_show=1, vertices=vertices)
    env.dt_move = 0.025
    
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
    
    red_agent_path = find_latest_agent_path(latest_log_dir, red_agent_id)
    blue_agent_path = find_latest_agent_path(latest_log_dir, blue_agent_id)
    if not red_agent_path or not blue_agent_path:
        raise FileNotFoundError(f"Found missing agent. Red:{red_agent_path}, Blue:{blue_agent_path}")

    print()
    print(f"Found log directory: {latest_log_dir}")
    print(f"Loading Red Agent (ID: {red_agent_id}) from: {red_agent_path}")
    print(f"Loading Blue Agent (ID: {blue_agent_id}) from: {blue_agent_path}")
    print()

    # 实例化红方
    actor_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict), action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(red_agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() 

    # 实例化蓝方
    enm_actor_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict), action_dims_dict, None, device).to(device)
    enm_actor_wrapper.load_state_dict(torch.load(blue_agent_path, map_location=device, weights_only=1), strict=False)
    enm_actor_wrapper.eval()

    # --- [修正] 移除重复的 env 初始化，直接配置已有的 env ---
    # env = ChooseStrategyEnv(env_args, tacview_show=1) 
    # env.tacview_show = 1
    # if env.tacview_show:
    #     env.tacview = Tacview()
    #     env.tacview.handshake()
    #     env.visualize_cage()

    env.shielded = 1
    env.no_out = 0 # 强制防止出界，训练的时候为0，测试的时候为1
    
    # --- 循环测试 ---
    t_bias = 0

    try:
        for i in range(5):
            print("\n" + "="*50)
            print(f"--- Starting Test: Self Play Test {i+1} ---")
            print("="*50)

            # 重置环境
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = create_initial_state()
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE, ego_side='r')

            done = False
            last_r_action_label = 0
            last_b_action_label = 0
            r_action_label = 0
            b_action_label = 0

            # --- 初始化数据记录 ---
            history = {
                'time': [],
                'r_ny': [], 'r_alpha': [], 'r_alt': [], 'r_mach': [],
                'b_ny': [], 'b_alpha': [], 'b_alt': [], 'b_mach': [],
            }

            # 回合仿真循环
            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)

                # 决策
                if count % action_cycle_multiplier == 0:
                    # --- 红方 (RL 智能体) ---
                    with torch.no_grad():
                        r_action_exec, _, _, r_action_check = actor_wrapper.get_action(
                            r_obs, explore={'cont':0, 'cat':0, 'bern':1}, check_obs=None, bern_threshold=0.4
                            ) # check_obs=r_check_obs, check_obs=None
                        
                    r_action_label = r_action_exec['cat'] # [0]
                    r_fire = r_action_exec['bern'][0]
                    last_r_action_label = r_action_label
                    print(f"红方(RL) 开火概率: {r_action_check['bern'][0]:.4f}")

                    if r_fire:
                        env.RUAV.about_to_fire = 1

                    # --- 蓝方 (RL 智能体 200) ---
                    with torch.no_grad():
                        b_action_exec, _, _, b_action_check = enm_actor_wrapper.get_action(
                            b_obs, explore={'cont':0, 'cat':0, 'bern':1}, check_obs=None, bern_threshold=0.4
                        )
                        
                    b_action_label = b_action_exec['cat'] # [0]
                    b_fire = b_action_exec['bern'][0]
                    last_b_action_label = b_action_label
                    if b_fire:
                        env.BUAV.about_to_fire = 1

                # 执行机动并步进
                r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                
                # 测试时限制开火后爬升
                if getattr(env.RUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'r', tabu=0, action_label=r_action_label) # r_action_label)
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', tabu=0, action_label=b_action_label) # b_action_label)
                    
                env.step(r_maneuver, b_maneuver)
                # 统计红方的奖励与状态
                done, _, _, _ = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)

                # --- 记录数据 ---
                history['time'].append(count * action_cycle_multiplier * dt_maneuver)
                history['r_ny'].append(env.RUAV.Ny)
                history['r_alpha'].append(env.RUAV.alpha_air * 180 / np.pi)
                history['r_alt'].append(env.RUAV.alt)
                history['r_mach'].append(env.RUAV.mach)
                history['b_ny'].append(env.BUAV.Ny)
                history['b_alpha'].append(env.BUAV.alpha_air * 180 / np.pi)
                history['b_alt'].append(env.BUAV.alt)
                history['b_mach'].append(env.BUAV.mach)

                env.render(t_bias=t_bias)

            # 报告结果
            result = "Draw"
            if env.win: result = "Win"
            elif env.lose: result = "Lose"
            print(f"\n--- Test Finished. Result for Red (Loaded Agent): {result} ---")
            
            env.clear_render(t_bias=t_bias)
            t_bias += env.t
            
            # input("Press Enter to continue to the next test...")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

