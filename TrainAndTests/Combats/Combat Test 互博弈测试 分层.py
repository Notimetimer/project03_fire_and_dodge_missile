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

# # --- 1. 项目路径和模块导入 ---

# project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(project_root)

from _context import *

# from BasicRules_new_hierarchical import basic_rules  # 不再使用规则智能体
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid as PPOPolicyNet, HybridActorWrapper as PPOActorWrapper # 纯MLP
from Algorithms.SACHybrid import PolicyNetHybrid as SACPolicyNet, HybridActorWrapper as SACActorWrapper
from 绘制回放曲线 import plot_replay

# --- [修正] 在此处直接定义缺失的常量 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2
# -----------------------------------------

# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def _is_off_policy(mission_name):
    """根据 mission 名称判断是否应使用 SAC 系列（SAC/TD3/DDPG）的 PolicyNet/ActorWrapper"""
    upper = (mission_name or '').upper()
    return any(tag in upper for tag in ('SAC', 'TD3', 'DDPG'))

def build_wrapper(state_dim, hidden_dim, action_dims_dict, device, mission_name, log_dir=None):
    """根据 mission_name 选择 PPO 或 SAC 的网络结构与 ActorWrapper"""
    if _is_off_policy(mission_name):
        # SAC 的 mask 配置会改变 fc_cat 输出维度(ver:5->13, hor:6/7->11)，
        # 必须从 checkpoint 目录的 actor.meta.json 推断，否则加载时维度不匹配
        net = SACPolicyNet(state_dim, hidden_dim, action_dims_dict, mask_search_dir=log_dir)
        return SACActorWrapper(net, action_dims_dict, None, device).to(device)
    else:
        net = PPOPolicyNet(state_dim, hidden_dim, action_dims_dict)
        return PPOActorWrapper(net, action_dims_dict, None, device).to(device)

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
    # "IL-SLA-PPO",
    # "IL-SE-SAC",
    # "IL-SL-PPO",
    # "IL-PPO",
    # "PPO"
    show_name = [
        "IL-SLA-PPO",
        "PPO",
    ]
    
    # 红方和蓝方分别使用不同的模型目录
    red_dir_name = "PPO0.3_flymask_v0h0_fireSL-run-20260930-125149"
    blue_dir_name = "NoIL_flymask_v0h0-run-20260924-224736"
    
    """
    PPO0.3_flymask_v0h0_fireSL-run-20260924-145554
    PPO0.3_flymask_v0h0_fireSL-run-20260921-194654
    PPO0.3_flymask_v0h0_fireSL-run-20260930-125149
    
    SAC0.3_flymask_v1h1-run-20260923-213238
    SAC0.3_flymask_v1h1-run-20260928-093645
    SAC0.3_flymask_v1h1-run-20260929-194715
    
    PPO0.3_flymask_v0h0-run-20260921-194617	
    PPO0.3_flymask_v0h0-run-20260928-111836
    PPO0.3_flymask_v0h0-run-20261001-152601
    
    切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260921-122428
    切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260928-233900
    切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260930-093137
    
    NoIL_flymask_v0h0-run-20260909-131801
    NoIL_flymask_v0h0-run-20260924-224736
    NoIL_flymask_v0h0-run-20260928-233924
    """

    parser = argparse.ArgumentParser("RL/IL Combat Test")
    parser.add_argument("--agent-id", type=int, default=None, help="Specific agent ID to test. If None, loads the latest.")
    args = parser.parse_args()    

    red_agent_id = None # 700
    blue_agent_id = None # 200
    
    # --- 环境和模型参数 (必须与训练时一致) ---
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3) # 55e3
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 初始化环境 ---
    # 构建场地边界
    vertices = None # 默认圆形边界
    # 南北长54km，东西宽100km的长方形边界
    # vertices = [[29.9e3, 50e3], [-29.9e3, 50e3], [-29.9e3, -50e3], [29.9e3, -50e3]]
    env = ChooseStrategyEnv(env_args, tacview_show=1, vertices=vertices)  # 0, 1
    env.dt_move = 0.050 # 0.05 # 0.04 # 25

    
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    # --- 查找并加载模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")

    red_log_dir = os.path.join(logs_root_dir, red_dir_name)
    blue_log_dir = os.path.join(logs_root_dir, blue_dir_name)

    if not os.path.exists(red_log_dir):
        raise FileNotFoundError(f"Red log directory not found: {red_log_dir}")
    if not os.path.exists(blue_log_dir):
        raise FileNotFoundError(f"Blue log directory not found: {blue_log_dir}")

    red_agent_path = find_latest_agent_path(red_log_dir, red_agent_id)
    blue_agent_path = find_latest_agent_path(blue_log_dir, blue_agent_id)
    if not red_agent_path or not blue_agent_path:
        raise FileNotFoundError(f"Found missing agent. Red:{red_agent_path}, Blue:{blue_agent_path}")

    print()
    print(f"Red log directory: {red_log_dir}")
    print(f"Blue log directory: {blue_log_dir}")
    print(f"Loading Red Agent (ID: {red_agent_id}) from: {red_agent_path}")
    print(f"Loading Blue Agent (ID: {blue_agent_id}) from: {blue_agent_path}")
    print()

    # 实例化红方（根据目录名判别 PPO / SAC）
    actor_wrapper = build_wrapper(state_dim, hidden_dim, action_dims_dict, device, red_dir_name, red_log_dir)
    actor_wrapper.load_state_dict(torch.load(red_agent_path, map_location=device, weights_only=1), strict=False)
    actor_wrapper.eval() 

    # 实例化蓝方（根据目录名判别 PPO / SAC）
    enm_actor_wrapper = build_wrapper(state_dim, hidden_dim, action_dims_dict, device, blue_dir_name, blue_log_dir)
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
        for i in range(1):
            print("\n" + "="*50)
            print(f"--- Starting Test: Self Play Test {i+1} ---")
            print("="*50)

            # 重置环境
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = None, None # create_initial_state()
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE, ego_side='r', 
                      red_init_ammo=6, blue_init_ammo=6)

            done = False
            last_r_action_label = 0
            last_b_action_label = 0
            r_action_label = 0
            b_action_label = 0

            # --- 初始化回放数据结构（用于绘制3D轨迹） ---
            replay_data = {
                'meta': {
                    'red_name': show_name[0],
                    'blue_name': show_name[1],
                    'result': '',
                },
                't': [],
                'RUAV': {'pos_': []},
                'BUAV': {'pos_': []},
                'RMIS': {},
                'BMIS': {},
            }

            fire_time = -120

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
                            r_obs, explore={'cont':0, 'cat':1, 'bern':1}, check_obs=r_check_obs, bern_threshold=0.4,
                            temperature={'cat':0.3, 'bern':1}
                            ) # check_obs=r_check_obs, check_obs=None 0.06
                    # print("中制导状态", r_obs[3])
                    r_action_label = r_action_exec['cat'] # [0]
                    r_fire = r_action_exec['bern'][0]
                    last_r_action_label = r_action_label
                    print(f"红方(RL) 开火概率: {r_action_check['bern'][0]:.4f}")

                    if r_fire:
                        env.RUAV.about_to_fire = 1
                        
                        print("开火瞬间状态观测", r_check_obs)
                        print("开火瞬间动作", r_action_label)

                    # --- 蓝方 (RL 智能体) ---
                    with torch.no_grad():
                        b_action_exec, _, _, b_action_check = enm_actor_wrapper.get_action(
                            b_obs, explore={'cont':0, 'cat':1, 'bern':1}, check_obs=b_check_obs, bern_threshold=0.4,
                            temperature={'cat':0.3, 'bern':1}
                        )
                    b_action_label = b_action_exec['cat']
                    b_fire = b_action_exec['bern'][0]
                    last_b_action_label = b_action_label
                    if b_fire:
                        env.BUAV.about_to_fire = 1

                # 执行机动并步进
                r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                
                # 测试时限制开火后爬升
                if getattr(env.RUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'r', tabu=1, action_label=None) # r_action_label)
                    print("Shoot")
                    print()
                    fire_time = env.t
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', tabu=1, action_label=None) # b_action_label)
                

                if (action_cycle_multiplier-1) * env.dt_maneuver <= env.t-fire_time < 2 * action_cycle_multiplier * env.dt_maneuver:
                    print("开火后瞬间观测", r_check_obs)
                    print("开火后动作", r_action_label)
                    print()

                env.step(r_maneuver, b_maneuver)
                # 统计红方的奖励与状态
                done, b_r1, b_r2, b_r3 = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)

                # if abs(env.t % 5) < 0.1:
                    # print("当前动作", r_action_exec)
                    # print("当前奖励函数", b_r1)
                    # print()

                # --- 记录回放数据（3D轨迹用） ---
                replay_data['t'].append(env.t)
                replay_data['RUAV']['pos_'].append(env.RUAV.pos_.tolist() + [env.t])
                replay_data['BUAV']['pos_'].append(env.BUAV.pos_.tolist() + [env.t])
                for m in env.alive_r_missiles:
                    key = str(m.id)
                    if key not in replay_data['RMIS']:
                        replay_data['RMIS'][key] = []
                    replay_data['RMIS'][key].append(m.pos_.tolist() + [env.t])
                for m in env.alive_b_missiles:
                    key = str(m.id)
                    if key not in replay_data['BMIS']:
                        replay_data['BMIS'][key] = []
                    replay_data['BMIS'][key].append(m.pos_.tolist() + [env.t])

                env.render(t_bias=t_bias)

            # 报告结果
            result = "Draw"
            if env.win: result = "Win"
            elif env.lose: result = "Lose"
            print(f"\n--- Test Finished. Result for Red (Loaded Agent): {result} ---")
            replay_data['meta']['result'] = result
            # 直接用战机的 got_hit 属性判断哪一方被命中（含双杀时双方都命中）
            replay_data['meta']['red_dead'] = bool(getattr(env.RUAV, 'got_hit', False))
            replay_data['meta']['blue_dead'] = bool(getattr(env.BUAV, 'got_hit', False))

            env.clear_render(t_bias=t_bias)
            t_bias += env.t

            # --- 绘制本回合 3D 回放轨迹 ---
            plot_replay(replay_data, save_path=None)

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

