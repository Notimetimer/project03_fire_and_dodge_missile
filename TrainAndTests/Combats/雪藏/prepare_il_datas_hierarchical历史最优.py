# d:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\TrainAndTests\Combats\prepare_il_datas_selfplay.py

import os
import sys
import numpy as np
import torch
import argparse
import pickle
from math import pi

# 导入环境和算法组件
from _context import *
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Algorithms.Utils import compute_monte_carlo_returns

def run_self_play_collection(num_episodes=100, gamma=0.995):
    # --- 1. 配置路径与模型 ---
    log_dir = r'd:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\IL_and_MixedPFSP_分阶段_挑战_并行_分层-run-20260326-172341'
    # 根据 elo_ratings.json，actor_rein751 分数最高 (1368.78)
    agent_path = os.path.join(log_dir, 'actor_rein751.pt')
    
    if not os.path.exists(agent_path):
        print(f"错误: 找不到模型文件 {agent_path}")
        return

    # --- 2. 初始化环境与模型 ---
    env_args = argparse.Namespace(max_episode_len=10*60, R_cage=45e3)
    hidden_dim = [128, 128, 128]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = ChooseStrategyEnv(env_args, tacview_show=False) # 采集时一般关闭可视化以加速
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
    
    # 加载 Actor
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    actor_wrapper.load_state_dict(torch.load(agent_path, map_location=device, weights_only=True), strict=False)
    actor_wrapper.eval()

    # --- 3. 采集循环 ---
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    action_cycle_multiplier = 30
    dt_maneuver = 0.2
    
    print(f"开始自博弈数据采集: 共 {num_episodes} 回合")
    
    try:
        for i_eps in range(num_episodes):
            # 随机化出生状态
            blue_psi = np.random.uniform(-pi, pi)
            red_psi = sub_of_radian(blue_psi + pi + np.random.uniform(-pi/4, pi/4))
            dist = np.random.uniform(40e3, 70e3)
            
            blue_pos = np.array([0, 9000, -dist/2])
            red_pos = np.array([0, 9000, dist/2])
            
            b_birth = {'position': blue_pos, 'psi': blue_psi}
            r_birth = {'position': red_pos, 'psi': red_psi}
            
            # 重置环境，双方均加载同个模型
            env.reset(red_birth_state=r_birth, blue_birth_state=b_birth)
            
            done = False
            last_b_obs, last_b_action = None, None
            steps = 0

            while env.running and not done:
                # 获取观测
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                
                if steps % action_cycle_multiplier == 0:
                    # 记录上一步到 transition_dict (仅记录蓝方作为演示数据)
                    if steps > 0:
                        transition_dict['states'].append(last_b_obs)
                        transition_dict['actions'].append(last_b_action)
                        transition_dict['rewards'].append(b_reward)
                        transition_dict['next_states'].append(b_obs)
                        transition_dict['dones'].append(False)

                    with torch.no_grad():
                        # 蓝方决策
                        b_act_exec, _, _, _ = actor_wrapper.get_action(b_obs, explore=True, check_obs=b_check_obs)
                        # 红方决策 (同样使用这个最强 Agent)
                        r_act_exec, _, _, _ = actor_wrapper.get_action(r_obs, explore=True, check_obs=r_check_obs)
                    
                    b_action_label = b_act_exec['cat'][0]
                    b_fire = b_act_exec['bern'][0]
                    r_action_label = r_act_exec['cat'][0]
                    r_fire = r_act_exec['bern'][0]
                    
                    if b_fire: launch_missile_immediately(env, 'b', tabu=1)
                    if r_fire: launch_missile_immediately(env, 'r', tabu=1)
                    
                    last_b_obs = b_obs
                    last_b_action = {'cat': b_act_exec['cat'], 'bern': b_act_exec['bern']}

                # 物理步进
                r_man = env.maneuver14LR(env.RUAV, r_action_label)
                b_man = env.maneuver14LR(env.BUAV, b_action_label)
                env.step(r_man, b_man)
                
                done, b_rew_event, b_rew_con, b_rew_sha = env.combat_terminate_and_reward('b', b_action_label, b_fire)
                b_reward = b_rew_event + b_rew_con + b_rew_sha
                steps += 1

            # 存储回合最后一步
            if last_b_obs is not None:
                transition_dict['states'].append(last_b_obs)
                transition_dict['actions'].append(last_b_action)
                transition_dict['rewards'].append(b_reward)
                transition_dict['next_states'].append(b_obs)
                transition_dict['dones'].append(True)
            
            if (i_eps + 1) % 10 == 0:
                print(f"已完成 {i_eps + 1}/{num_episodes} 回合")

        # --- 4. 计算回报并保存 ---
        il_transition_dict = {
            'states': transition_dict['states'],
            'actions': transition_dict['actions'],
            'returns': compute_monte_carlo_returns(gamma, transition_dict['rewards'], transition_dict['dones'])
        }
        
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IL_SelfPlay")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "il_transitions_top_agent_selfplay.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(il_transition_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(f"\n采集完成！数据已保存至: {save_path}")
        print(f"样本总数: {len(il_transition_dict['states'])}")

    except KeyboardInterrupt:
        print("\n中途停止采集")
    finally:
        env.close()

if __name__ == '__main__':
    run_self_play_collection(num_episodes=20) # 这里可以调大回合数
