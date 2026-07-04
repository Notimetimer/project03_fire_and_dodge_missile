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

from BasicRules_new_hierarchical import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper # 纯MLP

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
    dir_name = "SLWSPFSP0.3-run-20260618-221044"
   

    # 次要
    experiment_name = None    
    'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项'


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
    env = ChooseStrategyEnv(env_args, tacview_show=1, vertices=vertices)
    env.dt_move = 0.02 # 0.05 # 0.04 # 25

    
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

    env.shielded = 1
    env.no_out = 0 # 强制防止出界，训练的时候为0，测试的时候为1
    
    # --- 回放记录目录 ---
    logs_dir = os.path.dirname(logs_root_dir)  # logs/ 目录（logs_root_dir 的上一级）
    record_dir = os.path.join(logs_dir, "CombatRecord")
    os.makedirs(record_dir, exist_ok=True)
    agent_tag = os.path.basename(latest_log_dir)  # 用实验目录名作为文件名前缀

    # --- 循环测试 ---
    rule_opponents = [3,3,3] # [0,1,2]
    t_bias = 0

    try:
        for rule_num in rule_opponents:
            print("\n" + "="*50)
            print(f"--- Starting Test: Loaded Actor(Red) vs Rule_{rule_num}(Blue) ---")
            print("="*50)

            # 重置环境
            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = None, None # create_initial_state()
            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE, ego_side='r', 
                      red_init_ammo=4, blue_init_ammo=4)

            done = False
            last_r_action_label = 0
            last_b_action_label = 0
            r_action_label = 0
            b_action_label = 0

            # --- 初始化回放数据结构 ---
            replay_data = {
                'meta': {
                    'agent': agent_tag,
                    'rule_num': rule_num,
                    'dt': dt_maneuver,
                },
                't': [],
                'RUAV': {'pos_': []},
                'BUAV': {'pos_': []},
                'RMIS': {},
                'BMIS': {},
            }

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
                            r_obs, explore={'cont':0, 'cat':0, 'bern':1}, check_obs=r_check_obs, bern_threshold=0.06,
                            temperature={'cat':0.3, 'bern':1}
                            ) # check_obs=r_check_obs, check_obs=None
                        
                    r_action_label = r_action_exec['cat'] # [0]
                    r_fire = r_action_exec['bern'][0]
                    last_r_action_label = r_action_label
                    print(f"红方(RL) 开火概率: {r_action_check['bern'][0]:.4f}")

                    if r_fire:
                        env.RUAV.about_to_fire = 1

                    # --- 蓝方 (规则智能体) ---
                    b_state_check = env.unscale_state(b_check_obs)
                    b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
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
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', tabu=1, action_label=None) # b_action_label)
                    
                env.step(r_maneuver, b_maneuver)
                # 统计红方的奖励与状态
                done, b_r1, b_r2, b_r3 = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)

                # if abs(env.t % 5) < 0.1:
                    # print("当前动作", r_action_exec)
                    # print("当前奖励函数", b_r1)
                    # print()

                # --- 记录回放数据 ---
                # 独立时间 t 存储，同时每个位置点末尾追加当前时刻用于对照，格式 [x, y, z, t]
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

            # --- 导弹击毁判定与标注 ---
            # 位置点格式为 [x, y, z, t]，t 在最后；独立 t 列表也保留
            r_last_t = replay_data['RUAV']['pos_'][-1][3] if replay_data['RUAV']['pos_'] else None
            b_last_t = replay_data['BUAV']['pos_'][-1][3] if replay_data['BUAV']['pos_'] else None

            def mark_missile_hit(mis_dict, target_last_t, target_last_pos):
                for key, traj in mis_dict.items():
                    if not traj:
                        continue
                    mis_last = traj[-1]
                    mis_last_t = mis_last[3]
                    mis_last_pos = np.array(mis_last[:3])
                    # 导弹航迹与飞机航迹在同一时刻结束，且距离小于80m，则标记为击毁
                    if mis_last_t == target_last_t and target_last_pos is not None:
                        dist = np.linalg.norm(mis_last_pos - target_last_pos)
                        if dist < 80.0:
                            mis_last.append('*')  # 在末端添加星号标记

            r_last_pos = np.array(replay_data['RUAV']['pos_'][-1][:3]) if replay_data['RUAV']['pos_'] else None
            b_last_pos = np.array(replay_data['BUAV']['pos_'][-1][:3]) if replay_data['BUAV']['pos_'] else None
            mark_missile_hit(replay_data['BMIS'], r_last_t, r_last_pos)  # 蓝导弹打击红飞机
            mark_missile_hit(replay_data['RMIS'], b_last_t, b_last_pos)  # 红导弹打击蓝飞机

            # --- 保存回放 JSON ---
            replay_data['meta']['result'] = result
            json_name = f"{agent_tag}_vs_Rule{rule_num}_{result}.json"
            json_path = os.path.join(record_dir, json_name)
            try:
                import json as _json
                with open(json_path, 'w', encoding='utf-8') as _f:
                    _json.dump(replay_data, _f, ensure_ascii=False)
                print(f"回放已保存: {json_path}")
            except Exception as _e:
                print(f"保存回放失败: {_e}")

            # # --- 保存作战记录到 CSV ---
            # try:
            #     df_history = pd.DataFrame(history)
            #     save_name = f"CombatLog_vs_Rule{rule_num}.csv" #_{datetime.datetime.now().strftime('%H%M%S')}.csv"
            #     save_path = os.path.join(project_root, "logs", save_name)
            #     df_history.to_csv(save_path, index=False)
            #     print(f"Combat data for Rule {rule_num} saved to: {save_path}")
            # except Exception as e:
            #     print(f"Failed to save CSV: {e}")

            # --- 绘制曲线 ---
            plt.figure(figsize=(10, 10))
            plt.subplot(4, 1, 1)
            plt.plot(history['time'], history['r_ny'], label='Red Ny', color='crimson')
            plt.plot(history['time'], history['b_ny'], label='Blue Ny', color='royalblue', linestyle='--')
            plt.axhline(y=-3, color='black', linestyle=':', alpha=0.7, label='Ny limit (-3g)')
            plt.axhline(y=9, color='black', linestyle=':', alpha=0.7, label='Ny limit (9g)')
            plt.ylabel('Ny (g)')
            plt.title(f'Test vs Rule {rule_num}: Metrics')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 2)
            plt.plot(history['time'], history['r_alpha'], label='Red Alpha', color='crimson')
            plt.plot(history['time'], history['b_alpha'], label='Blue Alpha', color='royalblue', linestyle='--')
            plt.axhline(y=-8, color='black', linestyle=':', alpha=0.7, label='Alpha limit (-8°)')
            plt.axhline(y=26, color='black', linestyle=':', alpha=0.7, label='Alpha limit (26°)')
            plt.ylabel('Alpha (deg)')
            plt.title('Angle of Attack (Alpha)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 3)
            plt.plot(history['time'], history['r_mach'], label='Red Mach', color='crimson')
            plt.plot(history['time'], history['b_mach'], label='Blue Mach', color='royalblue', linestyle='--')
            plt.ylabel('Mach')
            plt.title('Flight Mach Number')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(4, 1, 4)
            plt.plot(history['time'], history['r_alt'], label='Red Alt', color='crimson')
            plt.plot(history['time'], history['b_alt'], label='Blue Alt', color='royalblue', linestyle='--')
            plt.ylabel('Alt (m)')
            plt.xlabel('Time (s)')
            plt.title('Altitude (Height)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            input("Press Enter to continue to the next test...")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

