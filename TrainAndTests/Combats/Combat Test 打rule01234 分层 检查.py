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
# from BasicRules_new_hierarchical2 import basic_rules
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 1218-104003
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Algorithms.PPOHybrid23_0 import HybridActorWrapper, infer_mask_cfg_from_actor_meta # 纯MLP
from 绘制回放曲线 import plot_replay

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

    gamma = 0.97

    from Algorithms.PPOHybrid23_0 import PolicyNetHybrid
    # from Algorithms.SACHybrid import PolicyNetHybrid
    
    tacview_show=1
    
    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = None
    dir_name = "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654"
    # "PPO0.3_flymask_v0h0-run-20260921-194617"
    # "SLWSPFSP0.3-run-20260807-212711"
    # "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654"
    
    
    


    
    # 次要
    experiment_name = None    
    # 'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项'


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
    env = ChooseStrategyEnv(env_args, tacview_show=tacview_show, vertices=vertices)
    env.dt_move = 0.05 # 025 # 2 # 0.05 # 0.04 # 25

    
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

    # [新增] 根据 checkpoint 所在目录的 actor.meta.json 自动推断 ver/hor 配置
    mask_cfg = None
    actor_meta_path = os.path.join(os.path.dirname(agent_path), "actor.meta.json")
    if os.path.exists(actor_meta_path):
        mask_cfg = infer_mask_cfg_from_actor_meta(actor_meta_path)

    # 实例化模型结构并加载权重
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict, mask_cfg=mask_cfg).to(device)
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
    
    # --- 循环测试 ---
    rule_opponents = [3] # [0,1,2,3,4] # [3]

    t_bias = 0

    try:
        for rule_num in rule_opponents:
            print("\n" + "="*50)
            print(f"--- Starting Test: Loaded Actor(Red) vs Rule_{rule_num}(Blue) ---")
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

            # --- 初始化数据记录 ---
            history = {
                'time': [],
                'r_ny': [], 'r_alpha': [], 'r_alt': [], 'r_mach': [],
                'b_ny': [], 'b_alpha': [], 'b_alt': [], 'b_mach': [],
                'r_ata': [], 'b_ata': [],                # 天线转动角 (度)
                'r_cat_entropy': [],
                'r_cat_conf': [], 'r_bern_fire_prob': [],
                'r_warning': [], 'r_missile_in_mid_term': [],
                'r_reward': [],
                'r_r_shaping': [],
                'r_count_12km': [], 'r_count_4km': [],
                'b_count_12km': [], 'b_count_4km': [],
                'distance': [],                 # 红蓝双方三维距离
                'r_fire_time': [], 'r_fire_dist': [],   # 红方开火时刻及对应距离
                'r_fire_alt': [], 'r_fire_mach': [],    # 红方开火时的高度/马赫数
                'b_fire_time': [], 'b_fire_dist': [],   # 蓝方开火时刻及对应距离
                'b_fire_alt': [], 'b_fire_mach': [],    # 蓝方开火时的高度/马赫数
            }

            # --- 初始化回放数据结构（用于绘制3D轨迹） ---
            replay_data = {
                'meta': {
                    'red_name': 'IL-SLA-PPO',
                    'blue_name': f'基准对手{rule_num + 1}',
                    'result': '',
                },
                't': [],
                'RUAV': {'pos_': []},
                'BUAV': {'pos_': []},
                'RMIS': {},
                'BMIS': {},
            }

            fire_time = -120
            r_cat_entropy = 0.0
            r_bern_fire_prob = 0.0
            r_cat_conf = 0.0
            r_warning = 0.0
            r_missile_in_mid_term = 0.0

            # 回合仿真循环
            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                r_state_check = env.unscale_state(r_check_obs)
                b_state_check = env.unscale_state(b_check_obs)
                r_warning = float(r_state_check['warning'])
                r_missile_in_mid_term = float(r_state_check['missile_in_mid_term'])

                # 决策
                if count % action_cycle_multiplier == 0:
                    # --- 红方 (RL 智能体) ---
                    with torch.no_grad():
                        r_action_exec, _, _, r_action_check = actor_wrapper.get_action(
                            r_obs, explore={'cont':0, 'cat':1, 'bern':0}, check_obs=r_check_obs, bern_threshold=0.82,
                            temperature={'cat':0.3, 'bern':1}
                            ) # check_obs=r_check_obs, check_obs=None 0.06
                    # print("中制导状态", r_obs[3])
                    r_action_label = r_action_exec['cat'] # [0]
                    r_fire = r_action_exec['bern'][0]
                    last_r_action_label = r_action_label

                    # 计算并记录 cat 熵与 bern 熵
                    eps = 1e-8
                    r_cat_entropy = 0.0
                    cat_top1_list = []
                    if 'cat' in r_action_check and r_action_check['cat'] is not None:
                        for cat_probs in r_action_check['cat']:
                            p = np.asarray(cat_probs).flatten()
                            p = np.clip(p, eps, 1.0)
                            r_cat_entropy += -np.sum(p * np.log(p))
                            cat_top1_list.append(np.max(p))
                    # cat 多头的 top-1 置信度（几何平均）
                    if len(cat_top1_list) > 0:
                        r_cat_conf = np.exp(np.mean(np.log(np.clip(cat_top1_list, eps, 1.0))))
                        print(f"红方(RL) cat 综合置信度: {r_cat_conf:.4f}")

                    r_bern_fire_prob = 0.0
                    if 'bern' in r_action_check and r_action_check['bern'] is not None:
                        p = np.asarray(r_action_check['bern']).flatten()
                        r_bern_fire_prob = float(p[0])
                        print(f"红方(RL) 开火概率: {r_bern_fire_prob:.4f}")

                    if r_fire:
                        env.RUAV.about_to_fire = 1
                        
                        print("开火瞬间状态观测", r_check_obs)
                        print("开火瞬间动作", r_action_label)

                    # --- 蓝方 (规则智能体) ---
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
                    fire_time = env.t
                    # 记录红方开火时刻及对应双方距离、高度、马赫数
                    _r_pos = np.array(env.RUAV.pos_)
                    _b_pos = np.array(env.BUAV.pos_)
                    history['r_fire_time'].append(env.t)
                    history['r_fire_dist'].append(np.linalg.norm(_r_pos - _b_pos))
                    history['r_fire_alt'].append(env.RUAV.alt)
                    history['r_fire_mach'].append(env.RUAV.mach)
                if getattr(env.BUAV, 'about_to_fire', 0):
                    # 蓝方弹药耗尽时，即使规则要求开火也不计入开火事件
                    _b_has_ammo = env.BUAV.ammo > 0
                    launch_missile_immediately(env, 'b', tabu=1, action_label=None) # b_action_label)
                    if _b_has_ammo and (not env.BUAV.dead) and not (env.RUAV.dead):
                        # 记录蓝方开火时刻及对应双方距离、高度、马赫数
                        _r_pos = np.array(env.RUAV.pos_)
                        _b_pos = np.array(env.BUAV.pos_)
                        history['b_fire_time'].append(env.t)
                        history['b_fire_dist'].append(np.linalg.norm(_r_pos - _b_pos))
                        history['b_fire_alt'].append(env.BUAV.alt)
                        history['b_fire_mach'].append(env.BUAV.mach)
                

                if (action_cycle_multiplier-1) * env.dt_maneuver <= env.t-fire_time < 2 * action_cycle_multiplier * env.dt_maneuver:
                    print("开火后瞬间观测", r_check_obs)
                    print("开火后动作", r_action_label)
                    print()

                env.step(r_maneuver, b_maneuver)
                # 统计红方的奖励与状态
                done, b_r1, b_r2, b_r3 = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)
                # [诊断] 补一次蓝方调用（返回值丢弃）：更新蓝方导弹 get_in_12km/get_in_4km 锁存与 BUAV._count 计数，
                # 与训练时双侧每步都调用保持一致；放在 'r' 调用之后，不影响已记录的红方奖励。
                env.combat_terminate_and_reward('b', b_action_label, b_fire, action_cycle_multiplier)

                # if abs(env.t % 5) < 0.1:
                    # print("当前动作", r_action_exec)
                    # print("当前奖励函数", b_r1)
                    # print()

                # --- 记录数据 ---
                history['time'].append(count * dt_maneuver)
                history['r_ny'].append(env.RUAV.Ny)
                history['r_alpha'].append(env.RUAV.alpha_air * 180 / np.pi)
                history['r_alt'].append(env.RUAV.alt)
                history['r_mach'].append(env.RUAV.mach)
                history['b_ny'].append(env.BUAV.Ny)
                history['b_alpha'].append(env.BUAV.alpha_air * 180 / np.pi)
                history['b_alt'].append(env.BUAV.alt)
                history['b_mach'].append(env.BUAV.mach)
                # 天线转动角 ATA（弧度 -> 度），直接调用 get_state 取，避免被 POMDP 干扰
                r_full_state = env.get_state('r')
                b_full_state = env.get_state('b')
                r_ti = r_full_state.get('target_information')
                b_ti = b_full_state.get('target_information')
                history['r_ata'].append(float(r_ti[4]) * 180 / np.pi if r_ti is not None else 0.0)
                history['b_ata'].append(float(b_ti[4]) * 180 / np.pi if b_ti is not None else 0.0)
                history['r_cat_entropy'].append(r_cat_entropy)
                history['r_bern_fire_prob'].append(r_bern_fire_prob)
                history['r_cat_conf'].append(r_cat_conf)
                history['r_warning'].append(r_warning)
                history['r_missile_in_mid_term'].append(r_missile_in_mid_term)
                history['r_reward'].append(b_r1)
                history['r_r_shaping'].append(env.RUAV.r_shaping)
                history['r_count_12km'].append(getattr(env.RUAV, '_count_12km', 0))
                history['r_count_4km'].append(getattr(env.RUAV, '_count_4km', 0))
                history['b_count_12km'].append(getattr(env.BUAV, '_count_12km', 0))
                history['b_count_4km'].append(getattr(env.BUAV, '_count_4km', 0))
                # 记录红蓝双方三维距离
                _r_pos = np.array(env.RUAV.pos_)
                _b_pos = np.array(env.BUAV.pos_)
                history['distance'].append(np.linalg.norm(_r_pos - _b_pos))

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
            
            # --- 保存作战记录到 CSV ---
            try:
                df_history = pd.DataFrame(history)
                save_name = f"CombatLog_vs_Rule_2609_{rule_num}.csv" #_{datetime.datetime.now().strftime('%H%M%S')}.csv"
                save_path = os.path.join(project_root, "logs", save_name)
                df_history.to_csv(save_path, index=False)
                print(f"Combat data for Rule {rule_num} saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")

            # --- 绘制曲线：三个独立 figure，一律中文 ---
            # 中文字体设置
            plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
            # 统一字号：轴标签、刻度、图例均用同一字号，避免图例被 matplotlib 默认放大
            _fs = 8
            plt.rcParams['font.size'] = _fs
            plt.rcParams['axes.labelsize'] = _fs
            plt.rcParams['xtick.labelsize'] = _fs
            plt.rcParams['ytick.labelsize'] = _fs
            plt.rcParams['legend.fontsize'] = _fs

            t = np.array(history['time'])

            # --- Figure 1: 红蓝双方高度变化曲线（均为实线），各自曲线上标开火点 ---
            plt.figure(1, figsize=(10/2, 4/2))
            plt.clf()
            plt.plot(t, [a / 1000 for a in history['r_alt']], label='红方高度', color='crimson', linestyle='-', linewidth=1.2)
            plt.plot(t, [a / 1000 for a in history['b_alt']], label='蓝方高度', color='royalblue', linestyle='-', linewidth=1.2)
            # 红方开火时刻画红色竖直虚线
            for i, ft in enumerate(history['r_fire_time']):
                plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='红方开火' if i == 0 else '')
            # 蓝方开火时刻画蓝色竖直虚线
            for i, ft in enumerate(history['b_fire_time']):
                plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='蓝方开火' if i == 0 else '')
            plt.xlabel('时间/(s)')
            plt.ylabel('高度/(km)')
            leg1 = plt.legend()
            leg1.set_draggable(True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- Figure 2: 红蓝双方马赫数曲线，各自曲线上标开火点 ---
            plt.figure(2, figsize=(10/2, 4/2))
            plt.clf()
            plt.plot(t, history['r_mach'], label='红方马赫数', color='crimson', linestyle='-', linewidth=1.2)
            plt.plot(t, history['b_mach'], label='蓝方马赫数', color='royalblue', linestyle='-', linewidth=1.2)
            # 红方开火时刻画红色竖直虚线
            for i, ft in enumerate(history['r_fire_time']):
                plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='红方开火' if i == 0 else '')
            # 蓝方开火时刻画蓝色竖直虚线
            for i, ft in enumerate(history['b_fire_time']):
                plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='蓝方开火' if i == 0 else '')
            plt.xlabel('时间/(s)')
            plt.ylabel('马赫数')
            leg2 = plt.legend()
            leg2.set_draggable(True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- Figure 3: 双方距离曲线（黑色），红/蓝开火时刻打红点/蓝点 ---
            plt.figure(3, figsize=(10/2, 4/2))
            plt.clf()
            plt.plot(t, [d / 10000 for d in history['distance']], label='双方距离', color='black', linestyle='-', linewidth=1.2)
            # 红方开火时刻画红色竖直虚线
            for i, ft in enumerate(history['r_fire_time']):
                plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='红方开火' if i == 0 else '')
            # 蓝方开火时刻画蓝色竖直虚线
            for i, ft in enumerate(history['b_fire_time']):
                plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='蓝方开火' if i == 0 else '')
            plt.xlabel('时间/(s)')
            plt.ylabel('双方距离/(10km)')
            # plt.title(f'红蓝双方距离曲线与开火时刻（vs 基准对手{rule_num + 1}）')
            leg3 = plt.legend()
            leg3.set_draggable(True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # --- Figure 4: 红蓝双方天线转动角 ATA 曲线（度） ---
            plt.figure(4, figsize=(10/2, 4/2))
            plt.clf()
            plt.plot(t, history['r_ata'], label='红方ATA', color='crimson', linestyle='-', linewidth=1.2)
            plt.plot(t, history['b_ata'], label='蓝方ATA', color='royalblue', linestyle='-', linewidth=1.2)
            # 红方开火时刻画红色竖直虚线
            for i, ft in enumerate(history['r_fire_time']):
                plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='红方开火' if i == 0 else '')
            # 蓝方开火时刻画蓝色竖直虚线
            for i, ft in enumerate(history['b_fire_time']):
                plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                            label='蓝方开火' if i == 0 else '')
            plt.xlabel('时间/(s)')
            plt.ylabel('天线转动角/(°)')
            leg4 = plt.legend()
            leg4.set_draggable(True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            

            # --- 绘制本回合 3D 回放轨迹 ---
            plot_replay(replay_data, save_path=None)
            
            
            plt.show()

            # input("Press Enter to continue to the next test...")

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()
        print("\nAll tests completed.")

