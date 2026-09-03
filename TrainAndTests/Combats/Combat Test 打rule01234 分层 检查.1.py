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
# from BasicRules_new_hierarchical2 import basic_rules
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

    dir_name = "SLWSPFSP0.3_flymask_1-run-20260903-173826"
        
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
    env.dt_move = 0.025 # 2 # 0.05 # 0.04 # 25

    
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

    # --- 4. 在内存中跑真实对抗，收集策略自己的轨迹 ---
    rule_opponents = [1,2,3]
    t_bias = 0
    episodes_data = []  # 记录每个 episode 的完整时间序列数据
    all_r_obs = []
    all_warning = []
    all_missile = []
    all_cat_probs = []
    all_bern_probs = []
    all_r_action_labels = []
    all_r_fire = []

    try:
        for rule_num in rule_opponents:
            print("\n" + "="*50)
            print(f"--- Starting Test: Loaded Actor(Red) vs Rule_{rule_num}(Blue) ---")
            print("="*50)

            env.reset(red_birth_state=None, blue_birth_state=None, ego_side='r',
                      red_init_ammo=4, blue_init_ammo=4)

            done = False
            last_b_action_label = 0
            r_action_label = (0, 0)
            b_action_label = (0, 0)
            fire_time = -120

            ep_times = []
            ep_cat_probs = []
            ep_bern_probs = []
            ep_r_action_labels = []
            ep_r_fire = []
            ep_r_obs = []
            ep_warning = []
            ep_missile = []

            for count in range(round(env_args.max_episode_len / dt_maneuver)):
                if not env.running or done:
                    break

                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                r_state_check = env.unscale_state(r_check_obs)
                r_warning = float(r_state_check['warning'])
                r_missile = float(r_state_check['missile_in_mid_term'])

                if count % action_cycle_multiplier == 0:
                    with torch.no_grad():
                        r_action_exec, _, _, r_action_check = actor_wrapper.get_action(
                            r_obs, explore={'cont':0, 'cat':1, 'bern':1}, check_obs=r_check_obs, bern_threshold=0.072,
                            temperature={'cat':0.999, 'bern':0.97}
                        )
                    r_action_label = r_action_exec['cat']
                    r_fire = r_action_exec['bern'][0]

                    if r_fire:
                        env.RUAV.about_to_fire = 1

                    b_state_check = env.unscale_state(b_check_obs)
                    b_action_label, b_fire = basic_rules(b_state_check, rule_num, last_action=last_b_action_label)
                    last_b_action_label = b_action_label
                    if b_fire:
                        env.BUAV.about_to_fire = 1

                    cat_probs = r_action_check.get('cat', [])
                    cat_probs_arr = [np.asarray(p) for p in cat_probs]
                    bern_prob_arr = np.asarray(r_action_check.get('bern', [0])).flatten()

                    ep_times.append(env.t)
                    ep_cat_probs.append(cat_probs_arr)
                    ep_bern_probs.append(bern_prob_arr)
                    ep_r_action_labels.append(r_action_label)
                    ep_r_fire.append(r_fire)
                    ep_r_obs.append(r_obs.copy())
                    ep_warning.append(r_warning)
                    ep_missile.append(r_missile)

                    all_cat_probs.append(cat_probs_arr)
                    all_bern_probs.append(bern_prob_arr)
                    all_r_action_labels.append(r_action_label)
                    all_r_fire.append(r_fire)
                    all_r_obs.append(r_obs.copy())
                    all_warning.append(r_warning)
                    all_missile.append(r_missile)

                r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)

                if getattr(env.RUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'r', tabu=1, action_label=None)
                    print("Shoot")
                    fire_time = env.t
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', tabu=0, action_label=None)

                if (action_cycle_multiplier-1) * env.dt_maneuver <= env.t-fire_time < 2 * action_cycle_multiplier * env.dt_maneuver:
                    print("开火后瞬间观测", r_check_obs)
                    print("开火后动作", r_action_label)

                env.step(r_maneuver, b_maneuver)
                done, b_r1, b_r2, b_r3 = env.combat_terminate_and_reward('r', r_action_label, r_fire, action_cycle_multiplier)

                env.render(t_bias=t_bias)

            result = "Draw"
            if env.win: result = "Win"
            elif env.lose: result = "Lose"
            print(f"\n--- Test Finished. Result for Red: {result} ---")
            env.clear_render(t_bias=t_bias)
            t_bias += env.t

            episodes_data.append({
                'rule_num': rule_num,
                'result': result,
                'times': np.array(ep_times),
                'cat_probs': ep_cat_probs,
                'bern_probs': np.array(ep_bern_probs),
                'action_labels': np.array(ep_r_action_labels),
                'fires': np.array(ep_r_fire),
                'warning': np.array(ep_warning),
                'missile': np.array(ep_missile),
            })

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    finally:
        env.end_render()

    # --- 5. 绘制与时间相关的动作分布归一化热度图 ---
    if len(episodes_data) > 0:
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            try:
                from Envs.Tasks.ChooseStrategyEnv2_0_hierarchical import action_optionsLR
            except Exception:
                action_optionsLR = {}

            # 分层动作头 Y 轴刻度标签精确映射定义
            HEAD_ACTION_LABELS = {
                # Head 0: 垂直方向 (5动作: 0到4依次为最大角度爬升到急俯冲)
                (0, 5): [
                    "0: +45° Climb (爬升)",
                    "1: +20° Climb (爬升)",
                    "2: Level/Track (平飞)",
                    "3: -30° Dive (俯冲)",
                    "4: Max Dive (急俯冲)"
                ],
                # Head 1: 水平方向 (7动作)
                (1, 7): [
                    "0: Track (追踪)",
                    "1: Left Crank (左crank)",
                    "2: Left 3-9 (左39)",
                    "3: Tail (尾后)",
                    "4: Right 3-9 (右39)",
                    "5: Right Crank (右crank)",
                    "6: Center (占中)"
                ]
            }

            HEAD_NAMES = {
                0: "Head 0 (Vertical/Pitch 垂直俯仰)",
                1: "Head 1 (Horizontal/Yaw 水平偏航)"
            }

            for ep_idx, ep in enumerate(episodes_data):
                times = ep['times']
                if len(times) == 0:
                    continue

                cat_probs_raw = ep['cat_probs']
                num_heads = len(cat_probs_raw[0])
                bern_probs = ep['bern_probs'].flatten()
                action_labels = ep['action_labels']
                fires = ep['fires']

                total_subplots = num_heads + 1
                fig, axes = plt.subplots(total_subplots, 1, figsize=(12, 3.5 * total_subplots), sharex=True)
                if total_subplots == 1:
                    axes = [axes]

                fig.suptitle(f"Episode {ep_idx+1}: Red vs Rule_{ep['rule_num']} (Result: {ep['result']})", fontsize=14, fontweight='bold')

                for head_i in range(num_heads):
                    ax = axes[head_i]
                    head_probs = np.stack([step_p[head_i] for step_p in cat_probs_raw], axis=0) # (T, n_classes)
                    T, n_classes = head_probs.shape

                    probs_matrix = head_probs.T  # (n_classes, T)

                    # 在每个时间点(列)对概率分布内部进行归一化
                    col_sums = np.sum(probs_matrix, axis=0, keepdims=True)
                    col_sums[col_sums == 0] = 1.0
                    probs_matrix_norm = probs_matrix / col_sums

                    t_min, t_max = times[0], times[-1] if len(times) > 1 else times[0] + 1.0
                    im = ax.imshow(
                        probs_matrix_norm,
                        aspect='auto',
                        cmap='viridis',
                        origin='lower',
                        extent=[t_min, t_max, -0.5, n_classes - 0.5],
                        vmin=0,
                        vmax=1
                    )
                    cbar = fig.colorbar(im, ax=ax, pad=0.02)
                    cbar.set_label('Norm Prob', fontsize=9)

                    ax.set_yticks(range(n_classes))
                    if (head_i, n_classes) in HEAD_ACTION_LABELS:
                        y_labels = HEAD_ACTION_LABELS[(head_i, n_classes)]
                    elif n_classes == 14:
                        y_labels = [action_optionsLR.get(j, f'A{j}') for j in range(n_classes)]
                    else:
                        y_labels = [f'Head{head_i}_A{j}' for j in range(n_classes)]
                    ax.set_yticklabels(y_labels, fontsize=8)

                    # 叠加实际执行的动作轨迹
                    exec_head = action_labels[:, head_i] if action_labels.ndim > 1 else action_labels
                    ax.plot(times, exec_head, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Executed Action')
                    ax.scatter(times, exec_head, color='red', s=12, zorder=5)

                    head_title_name = HEAD_NAMES.get(head_i, f'Cat Head {head_i}')
                    ax.set_title(f'{head_title_name} Probability Distribution over Time (Normalized per Timestep)', fontsize=11)
                    ax.set_ylabel('Action Class', fontsize=10)
                    ax.grid(True, alpha=0.2, linestyle=':')
                    ax.legend(loc='upper right', fontsize=8)

                # 绘制开火概率与开火事件
                ax_bern = axes[-1]
                ax_bern.plot(times, bern_probs, color='crimson', linewidth=2, label='Fire Prob (p_fire)')
                ax_bern.set_ylim(-0.05, 1.05)
                ax_bern.set_ylabel('p_fire', fontsize=10)
                ax_bern.set_xlabel('Time (s)', fontsize=10)
                ax_bern.set_title('Bernoulli Head: Fire Probability over Time', fontsize=11)
                ax_bern.grid(True, alpha=0.3)

                fire_mask = (fires > 0)
                if np.any(fire_mask):
                    ax_bern.scatter(times[fire_mask], bern_probs[fire_mask], color='gold', s=70, marker='*', zorder=6, label='Shoot Event')
                    for t_shoot in times[fire_mask]:
                        ax_bern.axvline(x=t_shoot, color='red', linestyle=':', alpha=0.6)

                ax_bern.legend(loc='upper right', fontsize=8)

                plt.tight_layout()

            plt.show()
        except Exception as e:
            import traceback
            print(f"[Time-Heatmap Analysis] 绘图失败: {e}")
            traceback.print_exc()
    else:
        print("[Time-Heatmap Analysis] 没有收集到对抗数据")

    print("\nTime-dependent probability heatmap analysis completed.")


