'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time  # 确保引入 time 模块
from datetime import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import traceback # [新增]
import random
import csv # [新增] 用于记录 PBT 详细指标

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from prepare_il_datas import run_rules
from VsBaseline_while_training import test_worker

EXEC_COUNT = 0

dt_move = 0.04

def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()

def load_il_and_transitions(folder, il_name, rl_name):
    if folder is None:
        folder = os.getcwd()
    il_path = os.path.join(folder, il_name)
    trans_path = os.path.join(folder, rl_name)
    il = None
    trans = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
        print(f"Loaded IL data from: {il_path}")
    else:
        print(f"File NOT found: {il_path}")
        
    if os.path.isfile(trans_path):
        with open(trans_path, "rb") as f:
            trans = pickle.load(f)
    return il, trans

# ==========================================
# 核心修复：数据结构重组函数
# ==========================================
def restructure_actions(actions_data):
    """
    将 list of dicts [{'fly': 1, 'fire': 0}, ...] 
    转换为 dict of arrays {'cat': array([[1],...]), 'bern': array([[0],...])}
    并确保维度是 (N, 1) 以适配 PPOHybrid2 的索引操作
    """
    # 如果已经是字典格式，直接返回
    if isinstance(actions_data, dict):
        return actions_data
    
    # 如果是列表且包含字典，进行转换
    if isinstance(actions_data, list) and len(actions_data) > 0:
        # print("Restructuring actions from List[Dict] to Dict[Array]...") # 频繁调用可注释掉以减少刷屏
        
        # 初始化容器
        new_actions = {'cat': [], 'bern': []}
        
        for item in actions_data:
            # 兼容处理：item 可能是 dict，也可能是包含 dict 的 numpy array
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item() # 提取 numpy 里的 dict
            
            # 映射 'fly' -> 'cat' (离散机动)
            # 映射 'fire' -> 'bern' (开关开火)
            if isinstance(act, dict):
                # 优先找 'fly'，找不到找 'cat'
                val_cat = act.get('fly', act.get('cat'))
                if val_cat is not None:
                    new_actions['cat'].append(val_cat)
                
                # 优先找 'fire'，找不到找 'bern'
                val_bern = act.get('fire', act.get('bern'))
                if val_bern is not None:
                    new_actions['bern'].append(val_bern)

            # 备用：如果数据意外变成了 list/tuple
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                 new_actions['cat'].append(act[0])
                 new_actions['bern'].append(act[1])

        # 转换为 Numpy Array 并调整形状为 (Batch, 1)
        # 这一点至关重要：PPOHybrid2 里的 expert_cat[:, i] 需要 expert_cat 是二维的
        
        # 1. 'cat': 离散动作，转为 int64，Reshape 为 (N, 1)
        cat_arr = np.array(new_actions['cat'], dtype=np.int64)
        if cat_arr.ndim == 1:
            cat_arr = cat_arr.reshape(-1, 1)
        
        # 2. 'bern': 伯努利动作，转为 float32 (BCE Loss需要)，Reshape 为 (N, 1)
        bern_arr = np.array(new_actions['bern'], dtype=np.float32)
        if bern_arr.ndim == 1:
            bern_arr = bern_arr.reshape(-1, 1)

        result = {
            'cat': cat_arr,
            'bern': bern_arr
        }
        
        print(f"Structure fixed: 'cat' shape={cat_arr.shape}, 'bern' shape={bern_arr.shape}")
        return result

    return actions_data

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

def summarize(il_dict):
    if il_dict is None: return
    print("\nData Summary:")
    for k in ('states', 'actions', 'returns'):
        v = il_dict.get(k)
        if isinstance(v, dict):
            for sk, sv in v.items():
                print(f"  {k}['{sk}']: shape={sv.shape}")
        else:
            print(f"  {k}: shape={getattr(v, 'shape', 'Unknown')}")
            
def append_experience(td, obs, state, action, reward, next_state, done, active_mask):
    """
    统一把一次蓝方经验追加到 transition_dict。
    修改：增加 obs 输入，用于存储局部观测
    """
    td['obs'].append(obs) # 新增：存储Actor用的局部观测
    td['states'].append(state) # 修改：这里存储Critic用的全局状态(pomdp=0)
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    td['active_masks'].append(active_mask) # 【新增】active_mask，转入多智能体
    return td



# 加载数据
original_il_transition_dict, _ = load_il_and_transitions(
    os.path.join(cur_dir, "IL"),
    "il_transitions_combat_LR.pkl",
    "transition_dict_combat_LR.pkl"
)
original_il_transition_dict0 = copy.deepcopy(original_il_transition_dict)

# --- 关键步骤：执行数据重构 ---
if original_il_transition_dict is not None:
    # 这里完成 (Batch, Key) -> (Key, Batch) 的转换
    original_il_transition_dict['actions'] = restructure_actions(original_il_transition_dict0['actions'])
    
    # 顺便确保 states 和 returns 也是标准的 float32 numpy array
    original_il_transition_dict['states'] = np.array(original_il_transition_dict0['states'], dtype=np.float32)
    original_il_transition_dict['returns'] = np.array(original_il_transition_dict0['returns'], dtype=np.float32)

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None, 
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta', 
                               rule_rate=0.5, deltaFSP_epsilon=0.5):
    """
    优化后的对手采样逻辑：
    1. 优先判定是否进入“规则复习”分支。
    2. 若未进入，则根据 SP_type 执行具体的采样策略。
    """
    # 【核心修改】在函数内部合并出一个临时的全集字典用于查询分数
    # 这样 keys 里的任何元素都能在这里找到对应的 ELO
    
    if hall_of_fame is not None:
        candidate_pool = hall_of_fame.copy()
        candidate_pool.update(elite_elo_ratings)
    else:
        candidate_pool = elite_elo_ratings
    
    keys = list(candidate_pool.keys())
    if not keys:
        return np.array([]), []

    # --- 第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护) ---
    # 只要 rule_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < rule_rate and rule_keys:
        candidate_pool_keys = rule_keys
        candidate_pool = {k: candidate_pool[k] for k in candidate_pool_keys}

        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    # --- 第二层判断：进入核心采样逻辑 ---
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        # if SP_type == 'PFSP_challenge':
        actual_target = np.max(elos)
        # elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
        #     actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        # else: # 默认通用的 'PFSP' 逻辑
        #     # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
        #     actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
        diffs = elos - actual_target
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
        return probs, keys

    # 2. 处理 FSP (全样本均匀分布)
    elif SP_type == 'FSP':
        probs = np.ones(len(keys)) / len(keys)
        return probs, keys

    # # 3. 处理 deltaFSP (新旧池切分)
    # elif SP_type == 'deltaFSP':
    #     n = len(keys)
    #     new_count = max(1, int(np.ceil(n * 0.2)))
    #     new_keys = keys[-new_count:]
    #     old_keys = keys[:-new_count]
        
    #     # 这里的 deltaFSP_epsilon 建议直接作为参数传入或使用全局变量
    #     if np.random.rand() < float(deltaFSP_epsilon) or not old_keys:
    #         target_keys = new_keys
    #     else:
    #         target_keys = old_keys
            
    #     probs = np.ones(len(target_keys)) / len(target_keys)
    #     return probs, target_keys

    # # 4. 处理 SP (最强/最新历史版本)
    # elif SP_type == 'SP':
    #     # rein_keys = [k for k in keys if k.startswith('actor_rein') and '_step_' not in k]
    #     # 严格匹配 actor_rein + 数字
    #     rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
    #     if not rein_keys: return np.array([]), []
        
    #     def extract_number(k):
    #         # try: return int(k.replace('actor_rein', ''))
    #         # except: return -1
    #         try: return int(re.search(r'actor_rein(\d+)', k).group(1))
    #         except: return -1
            
    #     best_key = max(rein_keys, key=extract_number)
    #     return np.array([1.0]), [best_key]

    # 5. 兜底逻辑: Rule 均匀采样 (None)
    else:
        if not rule_keys: return np.array([]), []
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = 8000
    red_height = 8000
    red_psi = -np.pi/2
    blue_psi = np.pi/2
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N = init_North
    red_E = 55e3 # 45e3
    blue_N = init_North
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process(rank, pipe, args, state_dim, hidden_dim, 
                   action_dims_dict, device_worker, dt_maneuver, 
                   seed, opp_greedy_rate, dt_move=0.05, no_crash=1, member_id=0):
    """
    常驻子进程：接收参数 -> 跑完一整场 -> 返回数据 -> 等待
    完整的 Worker 逻辑：包含环境初始化、模型加载、仿真循环、数据回传
    """
    try:  # <--- 【新增】添加此行，并将下方所有代码整体缩进
        # --- 1. 初始化阶段 (只运行一次) ---
        
        # 确保每个进程种子不同，避免所有环境生成完全一样的随机数
        worker_seed = seed + rank * 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        # 初始化环境 (关闭可视化以加速)
        env = ChooseStrategyEnv(args, tacview_show=False)
        env.shielded = no_crash # 假设默认开启防撞
        env.dt_move = dt_move
        env.dt_maneuver = dt_maneuver

        # 初始化本地网络 (CPU)
        local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        # 【修改 1】创建一个 dummy critic，仅为了满足 PPOHybrid 初始化要求
        local_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)
        local_agent = PPOHybrid(
            actor=HybridActorWrapper(local_actor, action_dims_dict, None, device_worker).to(device_worker),
            critic=local_dummy_critic,  # <--- 【修改】传入实体对象，而非 None
            actor_lr=0, critic_lr=0,    # 学习率为0，确保不会更新
            lmbda=0, eps=0, gamma=0, epochs=0, # 补全位置参数
            device=device_worker 
        )
        
        # 初始化对手网络
        adv_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        # 【修改 2】同样为对手创建一个 dummy critic
        adv_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)
        adv_agent = PPOHybrid(
            actor=HybridActorWrapper(adv_actor, action_dims_dict, None, device_worker).to(device_worker),
            critic=adv_dummy_critic,    # <--- 【修改】传入实体对象，而非 None
            actor_lr=0, critic_lr=0, 
            lmbda=0, eps=0, gamma=0, epochs=0, # 补全位置参数
            device=device_worker
        )

        # --- 2. 循环等待阶段 ---
        while True:
            # 阻塞等待指令
            cmd, packet = pipe.recv()
            
            if cmd == 'EXIT':
                env.close()
                break
                
            if cmd == 'RUN_EPISODE':
                # 解包数据
                (member_id, actor_weights, opponent_info, settings) = packet
                
                # A. 同步权重 (极快)
                local_agent.actor.load_state_dict(actor_weights)
                
                # B. 配置对手
                opp_name, opp_type, opp_data = opponent_info
                adv_is_rule = (opp_type == 'rule')
                rule_num = 0
                if adv_is_rule:
                    rule_num = opp_data
                else:
                    adv_agent.actor.load_state_dict(opp_data)

                # C. 准备本回合容器
                # Worker 收集完整的 ego_trans (用于 SIL) 和 enm_trans (用于 SIL)
                # local_trans 用于 PPO 更新 (只包含 Blue 视角)
                local_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                ego_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                enm_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}

                # D. 环境重置
                # randomized_birth = settings['randomized_birth']
                action_cycle_multiplier = settings['action_cycle_multiplier']
                reward_weight = settings['weight_reward']
                # 在子环境中重新计算出生状态
                # red_birth, blue_birth = create_initial_state_worker(randomized_birth)
                # 使用从master传来的出生状态
                red_birth = settings['red_birth']
                blue_birth = settings['blue_birth']
                env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_ammo=6, blue_init_ammo=6)
                
                # 状态变量初始化
                done = False
                last_decision_obs, last_decision_state = None, None
                last_enm_decision_obs, last_enm_decision_state = None, None
                current_action, current_action_exec, current_enm_action_exec = None, None, None
                
                steps_run = 0
                episode_return = 0 # 仅用于统计显示
                m_fired = 0
                
                dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}
                
                # --- E. 仿真循环 (核心物理逻辑) ---
                # 计算最大步数
                max_counts = int(args.max_episode_len / dt_maneuver)
                
                for count in range(max_counts):
                    if not env.running or done: break
                    
                    # 1. 获取观测
                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                    b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    r_state_global, _ = env.obs_1v1('r', reward_fn=1)

                    # 2. 决策点 (Action Cycle)
                    if steps_run % action_cycle_multiplier == 0:
                        # 2.1 存储【上一个】周期的经验
                        if steps_run > 0:
                            # 注意：这里调用你原文件里的 append_experience 辅助函数
                            # 确保 append_experience 在 这个函数 作用域外是可见的，或者复制进来
                            append_experience(local_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, b_state_global, False, not dead_dict['b'])
                            append_experience(ego_trans, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, b_state_global, False, not dead_dict['b'])
                            append_experience(enm_trans, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, r_state_global, False, not dead_dict['r'])

                        # 2.2 更新上一帧记录
                        last_decision_obs = b_obs
                        last_decision_state = b_state_global
                        last_enm_decision_obs = r_obs
                        last_enm_decision_state = r_state_global
                        
                        # 2.3 产生新动作 (No Grad)
                        with torch.no_grad():
                            # Blue Decision
                            b_action_exec, _, _, _ = local_agent.take_action(b_obs, explore=1)
                            b_action_label = b_action_exec['cat'][0]
                            b_fire = b_action_exec['bern'][0]
                            
                            # Red Decision
                            r_state_check = env.unscale_state(r_check_obs)
                            if adv_is_rule:
                                # 调用规则，假设 basic_rules 已导入
                                r_action_label, r_fire = basic_rules(r_state_check, rule_num, p_random=0.1)
                                r_action_exec = {'cat': np.array([r_action_label]), 'bern': np.array([r_fire], dtype=np.float32)}
                            else:
                                # 随机决定本局对手是否开启探索
                                adv_explore = 1 if np.random.rand() > opp_greedy_rate else 0
                                r_action_exec, _, _, _ = adv_agent.take_action(r_obs, explore={'cont':0, 'cat':adv_explore, 'bern':1})
                                r_action_label = r_action_exec['cat'][0]
                                r_fire = r_action_exec['bern'][0]

                        # 2.4 处理开火
                        b_m_id = launch_missile_immediately(env, 'b') if b_fire else None
                        r_m_id = launch_missile_immediately(env, 'r') if r_fire else None
                        if b_m_id: m_fired += 1
                        
                        # 2.5 记录当前动作供下一帧存储
                        current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                        current_action_exec = {'cat': b_action_exec['cat'], 'bern': np.array([b_m_id is not None])}
                        current_enm_action_exec = {'cat': r_action_exec['cat'], 'bern': np.array([r_m_id is not None])}

                    # 3. 物理步进
                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    steps_run += 1
                    
                    # 4. 奖励计算
                    done, b_rew_event, b_rew_constraint, b_rew_shaping = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier)
                    _, r_rew_event, r_rew_constraint, r_rew_shaping = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier)
                    
                    reward_for_learn = sum(np.array([b_rew_event, b_rew_constraint, b_rew_shaping]) * reward_weight)
                    reward_for_enm = sum(np.array([r_rew_event, r_rew_constraint, r_rew_shaping]) * reward_weight)
                    
                    if steps_run % action_cycle_multiplier == 0 or done:
                        episode_return += (b_rew_event + b_rew_constraint)
                    
                    # 5. 存活更新 (用于 Done 标记)
                    next_b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    next_r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                    dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

                # --- End of Simulation Loop ---
                
                # 6. 存储最后一步经验 (Terminal State)
                # 强制做一次终局判定
                done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False, action_cycle_multiplier)
                
                if last_decision_state is not None:
                    append_experience(local_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(ego_trans, last_decision_obs, last_decision_state, current_action_exec, reward_for_learn, next_b_state_global, True, not dead_dict['b'])
                    append_experience(enm_trans, last_enm_decision_obs, last_enm_decision_state, current_enm_action_exec, reward_for_enm, next_r_state_global, True, not dead_dict['r'])

                # 7. 打包结果
                result_packet = {
                    'trans': local_trans, # 用于 RL Update
                    'ego_trans': ego_trans, # 用于 SIL (win)
                    'enm_trans': enm_trans, # 用于 SIL (lose)
                    'metrics': {
                        'return': episode_return,
                        'steps': steps_run,
                        'win': env.win,
                        'lose': env.lose,
                        'draw': env.draw,
                        'm_fired': m_fired
                    },
                    'opp_name': opp_name,
                    'member_id': member_id,
                }
                
                # 8. 发送回 Master
                pipe.send(result_packet)

    except Exception: # [新增] 异常捕获与回传
        print(f"!!! Worker {rank} CRASHED !!!")
        tb = traceback.format_exc()
        print(tb)
        try: pipe.send({'error': tb})
        except: pass

# ==========================================
# PBT Population Helper Class
# ==========================================
class PBTMember:
    def __init__(self, agent_id, agent, elo=1200, sigma_elo=400.0):
        self.id = agent_id
        self.agent = agent  # PPOHybrid instance
        self.elo = elo
        self.sigma_elo = sigma_elo # 对手偏好超参数 (控制采样宽度)
        self.wins = 0
        self.matches = 0

    def reset_optimizer(self):
        """
        调用 PPOHybrid 内部定义的 reset_optimizer 
        以此来彻底重建 Adam 优化器，清除动量缓存
        """
        self.agent.reset_optimizer()
    
    # todo 不调模仿率了，改为调对手偏好
    

def run_MLP_simulation(
    num_workers=5, # 每个member的并行进程数，根据CPU核数调整，建议 10-20
    mission_name='无名',
    actor_lr=1e-4,
    critic_lr=5e-4,
    actor_lr_init_il = 1e-4,
    critic_lr_init_il = 5e-4,
    IL_epoches=180,
    max_steps=4 * 165e4,
    hidden_dim=None,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=None,
    il_batch_size=128,
    mini_batch_size_mixed=64,
    beta_mixed=1.0,
    label_smoothing=0.3,
    action_cycle_multiplier=30,
    trigger0=50e3,
    trigger_delta=50e3,
    weight_reward_0=None,
    no_crash=1,
    dt_move=0.05,
    max_episode_duration=10*60,
    R_cage = 62.00e3, # 45e3 # 55e3,
    dt_maneuver=0.2,
    transition_dict_threshold=1000,
    should_kick = True,
    init_elo_ratings = {
        "Rule_0": 1200,
        "Rule_1": 1200,
        "Rule_2": 1200,
    },
    self_play_type = 'PFSP', # FSP, SP, None 表示非自博弈
    hist_agent_as_opponent = 1, # 是否开始记录历史智能体
    sigma_elo = 400,
    WARM_UP_STEPS = 500e3,
    ADMISSION_THRESHOLD = 0.5,
    MAX_HISTORY_SIZE = 300,  # 100
    deltaFSP_epsilon = 0.8,
    rule_actor_rate = 0.2,
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
    save_interval = 1, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
    opp_greedy_rate = 0.5, # 对手贪婪率
    num_runs = 3, # 测试回合重复次数
    device = torch.device("cpu"),
    # --- PBT Params ---
    pop_size = 2,      # 种群大小
    interval_of_pbt=20, # pbt 进化频率
):

    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 2. 参数与环境配置 (Master 用于获取维度)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = ChooseStrategyEnv(args)
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    del dummy_env

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Master training device: {device}")

    # 3. 创建基础神经网络 (用于 MARWIL 预训练)
    # 我们先训练一个 base agent，然后复制给 population
    base_actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    base_critic_net = ValueNet(state_dim, hidden_dim).to(device)
    base_actor_wrapper = HybridActorWrapper(base_actor_net, action_dims_dict, None, device).to(device)

    base_agent = PPOHybrid(
        actor=base_actor_wrapper, 
        critic=base_critic_net, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device, 
        k_entropy=k_entropy, 
        max_std=label_smoothing
    )
    
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    
    save_meta_once(os.path.join(log_dir, "actor.meta.json"), base_agent.actor.state_dict())
    save_meta_once(os.path.join(log_dir, "critic.meta.json"), base_agent.critic.state_dict())

    # 保存未训练的onnx模型供结构检查
    dummy_state = torch.randn(1, state_dim).to(device)
    # ==========================================
    # 1. 导出 Actor 的底层网络（PolicyNetHybrid）
    # ==========================================
    actor_onnx_path = os.path.join(log_dir, "student_actor.onnx")
    try:
        torch.onnx.export(
            base_agent.actor.net,           # 只导出纯网络结构，避开 Wrapper里的复杂采样操作
            dummy_state,                       # 伪造的输入状态
            actor_onnx_path,                   # 输出的文件名 / 路径
            export_params=True,                # 是否连同参数一起导出（选 True 可以看权重信息）
            opset_version=11,                  # 建议使用 11 或以上的算子集
            do_constant_folding=True,          # 是否执行常量折叠优化
            input_names=['state'],             # 命名的输入节点名称
            output_names=['cat_output', 'bern_output'] # 按照返回顺序手动指定名字
        )
        print(f"Actor ONNX successfully exported to {actor_onnx_path}")
    except Exception as e:
        print(f"Error exporting Actor ONNX: {e}")
    # ==========================================
    # 2. 导出 Critic 的底层网络（ValueNet）
    # ==========================================
    critic_onnx_path = os.path.join(log_dir, "student_critic.onnx")
    try:
        torch.onnx.export(
            base_agent.critic,              
            dummy_state,                       
            critic_onnx_path,                  
            export_params=True,                
            opset_version=11,                 
            do_constant_folding=True,          
            input_names=['state'],             
            output_names=['value_estimate']    # Critic 返回的一般是标量价值
        )
        print(f"Critic ONNX successfully exported to {critic_onnx_path}")
    except Exception as e:
        print(f"Error exporting Critic ONNX: {e}")

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (只对 base_agent 做一次)
    print("Start MARWIL Training on Base Agent...")
    base_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = base_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, 
            label_smoothing=label_smoothing
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    print("IL Training Finished.")
    

    # ==============================================================================
    # PBT 初始化: 创建种群
    # ==============================================================================
    population = []
    print(f"Initializing Population of size {pop_size}...")
    for i in range(pop_size):
        # 创建新实例
        new_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        new_critic = ValueNet(state_dim, hidden_dim).to(device)
        new_wrapper = HybridActorWrapper(new_actor, action_dims_dict, None, device).to(device)
        
        # 复制预训练权重
        new_wrapper.load_state_dict(base_agent.actor.state_dict())
        new_critic.load_state_dict(base_agent.critic.state_dict())
        
        new_agent_obj = PPOHybrid(
            actor=new_wrapper, critic=new_critic,
            actor_lr=actor_lr, critic_lr=critic_lr,
            lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma, 
            device=device, k_entropy=k_entropy, max_std=label_smoothing
        )

        # 复制预训练的优化器状态 (动量)
        new_agent_obj.actor_optimizer.load_state_dict(base_agent.actor_optimizer.state_dict())
        new_agent_obj.critic_optimizer.load_state_dict(base_agent.critic_optimizer.state_dict())
        # 设置为强化学习的学习率
        new_agent_obj.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
        """
        # 保存权重和优化器状态到本地
        checkpoint = {
            'model_state_dict': student_agent.actor.state_dict(),
            'optimizer_state_dict': student_agent.actor_optimizer.state_dict(),
            'epoch': current_epoch,
            'elo': current_elo
        }
        torch.save(checkpoint, "checkpoint.pt")

        # 从本地读取模型和优化器
        checkpoint = torch.load("checkpoint.pt")
        student_agent.actor.load_state_dict(checkpoint['model_state_dict'])
        student_agent.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """
        # 初始化 sigma_elo: 在 [100, 800] 之间随机采样 (值变向量的单分量)
        init_sigma = 400 # np.random.uniform(100, 800) # debug 单member能否和原先保持一致性？
        
        member = PBTMember(i, new_agent_obj, elo=1200, sigma_elo=init_sigma)
        population.append(member)
        print(f"  Agent P{i}: sigma_elo={init_sigma:.1f}")

        student_agent = population[i].agent
    # # 清理 base_agent 节省显存
    # del base_agent
    # torch.cuda.empty_cache()

    # ----------------------------------------------------
    # 并行环境初始化 (Worker Setup)
    # ----------------------------------------------------
    
    # 7. 强化学习准备
    
    
    # 进程通信设置
    mp.set_start_method('spawn', force=True)
    
    # --- A. 启动并行测试进程池 (Async Test Pool) ---
    # 这个池子用于 periodic testing，不参与训练数据的生成
    test_pool = mp.Pool(processes=3, maxtasksperchild=10) 
    

    # --- B. 启动并行训练 Worker (Sync Training Workers) ---
    # 这些 Worker 与 Master 同步，负责生成训练数据
    workers = []
    pipes = []
    worker_device = torch.device('cpu') # Worker 使用 CPU 推理
    
    print(f"Initializing {num_workers} training workers...")

    # 初始化采样子进程
    for member_id in range(pop_size): # 加套一层
        for i in range(num_workers):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=worker_process, args=(
                i, child_conn, args, state_dim, hidden_dim, 
                action_dims_dict, worker_device, dt_maneuver, 
                seed, opp_greedy_rate, dt_move, no_crash, member_id
            ))
            p.start()
            workers.append(p)
            pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = copy.deepcopy(elo_ratings)
    hall_of_fame = {}
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # # 尝试加载历史，为了接着之前的训练（但从来都没有这么做）
    # if os.path.exists(full_json_path):
    #     with open(full_json_path, 'r', encoding='utf-8') as f: elo_ratings = json.load(f)
    # if os.path.exists(elite_json_path):
    #     with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)
    # if os.path.exists(hof_json_path):
    #     with open(hof_json_path, 'r', encoding='utf-8') as f: hall_of_fame = json.load(f)

    # --- [新增] PBT 详细指标 CSV 初始化 ---
    csv_paths = {
        'win': os.path.join(log_dir, "pbt_wins.csv"),
        'lose': os.path.join(log_dir, "pbt_loss.csv"),
        'draw': os.path.join(log_dir, "pbt_draw.csv"),
        'return': os.path.join(log_dir, "pbt_return.csv"),
        'sigma_elo': os.path.join(log_dir, "pbt_sigma_elo.csv"),
        'elo': os.path.join(log_dir, "pbt_elo.csv")
    }
    for label, path in csv_paths.items():
        while True:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ["batch_idx"] + [f"member_{i}" for i in range(pop_size)]
                    writer.writerow(header)
                break
            except PermissionError:
                print(f"Waiting for CSV initialization: {path} is locked (Excel?). Please close it...", end='\r')
                time.sleep(2)

    # 初始化种群成员的 Elo (如果 elo_ratings 中存了 __CURRENT_MAIN__，则以此为基准)
    init_main_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)
    for member in population:
        member.elo = init_main_elo

    # 初始对手
    if (not elo_ratings) or IL_epoches > 0:
        init_opponent_name = "actor_rein0"
        torch.save(base_agent.actor.state_dict(), os.path.join(log_dir, f"{init_opponent_name}.pt"))
        if self_play_type != 'None': elo_ratings[init_opponent_name] = 1200

    # 训练循环变量
    total_steps = 0
    pbt_generation_cnt = 0 # PBT 进化代数
    batch_idx = 0
    trigger = trigger0
    current_max_steps = int(max_steps)
    
    # 全局 Buffer (用于攒够 Batch 训练)
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    
    buffer = []

    for member in population:
        transition_dict = copy.deepcopy(empty_transition_dict)
        buffer.append(transition_dict)

    # # 历史记录初始化：用 P0 的参数存为初始对手
    # init_opponent_name = "actor_rein0"
    # torch.save(population[0].agent.actor.state_dict(), os.path.join(log_dir, f"{init_opponent_name}.pt"))
    # if self_play_type != 'None': elo_ratings[init_opponent_name] = 1200

    # =========================================================
    # 主循环 (Master Process) - PBT 结构
    # =========================================================
    while True: 
        while total_steps < current_max_steps:
            # --- 1. 并行测试 (Periodic Testing) ---
            # 为了简单起见，我们用当前 Elo 最高的 Agent 进行测试
            if total_steps >= trigger:
                best_member = max(population, key=lambda m: m.elo)
                print(f"\n>>> Triggering Parallel Test (Best Agent P{best_member.id}) at steps {total_steps}...")
                current_weights = {k: v.cpu().clone() for k, v in best_member.agent.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                test_tasks = []
                for r_idx in [0, 1, 2, 3, 4]:
                    obj = test_pool.apply_async(
                        test_worker, 
                        args=(current_weights, r_idx, args, 
                              state_dim, hidden_dim, action_dims_dict, 
                              dt_maneuver, 'cpu', num_runs, action_cycle_multiplier)
                    )
                    test_tasks.append(obj)
                # 等待所有测试进程结束
                test_results = [t.get() for t in test_tasks]

                outcomes = {rule_num: score for rule_num, score, result2 in test_results}
                outcomes_return = {rule_num: result2 for rule_num, score, result2 in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps)
                    logger.add(f"test/agent_vs_rule{r_num}_return", outcomes_return[r_num], total_steps)
                    print(f"  [Test Result] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

                # 名人堂判定：如果全胜则保存并加入池子
                if all(score > 0.5 for score in outcomes.values()):
                    # rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+$', k)]
                    rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+(_P\d+)?$', k)]
                    if rein_keys:
                        # 找到数值最大的编号（即最新的已保存智能体，忽略 _P 后缀）
                        hof_key = max(rein_keys, key=lambda k: int(k.split('_P')[0].replace("actor_rein", '')))
                        
                        if hof_key not in hall_of_fame:
                            hall_of_fame[hof_key] = elo_ratings.get(hof_key, best_member.elo)
                            print(f"!!! [Hall of Fame] New Hero Captured: {hof_key}")
                
                trigger += trigger_delta

            # ==================================================================
            # PBT Loop: 每个 Agent 轮流采样和训练
            # ==================================================================
            # 每个 Generation，所有成员都走一遍
            



            # --- 2. 准备训练 Batch (Synchronous) ---
            
            # 1. 计算当前排位 rank_pos
            valid_elo_values = [v for k, v in elite_elo_ratings.items() if not k.startswith("__")]
            if not valid_elo_values:
                rank_pos = 0.5
                min_elo, max_elo = init_main_elo, init_main_elo
            else:
                min_elo = np.min(valid_elo_values)
                max_elo = np.max(valid_elo_values)
                denom = float(max_elo - min_elo)
                # 如果分母为0，视为0.5中位
                if denom == 0.0:
                    rank_pos = 0.5
                else:
                    pop_max_elo = max(m.elo for m in population)
                    rank_pos = float((pop_max_elo - min_elo) / denom)


            # A. 获取当前策略权重 (CPU)
            all_current_actor_weights = []
            for member_id in range(pop_size):
                student_agent = population[member_id].agent
                current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
                all_current_actor_weights.append(current_actor_weights)

            # B. 分发任务给 Worker
            # 这一步 Master 决定每个 Worker 打谁
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            # [修正] 处理纯自博弈逻辑：当没有初始规则对手时，筛选分数最高的 MAX_HISTORY_SIZE 个对手作为匹配池
            if not init_elo_ratings:
                # 按照 Elo 分数降序排列，排除内部特殊键
                sorted_all_keys = [k for k in sorted(elo_ratings.keys(), 
                                                   key=lambda x: elo_ratings[x] if not x.startswith("__") else -1e9, 
                                                   reverse=True) if not k.startswith("__")]
                effective_pool = {k: elo_ratings[k] for k in sorted_all_keys[:MAX_HISTORY_SIZE]}
            else:
                effective_pool = elite_elo_ratings
            
            # 分发采样任务到子进程
            for rank in range(num_workers * pop_size):

                current_member_id = rank // num_workers
                # 拿出对应 member 的权重
                current_member_weights = all_current_actor_weights[current_member_id]

                # 采样对手 (使用该 member 独有的 sigma_elo)
                probs, opponent_keys = get_opponent_probabilities(
                    effective_pool,
                    hall_of_fame,
                    target_elo = max_elo,  
                    SP_type= 'PFSP_challenge', 
                    sigma=population[current_member_id].sigma_elo,
                    rule_rate=rule_actor_rate,
                    deltaFSP_epsilon=deltaFSP_epsilon,
                )
                selected_opponent_name = np.random.choice(opponent_keys, p=probs)
                
                # 准备对手数据
                opp_type = 'rule'
                opp_data = 0
                if "Rule" in selected_opponent_name:
                    try:
                        rule_num = int(selected_opponent_name.split('_')[1])
                    except:
                        rule_num = 0
                    opp_data = rule_num
                else:
                    opp_type = 'nn'
                    adv_path = os.path.join(log_dir, f"{selected_opponent_name}.pt")
                    if os.path.exists(adv_path):
                        opp_data = torch.load(adv_path, map_location='cpu', weights_only=1) # 传给 Worker 必须是 CPU Tensor
                    else:
                        # Fallback
                        opp_type = 'rule'
                        opp_data = 0
                
                opp_info = (selected_opponent_name, opp_type, opp_data)
                
                # 初始位置配置
                rb, bb = create_initial_state_worker(randomized_birth)
                settings = {
                    'randomized_birth': randomized_birth,
                    'action_cycle_multiplier': action_cycle_multiplier,
                    'weight_reward': weight_reward_0,
                    'red_birth': rb,
                    'blue_birth': bb
                }
                
                # 发送指令 pipe.send

                pipes[rank].send(('RUN_EPISODE', (current_member_id, current_member_weights, opp_info, settings)))

            # C. 等待所有 采样 Worker 完成 (Barrier)
            batch_results = []
            for rank in range(pop_size * num_workers):
                try: # <--- 【新增】
                    res = pipes[rank].recv() # 阻塞等待 pipe.recv
                except EOFError: # <--- 【新增】捕获管道断开错误
                    print(f"[Error] Worker {rank} crashed silently.")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed.")
                    
                # [新增] 检查 Worker 是否传回了奔溃信息
                if isinstance(res, dict) and 'error' in res:
                    print(f"--- Master received error from Worker {rank}, aborting. ---")
                    # 关闭所有子进程防止残留
                    for p in workers: p.terminate()
                    # 抛出具体的运行时错误
                    raise RuntimeError(f"Worker {rank} crashed with error:\n{res['error']}")
                    
                batch_results.append(res)
            
            # --- 3. 数据聚合与处理 ---
            batch_total_steps = np.zeros(pop_size)
            batch_wins = np.zeros(pop_size)
            batch_loss_cnt = np.zeros(pop_size)
            batch_draw_cnt = np.zeros(pop_size)        # 新增统计
            batch_total_return = np.zeros(pop_size)    # 新增统计
            batch_total_m_fired = np.zeros(pop_size)   # 新增统计
            
            # 在这里，已经把每个worker的经验都加到对应的member的buffer里了
            for res in batch_results:
                # res 结构: {'trans':..., 'ego_tr':..., 'enm_tr':..., 'metrics':..., 'opp_name':...}
                l_tr = res['trans'] # PPO 训练数据 (含探索)
                ego_tr = res['ego_trans'] # SIL 蓝方数据
                enm_tr = res['enm_trans'] # SIL 红方数据
                metrics = res['metrics']
                opp_name = res['opp_name']

                member_id = res['member_id'] # member 编号

                # 3.1 聚合 PPO 数据到全局 Buffer
                transition_dict = buffer[member_id]

                for k in transition_dict:
                    transition_dict[k].extend(l_tr[k])
                
                # [新增] 填充 buffer 用户打印详情
                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")
                
                batch_total_steps[member_id] += metrics['steps']
                batch_total_return[member_id] += metrics['return']
                batch_total_m_fired[member_id] += metrics['m_fired']

                if metrics['win']: batch_wins[member_id] += 1
                elif metrics['lose']: batch_loss_cnt[member_id] += 1
                else: batch_draw_cnt[member_id] += 1
                
                
                
                
                # 3.3 ELO 更新 (实时更新)
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                
                is_rule_agent = "Rule" in opp_name
                # 简单的踢出逻辑 check
                should_update = True
                if should_kick and not is_rule_agent:
                    # 如果对手表现极差（例如无脑不开火且输了），可以不更新 ELO 甚至踢出
                    # 这里简化处理，暂时都更新
                    pass
                
                if opp_name in elo_ratings:
                    prev_member_elo = population[member_id].elo
                    adv_elo = elo_ratings[opp_name]
                    
                    # 更新该成员的 Elo 分
                    new_member_elo = update_elo(prev_member_elo, adv_elo, actual_score, K_FACTOR)
                    population[member_id].elo = new_member_elo
                    
                    # 更新对手 Elo 分
                    new_adv_elo = update_elo(adv_elo, prev_member_elo, 1.0 - actual_score, K_FACTOR)
                    elo_ratings[opp_name] = new_adv_elo
                    
                    # 记录主分数（用于兼容旧逻辑，保存当前最高分为主智能体分）
                    elo_ratings["__CURRENT_MAIN__"] = max(m.elo for m in population)
                    
                    if opp_name in elite_elo_ratings:
                        elite_elo_ratings[opp_name] = new_adv_elo
                else:
                    print(f'警告，elo_ratings没有收录对手: {opp_name}!!!')
            
            # [新增] 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps += np.sum(batch_total_steps)  # 如果用累加步数的话会吃亏，因为分割经验池导致数据利用率低了
            batch_idx += 1
            
            # --- 3.5 [新增] 记录所有 Member 分项指标 ---
            # A. 写入 CSV (以迭代次数 batch_idx 为行，member_id 为列)
            csv_data = {
                'win': batch_wins / num_workers,
                'lose': batch_loss_cnt / num_workers,
                'draw': batch_draw_cnt / num_workers,
                'return': batch_total_return / num_workers,
                'sigma_elo': np.array([m.sigma_elo for m in population]),
                'elo': np.array([m.elo for m in population])
            }
            for label, values in csv_data.items():
                while True:
                    try:
                        with open(csv_paths[label], 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([batch_idx] + list(values))
                        break # 写入成功，退出循环
                    except PermissionError:
                        # 如果名被 Excel 占用，循环等待并提示
                        print(f"Warning: CSV file {label} is locked by another program (Excel?). Please close it. Retrying in 2 seconds...", end='\r')
                        time.sleep(2)
            
            # # B. 写入 TensorBoard (分成员记录，方便对比)
            # for m_id in range(pop_size):
            #     m_tag = f"pbt_member/P{m_id}"
            #     logger.add(f"{m_tag}/win", batch_wins[m_id] / num_workers, total_steps)
            #     logger.add(f"{m_tag}/return", batch_total_return[m_id] / num_workers, total_steps)

            # C. 记录原有聚合指标
            logger.add("special/0 发射的导弹总数", np.mean(batch_total_m_fired), total_steps)
            # 记录平均回报与胜率
            best_member = np.argmax(batch_wins) # 记录胜率最高的智能体的回报
            logger.add("train/1 avg_episode_return", batch_total_return[best_member] / num_workers, total_steps)
            logger.add("train/2 win", batch_wins[best_member] / num_workers, total_steps)
            logger.add("train/2 lose", batch_loss_cnt[best_member] / num_workers, total_steps)
            logger.add("train/2 draw", batch_draw_cnt[best_member] / num_workers, total_steps)
            logger.add("train/11 episode/step", batch_idx * num_workers, total_steps)


            # --- 5. 更新，保存与维护 (Checkpoint & Pool) ---
            if batch_idx % save_interval == 0 and \
                any(len(buffer[m_id]['dones']) >= transition_dict_threshold for m_id in range(pop_size)):
                
                # 遍历每一个 member 进行更新
                for m_id in range(pop_size):
                    m_trans_dict = buffer[m_id]
                    m_member = population[m_id]
                    m_agent = m_member.agent
                    
                    # 重构 Action 结构
                    m_trans_dict['actions'] = restructure_actions(m_trans_dict['actions'])
                    
                    # --- 计算该成员相对于对手池的 Elo 差值，用于 超参数 调节 ---
                    all_keys = list(elo_ratings.keys())
                    rule_keys = [k for k in all_keys if k.startswith('Rule')]
                    rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                    # 取最后（最新插入）的300个
                    latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                    target_pool_keys = rule_keys + latest_rein_keys
                    avg_pool_elo = np.mean([elo_ratings[k] for k in target_pool_keys])
                    x_elo_diff = m_member.elo - avg_pool_elo
                    
                    # 执行 PPO 更新
                    m_agent.update(m_trans_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed)
                    
                    # 记录 Log (按 Member 区分)
                    m_tag_prefix = f"{m_id}"
                    logger.add(f"train/{m_tag_prefix}-actor_loss", m_agent.actor_loss, total_steps)
                    logger.add(f"train/{m_tag_prefix}-critic_loss", m_agent.critic_loss, total_steps)
                    logger.add(f"train/{m_tag_prefix}-entropy_cat", m_agent.entropy_cat, total_steps)
                    logger.add(f"train/{m_tag_prefix}-entropy_bern", m_agent.entropy_bern, total_steps)
                    logger.add(f"train/{m_tag_prefix}-elo", m_member.elo, total_steps)
                    logger.add(f"train/{m_tag_prefix}-elo_diff_x", x_elo_diff, total_steps)
                    logger.add(f"train/{m_tag_prefix}-sigma_elo", m_member.sigma_elo, total_steps)
                    
                    # 清空该成员的 Buffer
                    buffer[m_id] = copy.deepcopy(empty_transition_dict)
                    
                    # A. 保存模型
                    actor_key = f"actor_rein{batch_idx}_P{m_id}"
                    torch.save(m_agent.actor.state_dict(), os.path.join(log_dir, f"{actor_key}.pt"))
                    # Critic 如果是共享的话可以只存一份，或者每个 member 存一份
                    torch.save(m_agent.critic.state_dict(), os.path.join(log_dir, f"critic_P{m_id}.pt"))

                    # B. 维护精英池 (每个成员独立判断是否入池)
                    if hist_agent_as_opponent and total_steps >= WARM_UP_STEPS:
                        valid_elos = [v for k, v in elite_elo_ratings.items() if k.startswith("Rule")]
                        if not valid_elos: valid_elos = [1200]
                        r_min, r_max = min(valid_elos), max(valid_elos)
                        denom = r_max - r_min if r_max != r_min else 1.0
                        m_rank = (m_member.elo - r_min) / denom
                        
                        if m_rank >= ADMISSION_THRESHOLD:
                            history_keys = [k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")]
                            while len(history_keys) >= MAX_HISTORY_SIZE:
                                weakest_history_key = min(history_keys, key=lambda k: elite_elo_ratings[k])
                                del elite_elo_ratings[weakest_history_key]
                                history_keys.remove(weakest_history_key)
                            elite_elo_ratings[actor_key] = m_member.elo
                            print(f"Accepted {actor_key} into Elite Pool (Rank: {m_rank:.2f}).")
                    
                    # 无论是否进入精英池，全量表都要记录
                    elo_ratings[actor_key] = m_member.elo

                # 全量日志与快照保存
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

                # -----------------------------------------------------------
                # 逻辑分支 C: 保存“精英池快照” (Elite JSON)
                # -----------------------------------------------------------
                # 这才是下次训练 resume 时应该读取的文件
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = max(m.elo for m in population)
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                
                print(f"Step {total_steps}: All members updated. Max ELO {max(m.elo for m in population):.0f}")

                # =========================================================
                # PBT 进化 (Exploit & Explore)
                # =========================================================
                # 此处暂时将进化频率硬编码为 20 次 batch，之后你可以基于步数或自由调节
                if batch_idx > 0 and batch_idx % interval_of_pbt == 0: # 20 == 0:
                    print(f"\n--- PBT Evolution Step at Batch {batch_idx} ---")
                    
                    # 1. 评估种群 (依据 Elo 分数升序排序)
                    sorted_population = sorted(population, key=lambda m: m.elo)
                    worst_member = sorted_population[0]
                    best_member = sorted_population[-1]
                    
                    print(f"  Best Member: P{best_member.id} (Elo: {best_member.elo:.1f})")
                    print(f"  Worst Member: P{worst_member.id} (Elo: {worst_member.elo:.1f})")
                    
                    # 2. 淘汰与替换 (Exploit & Explore)
                    # 只有最强和最弱差距足够大时才执行进化，避免频繁震荡
                    if best_member.elo - worst_member.elo >= 200:
                        print(f"  -> P{worst_member.id} (Weak) will exploit P{best_member.id} (Best)")
                        
                        # 权重替换 (Exploit)
                        worst_member.agent.actor.net.load_state_dict(best_member.agent.actor.net.state_dict())
                        worst_member.agent.critic.net.load_state_dict(best_member.agent.critic.net.state_dict())
                        
                        # 超参数替换与扰动 (Explore)
                        # 采用最佳成员的 sigma_elo 基础上给予 0.8 或 1.2 倍的扰动
                        mutation = np.random.choice([0.8, 1.2])
                        new_sigma = np.clip(best_member.sigma_elo * mutation, 50, 1000)
                        worst_member.sigma_elo = new_sigma
                        print(f"  -> Hyperparams: Mutated sigma_elo to {new_sigma:.1f} (from {best_member.sigma_elo:.1f})")

                        # 3. 重置优化器 (重要！清楚旧的参数优化动量)
                        worst_member.agent.reset_optimizer()
                        
                        # 同步 Elo 分数
                        worst_member.elo = best_member.elo
                        print(f"  -> Params & Elo of P{worst_member.id} replaced with P{best_member.id}\n")
            
                # -----------------------------------------------------------
                # 逻辑分支 D: 保存名人堂 (hall_of_fame.json)
                # -----------------------------------------------------------
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                # --- 日志记录 (Logging) - 保持不变，展示的是精英池状态 ---
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    pop_max_elo = max(m.elo for m in population)
                    logger.add("Elo/Main_Agent_Raw", pop_max_elo, total_steps)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    denom = float(max_elo - min_elo)
                    if denom == 0.0:
                        rank_pos = 0.5
                    else:
                        rank_pos = float((pop_max_elo - min_elo) / denom)
                    # 现有日志
                    logger.add("Elo_Centered/Current_Rank %", rank_pos*100, total_steps)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps)
                    
                    rule_vals = [v for k, v in valid_elos.items() if k.startswith("Rule")]
                    if rule_vals:
                        r_min, r_max = np.min(rule_vals), np.max(rule_vals)
                        denom = float(r_max - r_min)
                        curr_rank = 0.5 if denom == 0 else (pop_max_elo - r_min) / denom
                        logger.add("Elo_Centered/Current_Rank %", curr_rank * 100, total_steps)
                    
                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps)

                    # 记录详细分数
                    # best_m_id = max(range(pop_size), key=lambda i: population[i].elo)
                    # 记录最强个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", pop_max_elo - mean_elo, total_steps)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps)
                        logger.add(f"Elo_Centered/{rk}", rule_elo - mean_elo, total_steps)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_Best_vs_{rk}", pop_max_elo - rule_elo, total_steps)

        # --- [新增] 达到 max_steps 后的交互逻辑 ---
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        # 【新增】明确告知用户 Worker 当前状态
        print("All simulation workers are now idling safely. System is paused.") 
        
        inp = input(f"Enter new max_steps (current {total_steps}), or press Enter to exit: ")
        
        if not inp.strip():
            print("No input provided. Exiting training.")
            break # 退出外层 while True 循环
        try:
            new_max = int(inp)
            if new_max > total_steps:
                current_max_steps = new_max
                print(f"Continuing training until {current_max_steps} steps.")
            else:
                print("Input steps less than current. Exiting...")
                break # 退出外层 while True 循环
        except ValueError:
            print("Invalid input (not a number). Exiting...")
            break

    # Cleanup
    print("Closing workers...")
    for pipe in pipes:
        try: # 【新增】防止管道已断开导致的报错
            pipe.send(('EXIT', None))
        except:
            pass
            
    for p in workers:
        p.join(timeout=5) # 【修改】给子进程 5 秒优雅退出的时间
        if p.is_alive():
            p.terminate() # 如果没死，强制结束
    
    # 【修改】调整顺序：先尝试关闭测试池
    try:
        test_pool.close()
        test_pool.join()
    except:
        pass
    
    logger.close()
    print("Training Finished.")