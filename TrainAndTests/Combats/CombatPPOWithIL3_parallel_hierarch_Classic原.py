'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

from typing import final
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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new_hierarchical import *
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from VsBaseline_while_training_hierarch import test_worker

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
    td['states'].append(state) # 修改：这里存储Critic用的全局状态
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    td['active_masks'].append(active_mask) # 【新增】active_mask，转入多智能体
    return td

# ==========================================
# 新增：混合缓冲区类
# ==========================================
class IL_transition_buffer:
    def __init__(self, init_dict, max_size=10000):
        """
        内部存储全部采用 List，确保状态和动作在最外层长度绝对对等。
        """
        self.max_size = max_size
        self.addon_dict = {}
        
        # 无论 init_dict 是否为空，都显式初始化所有键，确保 'obs' 一定存在
        src = init_dict if init_dict is not None else {}
        
        # 强制转换成 list 存储。如果 src 里没有 'obs'，就用 'states' 代替
        self.addon_dict['obs'] = list(src.get('obs', src.get('states', [])))
        self.addon_dict['states'] = list(src.get('states', []))
        self.addon_dict['returns'] = list(src.get('returns', []))
        self.addon_dict['actions'] = list(src.get('actions', []))
        
        # 特殊处理：如果 returns 是 torch.tensor (如你截图中所示)，转为 list 存储
        if torch.is_tensor(src.get('returns')):
            self.addon_dict['returns'] = src['returns'].tolist()

        # 打印初始长度以供检查
        print(f"[IL_transition_buffer] Initialized. Size: {len(self.addon_dict['states'])}")
        
    def add(self, data):
        """
        data: 包含 'obs', 'states', 'actions', 'returns' 的字典，值应为 List。
        """
        # 1. 提取新数据并确保格式为 list (防止 data 缺失 'obs'，逻辑同 init)
        new_obs = list(data.get('obs', data.get('states', [])))
        new_states = list(data.get('states', []))
        new_returns = list(data.get('returns', []))
        new_actions = list(data.get('actions', []))
        
        # 处理可能传入的 tensor
        if torch.is_tensor(data.get('returns')):
            new_returns = data['returns'].tolist()
        
        # 使用 .extend() 拼接列表元素
        self.addon_dict['obs'].extend(new_obs)
        self.addon_dict['states'].extend(new_states)
        self.addon_dict['returns'].extend(new_returns)
        self.addon_dict['actions'].extend(new_actions)
        
        # 2. 基于添加顺序的剪裁（保留最后/最新的 max_size 条）
        current_len = len(self.addon_dict['states'])
        max_size_int = int(self.max_size)  # 确保 max_size 是整数
        if current_len > max_size_int:
            keep_from = current_len - max_size_int
            self.addon_dict['obs'] = self.addon_dict['obs'][keep_from:]
            self.addon_dict['states'] = self.addon_dict['states'][keep_from:]
            self.addon_dict['returns'] = self.addon_dict['returns'][keep_from:]
            self.addon_dict['actions'] = self.addon_dict['actions'][keep_from:]
    

    
    def read(self, batch_size):
        """
        随机采样并进行格式转换。
        """
        total_len = len(self.addon_dict['states'])
        if total_len == 0:
            raise ValueError("IL_transition_buffer is empty.")
            
        indices = np.random.randint(0, total_len, size=min(int(batch_size), total_len))
        
        # 采样（列表推导式，保持原始元素格式）
        sampled_obs = [self.addon_dict['obs'][i] for i in indices]
        sampled_states = [self.addon_dict['states'][i] for i in indices]
        sampled_returns = [self.addon_dict['returns'][i] for i in indices]
        sampled_actions_raw = [self.addon_dict['actions'][i] for i in indices]
        
        # 此时才将采样的动作列表转换为算法需要的 dict-of-arrays 格式
        return {
            'obs': np.array(sampled_obs, dtype=np.float32),
            'states': np.array(sampled_states, dtype=np.float32),
            'returns': np.array(sampled_returns, dtype=np.float32),
            'actions': restructure_actions(sampled_actions_raw) 
        }

    def clear(self):
        for k in self.addon_dict:
            self.addon_dict[k] = []
        print("[IL_transition_buffer] Buffer cleared.")
        


# calculate_expected_score 概率计算

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(WinRates=None, Elite_WinRates=None, 
                               SP_type='PFSP_classic', 
                               rule_rate=0.5, p_factor=0.3, deltaFSP_epsilon=0.5):
    """
    经典 PFSP 采样逻辑：
    WinRates: dict, 记录 Learner 对阵各对手的胜率 P
    p_factor: 经典公式 (1-P)^p 中的指数，通常取 1.0~3.0，越高越针对强敌
    """
    # 合并池子以便查询
    if WinRates is None:
        WinRates = {}
    if Elite_WinRates is None:
        Elite_WinRates = {}
    candidate_pool = Elite_WinRates if Elite_WinRates else WinRates
    keys = list(candidate_pool.keys())
    
    if not keys: return np.array([]), []

    # 1. 规则复习分支 (保持不变)
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < rule_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys

    # 2. 经典 PFSP 采样核心
    # 正确判断不同 SP_type，PFSP 核心在 SP_type 为这两者之一时生效
    if SP_type == 'PFSP_classic' or SP_type == 'PFSP_challenge':
        # 获取所有候选对手的胜率 P
        ps = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
        
        # 计算权重：(1 - P)^p
        # 解释：胜率越低（越打不过），权重越高，但太强太弱的对手被过滤掉
        opponent_mask = np.ones_like(ps)
        for i in range(len(ps)):
            if ps[i] > 0.2 and ps[i] < 0.8:
                opponent_mask[i] = 1.0
            else:
                opponent_mask[i] = 0.01
        weights = np.power((1.0 - ps), p_factor)
        weights = weights * opponent_mask
        
        # 防止全为 0 的情况（例如胜率全是 1）
        if weights.sum() < 1e-8:
            probs = np.ones(len(keys)) / len(keys)
        else:
            probs = weights / weights.sum()
            
        return probs, keys

    # 2. 处理 FSP (全样本均匀分布)
    elif SP_type == 'FSP':
        probs = np.ones(len(keys)) / len(keys)
        return probs, keys

    # 3. 处理 deltaFSP (新旧池切分)
    elif SP_type == 'deltaFSP':
        n = len(keys)
        new_count = max(1, int(np.ceil(n * 0.2)))
        new_keys = keys[-new_count:]
        old_keys = keys[:-new_count]
        
        # 这里的 deltaFSP_epsilon 建议直接作为参数传入或使用全局变量
        if np.random.rand() < float(deltaFSP_epsilon) or not old_keys:
            target_keys = new_keys
        else:
            target_keys = old_keys
            
        probs = np.ones(len(target_keys)) / len(target_keys)
        return probs, target_keys

    # 4. 处理 SP (最新历史版本)
    elif SP_type == 'SP':
        # rein_keys = [k for k in keys if k.startswith('actor_rein') and '_step_' not in k]
        # 严格匹配 actor_rein + 数字
        rein_keys = [k for k in keys if re.match(r'^actor_rein\d+$', k)]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            # try: return int(k.replace('actor_rein', ''))
            # except: return -1
            try: return int(re.search(r'actor_rein(\d+)', k).group(1))
            except: return -1
            
        best_key = max(rein_keys, key=extract_number)
        return np.array([1.0]), [best_key]

    # 5. 兜底逻辑: Rule 均匀采样 (None)
    else:
        if not rule_keys: return np.array([]), []
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys


# 辅助：需要把 create_initial_state 定义在 worker 能访问的地方，或者 copy 进去
def create_initial_state_worker(randomized=0):
    # (复制原本的 create_initial_state 逻辑)
    blue_height = np.random.uniform(6000.0, 9000.0) * int(randomized) + \
                9000.0 * (1-int(randomized))
    red_height = blue_height
    # 初始航向随机化
    red_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-pi/3, pi/3))
    blue_psi = sub_of_radian(np.pi/2 + np.random.uniform(-pi/3, pi/3))
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
                   seed, opp_greedy_rate, dt_move=0.05, no_crash=1, pomdp=1, vertices=None):
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

        # args.R_cage = np.random.uniform(30e3, 45e3) # 已移除：放在这里会导致同一个Worker的所有episode的环境大小不变

        # 初始化环境 (关闭可视化以加速)
        env = ChooseStrategyEnv(args, tacview_show=False, vertices=vertices)
        env.shielded = no_crash # 假设默认开启防撞
        env.dt_move = dt_move
        env.dt_maneuver = dt_maneuver
        
        env.no_out = 0 # 训练时该出界必须出界

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
                (actor_weights, opponent_info, settings) = packet
                
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
                # randomized_birth = settings['randomized_birth']  # 改在外面随机，里面不需要
                action_cycle_multiplier = settings['action_cycle_multiplier']
                reward_weight = settings['weight_reward']
                # 在子环境中重新计算出生状态
                # red_birth, blue_birth = create_initial_state_worker(randomized_birth)
                # 使用从master传来的出生状态
                red_birth = settings['red_birth']
                blue_birth = settings['blue_birth']
                
                # 每次重新运行对局前，根据Master指定的范围随机化当前环境大小
                # r_min, r_max = settings.get('R_cage_range', (55.00e3, 55.00e3))
                fire_mask = settings.get('fire_mask', 1)
                # env.R_cage = np.random.uniform(r_min, r_max)
                
                # 进场瞬间给全信息
                env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_ammo=6, blue_init_ammo=6, pomdp=0)
                
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
                    r_obs, r_check_obs = env.obs_1v1('r', pomdp=pomdp)
                    b_obs, b_check_obs = env.obs_1v1('b', pomdp=pomdp)
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
                            # “警报响起，少整活、多保命”
                            b_state_check = env.unscale_state(b_check_obs)
                            if b_state_check["warning"]:
                                temperature = 1 # 0.3
                            else:
                                temperature = 1
                            b_action_exec, _, _, _ = local_agent.take_action(b_obs, explore=1, temperature=temperature, mask_on=fire_mask)
                            b_action_label = b_action_exec['cat'] # [0]
                            b_fire = b_action_exec['bern'][0]
                            
                            # Red Decision
                            r_state_check = env.unscale_state(r_check_obs)
                            if r_state_check["warning"]:
                                temperature = 1 # 0.3
                            else:
                                temperature = 1
                            if adv_is_rule:
                                # 调用规则，假设 basic_rules 已导入
                                r_action_label, r_fire = basic_rules(r_state_check, rule_num, p_random=0.1)
                                r_action_exec = {'cat': r_action_label, 'bern': np.array([r_fire], dtype=np.float32)}
                            else:
                                # 随机决定本局对手是否开启探索
                                adv_explore = 1 if np.random.rand() > opp_greedy_rate else 0
                                r_action_exec, _, _, _ = adv_agent.take_action(r_obs, explore={'cont':0, 'cat':adv_explore, 'bern':1}, temperature=temperature, mask_on=fire_mask)
                                r_action_label = r_action_exec['cat'] #[0]
                                r_fire = r_action_exec['bern'][0]

                        # 2.4 处理开火 (改为置位标志，由后续物理循环尝试发射)
                        b_is_firing = 0
                        r_is_firing = 0
                        if b_fire: 
                            env.BUAV.about_to_fire = 1
                            b_is_firing = env.has_ammo_to_fire('b')
                        if r_fire: 
                            env.RUAV.about_to_fire = 1
                            r_is_firing = env.has_ammo_to_fire('r')
                        
                        # 2.5 记录当前动作供下一帧存储 (初值设为未发射，若后续周期内发射成功则更新)
                        current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                        current_action_exec = {'cat': b_action_exec['cat'], 'bern': np.array([b_is_firing])}
                        current_enm_action_exec = {'cat': r_action_exec['cat'], 'bern': np.array([r_is_firing])}

                    # 3. 物理步进与尝试发射
                     # 采样的时候如果限制动作次序，会妨碍“试错”，到测试时也必须开启  r_action_label  b_action_label None
                    b_m_id = launch_missile_immediately(env, 'b', action_label=b_action_label) if getattr(env.BUAV, 'about_to_fire', 0) else None
                    r_m_id = launch_missile_immediately(env, 'r', action_label=r_action_label) if getattr(env.RUAV, 'about_to_fire', 0) else None
                    
                    if b_m_id: 
                        m_fired += 1
                    if r_m_id:
                        pass
                    
                    # debug
                    if r_action_label[0] > 4:
                        print("数值超出范围", r_action_label[0], r_action_label[1])

                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    steps_run += 1
                    
                    # 4. 奖励计算
                    done, b_reward1, b_reward2, b_reward3 = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier)
                    _, r_reward1, r_reward2, r_reward3 = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier)
                    
                    reward_for_learn = sum(np.array([b_reward1, b_reward2, b_reward3]) * reward_weight)
                    reward_for_enm = sum(np.array([r_reward1, r_reward2, r_reward3]) * reward_weight)
                    
                    if steps_run % action_cycle_multiplier == 0 or done:
                        episode_return += b_reward1
                    
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

                # --- 序列时空修正 (Credit Assignment Fix) ---死后时间压缩
                # 由于代理死亡后 active_mask 变为 False，回合结束时的同归于尽补偿等延迟奖励
                # 无法回传(会乘以 mask=0)。因此将死亡后产生的所有收益收束到最后一个存活步，并截断死亡后的冗余数据。
                def truncate_and_shift(td):
                    if not td or len(td.get('rewards', [])) == 0:
                        return td
                    
                    active = td['active_masks']
                    last_idx = -1
                    for i in range(len(active)-1, -1, -1):
                        if active[i]:
                            last_idx = i
                            break
                    
                    if last_idx != -1 and last_idx < len(td['rewards']) - 1:
                        # 收束后续所有收益到最后一个决策点
                        td['rewards'][last_idx] += sum(td['rewards'][last_idx+1:])
                        td['dones'][last_idx] = True # 将此步标识为事实上的终止步
                        # 不用传送同归于尽的最后一步next_states到先死瞬间，因为 td_target = rewards + self.gamma * next_vals * (1 - dones)
                        
                        # 截断死亡后的垫充帧
                        for k in td.keys():
                            if isinstance(td[k], list):
                                td[k] = td[k][:last_idx+1]
                    return td
                
                local_trans = truncate_and_shift(local_trans)
                ego_trans = truncate_and_shift(ego_trans)
                enm_trans = truncate_and_shift(enm_trans)
                # ----------------------------------------------

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
                    'opp_name': opp_name
                }
                
                # 8. 发送回 Master
                pipe.send(result_packet)

    except Exception: # [新增] 异常捕获与回传
        print(f"!!! Worker {rank} CRASHED !!!")
        tb = traceback.format_exc()
        print(tb)
        try: pipe.send({'error': tb})
        except: pass
            


def run_MLP_simulation(
    k_nonlinear,
    num_workers=10, # 并行进程数，根据CPU核数调整，建议 10-20
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
    alpha_il=1.0,
    il_batch_size=128,
    il_batch_size2=1e4,
    il_buffer_max_size=2e4,
    mini_batch_size_mixed=64,
    beta_mixed=1.0,
    label_smoothing=0.3,
    label_smoothing_mixed=0.01,
    action_cycle_multiplier=30,
    trigger0=50e3,
    trigger_delta=50e3,
    weight_reward_0=None,
    IL_rule=2,
    no_crash=1,
    dt_move=0.05,
    max_episode_duration=15*60,
    R_cage = 62.00e3, # 45e3 # 55e3,
    dt_maneuver=0.2,
    transition_dict_threshold=1000,
    should_kick = True,
    use_init_data = False,
    init_elo_ratings = {
        "Rule_0": 1200,
        "Rule_1": 1200,
        "Rule_2": 1200,
    },
    self_play_type = 'PFSP', # FSP, SP, None 表示非自博弈
    hist_agent_as_opponent = 1, # 是否开始记录历史智能体
    use_sil = True,
    sil_only_maneuver = 1, # 自模仿只包含机动还是也包含开火
    p_factor = 0.3,
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
    max_il_exponent = -2.0,
    k_shape_il = 0.004,
    R_cage_range = (55.00e3, 55.00e3), # 新增：环境随机化范围
    vertices = None,
    resume_dir = None,
    init_il_data = None, # [新增] 从外部传入预拉取的数据集
    POMDP = 0, # 0全信息，1部分信息
):

    actor_lr0 = actor_lr
    critic_lr0 = critic_lr
    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # --- [修改] 灵活加载模仿学习数据集 ---
    if init_il_data is not None:
        print("Using externally provided IL dataset.")
        original_il_transition_dict = copy.deepcopy(init_il_data)
        original_il_transition_dict0 = copy.deepcopy(init_il_data)
    else:
        # 如果外面没传，则走老路子，从本地文件加载
        original_il_transition_dict, _ = load_il_and_transitions(
            os.path.join(cur_dir, "IL"),
            "il_transitions_combat_LR.pkl",
            "transition_dict_combat_LR.pkl"
        )
        original_il_transition_dict0 = copy.deepcopy(original_il_transition_dict)
    
    # 对加载/传入的数据进行必要的重构
    if original_il_transition_dict is not None:
        original_il_transition_dict['actions'] = restructure_actions(original_il_transition_dict['actions'])
        # 顺便确保 states 和 returns 也是标准的 float32 numpy array
        if 'states' in original_il_transition_dict:
            original_il_transition_dict['states'] = np.array(original_il_transition_dict['states'], dtype=np.float32)
        if 'returns' in original_il_transition_dict:
            original_il_transition_dict['returns'] = np.array(original_il_transition_dict['returns'], dtype=np.float32)
        print(f"IL dataset processed. Samples: {len(original_il_transition_dict['states'] if original_il_transition_dict['states'] is not None else [])}")
    
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

    # 3. 创建神经网络
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)

    student_agent = PPOHybrid(
        actor=actor_wrapper, 
        critic=critic_net, 
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
    if resume_dir is not None and os.path.exists(resume_dir):
        log_dir = resume_dir
        print(f"Resuming from directory: {log_dir}")
        IL_epoches = 0  # 中断续训跳过预训练
    else:
        log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
    
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())

    # 中断续训
    if resume_dir is not None and os.path.exists(resume_dir):
        actor_files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
        if len(actor_files) > 0:
            def extract_num(f):
                m = re.search(r'actor_rein(\d+)\.pt$', f)
                return int(m.group(1)) if m else -1
            latest_actor = max(actor_files, key=extract_num)
            student_agent.actor.load_state_dict(torch.load(latest_actor, map_location=device))
            print(f"Loaded actor from: {latest_actor}")
            
            critic_path = os.path.join(log_dir, "critic.pt")
            if os.path.exists(critic_path):
                student_agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print(f"Loaded critic from: {critic_path}")
            
            opt_path = os.path.join(log_dir, "optimizers_state.pt")
            if os.path.exists(opt_path):
                try:
                    opt_states = torch.load(opt_path, map_location=device)
                    student_agent.actor_optimizer.load_state_dict(opt_states['actor_optimizer'])
                    student_agent.critic_optimizer.load_state_dict(opt_states['critic_optimizer'])
                    print("Loaded optimizer states.")
                except Exception as e:
                    print(f"Failed to load optimizers: {e}")
    
    # 保存onnx模型
    # 前提：假设此时 student_agent 已经创建好，且 state_dim 已经定义
    # 构建一个与 state 维度相同的 dummy input (batch_size=1)
    dummy_state = torch.randn(1, state_dim).to(device)
    # ==========================================
    # 1. 导出 Actor 的底层网络（PolicyNetHybrid）
    # ==========================================
    actor_onnx_path = os.path.join(log_dir, "student_actor.onnx")
    # 对于你的 PolicyNetHybrid，它返回的是一个 dict {'cont': ..., 'cat': ..., 'bern': ...}
    # 在高版本的 PyTorch 中，ONNX 对返回 dict 有支持（自动解包），或者你可以写一个简单的 wrapper 解包
    try:
        torch.onnx.export(
            student_agent.actor.net,           # 只导出纯网络结构，避开 Wrapper里的复杂采样操作
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
            student_agent.critic,              
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

    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (Serial Execution on Master)
    if IL_epoches > 0:
        print("Start IL Training...")

    student_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    # === 模仿训练循环 ===
    # 现在 original_il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, # 显存如果够大可以适当调大
            label_smoothing=label_smoothing,
            no_bern = 0,
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            # [新增] MARWIL 监控：每个动作头独立的 NLL 与策略熵 (全采样固定 batch, no_grad)
            for _name, _val in [
                ("il_train/nll_cont", getattr(student_agent, "marwil_nll_cont", None)),
                ("il_train/nll_cat", getattr(student_agent, "marwil_nll_cat", None)),
                # ("il_train/nll_bern", getattr(student_agent, "marwil_nll_bern", None)),
                ("il_train/entropy_cont", getattr(student_agent, "marwil_entropy_cont", None)),
                ("il_train/entropy_cat", getattr(student_agent, "marwil_entropy_cat", None)),
                # ("il_train/entropy_bern", getattr(student_agent, "marwil_entropy_bern", None)),
                ("il_train/weight_mean", getattr(student_agent, "marwil_weight_mean", None)),
                # ("il_train/weight_max", getattr(student_agent, "marwil_weight_max", None)),
                # ("il_train/weight_min", getattr(student_agent, "marwil_weight_min", None)),
                ("il_train/weight_clip_frac", getattr(student_agent, "marwil_weight_clip_frac", None)),
                ("il_train/adv_std", getattr(student_agent, "marwil_adv_std", None)),
                # ("il_train/adv_p95", getattr(student_agent, "marwil_adv_p95", None)),
                ("il_train/adv_max", getattr(student_agent, "marwil_adv_max", None)),
                ("il_train/adv_mean", getattr(student_agent, "marwil_adv_mean", None)),
                ("il_train/adv_positive_frac", getattr(student_agent, "marwil_adv_positive_frac", None)),
            ]:
                if _val is not None:
                    logger.add(_name, _val, epoch)

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")
    
    if IL_epoches > 0:
        print("IL Training Finished.")
    else:
        print("No IL")
    
    # 存储在线训练前的网络参数
    int_agent_name = "actor_rein0"
    torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{int_agent_name}.pt"))


    # --- 新增：实例化混合缓冲区 ---
    il_transition_buffer = None
    if IL_epoches + use_sil > 0:  # 只要出现模仿学习就得准备好初始的模仿池
        print("Initializing IL Transition Buffer...")
        original_data_input = original_il_transition_dict0 if use_init_data else None
        il_transition_buffer = IL_transition_buffer(original_data_input, max_size=il_buffer_max_size)

    # ==============================================================================
    # 强化学习 (Self-Play / PFSP) 阶段
    # ==============================================================================
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
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
    
    args.max_episode_len = max_episode_duration
    # args.R_cage = 45e3 # np.random.uniform(30e3, 45e3) # 环境大小随机化
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, 
                       kwargs={
                           'rank': i,
                           'pipe': child_conn,
                           'args': args,
                           'state_dim': state_dim,
                           'hidden_dim': hidden_dim,
                           'action_dims_dict': action_dims_dict,
                           'device_worker': worker_device,
                           'dt_maneuver': dt_maneuver,
                           'seed': seed,
                           'opp_greedy_rate': opp_greedy_rate,
                           'dt_move': dt_move,
                           'no_crash': no_crash,
                           'pomdp': POMDP,
                           'vertices': vertices,
                       })
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = {} # {copy.deepcopy(elo_ratings)}
    hall_of_fame = {}
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")
    # 新增：胜率表文件路径
    WinRates_path = os.path.join(log_dir, "WinRates.json")
    Elite_WinRates_path = os.path.join(log_dir, "Elite_WinRates.json")
    # 初始化胜率字典
    WinRates = {}
    Elite_WinRates = {}

    # 尝试加载历史 # 中断续训
    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f: elo_ratings = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f: hall_of_fame = json.load(f)
    if os.path.exists(WinRates_path):
        with open(WinRates_path, 'r') as f: WinRates = json.load(f)
    if os.path.exists(Elite_WinRates_path):
        with open(Elite_WinRates_path, 'r') as f: Elite_WinRates = json.load(f)

    main_agent_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)

    
    # 不论如何，记录在线训练前的网络参数
    if (not elite_elo_ratings): # 如果是从零开始训练的
        # 初始对手胜率一律当做0.5
        for k in init_elo_ratings.keys():
            WinRates[k] = 0.5
        # 初始对手Elo一律当1200
        for k in init_elo_ratings.keys():
            elo_ratings[k] = main_agent_elo    

        # 自博弈开启时的初始分值填充
        if hist_agent_as_opponent:
            elo_ratings[int_agent_name] = main_agent_elo
            WinRates[int_agent_name] = 0.5
        
        # 从零开始不论对手有多烂都要加入Elite池            
        Elite_WinRates = copy.deepcopy(WinRates)
        elite_elo_ratings = copy.deepcopy(elo_ratings)

    
    # # 初始对手(存储IL后的网络参数)
    # if (not elo_ratings) and hist_agent_as_opponent: # or IL_epoches > 0: # 没有模仿学习也要存
    #     elo_ratings[int_agent_name] = 1200
        
    # # 初始化 init_elo_ratings 里面的元素胜率 以及 actor_rein0
    # if not WinRates:
    #     for k in init_elo_ratings.keys():
    #         WinRates[k] = 0.5
    #     WinRates["actor_rein0"] = 0.5

    # 训练循环变量
    total_steps = elo_ratings.get("__LAST_UPDATE_STEP__", 0)
    batch_idx = elo_ratings.get("__LAST_UPDATE_BATCH__", 0)
    trigger = trigger0 + (total_steps // trigger_delta) * trigger_delta
    
    current_max_steps = int(max_steps)
    
    # 全局 Buffer (用于攒够 Batch 训练)
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    transition_dict = copy.deepcopy(empty_transition_dict)

    # 初始化基于胜率的在线 EMA 变量
    ema_score = 0.5
    ema_step = 0
    target_p1 = 0.65

    # =========================================================
    # 主循环 (Master Process)
    # =========================================================
    while True:
        while total_steps < current_max_steps:
            # 先尝尝乱开或的后果，再mask掉错误开火
            if total_steps < 5e3:
                fire_mask = 1 # 0 # 全程开启开火mask
            else:
                fire_mask = 1
            # --- 【修改】同步并行测试阶段 ---
            # 只有测试跑完并处理完名人堂，才进入下一步的采样和仿真
            # --- 1. 并行测试触发逻辑 (Async) ---
            if total_steps >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
                # 1. 深度拷贝当前 Actor 权重到 CPU 内存
                current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                test_tasks = []
                for r_idx in [0, 1, 2, 3, 4]:
                    obj = test_pool.apply_async(
                        test_worker, 
                        # args=(current_weights, r_idx, args, 
                        #     state_dim, hidden_dim, action_dims_dict, 
                        #     dt_maneuver, 'cpu', num_runs, action_cycle_multiplier)
                        kwds={
                            'model_state_dict': current_weights,
                            'rule_num': r_idx,
                            'env_args': args,
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dims_dict': action_dims_dict,
                            'dt_maneuver_val': dt_maneuver,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'no_out': 0,  # 这里可以根据需要设为 1
                            'deterministic': False,
                            'restrict_fire': True, # False, 和采样保持一致
                            'vertices': vertices,
                        }
                    )
                    test_tasks.append(obj)
                
                # 第二种形式：追加额外测试 (机动动作确定化 + 动作次序限制打开)
                test_tasks_no_random = []
                for r_idx in [0, 1, 2, 3, 4]:
                    obj = test_pool.apply_async(
                        test_worker, 
                        kwds={
                            'model_state_dict': current_weights,
                            'rule_num': r_idx,
                            'env_args': args,
                            'state_dim': state_dim,
                            'hidden_dim': hidden_dim,
                            'action_dims_dict': action_dims_dict,
                            'dt_maneuver_val': dt_maneuver,
                            'device_name': 'cpu',
                            'num_runs': num_runs,
                            'action_cycle_multiplier': action_cycle_multiplier,
                            'no_out': 0,
                            'deterministic': True,     # 机动动作确定化
                            'restrict_fire': True      # 动作次序限制打开
                        }
                    )
                    test_tasks_no_random.append(obj)

                # 等待所有测试进程结束
                test_results = [t.get() for t in test_tasks]
                test_results_no_random = [t.get() for t in test_tasks_no_random]

                # 记录第一种测试结果
                outcomes = {rule_num: score for rule_num, score, result2, wins, loses, draws in test_results}
                outcomes_return = {rule_num: result2 for rule_num, score, result2, wins, loses, draws in test_results}

                for r_num, score in outcomes.items():
                    logger.add(f"test/agent_vs_rule{r_num}", score, total_steps)
                    # logger.add(f"test/agent_vs_rule{r_num}_return", outcomes_return[r_num], total_steps)
                    print(f"  [Test Result] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

                # 记录第二种测试结果 (test_No_random)
                outcomes_nr = {rule_num: score for rule_num, score, result2, wins, loses, draws in test_results_no_random}
                outcomes_return_nr = {rule_num: result2 for rule_num, score, result2, wins, loses, draws in test_results_no_random}

                for r_num, score in outcomes_nr.items():
                    logger.add(f"test_No_random/agent_vs_rule{r_num}", score, total_steps)
                    # logger.add(f"test_No_random/agent_vs_rule{r_num}_return", outcomes_return_nr[r_num], total_steps)
                    print(f"  [Test No Random] Rule_{r_num}: {score} (return: {outcomes_return_nr[r_num]:.2f})")

                # 名人堂判定：如果全胜则保存并加入池子
                if all(score > 0.5 for score in outcomes.values()):
                    # 【核心修改】从全量注册表 elo_ratings 中寻找最新的已保存编号
                    rein_keys = [k for k in elo_ratings.keys() if re.match(r'^actor_rein\d+$', k)]
                    
                    if rein_keys:
                        # 找到数值最大的编号（即最新的已保存智能体）
                        hof_key = max(rein_keys, key=lambda k: int(k.replace('actor_rein', '')))
                        
                        if hof_key not in hall_of_fame:
                            # 存入字典，分数优先取全量表中的记录
                            hall_of_fame[hof_key] = elo_ratings.get(hof_key, main_agent_elo)
                            print(f"!!! [Hall of Fame] New Hero Captured: {hof_key}")
                
                trigger += trigger_delta

            # --- 2. 准备训练 Batch (Synchronous) ---
            

            # A. 获取当前策略权重 (CPU)
            current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
            
            # B. 分发任务给 Worker
            # 这一步 Master 决定每个 Worker 打谁
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            # # [修正] 处理纯自博弈逻辑：当没有初始规则对手时，筛选分数最高的 MAX_HISTORY_SIZE 个对手作为匹配池
            # if not init_elo_ratings:
            #     # 按照 Elo 分数降序排列，排除内部特殊键
            #     sorted_all_keys = [k for k in sorted(elo_ratings.keys(), 
            #                                     key=lambda x: elo_ratings[x] if not x.startswith("__") else -1e9, 
            #                                     reverse=True) if not k.startswith("__")]
            #     effective_pool = {k: elo_ratings[k] for k in sorted_all_keys[:MAX_HISTORY_SIZE]}
            # else:
            #     effective_pool = elite_elo_ratings
                
            for rank in range(num_workers):
                # 采样对手
                # 注意：采样始终基于 win_rates（Elite_WinRates 仅为记录用）
                probs, opponent_keys = get_opponent_probabilities(
                    WinRates,
                    Elite_WinRates,
                    SP_type=self_play_type,
                    rule_rate=rule_actor_rate,
                    p_factor=p_factor,
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
                    'blue_birth': bb,
                    # 'R_cage_range': R_cage_range, # 将范围传给Worker
                    'fire_mask': fire_mask,
                }
                
                # 发送指令 pipe.send
                pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # C. 等待所有 Worker 完成 (Barrier)
            batch_results = []
            for rank in range(num_workers):
                try: # <--- 【新增】
                    res = pipes[rank].recv() # 阻塞等待  # pipe.send
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
            batch_total_steps = 0
            batch_wins = 0
            batch_loss_cnt = 0
            batch_draw_cnt = 0        # 新增统计
            batch_total_return = 0    # 新增统计
            batch_total_m_fired = 0   # 新增统计
            
            for res in batch_results:
                # res 结构: {'trans':..., 'ego_tr':..., 'enm_tr':..., 'metrics':..., 'opp_name':...}
                l_tr = res['trans'] # PPO 训练数据 (含探索)
                ego_tr = res['ego_trans'] # SIL 蓝方数据
                enm_tr = res['enm_trans'] # SIL 红方数据
                metrics = res['metrics']
                opp_name = res['opp_name']
                
                # [新增] 填充 buffer 用户打印详情
                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(f"{opp_name}: {result_str}")
                
                batch_total_steps += metrics['steps']
                batch_total_return += metrics['return']
                batch_total_m_fired += metrics['m_fired']

                if metrics['win']: batch_wins += 1
                elif metrics['lose']: batch_loss_cnt += 1
                else: batch_draw_cnt += 1
                
                # 3.1 聚合 PPO 数据到全局 Buffer
                for k in transition_dict:
                    transition_dict[k].extend(l_tr[k])
                
                # 3.2 SIL 数据收集 (需计算 return)
                if use_sil:
                    ego_tr['returns'] = compute_monte_carlo_returns(gamma, ego_tr['rewards'], ego_tr['dones'])
                    il_transition_buffer.add(ego_tr)  # 优化无望，改回原论文做法用来对比

                    # if not metrics['lose']: # 赢或平，学自己
                    #     # 计算回报 (Master 端计算)
                    #     ego_tr['returns'] = compute_monte_carlo_returns(gamma, ego_tr['rewards'], ego_tr['dones'])
                    #     il_transition_buffer.add(ego_tr)
                    
                    # if not metrics['win']: # 输或平，学对手
                    #     enm_tr['returns'] = compute_monte_carlo_returns(gamma, enm_tr['rewards'], enm_tr['dones'])
                    #     il_transition_buffer.add(enm_tr)
                
                # 3.3 ELO 与 胜率 更新 (实时更新)
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                
                # --- 胜率更新 (经典滑动平均) ---
                alpha_win = 0.1
                # 全记录
                # 需要记录的是我对这个对手的胜率，数值越高表示对手相对我越弱
                if opp_name in WinRates:
                    old_p = WinRates[opp_name]
                    WinRates[opp_name] = (1 - alpha_win) * old_p + alpha_win * actual_score
                    # 同步更新 Elite 池中已有的对手
                    if opp_name in Elite_WinRates:
                        Elite_WinRates[opp_name] = WinRates[opp_name]
                else:
                    # 只有 hist_agent_as_opponent=1 时，才允许 WinRates 增长
                    if hist_agent_as_opponent:
                        WinRates[opp_name] = 0.5
                
                # --- Elo 分值更新 (始终扩充，不受 hist_agent_as_opponent 限制) ---
                if opp_name in elo_ratings:
                    prev_main_elo = main_agent_elo
                    adv_elo = elo_ratings[opp_name]
                    
                    # 更新主智能体Elo分
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                    # 更新对手Elo分
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                    elo_ratings[opp_name] = new_adv_elo
                    elo_ratings["__CURRENT_MAIN__"] = main_agent_elo
                    # 同步更新 Elite 池中已有的对手
                    if opp_name in elite_elo_ratings:
                        elite_elo_ratings[opp_name] = new_adv_elo
                else:
                    # 新对手：始终添加到 elo_ratings
                    elo_ratings[opp_name] = main_agent_elo
            
            # [新增] 在 PPO 更新前打印本轮详细战况
            if batch_idx % 1 == 0:
                print(f"  [Batch {batch_idx}] Results: {', '.join(worker_metrics_buffer)}")

            # 更新全局计数
            total_steps += batch_total_steps
            batch_idx += 1
            
            # --- 3.5 记录批次聚合指标 ---
            # 更新在线胜率 EMA 并且加入偏差修正（既稳又抗滞后）
            batch_score = (batch_wins + batch_draw_cnt * 0.5) / num_workers
            ema_step += 1
            alpha_ema = 0.1  # 平滑系数，较大可防止滞后，和tensorboard的smoothing是补数的关系
            # 若第一步则直接初始化，避免前期偏差
            if ema_step == 1:
                ema_score = batch_score
            else:
                ema_score = (1 - alpha_ema) * ema_score + alpha_ema * batch_score
                

            # 使用带有偏差修正的滤波值
            filtered_score = ema_score
            logger.add("train_plus/batch_score", batch_score, total_steps)
            logger.add("train_plus/filtered_score", filtered_score, total_steps)
            logger.add("train_plus/target_p1", target_p1, total_steps)

            # 记录导弹发射平均数量或总数
            logger.add("special/0 发射的导弹总数", batch_total_m_fired, total_steps)
            # 记录平均回报与胜率
            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps)
            logger.add("train/2 win", batch_wins / num_workers, total_steps)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps)
            # 找最好的智能体
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps)
            logger.add("agent/ batch_step", batch_idx, total_steps)

            # --- 5. 更新，保存与维护 (Checkpoint & Pool) ---
            if batch_idx % save_interval == 0 and \
                len(transition_dict['dones']) >= transition_dict_threshold:
                # # --- 4. 执行训练 (PPO Update) ---
                # # 当收集的数据量超过 capacity 时更新
                # if len(transition_dict['dones']) >= transition_dict_threshold:
                # 重构 Action 结构 (List[Dict] -> Dict[Array])
                transition_dict['actions'] = restructure_actions(transition_dict['actions'])
                
                '记录ELo相对位置'
                # [新增] 调节alpha_il
                # --- [新增] 动态计算 alpha_il ---
                # 1. 筛选对手池：Rule开头的所有Key + actor_rein开头的最后300个Key
                all_keys = list(elo_ratings.keys())
                rule_keys = [k for k in all_keys if k.startswith('Rule')]
                rein_keys = [k for k in all_keys if k.startswith('actor_rein')]
                # 取最后（最新插入）的300个
                latest_rein_keys = rein_keys[-300:] if len(rein_keys) > 300 else rein_keys
                
                target_pool_keys = rule_keys + latest_rein_keys
                # 计算池子滑动平均分
                avg_pool_elo = np.mean([elo_ratings[k] for k in target_pool_keys])
                # 计算 Elo 差值 x (当前主分 - 池子均分)
                x_elo_diff = main_agent_elo - avg_pool_elo
                logger.add("train_plus/elo_diff_x", x_elo_diff, total_steps)
                
                # 学习率warm_up
                actor_lr = min(actor_lr0, actor_lr0 * total_steps/1e6)
                critic_lr = min(critic_lr0, critic_lr0 * total_steps/1e6)
                student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

                student_agent.update(transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed, target_p1=target_p1, 
                                     k_nonlinear=k_nonlinear, mask_on=fire_mask)
                #====================
                # 记录 Log

                # [Modification] 保留原有梯度监控代码
                actor_pre_clip_grad = student_agent.pre_clip_actor_grad
                critic_pre_clip_grad = student_agent.pre_clip_critic_grad

                # 梯度监控
                logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
                logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
                # 损失函数监控
                logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
                logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
                # 强化学习actor特殊项监控
                logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
                logger.add("train/9 entropy_cat", student_agent.entropy_cat, total_steps)
                logger.add("train/9 entropy_bern", student_agent.entropy_bern, total_steps)
                
                logger.add("train/10 advantage", student_agent.advantage, total_steps) 
                # 强化学习
                logger.add("train/10 explained_var", student_agent.explained_var, total_steps)
                logger.add("train/10 approx_kl", student_agent.approx_kl, total_steps)
                logger.add("train/10 clip_frac", student_agent.clip_frac, total_steps)
                
                # [新增] 诊断监控
                logger.add("train_plus/td_error_var", student_agent.td_error_var, total_steps)
                logger.add("train_plus/grad_norm_ratio", student_agent.grad_norm_ratio, total_steps)
                
                # IL-PPO信号强度对比
                # 错误做法，更新强度数量级和样本数无关
                # if use_sil:
                #     logger.add("train_plus/原始信号强度对比IL-PPO", student_agent.IL_samples/student_agent.PPO_samples*alpha_il, total_steps)
                #     logger.add("train_plus/滤波后信号强度对比IL-PPO", student_agent.IL_valid_samples/student_agent.PPO_valid_samples*alpha_il, total_steps)
                    
                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # 清空 Buffer
                transition_dict = copy.deepcopy(empty_transition_dict)
                
                # A. 保存模型
                actor_key = f"actor_rein{batch_idx}"
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{actor_key}.pt"))
                torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                print(f"Saved Checkpoint: {actor_key}")

                # B. 经典胜率精英池维护
                # 只有自博弈能够更新精英Elo和胜率表，否则只能更新普通胜率和Elo表
                if total_steps >= WARM_UP_STEPS:
                    # 1. 产生新 Checkpoint 时，初始化其在胜率表中的地位
                    WinRates[actor_key] = 0.5  # Learner 的镜像初始胜率为 0.5
                    elo_ratings[opp_name] = main_agent_elo

                    if hist_agent_as_opponent:
                        agent_keys = [k for k in WinRates.keys() if k.startswith("actor_rein")]
                        sorted_agents = sorted(agent_keys, key=lambda k: WinRates[k])
                        # 计算 rule 的最大 win_rate（即 Learner 对规则对手中赢得最多的那个值）
                        # 新的门槛规则：候选 NN 必须比最弱的 Rule（对 Learner 来说最容易打的 Rule）更难
                        # 换句话说：要求 WinRates[agent] < max_rule_win
                        rule_keys_present = [k for k in WinRates.keys() if k.startswith('Rule')]
                        max_rule_win = None
                        if rule_keys_present:
                            max_rule_win = max([WinRates[k] for k in rule_keys_present])

                        # 过滤出满足硬性门槛的 agents（如果有 rule 则必须使 Learner 胜率小于 rules 中的最高值）
                        qualified_agents = []
                        for a in sorted_agents:
                            if max_rule_win is None:
                                qualified_agents.append(a)
                            else:
                                # 要求：Learner 对 agent 的胜率 < rules 中的最高 Learner 胜率
                                if WinRates.get(a, 1.0) < float(max_rule_win):
                                    qualified_agents.append(a)

                        toughest_agents = qualified_agents[:MAX_HISTORY_SIZE]

                        # # --- 新增逻辑：同步移除数值过高的“过期”Agent ---
                        # if max_rule_win is not None:
                        #     purge_threshold = float(max_rule_win) + 0.1
                        #     # 仅保留胜率没超过阈值的 NN 智能体
                        #     toughest_agents = [a for a in toughest_agents if WinRates.get(a, 1.0) <= purge_threshold]
                        # # ----------------------------------------------

                        # 3. 更新 Elite 表格（仅包含通过硬性门槛的 NN），并保证 init_elo_ratings 中的 Rule 被保留
                        Elite_WinRates = {k: WinRates[k] for k in toughest_agents}
                        # 根据传入的 init_elo_ratings 决定加入的 Rule（保证按需包含）
                        for rk in [k for k in init_elo_ratings.keys() if k.startswith("Rule")]:
                            if rk in WinRates:
                                Elite_WinRates[rk] = WinRates[rk]
                            
                        # 4. 让 Elite_Elo_ratings 从属于 Elite_WinRates (用于对比记录)
                        elite_elo_ratings = {k: elo_ratings.get(k, 1200) for k in Elite_WinRates.keys()}
                        print(f"Accepted {actor_key}. Elite pool size: {len(Elite_WinRates)} (NN: {len(toughest_agents)})")
                    
                # -----------------------------------------------------------
                # 逻辑分支 B: 维护“全量历史记录” (Full JSON)
                # -----------------------------------------------------------
                # 目标：记录所有产生过的 Agent 的最后一次已知 Elo，无论它是否在精英池里
                # 无论是否进入精英池，全量表都要记录
                elo_ratings[actor_key] = main_agent_elo
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                
                # 5. 保存全量日志
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

                # -----------------------------------------------------------
                # 逻辑分支 C: 保存“精英池快照” (Elite JSON)
                # -----------------------------------------------------------
                # 这才是下次训练 resume 时应该读取的文件
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f:
                    json.dump(save_elite, f, ensure_ascii=False, indent=2)
                
                # -----------------------------------------------------------
                # 逻辑分支 D: 保存名人堂 (hall_of_fame.json)
                # -----------------------------------------------------------
                with open(hof_json_path, "w", encoding="utf-8") as f:
                    json.dump(hall_of_fame, f, ensure_ascii=False, indent=2)

                # -----------------------------------------------------------
                # 逻辑分支 E: 保存胜率表 (用于核心更新与选择)
                # -----------------------------------------------------------
                with open(WinRates_path, "w", encoding="utf-8") as f:
                    json.dump(WinRates, f, ensure_ascii=False, indent=2)
                with open(Elite_WinRates_path, "w", encoding="utf-8") as f:
                    json.dump(Elite_WinRates, f, ensure_ascii=False, indent=2)

                # --- 日志记录 (Logging) - 保持不变，展示的是精英池状态 ---
                valid_elos = {k: v for k, v in elite_elo_ratings.items() if not k.startswith("__")}
                if valid_elos:
                    mean_elo = np.mean(list(valid_elos.values()))
                    # 排序 (Rule 在前，rein 按数字) - 简单按 key 字符串排序即可，或者 lambda
                    # 这里为了简单，直接遍历
                    # sorted_keys = sorted(valid_elos.keys())
                    
                    logger.add("Elo/Main_Agent_Raw", main_agent_elo, total_steps)
                    
                    # 记录主智能体在当前所有 ELO 中的归一化排名位置：
                    # (主elo - min_elo) / (max_elo - min_elo)，当分母为0时取0.5
                    min_elo = np.min(list(valid_elos.values()))
                    max_elo = np.max(list(valid_elos.values()))
                    
                    rule_elos = [v for k, v in elite_elo_ratings.items() if k.startswith("Rule")]
                    if not rule_elos:
                        min_rule_elo, max_rule_elo = main_agent_elo, main_agent_elo
                    else:
                        min_rule_elo = np.min(rule_elos)
                        max_rule_elo = np.max(rule_elos)
                    rule_elo_thres = ADMISSION_THRESHOLD * max_rule_elo +\
                        (1-ADMISSION_THRESHOLD) * min_rule_elo

                    # 记录与 Rule 阈值的差值 (维持旧指标名)
                    elo_diff_to_thres = main_agent_elo - rule_elo_thres
                    logger.add("Elo_Centered/EloDiffToMean", elo_diff_to_thres, total_steps)
                    
                    # 新增：记录 ELO 极差（max - min），用于判断 PFSP sigma 是否合适...
                    elo_spread = float(max_elo - min_elo)
                    print('elo分极差：', elo_spread)
                    logger.add("Elo/Spread", elo_spread, total_steps)

                    curr_rank = 0.5 if elo_spread == 0 else (main_agent_elo - min_elo) / elo_spread

                    # Elo分数没有稳定不许记录
                    if total_steps >= WARM_UP_STEPS:
                        logger.add("Elo_Centered/Current_rank_normed %", curr_rank * 100, total_steps)
                    

                    # # 动态学习率调节
                    # actor_lr = 1e-4 + np.clip(curr_rank, 0, 1) * (1e-5 - 1e-4)
                    # critic_lr = actor_lr * 5
                    # student_agent.set_learning_rate(actor_lr, critic_lr)

                    hist_count = len([k for k in valid_elos if not k.startswith("Rule")])
                    logger.add("Elo/History_Pool_Size", hist_count, total_steps)

                    # 记录详细分数
                    # 记录最新个体的绝对分和相对均值的居中分
                    logger.add("Elo_Centered/Latest_Best", main_agent_elo - mean_elo, total_steps)

                    # 3. 提取所有的 Rule 对手并遍历，一步到位记录 Rule信息 和 差值
                    rule_keys = [k for k in valid_elos.keys() if k.startswith("Rule_")]
                    for rk in sorted(rule_keys):
                        rule_elo = float(valid_elos[rk])
                        
                        # 记录 Rule 的绝对分与居中分
                        logger.add(f"Elo_Raw/{rk}", rule_elo, total_steps)
                        
                        # 直接计算并记录 最强个体 vs 规则对手 的差值
                        logger.add(f"Elo_Diff/Latest_vs_{rk}", main_agent_elo - rule_elo, total_steps)

                # --- 例行保存优化器状态和步数 ---
                torch.save({
                    'actor_optimizer': student_agent.actor_optimizer.state_dict(),
                    'critic_optimizer': student_agent.critic_optimizer.state_dict(),
                }, os.path.join(log_dir, "optimizers_state.pt"))
                # print(f"Optimizers routinely saved to optimizers_state.pt")
                elo_ratings["__LAST_UPDATE_STEP__"] = total_steps
                elo_ratings["__LAST_UPDATE_BATCH__"] = batch_idx
                with open(full_json_path, "w", encoding="utf-8") as f:
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)

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