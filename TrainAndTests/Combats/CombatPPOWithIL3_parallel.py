'''
并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
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

dt_move = 0.05

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
        if current_len > self.max_size:
            keep_from = current_len - self.max_size
            self.addon_dict['obs'] = self.addon_dict['obs'][keep_from:]
            self.addon_dict['states'] = self.addon_dict['states'][keep_from:]
            self.addon_dict['returns'] = self.addon_dict['returns'][keep_from:]
            self.addon_dict['actions'] = self.addon_dict['actions'][keep_from:]
    
        # # 2. 基于 Return 的原子化排序与剪裁
        # # 通过 zip 绑定每一行数据，确保 S, A, R 在排序和删除时永远同步
        # combined = list(zip(
        #     self.addon_dict['obs'],
        #     self.addon_dict['states'],
        #     self.addon_dict['returns'],
        #     self.addon_dict['actions']
        # ))
        # # 按 returns (索引为 2) 降序排列（从高质量到低质量）
        # combined.sort(key=lambda x: x[2], reverse=True)
        
        # # 保留前 max_size 条高质量经验
        # top_data = combined[:self.max_size]
        
        # # 解包回列表
        # unzipped = list(zip(*top_data)) if top_data else ([], [], [], [])
        # self.addon_dict['obs'] = list(unzipped[0])
        # self.addon_dict['states'] = list(unzipped[1])
        # self.addon_dict['returns'] = list(unzipped[2])
        # self.addon_dict['actions'] = list(unzipped[3])
    
    def read(self, batch_size):
        """
        随机采样并进行格式转换。
        """
        total_len = len(self.addon_dict['states'])
        if total_len == 0:
            raise ValueError("IL_transition_buffer is empty.")
            
        indices = np.random.randint(0, total_len, size=batch_size)
        
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


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame_keys=None, target_elo=None, sigma=400, SP_type='PFSP_with_delta', rule_rate=0.5, deltaFSP_epsilon=0.5, ):
    """
    优化后的对手采样逻辑：
    1. 优先判定是否进入“规则复习”分支。
    2. 若未进入，则根据 SP_type 执行具体的采样策略。
    """
    keys = list(elite_elo_ratings.keys())
    if not keys:
        return np.array([]), []
    
    # 2. [新增] 简单合并：将名人堂成员塞进候选名单
    # 这样它们就会根据在 elo_ratings.json 里的分数参与正常的 PFSP 采样
    for hof_key in hall_of_fame_keys:
        if hof_key not in keys:
            keys.append(hof_key)

    # --- 第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护) ---
    # 只要 rule_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < rule_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys

    # --- 第二层判断：进入核心采样逻辑 ---
    elos = np.array([elite_elo_ratings[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = np.max(elos)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
        diffs = elos - actual_target
        scores = np.exp(-0.5 * (diffs / float(sigma))**2)
        probs = scores / (scores.sum() + 1e-12)
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

    # 4. 处理 SP (最强/最新历史版本)
    elif SP_type == 'SP':
        rein_keys = [k for k in keys if k.startswith('actor_rein')]
        if not rein_keys: return np.array([]), []
        
        def extract_number(k):
            try: return int(k.replace('actor_rein', ''))
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
    blue_height = 9000
    red_height = 9000
    red_psi = -np.pi/2
    blue_psi = np.pi/2
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N = init_North
    red_E = 45e3
    blue_N = init_North
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE


def worker_process(rank, pipe, args, state_dim, hidden_dim, action_dims_dict, device_worker, dt_maneuver, seed):
    """
    常驻子进程：接收参数 -> 跑完一整场 -> 返回数据 -> 等待
    完整的 Worker 逻辑：包含环境初始化、模型加载、仿真循环、数据回传
    """
    # --- 1. 初始化阶段 (只运行一次) ---
    import random
    # 确保每个进程种子不同，避免所有环境生成完全一样的随机数
    worker_seed = seed + rank * 1000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    # 初始化环境 (关闭可视化以加速)
    env = ChooseStrategyEnv(args, tacview_show=False)
    env.shielded = 1 # 假设默认开启防撞
    env.dt_move = 0.05 
    env.dt_maneuver = dt_maneuver

    # 初始化本地网络 (CPU)
    local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
    local_agent = PPOHybrid(
        actor=HybridActorWrapper(local_actor, action_dims_dict, None, device_worker).to(device_worker),
        critic=None, 
        actor_lr=0, critic_lr=0, device=device_worker # Worker不训练，LR无所谓
    )
    
    # 初始化对手网络
    adv_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
    adv_agent = PPOHybrid(
        actor=HybridActorWrapper(adv_actor, action_dims_dict, None, device_worker).to(device_worker),
        critic=None, 
        actor_lr=0, critic_lr=0, device=device_worker
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
            randomized_birth = settings['randomized_birth']
            action_cycle_multiplier = settings['action_cycle_multiplier']
            reward_weight = settings['weight_reward']
            
            red_birth, blue_birth = create_initial_state_worker(randomized_birth)
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
                        # 确保 append_experience 在 worker_process 作用域外是可见的，或者复制进来
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
                            r_action_label, r_fire = basic_rules(r_state_check, rule_num, last_action=0)
                            r_action_exec = {'cat': np.array([r_action_label]), 'bern': np.array([r_fire], dtype=np.float32)}
                        else:
                            # 设定一个对手贪婪率，比如 0.5
                            opp_greedy_rate = 0.5 
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
                
                if steps_run % action_cycle_multiplier == 0:
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
                'opp_name': opp_name
            }
            
            # 8. 发送回 Master
            pipe.send(result_packet)
            
            


def run_MLP_simulation(
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
    il_batch_size2=128,
    il_buffer_max_size=20000,
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
    max_episode_duration=10*60,
    R_cage = 45e3, # 55e3,
    dt_maneuver=0.2,
    transition_dict_capacity=2000, # 稍微调大一点，因为现在是批量进数据
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
    sigma_elo = 400,
    WARM_UP_STEPS = 500e3,
    ADMISSION_THRESHOLD = 0.5,
    MAX_HISTORY_SIZE = 300,  # 100
    deltaFSP_epsilon = 0.8,
    rule_actor_rate = 0.2,
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
    save_interval = 2, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (Serial Execution on Master)
    print("Start MARWIL Training...")

    student_agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    # === 模仿训练循环 ===
    # 现在 original_il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(IL_epoches): 
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
            original_il_transition_dict, 
            beta=beta_mixed, 
            batch_size=il_batch_size, # 显存如果够大可以适当调大
            label_smoothing=label_smoothing
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    print("IL Training Finished.")
    
    # --- 新增：实例化混合缓冲区 ---
    il_transition_buffer = None
    if IL_epoches > 0:
        print("Initializing IL Transition Buffer...")
        il_transition_buffer = IL_transition_buffer(original_il_transition_dict0, max_size=il_buffer_max_size)

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
    pending_tests = []
    test_results_aggregator = {}

    # --- B. 启动并行训练 Worker (Sync Training Workers) ---
    # 这些 Worker 与 Master 同步，负责生成训练数据
    workers = []
    pipes = []
    worker_device = torch.device('cpu') # Worker 使用 CPU 推理
    
    print(f"Initializing {num_workers} training workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(
            i, child_conn, args, state_dim, hidden_dim, action_dims_dict, worker_device, dt_maneuver, seed
        ))
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    # ELO 初始化
    elo_ratings = copy.deepcopy(init_elo_ratings)
    elite_elo_ratings = copy.deepcopy(elo_ratings)
    hall_of_fame_keys = []
    
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # 尝试加载历史
    if os.path.exists(full_json_path):
        with open(full_json_path, 'r', encoding='utf-8') as f: elo_ratings = json.load(f)
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f: elite_elo_ratings = json.load(f)
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f: hall_of_fame_keys = json.load(f)

    main_agent_elo = elo_ratings.get("__CURRENT_MAIN__", 1200)

    # 初始对手
    if (not elo_ratings) or IL_epoches > 0:
        init_opponent_name = "actor_rein0"
        torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{init_opponent_name}.pt"))
        if self_play_type != 'None': elo_ratings[init_opponent_name] = 1200

    # 训练循环变量
    total_steps = 0
    batch_idx = 0
    trigger = trigger0
    current_max_steps = int(max_steps)
    
    # 全局 Buffer (用于攒够 Batch 训练)
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    global_transition_dict = copy.deepcopy(empty_transition_dict)

    # =========================================================
    # 主循环 (Master Process)
    # =========================================================
    while True: 
        while total_steps < current_max_steps:
            
            # --- 1. 并行测试触发逻辑 (Async) ---
            if total_steps >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
                # 1. 深度拷贝当前 Actor 权重到 CPU 内存
                current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}
                for r_idx in [0, 1, 2, 3, 4]:
                    res_obj = test_pool.apply_async(
                        test_worker, 
                        args=(current_weights, r_idx, args, state_dim, hidden_dim, action_dims_dict, dt_maneuver, 'cpu') # [修改] 显式传入 dt_maneuver
                    )
                    pending_tests.append((res_obj, total_steps)) # 记录任务对象和触发时的步数
                
                trigger += trigger_delta # 更新下一次触发阈值

            # 处理测试结果
            if len(pending_tests) > 0:
                finished_tasks = []
                for task in pending_tests:
                    res_obj, recorded_step = task
                    if res_obj.ready():
                        rule_num, outcome = res_obj.get()
                        logger.add(f"test/agent_vs_rule{rule_num}", outcome, recorded_step)
                        print(f"  [Async Test Result] Rule_{rule_num}: {outcome} (Triggered at {recorded_step})")
                        
                        # B. 名人堂逻辑
                        if recorded_step not in test_results_aggregator:
                            test_results_aggregator[recorded_step] = {}
                        test_results_aggregator[recorded_step][rule_num] = outcome
                        
                        # C. [新增] 检查是否达成“名人堂”成就
                        # 假设你测试的是 Rule 0, 1, 2, 3, 4 共 5 个规则
                        expected_rule_count = len([k for k in init_elo_ratings.keys() if k.startswith("Rule")])
                        if len(test_results_aggregator[recorded_step]) == expected_rule_count:
                            all_outcomes = test_results_aggregator[recorded_step].values()
                            # 如果 5 场全胜 (1.0 代表胜)
                            if all(res == 1.0 for res in all_outcomes):
                                hof_key = f"actor_rein_step_{recorded_step}" # 定义一个唯一的标识符
                                if hof_key not in hall_of_fame_keys:
                                    hall_of_fame_keys.append(hof_key)
                                    print(f"!!! [Hall of Fame] New Hero Captured: {hof_key}")
                            
                            # 完成后清理该步数的汇总数据，释放内存
                            del test_results_aggregator[recorded_step]
                        
                        finished_tasks.append(task)
                
                # 清理已完成的任务
                for task in finished_tasks:
                    pending_tests.remove(task)

            # --- 2. 准备训练 Batch (Synchronous) ---
            
            # 1. 计算当前排位 rank_pos
            valid_elo_values = [v for k, v in elite_elo_ratings.items() if not k.startswith("__")]
            if not valid_elo_values:
                rank_pos = 0.5
                min_elo, max_elo = main_agent_elo, main_agent_elo
            else:
                min_elo = np.min(valid_elo_values)
                max_elo = np.max(valid_elo_values)
                denom = float(max_elo - min_elo)
                # 如果分母为0，视为0.5中位
                if denom == 0.0:
                    rank_pos = 0.5
                else:
                    rank_pos = float((main_agent_elo - min_elo) / denom)


            # A. 获取当前策略权重 (CPU)
            current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
            
            # B. 分发任务给 Worker
            # 这一步 Master 决定每个 Worker 打谁
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            for rank in range(num_workers):
                # 采样对手
                probs, opponent_keys = get_opponent_probabilities(elite_elo_ratings, target_elo=main_agent_elo, SP_type=self_play_type, sigma=sigma_elo)
                selected_opponent_name = np.random.choice(opponent_keys, p=probs)
                
                # 准备对手数据
                opp_type = 'rule'
                opp_data = 0
                if "Rule" in selected_opponent_name:
                    try: rule_num = int(selected_opponent_name.split('_')[1])
                    except: rule_num = 0
                    opp_data = rule_num
                else:
                    opp_type = 'nn'
                    adv_path = os.path.join(log_dir, f"{selected_opponent_name}.pt")
                    if os.path.exists(adv_path):
                        opp_data = torch.load(adv_path, map_location='cpu') # 传给 Worker 必须是 CPU Tensor
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
                
                # 发送指令
                pipes[rank].send(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # C. 等待所有 Worker 完成 (Barrier)
            batch_results = []
            for rank in range(num_workers):
                res = pipes[rank].recv() # 阻塞等待
                batch_results.append(res)
            
            # --- 3. 数据聚合与处理 ---
            batch_total_steps = 0
            batch_wins = 0
            batch_loss_cnt = 0
            
            for res in batch_results:
                # res 结构: {'trans':..., 'ego_tr':..., 'enm_tr':..., 'metrics':..., 'opp_name':...}
                l_tr = res['trans'] # PPO 训练数据 (含探索)
                ego_tr = res['ego_trans'] # SIL 蓝方数据
                enm_tr = res['enm_trans'] # SIL 红方数据
                metrics = res['metrics']
                opp_name = res['opp_name']
                
                batch_total_steps += metrics['steps']
                if metrics['win']: batch_wins += 1
                if metrics['lose']: batch_loss_cnt += 1
                
                # 3.1 聚合 PPO 数据到全局 Buffer
                for k in global_transition_dict:
                    global_transition_dict[k].extend(l_tr[k])
                
                # 3.2 SIL 数据收集 (需计算 return)
                if use_sil:
                    if not metrics['lose']: # 赢或平，学自己
                        # 计算回报 (Master 端计算)
                        ego_tr['returns'] = compute_monte_carlo_returns(gamma, ego_tr['rewards'], ego_tr['dones'])
                        il_transition_buffer.add(ego_tr)
                    
                    if not metrics['win']: # 输或平，学对手
                        enm_tr['returns'] = compute_monte_carlo_returns(gamma, enm_tr['rewards'], enm_tr['dones'])
                        il_transition_buffer.add(enm_tr)
                
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

                if opp_name in elite_elo_ratings and should_update:
                    prev_main_elo = main_agent_elo
                    adv_elo = elite_elo_ratings[opp_name]
                    
                    # 更新主智能体
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score, K_FACTOR)
                    # 更新对手
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score, K_FACTOR)
                    
                    elite_elo_ratings[opp_name] = new_adv_elo
                    elo_ratings[opp_name] = new_adv_elo
                    elo_ratings["__CURRENT_MAIN__"] = main_agent_elo
            
            # 更新全局计数
            total_steps += batch_total_steps
            batch_idx += 1
            
            # --- 4. 执行训练 (PPO Update) ---
            # 当收集的数据量超过 capacity 时更新
            if len(global_transition_dict['dones']) >= transition_dict_capacity:
                
                # 重构 Action 结构 (List[Dict] -> Dict[Array])
                global_transition_dict['actions'] = restructure_actions(global_transition_dict['actions'])
                
                if use_sil:
                    # 读取 IL 数据
                    il_data = il_transition_buffer.read(il_batch_size2)
                    
                    # 混合更新
                    student_agent.mixed_update(
                        global_transition_dict,
                        il_data,
                        init_il_transition_dict = original_il_transition_dict0 if use_init_data else None,
                        eta = np.clip(1 - total_steps/3e6, 0, 1),
                        adv_normed=True,
                        label_smoothing=label_smoothing_mixed,
                        alpha=alpha_il,
                        beta=beta_mixed,
                        sil_only_maneuver = sil_only_maneuver,
                        mini_batch_size = mini_batch_size_mixed
                    )
                else:
                    student_agent.update(global_transition_dict, adv_normed=1, mini_batch_size=64)
                
                # 记录 Log
                logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
                logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
                logger.add("train/10 entropy", student_agent.entropy_mean, total_steps)
                logger.add("train/batch_win_rate", batch_wins / num_workers, total_steps)
                logger.add("Elo/Main_Agent", main_agent_elo, total_steps)
                
                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}, ELO {main_agent_elo:.0f}")

                # 清空 Buffer
                global_transition_dict = copy.deepcopy(empty_transition_dict)

            # --- 5. 保存与维护 (Checkpoint & Pool) ---
            if batch_idx % save_interval == 0:
                # A. 保存模型
                actor_key = f"actor_rein{batch_idx}"
                torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{actor_key}.pt"))
                torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, "critic.pt"))
                print(f"Saved Checkpoint: {actor_key}")

                # B. 维护精英池 (Admission)
                if hist_agent_as_opponent and total_steps >= WARM_UP_STEPS:
                    # 计算 Rank
                    valid_elos = [v for k, v in elite_elo_ratings.items() if k.startswith("Rule")]
                    if not valid_elos: valid_elos = [1200]
                    r_min, r_max = min(valid_elos), max(valid_elos)
                    denom = r_max - r_min if r_max != r_min else 1.0
                    rank = (main_agent_elo - r_min) / denom
                    
                    if rank >= ADMISSION_THRESHOLD:
                        # 满员清理
                        history_keys = [k for k in elite_elo_ratings.keys() if not k.startswith("Rule") and not k.startswith("__")]
                        while len(history_keys) >= MAX_HISTORY_SIZE:
                            weakest = min(history_keys, key=lambda k: elite_elo_ratings[k])
                            del elite_elo_ratings[weakest]
                            history_keys.remove(weakest)
                            print(f"Kicked weakest: {weakest}")
                        
                        elite_elo_ratings[actor_key] = main_agent_elo
                        print(f"Accepted {actor_key} into Elite Pool.")
                
                # C. 保存 JSON
                elo_ratings[actor_key] = main_agent_elo
                with open(full_json_path, "w", encoding="utf-8") as f: json.dump(elo_ratings, f, indent=2)
                
                save_elite = copy.deepcopy(elite_elo_ratings)
                save_elite["__CURRENT_MAIN__"] = main_agent_elo
                with open(elite_json_path, "w", encoding="utf-8") as f: json.dump(save_elite, f, indent=2)
                with open(hof_json_path, "w", encoding="utf-8") as f: json.dump(hall_of_fame_keys, f, indent=2)

        # --- 交互式延长训练 ---
        print(f"\n--- Target steps reached: {total_steps} / {current_max_steps} ---")
        try:
            inp = input(f"Enter new max_steps (current {total_steps}): ")
            if not inp.strip(): break
            new_max = int(inp)
            if new_max > total_steps:
                current_max_steps = new_max
            else:
                break
        except:
            break

    # Cleanup
    print("Closing workers...")
    for pipe in pipes: pipe.send(('EXIT', None))
    for p in workers: p.join()
    test_pool.close()
    test_pool.join()
    logger.close()
    print("Training Finished.")