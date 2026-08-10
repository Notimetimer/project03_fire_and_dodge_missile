'''
Multi-agent hierarchical policy gradient for Air Combat Tactics emergence via self-play
2021

'''

from typing import final
import os
import sys
import numpy as np
from math import pi
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
# 必须在任何 sklearn 导入之前执行！
try:
    import threadpoolctl
    # 彻底让信息查询返回空列表
    threadpoolctl.threadpool_info = lambda *args, **kwargs: []
    # 核心修复：直接将 threadpool_limits 变为一个什么都不做的空上下文管理器
    class DummyContextManager:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def _set_threadpool_limits(self): return []
    threadpoolctl.threadpool_limits = DummyContextManager
    # 针对 3.x 版本的控制器拦截
    if hasattr(threadpoolctl, 'ThreadpoolController'):
        threadpoolctl.ThreadpoolController.info = lambda self, *args, **kwargs: []
        threadpoolctl.ThreadpoolController.limit = lambda self, *args, **kwargs: DummyContextManager()
    print("[Patch] threadpoolctl 全版本上下文拦截补丁已成功强行注入。")
except Exception as e:
    print(f"[Patch] 补丁注入失败: {e}，尝试继续运行...")
from sklearn.cluster import KMeans

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new_hierarchical import *
# 必须先import环境再import算法，否则算法可能无法指向设置的算法模块
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical_compare_3 import * # 奖励函数 对比
from Algorithms.PPOHybrid23_0MA import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from VsBaseline_while_training_hierarch_plus import test_worker
from RewardWeightController import FireRewardWeightController
# 强制显式绑定：防止调用链中其它 import 污染 ChooseStrategyEnv
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical_compare_3 import ChooseStrategyEnv

dt_move = 0.04


class RuleTeacherWrapper:
    """
    基于规则(basic_rules)的教师包装器，用于 RDistill。

    对齐 PolicyNetHybrid.forward 的输出格式：
      - 'cat': list[Tensor]，每个动作头一个软 one-hot 概率分布 (B, n_classes)
      - 'bern': Tensor，开火头 logits (B, 1)
      - 'cont': None
    软 one-hot 构造方式与 PPOHybrid.compute_il_loss 的 pre_training 分支一致：
      正确动作概率 = 1 - label_smoothing，其余均分 label_smoothing/(n-1)。

    通过 env.obs2obs_check + env.unscale_state 把网络局部观测(obs)还原为
    basic_rules 所需的 state_check，再逐样本调用规则得到 (action, fire)。
    """
    is_rule_teacher = True

    def __init__(self, env, rule_num, action_dims_dict, device, label_smoothing=0.1):
        self.env = env
        self.rule_num = rule_num
        self.action_dims_dict = action_dims_dict
        self.device = device
        self.label_smoothing = label_smoothing
        cat_dims = action_dims_dict.get('cat', [])
        # 兼容 int / list 两种写法
        self.cat_dims = list(cat_dims) if isinstance(cat_dims, (list, tuple)) else [cat_dims]

    def eval(self):
        return self

    @torch.no_grad()
    def predict_distributions(self, states):
        obs_np = states.detach().cpu().numpy()
        B = obs_np.shape[0]
        ls = self.label_smoothing
        n_heads = len(self.cat_dims)

        # 初始化为平滑背景概率
        cat_probs = [np.full((B, nc), ls / max(nc - 1, 1), dtype=np.float32) for nc in self.cat_dims]
        bern_logits = np.full((B, 1), -3.0, dtype=np.float32)

        for i in range(B):
            check_obs = self.env.obs2obs_check(obs_np[i])
            state_check = self.env.unscale_state(check_obs)
            action_number, fire = basic_rules(state_check, self.rule_num, p_random=0)

            if isinstance(action_number, (list, tuple, np.ndarray)):
                act_list = list(np.asarray(action_number).reshape(-1))
            else:
                act_list = [action_number]

            for h in range(n_heads):
                nc = self.cat_dims[h]
                idx = int(act_list[h]) if h < len(act_list) else 0
                idx = int(np.clip(idx, 0, nc - 1))
                cat_probs[h][i, :] = ls / max(nc - 1, 1)
                cat_probs[h][i, idx] = 1.0 - ls

            bern_logits[i, 0] = 3.0 if fire else -3.0

        cat_tensors = [torch.tensor(cp, dtype=torch.float, device=self.device) for cp in cat_probs]
        bern_tensor = torch.tensor(bern_logits, dtype=torch.float, device=self.device)
        return {'cont': None, 'cat': cat_tensors, 'bern': bern_tensor}

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

    def save(self, path):
        """直接序列化整个 buffer 实例"""
        torch.save(self, path)
        print(f"[IL_transition_buffer] Saved to {path}. Size: {len(self.addon_dict['states'])}")

    @staticmethod # 静态方法调用,不需要实例化
    def load(path):
        """从磁盘加载 buffer 实例"""
        if not os.path.exists(path):
            return None
        buffer = torch.load(path, map_location='cpu')
        print(f"[IL_transition_buffer] Loaded from {path}. Size: {len(buffer.addon_dict['states'])}")
        return buffer
        


# calculate_expected_score 概率计算

def calculate_expected_score(player_elo, opponent_elo):
    """计算期望得分"""
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)) # 这个数是约定俗成的,别改

def update_elo(player_elo, opponent_elo, score, K_FACTOR):
    """更新ELO分数. score: 1 for win, 0 for loss, 0.5 for draw."""
    expected = calculate_expected_score(player_elo, opponent_elo)
    return player_elo + K_FACTOR * (score - expected)


def get_opponent_probabilities(elite_elo_ratings, hall_of_fame=None, 
                               target_elo=None, sigma=400, SP_type='PFSP_with_delta', 
                               compete_old_rate=0.5, deltaFSP_epsilon=0.5,):
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
    
    if not keys: return np.array([]), []

    # --- 第一层判断：规则复习分支 (Epsilon-Greedy 锚点保护) ---
    # 只要 compete_old_rate > 0，就有概率强行进入规则池采样，防止“策略遗忘”
    rule_keys = [k for k in keys if k.startswith('Rule')]
    if np.random.rand() < compete_old_rate and rule_keys:
        probs = np.ones(len(rule_keys)) / len(rule_keys)
        return probs, rule_keys
    
    # --- 第二层判断：进入核心采样逻辑 ---
    # 【核心修改】统一从 candidate_pool 取分，彻底避免 KeyError
    elos = np.array([candidate_pool[k] for k in keys], dtype=np.float64)
    
    # 1. 处理 PFSP 系列 (高斯核采样)
    if SP_type.startswith('PFSP'):
        if SP_type == 'PFSP_challenge':
            actual_target = min(np.max(elos), float(target_elo) + 300)
        elif SP_type == 'PFSP_balanced' or SP_type == 'PFSP_with_delta':
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
        else: # 默认通用的 'PFSP' 逻辑
            actual_target = float(target_elo) if target_elo is not None else np.mean(elos)
            # # 你之前的逻辑：取 0.5 均值 + 0.5 最大值，作为一个偏向挑战的平衡点
            # actual_target = 0.5 * (float(target_elo) if target_elo is not None else np.mean(elos)) + 0.5 * np.max(elos)
        
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
                8000.0 * (1-int(randomized))
    if np.random.uniform(0,1) < 0.2 and int(randomized): # 以很低概率在更大的范围随机初始高度
        blue_height = np.random.uniform(2000.0, 12000.0)
    red_height = blue_height
    # 初始航向随机化
    red_psi = sub_of_radian(-np.pi/2 + np.random.uniform(-pi*2/3, pi*2/3)) # -pi/3, pi/3
    blue_psi = sub_of_radian(np.pi/2 + np.random.uniform(-pi*2/3, pi*2/3))
    init_North = np.random.uniform(-30e3, 30e3) * int(randomized)
    red_N = init_North
    red_E = 55e3 # 45e3
    if np.random.uniform(0,1) < 0.2: # 以低概率随机初始距离
        red_E = np.random.uniform(10e3,55e3)
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
        # 在子进程内部再次显式绑定 ChooseStrategyEnv，避免全局命名空间被其它 import 污染
        from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical_compare_3 import ChooseStrategyEnv
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
        # Worker 仅做推理：直接用 HybridActorWrapper（SAC 与 PPO 共用同一套 actor 接口），无需构建完整 SAC/Q 网络
        # [MAPPO] 红蓝双方都是学习方，各自一套 actor
        blue_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        blue_agent = HybridActorWrapper(blue_actor, action_dims_dict, None, device_worker).to(device_worker)
        
        red_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        red_agent = HybridActorWrapper(red_actor, action_dims_dict, None, device_worker).to(device_worker)

        # --- 2. 循环等待阶段 ---
        while True:
            # 阻塞等待指令
            cmd, packet = pipe.recv()
            
            if cmd == 'EXIT':
                env.close()
                break
                
            if cmd == 'RUN_EPISODE':
                # 解包数据 [MAPPO] 红蓝双方都用最新权重，顺序统一先红后蓝
                (red_actor_weights, blue_actor_weights, settings) = packet
                
                # A. 同步权重 (极快)
                red_agent.load_state_dict(red_actor_weights)
                blue_agent.load_state_dict(blue_actor_weights)

                # C. 准备本回合容器
                # [MAPPO] 红蓝双方各自的 PPO 经验容器
                # 'obs' 为各自的局部观测(Actor输入)，'states' 为红蓝全局状态拼接(Critic输入，先红后蓝)
                blue_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
                red_trans = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}

                # 新增: 开火角度参数及导弹存活期 ATA 的回合统计容器
                # 红方（对手）统计
                episode_red_fire_thetas = []      # 开火瞬间俯仰角
                episode_red_ATAs = []             # 开火后30s的ATA
                episode_red_delta_psi_threats = [] # 收到告警后的delta_psi_threat
                episode_red_delta_thetas = []     # 开火后30s内的delta_theta
                episode_red_delta_psis = []       # 开火后30s内的delta_psi
                # 蓝方（本方学习代理）统计
                episode_blue_fire_thetas = []     # 开火瞬间俯仰角
                episode_blue_ATAs = []            # 开火后30s的ATA
                episode_blue_delta_psi_threats = [] # 收到告警后的delta_psi_threat
                episode_blue_delta_thetas = []    # 开火后30s内的delta_theta
                episode_blue_delta_psis = []      # 开火后30s内的delta_psi
                
                # 新增: 开火策略指标统计容器
                # 蓝方（本方）开火策略指标
                episode_blue_fire_intervals = []  # 开火间隔时长
                episode_blue_fire_delta_psis = [] # 开火瞬间的abs(delta_psi)
                episode_blue_fire_distances = []  # 开火距离
                episode_blue_fire_AA_hors = []    # 开火瞬间的abs(AA_hor)
                episode_blue_fire_alts = []           # 开火瞬间高度
                
                # 记录上次开火时间用于计算间隔
                last_blue_fire_time = -1.0
                

                # D. 环境重置
                # randomized_birth = settings['randomized_birth']  # 改在外面随机，里面不需要
                action_cycle_multiplier = settings['action_cycle_multiplier']
                reward_weight = settings['weight_reward']
                # 在子环境中重新计算出生状态
                # red_birth, blue_birth = create_initial_state_worker(randomized_birth)
                # 使用从master传来的出生状态
                red_birth = settings['red_birth']
                blue_birth = settings['blue_birth']
                end_reward_weight = settings['end_reward_weight']
                
                # 每次重新运行对局前，根据Master指定的范围随机化当前环境大小
                # r_min, r_max = settings.get('R_cage_range', (55.00e3, 55.00e3))
                fire_mask = settings.get('fire_mask', 1)
                # env.R_cage = np.random.uniform(r_min, r_max)

                fire_inside_weight = settings.get('fire_inside_weight', None)
                fire_reward_weight = settings.get('fire_reward_weight', None)
                
                # 进场瞬间给全信息
                red_init_ammo=6
                blue_init_ammo=6
                # 残局训练
                if np.random.uniform(0,1) < 0.3:
                    red_init_ammo = int(np.round(np.random.uniform(0,3)))
                    blue_init_ammo = int(np.round(np.random.uniform(0,3)))
                env.reset(red_birth_state=red_birth, blue_birth_state=blue_birth, red_init_ammo=red_init_ammo, blue_init_ammo=blue_init_ammo, pomdp=0)
                
                # 状态变量初始化
                done = False
                # [MAPPO] last_decision_state 统一为红蓝全局状态拼接(先红后蓝)
                last_decision_obs, last_enm_decision_obs = None, None
                last_decision_state = None
                current_action, current_enm_action = None, None
                
                steps_run = 0
                episode_return = 0 # 仅用于统计显示
                episode_return_dense = 0
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

                    # 收到告警后的delta_psi_threat - 红方（对手）
                    r_state_check = env.unscale_state(r_check_obs)
                    delta_psi_threat = np.arccos(r_check_obs["threat"][0])
                    if r_state_check["warning"]:
                        episode_red_delta_psi_threats.append(float(delta_psi_threat))
                    
                    # 收到告警后的delta_psi_threat - 蓝方（本方）
                    b_state_check = env.unscale_state(b_check_obs)
                    blue_delta_psi_threat = np.arccos(b_check_obs["threat"][0])
                    if b_state_check["warning"]:
                        episode_blue_delta_psi_threats.append(float(blue_delta_psi_threat))
                    
                    # 记录开火后30s内的角度数据（与ATA同级别）- 红方（对手）
                    if r_check_obs["weapon"] * 120 <= 30 and not r_state_check["warning"]:  # 有存活导弹（30s内）且没有告警
                        if "target_information" in r_state_check and len(r_state_check["target_information"]) > 0:
                            red_delta_psi = np.arctan2(r_state_check["target_information"][1], r_state_check["target_information"][0])
                            red_delta_theta = r_state_check["target_information"][2]
                            red_ATA = r_state_check["target_information"][4]
                            episode_red_ATAs.append(float(red_ATA))
                            episode_red_delta_thetas.append(float(red_delta_theta))
                            episode_red_delta_psis.append(float(red_delta_psi))
                    
                    # 记录开火后30s内的角度数据 - 蓝方（本方）
                    if b_check_obs["weapon"] * 120 <= 30 and not b_state_check["warning"]:  # 有存活导弹（30s内）且没有告警
                        if "target_information" in b_state_check and len(b_state_check["target_information"]) > 0:
                            blue_delta_psi = np.arctan2(b_state_check["target_information"][1], b_state_check["target_information"][0])
                            blue_delta_theta = b_state_check["target_information"][2]
                            blue_ATA = b_state_check["target_information"][4]
                            episode_blue_ATAs.append(float(blue_ATA))
                            episode_blue_delta_thetas.append(float(blue_delta_theta))
                            episode_blue_delta_psis.append(float(blue_delta_psi))


                    # [MAPPO] 集中式 Critic 输入：红蓝全局状态拼接（先红后蓝）
                    global_state = np.concatenate([r_state_global, b_state_global])

                    # 2. 决策点 (Action Cycle)
                    if steps_run % action_cycle_multiplier == 0:
                        # 2.1 存储【上一个】周期的经验
                        if steps_run > 0:
                            # 注意：这里调用你原文件里的 append_experience 辅助函数
                            # 确保 append_experience 在 这个函数 作用域外是可见的，或者复制进来
                            append_experience(blue_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, global_state, False, not dead_dict['b'])
                            append_experience(red_trans, last_enm_decision_obs, last_decision_state, current_enm_action, reward_for_enm, global_state, False, not dead_dict['r'])

                        # 2.2 更新上一帧记录
                        last_decision_obs = b_obs
                        last_enm_decision_obs = r_obs
                        last_decision_state = global_state
                        
                        # 2.3 产生新动作 (No Grad)
                        with torch.no_grad():
                            # Blue Decision
                            b_state_check = env.unscale_state(b_check_obs)
                            b_action_exec, _, _, _ = blue_agent.get_action(b_obs, explore=1, mask_on=fire_mask)
                            # b_action_exec, _, _, _ = blue_agent.get_action(b_obs, explore=1, check_obs=b_check_obs, mask_on=fire_mask) # 不建议采样也启用mask
                            b_action_label = b_action_exec['cat'] # [0]
                            b_fire = b_action_exec['bern'][0]
                            
                            # Red Decision [MAPPO] 红方同为学习方，与蓝方对称采样
                            r_state_check = env.unscale_state(r_check_obs)
                            r_action_exec, _, _, _ = red_agent.get_action(r_obs, explore=1, mask_on=fire_mask)
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
                        
                        # 2.5 记录当前动作供下一帧存储 (存采样动作，用于 PPO 更新)
                        current_action = {'cat': b_action_exec['cat'], 'bern': b_action_exec['bern']}
                        current_enm_action = {'cat': r_action_exec['cat'], 'bern': r_action_exec['bern']}

                    # 3. 物理步进与尝试发射
                     # 采样的时候如果限制动作次序，会妨碍“试错”，到测试时也必须开启  r_action_label  b_action_label None

                    r_action_label_fire=None
                    b_action_label_fire=None
                    
                    r_m_id = launch_missile_immediately(env, 'r', action_label=r_action_label_fire) if getattr(env.RUAV, 'about_to_fire', 0) else None
                    b_m_id = launch_missile_immediately(env, 'b', action_label=b_action_label_fire) if getattr(env.BUAV, 'about_to_fire', 0) else None
                    
                    if b_m_id: 
                        m_fired += 1
                        # 记录蓝方（本方）开火俯仰角
                        blue_fire_theta = float(env.BUAV.theta)
                        episode_blue_fire_thetas.append(blue_fire_theta)
                        
                        # 记录开火瞬间高度
                        blue_fire_alt = float(env.BUAV.alt)
                        episode_blue_fire_alts.append(blue_fire_alt)
                        
                        # 记录蓝方开火策略指标
                        current_time = steps_run * dt_maneuver
                        if last_blue_fire_time >= 0:
                            fire_interval = current_time - last_blue_fire_time
                            episode_blue_fire_intervals.append(fire_interval)
                        last_blue_fire_time = current_time
                        
                        # 记录开火瞬间的abs(delta_psi)、距离、AA_hor
                        if "target_information" in b_state_check and len(b_state_check["target_information"]) > 0:
                            # delta_psi: target_information[0]
                            fire_delta_psi = np.arccos(b_state_check["target_information"][0])
                            episode_blue_fire_delta_psis.append(float(fire_delta_psi))
                            
                            # 距离: target_information[1]
                            fire_distance = b_state_check["target_information"][3]
                            episode_blue_fire_distances.append(float(fire_distance))
                            
                            # AA_hor: target_information[6]
                            fire_AA_hor = b_state_check["target_information"][6]
                            episode_blue_fire_AA_hors.append(abs(float(fire_AA_hor)))
                        
                    if r_m_id is not None:
                        fire_theta = float(env.RUAV.theta)
                        episode_red_fire_thetas.append(fire_theta)
                    
                    # debug
                    if r_action_label[0] > 4:
                        print("数值超出范围", r_action_label[0], r_action_label[1])

                    r_maneuver = env.maneuver14LR(env.RUAV, r_action_label)
                    b_maneuver = env.maneuver14LR(env.BUAV, b_action_label)
                    env.step(r_maneuver, b_maneuver)
                    steps_run += 1
                    
                    # 4. 奖励计算
                    done, b_reward1, b_reward2, b_reward3 = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, 
                                                            action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)
                    _, r_reward1, r_reward2, r_reward3 = env.combat_terminate_and_reward('r', r_action_label, r_m_id is not None, action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)
                    _, b_dense_reward, _, _ = env.combat_terminate_and_reward('b', b_action_label, b_m_id is not None, action_cycle_multiplier, end_reward_weight=0,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)

                    reward_for_learn = sum(np.array([b_reward1, b_reward2, b_reward3]) * reward_weight)
                    reward_for_enm = sum(np.array([r_reward1, r_reward2, r_reward3]) * reward_weight)
                    
                    if steps_run % action_cycle_multiplier == 0 or done:
                        episode_return += b_reward1
                        episode_return_dense += b_dense_reward
                    
                    # 5. 存活更新 (用于 Done 标记)
                    next_b_state_global, _ = env.obs_1v1('b', reward_fn=1)
                    next_r_state_global, _ = env.obs_1v1('r', reward_fn=1)
                    dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

                # --- End of Simulation Loop ---
                
                # 6. 存储最后一步经验 (Terminal State)
                # 强制做一次终局判定
                done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False, action_cycle_multiplier, end_reward_weight=end_reward_weight,
                                                            fire_reward_weight=fire_reward_weight,
                                                            fire_inside_weight=fire_inside_weight)
                
                if last_decision_state is not None:
                    next_global_state = np.concatenate([next_r_state_global, next_b_state_global])
                    append_experience(blue_trans, last_decision_obs, last_decision_state, current_action, reward_for_learn, next_global_state, True, not dead_dict['b'])
                    append_experience(red_trans, last_enm_decision_obs, last_decision_state, current_enm_action, reward_for_enm, next_global_state, True, not dead_dict['r'])

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
                
                blue_trans = truncate_and_shift(blue_trans)
                red_trans = truncate_and_shift(red_trans)
                # ----------------------------------------------

                # 计算本回合红方开火与角度参数指标
                if len(episode_red_fire_thetas) > 0:
                    ep_avg_fire_theta = float(np.mean(episode_red_fire_thetas))
                else:
                    ep_avg_fire_theta = None
                    
                if len(episode_red_ATAs) > 0:
                    ep_avg_ATA = float(np.mean(episode_red_ATAs))
                else:
                    ep_avg_ATA = None
                    
                if len(episode_red_delta_psi_threats) > 0:
                    ep_avg_delta_psi_threat = float(np.mean(episode_red_delta_psi_threats))
                else:
                    ep_avg_delta_psi_threat = None
                    
                if len(episode_red_delta_thetas) > 0:
                    ep_avg_delta_theta = float(np.mean(episode_red_delta_thetas))
                else:
                    ep_avg_delta_theta = None
                    
                if len(episode_red_delta_psis) > 0:
                    ep_avg_delta_psi = float(np.mean(episode_red_delta_psis))
                else:
                    ep_avg_delta_psi = None

                # 计算本回合蓝方（本方）开火与角度参数指标
                if len(episode_blue_fire_thetas) > 0:
                    ep_blue_avg_fire_theta = float(np.mean(episode_blue_fire_thetas))
                else:
                    ep_blue_avg_fire_theta = None
                    
                if len(episode_blue_ATAs) > 0:
                    ep_blue_avg_ATA = float(np.mean(episode_blue_ATAs))
                else:
                    ep_blue_avg_ATA = None
                    
                if len(episode_blue_delta_psi_threats) > 0:
                    ep_blue_avg_delta_psi_threat = float(np.mean(episode_blue_delta_psi_threats))
                else:
                    ep_blue_avg_delta_psi_threat = None
                    
                if len(episode_blue_delta_thetas) > 0:
                    ep_blue_avg_delta_theta = float(np.mean(episode_blue_delta_thetas))
                else:
                    ep_blue_avg_delta_theta = None
                    
                if len(episode_blue_delta_psis) > 0:
                    ep_blue_avg_delta_psi = float(np.mean(episode_blue_delta_psis))
                else:
                    ep_blue_avg_delta_psi = None

                # 计算本回合蓝方（本方）开火策略指标
                if len(episode_blue_fire_intervals) > 0:
                    ep_blue_avg_fire_interval = float(np.mean(episode_blue_fire_intervals))
                else:
                    ep_blue_avg_fire_interval = None
                    
                if len(episode_blue_fire_delta_psis) > 0:
                    ep_blue_avg_fire_delta_psi = float(np.mean(episode_blue_fire_delta_psis))
                else:
                    ep_blue_avg_fire_delta_psi = None
                    
                if len(episode_blue_fire_distances) > 0:
                    ep_blue_avg_fire_distance = float(np.mean(episode_blue_fire_distances))
                else:
                    ep_blue_avg_fire_distance = None
                    
                if len(episode_blue_fire_AA_hors) > 0:
                    ep_blue_avg_fire_AA_hor = float(np.mean(episode_blue_fire_AA_hors))
                else:
                    ep_blue_avg_fire_AA_hor = None
                
                if len(episode_blue_fire_alts) > 0:
                    ep_blue_avg_fire_altitude = float(max(episode_blue_fire_alts)) # 不再记录平均开火高度，改为记录最大开火高度
                else:
                    ep_blue_avg_fire_altitude = None

                WVR = env.close_range_kill()
                BVR_perish_together = (not WVR) and env.draw

                # 7. 打包结果
                result_packet = {
                    'blue_trans': blue_trans, # 蓝方 PPO 训练数据
                    'red_trans': red_trans,   # 红方 PPO 训练数据
                    'metrics': {
                        'return': episode_return,
                        'dense_return': b_dense_reward,
                        'steps': steps_run,
                        'win': env.win,
                        'lose': env.lose,
                        'draw': env.draw,
                        'm_fired': m_fired,
                        'BVR_perish_together': BVR_perish_together
                    },
                    # 新增: 本回合红方开火角度参数统计 [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                    'ep_avg_fire_theta': ep_avg_fire_theta,
                    'ep_avg_ATA': ep_avg_ATA,
                    'ep_avg_delta_psi_threat': ep_avg_delta_psi_threat,
                    'ep_avg_delta_theta': ep_avg_delta_theta,
                    'ep_avg_delta_psi': ep_avg_delta_psi,
                    # 新增: 本回合蓝方（本方）开火角度参数统计
                    'ep_blue_avg_fire_theta': ep_blue_avg_fire_theta,
                    'ep_blue_avg_ATA': ep_blue_avg_ATA,
                    'ep_blue_avg_delta_psi_threat': ep_blue_avg_delta_psi_threat,
                    'ep_blue_avg_delta_theta': ep_blue_avg_delta_theta,
                    'ep_blue_avg_delta_psi': ep_blue_avg_delta_psi,
                    # 新增: 本回合蓝方（本方）开火策略指标统计
                    'ep_blue_avg_fire_interval': ep_blue_avg_fire_interval,
                    'ep_blue_avg_fire_delta_psi': ep_blue_avg_fire_delta_psi,
                    'ep_blue_avg_fire_distance': ep_blue_avg_fire_distance,
                    'ep_blue_avg_fire_AA_hor': ep_blue_avg_fire_AA_hor,
                    'ep_blue_avg_fire_altitude': ep_blue_avg_fire_altitude
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
    collape_recover={
        "collapsed": False,
        "best_actor_name": None,
        "actor_frozen_batchs": 5,
    },
    num_workers=10, # 并行进程数，根据CPU核数调整，建议 10-20
    n_clusters=5,
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
    alpha_il=0.05,
    il_batch_size=128,
    il_batch_size2=None,
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
    chosen_quantile = 0.2, 
    DARK_SIDE = 1,  # sil默认找最差
    p_factor = None, # 无效接口
    sigma_elo = 400,
    WARM_UP_STEPS = 500e3,
    ADMISSION_THRESHOLD = 0.5,
    MAX_HISTORY_SIZE = 50, # 300 # 100
    deltaFSP_epsilon = 0.8,
    compete_old_rate = 0.2,
    K_FACTOR = 16,  # 32 原先振荡太大了
    randomized_birth = 1,
    save_interval = 1, # 注意：现在的含义是经过多少次 Batch (每Batch = num_workers个回合)
    opp_greedy_rate = 0, # 对手贪婪率
    num_runs = 3, # 测试回合重复次数
    device = torch.device("cpu"),
    max_il_exponent = -2.0,
    k_shape_il = 0.004,
    R_cage_range = (55.00e3, 55.00e3), # 新增：环境随机化范围
    vertices = None,
    resume_dir = None,
    init_il_data = None, # [新增] 从外部传入预拉取的数据集
    POMDP = 0, # 0全信息，1部分信息
    should_stir = 0, # 是否搅拌策略参数后存储
    adj_r_w = 0, # 是否允许奖励函数权重浮动
    use_RND = 0, # 好奇心机制
    beta_RND = 0.3,
    use_ADistill = 0, # 温和蒸馏机制
    beta_ADistill = 0.04,
    no_bern_distill = 1, # 1: RDistill时不计算bern的KL散度
    distill_learn_type = "dual_prob", # RDistill奖励构造方式: dual_prob(师生KL) 或 single_prob(teacher对真实动作NLL)
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
    # 保留一个常驻 env 供规则教师(RuleTeacherWrapper)做 obs->check_obs 的还原（仅用其无状态的缩放方法）
    rule_teacher_env = copy.deepcopy(dummy_env)
    del dummy_env

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Master training device: {device}")

    # 3. 创建神经网络
    # [MAPPO] 红蓝双方都是 student_agent 的不同实例
    # Critic 输入为红蓝双方全局状态的拼接（先红后蓝），维度 2*state_dim
    global_state_dim = state_dim * 2

    def build_student_agent():
        actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        critic_net = ValueNet(global_state_dim, hidden_dim).to(device)
        actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
        return PPOHybrid(
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
            max_std=label_smoothing,
            rnd_state_dim=global_state_dim,
        )

    student_agent_r = build_student_agent()
    student_agent_b = build_student_agent()
    # 统一顺序：先红后蓝
    student_agents = {'r': student_agent_r, 'b': student_agent_b}
    
    
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
    
    save_meta_once(actor_meta_path, student_agent_b.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent_b.critic.state_dict())

    # [新增] （训练方）蓝方开火策略指标的 EMA 变量（指数=0.2，即 1-0.8）
    ema_fire_interval = None # 50
    ema_fire_delta_psi = None # 30
    ema_fire_distance = None # 50e3
    ema_fire_AA_hor = None # 145
    ema_fire_altitude = None # 3e3
    ema_fire_theta = None # -5
    ema_ATA = None # 37
    ema_delta_psi_threat = None # 135
    ema_delta_theta = None # 4
    EMA_ALPHA = 0.2
    
    fire_inside_weight = None
    fire_reward_weight = None

    RWController = FireRewardWeightController(initial_fire_reward_weight=1.0)

    # 中断续训 [MAPPO] 红蓝双方各自加载
    if resume_dir is not None and os.path.exists(resume_dir):
        for tag, agent in student_agents.items():
            current_actor_path = os.path.join(log_dir, f"current_actor_{tag}.pt")
            if os.path.exists(current_actor_path):
                agent.actor.load_state_dict(torch.load(current_actor_path, map_location=device))
                print(f"Loaded current actor ({tag}) from: {current_actor_path}")
            else:
                # 如果current_actor不存在，回退到最新的 actor_rein 检查点
                actor_files = glob.glob(os.path.join(log_dir, f"actor_rein*_{tag}.pt"))
                if len(actor_files) > 0:
                    def extract_num(f):
                        m = re.search(r'actor_rein(\d+)_[rb]\.pt$', f)
                        return int(m.group(1)) if m else -1
                    latest_actor = max(actor_files, key=extract_num)
                    agent.actor.load_state_dict(torch.load(latest_actor, map_location=device))
                    print(f"Loaded actor ({tag}) from: {latest_actor}")
            
            critic_path = os.path.join(log_dir, f"critic_{tag}.pt")
            if os.path.exists(critic_path):
                agent.critic.load_state_dict(torch.load(critic_path, map_location=device))
                print(f"Loaded critic ({tag}) from: {critic_path}")
        
        opt_path = os.path.join(log_dir, "optimizers_state.pt")
        if os.path.exists(opt_path):
            try:
                opt_states_all = torch.load(opt_path, map_location=device)
                for tag, agent in student_agents.items():
                    opt_states = opt_states_all.get(tag, None)
                    if opt_states is None:
                        continue
                    agent.actor_optimizer.load_state_dict(opt_states['actor_optimizer'])
                    agent.critic_optimizer.load_state_dict(opt_states['critic_optimizer'])
                    print(f"Loaded optimizer states ({tag}).")
                    if agent.rnd_target is not None and opt_states.get('rnd_target') is not None:
                        agent.rnd_target.load_state_dict(opt_states['rnd_target'])
                        print(f"Loaded RND target state ({tag}).")
                    if agent.rnd_prediction is not None and opt_states.get('rnd_prediction') is not None:
                        agent.rnd_prediction.load_state_dict(opt_states['rnd_prediction'])
                        print(f"Loaded RND prediction state ({tag}).")
                    if agent.rnd_optimizer is not None and opt_states.get('rnd_optimizer') is not None:
                        agent.rnd_optimizer.load_state_dict(opt_states['rnd_optimizer'])
                        print(f"Loaded RND optimizer state ({tag}).")
            except Exception as e:
                print(f"Failed to load optimizers: {e}")
        
        # [新增] 恢复 special EMA 状态和控制器状态
        special_json_path = os.path.join(log_dir, "special.json")
        if os.path.exists(special_json_path):
            with open(special_json_path, "r", encoding="utf-8") as f:
                special_data = json.load(f)
            ema_fire_interval = special_data.get("ema_fire_interval", None)
            ema_fire_delta_psi = special_data.get("ema_fire_delta_psi", None)
            ema_fire_distance = special_data.get("ema_fire_distance", None)
            ema_fire_AA_hor = special_data.get("ema_fire_AA_hor", None)
            ema_fire_altitude = special_data.get("ema_fire_altitude", None)
            ema_fire_theta = special_data.get("ema_fire_theta", None)
            ema_ATA = special_data.get("ema_ATA", None)
            ema_delta_psi_threat = special_data.get("ema_delta_psi_threat", None)
            ema_delta_theta = special_data.get("ema_delta_theta", None)
            # [新增] 恢复控制器状态
            if "controller_state" in special_data:
                RWController.load_state_dict(special_data["controller_state"])
                print(f"Loaded controller state from: {special_json_path}")
            print(f"Loaded special EMA states from: {special_json_path}")
    
    # 保存onnx模型 [MAPPO] 红蓝各导出一份
    # 构建 dummy input (batch_size=1)：Actor 用局部观测维度，Critic 用全局拼接维度
    dummy_state = torch.randn(1, state_dim).to(device)
    dummy_global_state = torch.randn(1, global_state_dim).to(device)
    for tag, agent in student_agents.items():
        # ==========================================
        # 1. 导出 Actor 的底层网络（PolicyNetHybrid）
        # ==========================================
        actor_onnx_path = os.path.join(log_dir, f"student_actor_{tag}.onnx")
        # 对于你的 PolicyNetHybrid，它返回的是一个 dict {'cont': ..., 'cat': ..., 'bern': ...}
        # 在高版本的 PyTorch 中，ONNX 对返回 dict 有支持（自动解包），或者你可以写一个简单的 wrapper 解包
        try:
            torch.onnx.export(
                agent.actor.net,                   # 只导出纯网络结构，避开 Wrapper里的复杂采样操作
                dummy_state,                       # 伪造的输入状态
                actor_onnx_path,                   # 输出的文件名 / 路径
                export_params=True,                # 是否连同参数一起导出（选 True 可以看权重信息）
                opset_version=11,                  # 建议使用 11 或以上的算子集
                do_constant_folding=True,          # 是否执行常量折叠优化
                input_names=['state'],             # 命名的输入节点名称
                output_names=['cat_output', 'bern_output'] # 按照返回顺序手动指定名字
            )
            print(f"Actor ONNX ({tag}) successfully exported to {actor_onnx_path}")
        except Exception as e:
            print(f"Error exporting Actor ONNX ({tag}): {e}")
        # ==========================================
        # 2. 导出 Critic 的底层网络（ValueNet）
        # ==========================================
        critic_onnx_path = os.path.join(log_dir, f"student_critic_{tag}.onnx")
        try:
            torch.onnx.export(
                agent.critic,              
                dummy_global_state,                
                critic_onnx_path,                  
                export_params=True,                
                opset_version=11,                 
                do_constant_folding=True,          
                input_names=['state'],             
                output_names=['value_estimate']    # Critic 返回的一般是标量价值
            )
            print(f"Critic ONNX ({tag}) successfully exported to {critic_onnx_path}")
        except Exception as e:
            print(f"Error exporting Critic ONNX ({tag}): {e}")

    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)

    # 5. 模仿学习预训练 (Serial Execution on Master)
    # [MAPPO] 集中式 Critic 输入为红蓝拼接，IL 数据只有单方 state，因此只预训练 actor；
    # critic 输入用 [state, state] 复制拼接，仅用于 MARWIL 的权重计算 (train_critic=False)
    if IL_epoches > 0:
        print("Start IL Training (actor only, MAPPO)...")

    il_dict_ma = None
    if original_il_transition_dict is not None and IL_epoches > 0:
        il_states_np = np.array(original_il_transition_dict['states'], dtype=np.float32)
        il_dict_ma = dict(original_il_transition_dict)
        il_dict_ma['obs'] = il_states_np  # Actor 用局部观测
        il_dict_ma['states'] = np.concatenate([il_states_np, il_states_np], axis=1)  # Critic 占位输入

    for tag, agent in student_agents.items():
        agent.set_learning_rate(actor_lr=actor_lr_init_il, critic_lr=critic_lr_init_il)
    
    # === 模仿训练循环 ===
    # 现在 original_il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(IL_epoches): 
        for tag, agent in student_agents.items():
            avg_actor_loss, avg_critic_loss, c = agent.MARWIL_update(
                il_dict_ma, 
                beta=beta_mixed, 
                batch_size=il_batch_size, # 显存如果够大可以适当调大
                label_smoothing=label_smoothing,
                no_bern = 0, # 0
                train_critic=False, # [MAPPO] 只预训练 actor
            )
            
            # 记录
            if epoch % 1 == 0:
                logger.add(f"il_train/avg_actor_loss_{tag}", avg_actor_loss, epoch)

                # [新增] MARWIL 监控：每个动作头独立的 NLL 与策略熵 (全采样固定 batch, no_grad)
                for _name, _val in [
                    (f"il_train/nll_cont_{tag}", getattr(agent, "marwil_nll_cont", None)),
                    (f"il_train/nll_cat_{tag}", getattr(agent, "marwil_nll_cat", None)),
                    (f"il_train/entropy_cont_{tag}", getattr(agent, "marwil_entropy_cont", None)),
                    (f"il_train/entropy_cat_{tag}", getattr(agent, "marwil_entropy_cat", None)),
                    (f"il_train/accuracy_cont_{tag}", getattr(agent, "marwil_accuracy_cont", None)),
                    (f"il_train/accuracy_cat_{tag}", getattr(agent, "marwil_accuracy_cat", None)),
                    (f"il_train/accuracy_bern_{tag}", getattr(agent, "marwil_accuracy_bern", None)),
                    (f"il_train/weight_mean_{tag}", getattr(agent, "marwil_weight_mean", None)),
                    (f"il_train/weight_clip_frac_{tag}", getattr(agent, "marwil_weight_clip_frac", None)),
                    (f"il_train/adv_std_{tag}", getattr(agent, "marwil_adv_std", None)),
                    (f"il_train/adv_max_{tag}", getattr(agent, "marwil_adv_max", None)),
                    (f"il_train/adv_mean_{tag}", getattr(agent, "marwil_adv_mean", None)),
                    (f"il_train/adv_positive_frac_{tag}", getattr(agent, "marwil_adv_positive_frac", None)),
                ]:
                    if _val is not None:
                        logger.add(_name, _val, epoch)

                print(f"Epoch {epoch} ({tag}): Actor Loss: {avg_actor_loss:.4f}")
    
    # MARWIL 结束后恢复 bern bias，防止稀疏开火动作被拉向高熵中间态（无济于事）
    for tag, agent in student_agents.items():
        if hasattr(agent.actor.net, 'fc_bern'):
            with torch.no_grad():
                agent.actor.net.fc_bern[-1].bias.clamp_(max=-2.5)
    
    if IL_epoches > 0:
        print("IL Training Finished.")
    else:
        print("No IL")
    
    # 存储在线训练前的网络参数
    for tag, agent in student_agents.items():
        torch.save(agent.actor.state_dict(), os.path.join(log_dir, f"actor_rein0_{tag}.pt"))

    # [MAPPO] SIL 已随 PFSP 一并屏蔽，不再维护 IL 混合缓冲区
    il_transition_buffer = None

    # ==============================================================================
    # 强化学习 (MAPPO 纯自博弈: 红方最新 vs 蓝方最新) 阶段
    # ==============================================================================
    for tag, agent in student_agents.items():
        agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
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

    # [MAPPO] 不再维护 ELO/PFSP/名人堂，仅保留少量统计
    Elite_Fire_Stats_path = os.path.join(log_dir, "Elite_Fire_Stats.json")
    train_state_path = os.path.join(log_dir, "train_state.json")
    Elite_Fire_Stats = {}

    # 尝试加载历史
    total_steps = 0
    batch_idx = 0
    if os.path.exists(train_state_path):
        with open(train_state_path, 'r', encoding='utf-8') as f:
            train_state = json.load(f)
        total_steps = train_state.get("__LAST_UPDATE_STEP__", 0)
        batch_idx = train_state.get("__LAST_UPDATE_BATCH__", 0)
        print(f"Resumed training state: steps={total_steps}, batch={batch_idx}")
    if os.path.exists(Elite_Fire_Stats_path):
        with open(Elite_Fire_Stats_path, 'r', encoding='utf-8') as f:
            Elite_Fire_Stats = json.load(f)

    if collape_recover["collapsed"]:
        actor_freeze_until = batch_idx + int(collape_recover["actor_frozen_batchs"])
        for agent in student_agents.values():
            agent.reset_optimizer() # 恢复训练清除动量
    else:
        actor_freeze_until = -1
    trigger = trigger0 + (total_steps // trigger_delta) * trigger_delta
    
    current_max_steps = int(max_steps)
    
    # 全局 Buffer (用于攒够 Batch 训练)
    empty_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
    # [MAPPO] 红蓝各自的训练 buffer
    red_transition_dict = copy.deepcopy(empty_transition_dict)
    blue_transition_dict = copy.deepcopy(empty_transition_dict)

    # 初始化基于胜率的在线 EMA 变量
    ema_score = 0.5
    ema_step = 0
    target_p1 = 0.65
    ppo_grad_ema = None  # [新增] 初始化 PPO 梯度 EMA 缓存
    rnd_mse = None       # RND 原始 MSE（上一批次的值，首批为 None）

    # =========================================================
    # 主循环辅助函数 (避免在循环内部定义函数)
    # =========================================================
    def launch_test_tasks(weights, tag, deterministic):
        tasks = []
        for r_idx in [0, 1, 2, 3, 4]:
            obj = test_pool.apply_async(
                test_worker,
                kwds={
                    'model_state_dict': weights,
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
                    'deterministic': deterministic,
                    'restrict_fire': True,
                    'vertices': vertices,
                }
            )
            tasks.append(obj)
        return [t.get() for t in tasks]

    def log_test_results(test_results, prefix, total_steps_):
        outcomes = {rule_num: score for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
        outcomes_return = {rule_num: result2 for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}
        outcomes_perish = {rule_num: p_t_ for rule_num, score, result2, wins, loses, draws, p_t_ in test_results}

        for r_num, score in outcomes.items():
            logger.add(f"{prefix}/agent_vs_rule{r_num}", score, total_steps_)
            print(f"  [{prefix}] Rule_{r_num}: {score} (return: {outcomes_return[r_num]:.2f})")

        avg_score = np.mean(list(outcomes.values()))
        avg_perish_together = np.mean(list(outcomes_perish.values()))
        logger.add(f"{prefix}/avg_score", avg_score, total_steps_)
        logger.add(f"{prefix}/BVR perish together", avg_perish_together, total_steps_)

    def _ema_update(ema_val, batch_val, alpha=EMA_ALPHA):
        if batch_val is None:
            return ema_val
        return batch_val if ema_val is None else (1 - alpha) * ema_val + alpha * batch_val

    # =========================================================
    # 主循环 (Master Process)
    # =========================================================
    while True:
        while total_steps < current_max_steps:
            if total_steps < 5e3:
                fire_mask = 0 # 0 # 全程开启开火mask
            else:
                fire_mask = 0
            # --- 【修改】同步并行测试阶段 ---
            # 只有测试跑完并处理完名人堂，才进入下一步的采样和仿真
            # --- 1. 并行测试触发逻辑 (Async) ---
            if total_steps >= trigger:
                print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
                # 1. 深度拷贝当前红蓝 Actor 权重到 CPU 内存
                red_weights = {k: v.cpu().clone() for k, v in student_agent_r.actor.state_dict().items()}
                blue_weights = {k: v.cpu().clone() for k, v in student_agent_b.actor.state_dict().items()}

                # 2. 分发测试任务并【立即阻塞等待】
                # # 注意：这里直接用 list comprehension 配合 .get() 实现阻塞
                # test_tasks = []
                # for r_idx in [0, 1, 2, 3, 4, 5, 6]:
                #     obj = test_pool.apply_async(
                #         test_worker, 
                #         # args=(current_weights, r_idx, args, 
                #         #     state_dim, hidden_dim, action_dims_dict, 
                #         #     dt_maneuver, 'cpu', num_runs, action_cycle_multiplier)
                #         kwds={
                #             'model_state_dict': current_weights,
                #             'rule_num': r_idx,
                #             'env_args': args,
                #             'state_dim': state_dim,
                #             'hidden_dim': hidden_dim,
                #             'action_dims_dict': action_dims_dict,
                #             'dt_maneuver_val': dt_maneuver,
                #             'device_name': 'cpu',
                #             'num_runs': num_runs,
                #             'action_cycle_multiplier': action_cycle_multiplier,
                #             'no_out': 0,  # 这里可以根据需要设为 1
                #             'deterministic': False,
                #             'restrict_fire': True, # False, 和采样保持一致
                #             'vertices': vertices,
                #         }
                #     )
                #     test_tasks.append(obj)
                
                # # 第二种形式：追加额外测试 (机动动作确定化 + 动作次序限制打开)
                # test_tasks_no_random = []
                # for r_idx in [0, 1, 2, 3, 4, 5, 6]:
                #     obj = test_pool.apply_async(
                #         test_worker, 
                #         kwds={
                #             'model_state_dict': current_weights,
                #             'rule_num': r_idx,
                #             'env_args': args,
                #             'state_dim': state_dim,
                #             'hidden_dim': hidden_dim,
                #             'action_dims_dict': action_dims_dict,
                #             'dt_maneuver_val': dt_maneuver,
                #             'device_name': 'cpu',
                #             'num_runs': num_runs,
                #             'action_cycle_multiplier': action_cycle_multiplier,
                #             'no_out': 0,
                #             'deterministic': True,     # 机动动作确定化
                #             'restrict_fire': True,      # 动作次序限制打开
                #             'vertices': vertices,
                #         }
                #     )
                #     test_tasks_no_random.append(obj)
                
                # [MAPPO] 红蓝双方都测，每种确定性设置下各 5 个 rule
                test_results_b = launch_test_tasks(blue_weights, 'b', False)
                test_results_b_nr = launch_test_tasks(blue_weights, 'b', True)
                test_results_r = launch_test_tasks(red_weights, 'r', False)
                test_results_r_nr = launch_test_tasks(red_weights, 'r', True)

                log_test_results(test_results_b, "test_b", total_steps)
                log_test_results(test_results_b_nr, "test_b_No_random", total_steps)
                log_test_results(test_results_r, "test_r", total_steps)
                log_test_results(test_results_r_nr, "test_r_No_random", total_steps)

                trigger += trigger_delta

            # --- 2. 准备训练 Batch (Synchronous) ---
            # 改变环境奖励权重，超过100轮采样再更新权重，每次权重维持5轮采样
            if adj_r_w and batch_idx > 10:
                fire_inside_weight, fire_reward_weight = RWController.update({
                    'ema_fire_interval': ema_fire_interval,
                    'ema_fire_distance': ema_fire_distance,
                    'ema_fire_altitude': ema_fire_altitude,
                    'ema_fire_delta_psi': ema_fire_delta_psi,
                    'ema_fire_theta': ema_fire_theta,
                    'ema_ATA': ema_ATA,
                    'ema_delta_psi_threat': ema_delta_psi_threat,
                    'ema_delta_theta': ema_delta_theta
                })
                logger.add(f"SPECIAL/开火权重", fire_reward_weight, total_steps)
                logger.add(f"SPECIAL/0 W_d_fire", fire_inside_weight[0], total_steps)
                logger.add(f"SPECIAL/1 W_t_since_fire", fire_inside_weight[1], total_steps)
                logger.add(f"SPECIAL/2 W_AA_fire", fire_inside_weight[2], total_steps)
                logger.add(f"SPECIAL/3 W_psi_fire", fire_inside_weight[3], total_steps)
                logger.add(f"SPECIAL/4 W_v_fire", fire_inside_weight[4], total_steps)
                logger.add(f"SPECIAL/5 W_theta_fire", fire_inside_weight[5], total_steps)
            
            if not adj_r_w:
                fire_inside_weight = None
                fire_reward_weight = None

            # A. 获取当前策略权重 (CPU)
            # [MAPPO] 红蓝双方都用最新权重
            red_actor_weights = {k: v.cpu() for k, v in student_agent_r.actor.state_dict().items()}
            blue_actor_weights = {k: v.cpu() for k, v in student_agent_b.actor.state_dict().items()}
            
            # B. 分发任务给 Worker
            # [MAPPO] 不再采样对手，红蓝双方最新的 agent 直接对局
            worker_metrics_buffer = [] # 暂存本轮 metrics 方便打印
            
            for rank in range(num_workers):
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
                    'end_reward_weight': 0.556, # np.clip(total_steps/5e3, 0, 0.5),
                    'fire_inside_weight': fire_inside_weight,
                    'fire_reward_weight': fire_reward_weight,
                }
                
                # 发送指令 pipe.send [MAPPO] 发送红蓝双方最新权重
                pipes[rank].send(('RUN_EPISODE', (red_actor_weights, blue_actor_weights, settings)))

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
            batch_bvr_perish_together_cnt = 0 # 新增统计
            batch_total_return = 0    # 新增统计
            batch_total_dense_return = 0
            batch_total_m_fired = 0   # 新增统计
            
            # 新增: 批次开火策略指标统计
            batch_blue_fire_intervals = []
            batch_blue_fire_delta_psis = []
            batch_blue_fire_distances = []
            batch_blue_fire_AA_hors = []
            batch_blue_fire_alts = []
            batch_blue_fire_thetas = []
            batch_blue_ATAs = []
            batch_blue_delta_psi_threats = []
            batch_blue_delta_thetas = []
            
            for res in batch_results:
                # res 结构: {'blue_trans':..., 'red_trans':..., 'metrics':...}
                b_tr = res['blue_trans'] # 蓝方 PPO 训练数据
                r_tr = res['red_trans']  # 红方 PPO 训练数据
                metrics = res['metrics']
                
                # --- 新增: 更新红方开火角度参数的 EMA ---
                ep_avg_fire_theta = res.get('ep_avg_fire_theta')
                ep_avg_ATA = res.get('ep_avg_ATA')
                ep_avg_delta_psi_threat = res.get('ep_avg_delta_psi_threat')
                ep_avg_delta_theta = res.get('ep_avg_delta_theta')
                ep_avg_delta_psi = res.get('ep_avg_delta_psi')
                
                # 更新 EMA [fire_theta, ATA, delta_psi_threat, delta_theta, delta_psi]
                alpha_ema = 0.1
                red_main_key = "__RED_MAIN__"
                if red_main_key not in Elite_Fire_Stats:
                    Elite_Fire_Stats[red_main_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    
                old_red_stats = Elite_Fire_Stats[red_main_key]
                new_red_stats = list(old_red_stats)
                
                if ep_avg_fire_theta is not None:
                    new_red_stats[0] = alpha_ema * ep_avg_fire_theta + (1 - alpha_ema) * old_red_stats[0]
                if ep_avg_ATA is not None:
                    new_red_stats[1] = alpha_ema * ep_avg_ATA + (1 - alpha_ema) * old_red_stats[1]
                if ep_avg_delta_psi_threat is not None:
                    new_red_stats[2] = alpha_ema * ep_avg_delta_psi_threat + (1 - alpha_ema) * old_red_stats[2]
                if ep_avg_delta_theta is not None:
                    new_red_stats[3] = alpha_ema * ep_avg_delta_theta + (1 - alpha_ema) * old_red_stats[3]
                if ep_avg_delta_psi is not None:
                    new_red_stats[4] = alpha_ema * ep_avg_delta_psi + (1 - alpha_ema) * old_red_stats[4]
                    
                Elite_Fire_Stats[red_main_key] = new_red_stats
                
                # --- 新增: 更新蓝方（本方/主代理）开火角度参数的 EMA ---
                ep_blue_avg_fire_theta = res.get('ep_blue_avg_fire_theta')
                ep_blue_avg_ATA = res.get('ep_blue_avg_ATA')
                ep_blue_avg_delta_psi_threat = res.get('ep_blue_avg_delta_psi_threat')
                ep_blue_avg_delta_theta = res.get('ep_blue_avg_delta_theta')
                ep_blue_avg_delta_psi = res.get('ep_blue_avg_delta_psi')
                
                # 使用当前主代理的key来记录本方行为
                main_agent_key = "__CURRENT_MAIN__"
                if main_agent_key not in Elite_Fire_Stats:
                    Elite_Fire_Stats[main_agent_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                old_main_stats = Elite_Fire_Stats[main_agent_key]
                new_main_stats = list(old_main_stats)
                
                if ep_blue_avg_fire_theta is not None:
                    new_main_stats[0] = alpha_ema * ep_blue_avg_fire_theta + (1 - alpha_ema) * old_main_stats[0]
                if ep_blue_avg_ATA is not None:
                    new_main_stats[1] = alpha_ema * ep_blue_avg_ATA + (1 - alpha_ema) * old_main_stats[1]
                if ep_blue_avg_delta_psi_threat is not None:
                    new_main_stats[2] = alpha_ema * ep_blue_avg_delta_psi_threat + (1 - alpha_ema) * old_main_stats[2]
                if ep_blue_avg_delta_theta is not None:
                    new_main_stats[3] = alpha_ema * ep_blue_avg_delta_theta + (1 - alpha_ema) * old_main_stats[3]
                if ep_blue_avg_delta_psi is not None:
                    new_main_stats[4] = alpha_ema * ep_blue_avg_delta_psi + (1 - alpha_ema) * old_main_stats[4]
                    
                Elite_Fire_Stats[main_agent_key] = new_main_stats
                
                # [新增] 填充 buffer 用户打印详情
                result_str = "Win" if metrics['win'] else ("Lose" if metrics['lose'] else "Draw")
                worker_metrics_buffer.append(result_str)
                
                batch_total_steps += metrics['steps']
                batch_total_return += metrics['return']
                batch_total_dense_return += metrics['dense_return']
                batch_total_m_fired += metrics['m_fired']
                if metrics.get('BVR_perish_together', False):
                    batch_bvr_perish_together_cnt += 1

                # 收集蓝方开火策略指标
                ep_blue_avg_fire_interval = res.get('ep_blue_avg_fire_interval')
                ep_blue_avg_fire_delta_psi = res.get('ep_blue_avg_fire_delta_psi')
                ep_blue_avg_fire_distance = res.get('ep_blue_avg_fire_distance')
                ep_blue_avg_fire_AA_hor = res.get('ep_blue_avg_fire_AA_hor')
                ep_blue_avg_fire_altitude = res.get('ep_blue_avg_fire_altitude')
                ep_blue_avg_fire_theta = res.get('ep_blue_avg_fire_theta')
                
                if ep_blue_avg_fire_interval is not None:
                    batch_blue_fire_intervals.append(ep_blue_avg_fire_interval)
                if ep_blue_avg_fire_delta_psi is not None:
                    batch_blue_fire_delta_psis.append(ep_blue_avg_fire_delta_psi)
                if ep_blue_avg_fire_distance is not None:
                    batch_blue_fire_distances.append(ep_blue_avg_fire_distance)
                if ep_blue_avg_fire_AA_hor is not None:
                    batch_blue_fire_AA_hors.append(ep_blue_avg_fire_AA_hor)
                if ep_blue_avg_fire_altitude is not None:
                    batch_blue_fire_alts.append(ep_blue_avg_fire_altitude)
                if ep_blue_avg_fire_theta is not None:
                    batch_blue_fire_thetas.append(ep_blue_avg_fire_theta)
                
                if ep_blue_avg_ATA is not None:
                    batch_blue_ATAs.append(ep_blue_avg_ATA)
                if ep_blue_avg_delta_psi_threat is not None:
                    batch_blue_delta_psi_threats.append(ep_blue_avg_delta_psi_threat)
                if ep_blue_avg_delta_theta is not None:
                    batch_blue_delta_thetas.append(ep_blue_avg_delta_theta)

                if metrics['win']: batch_wins += 1
                elif metrics['lose']: batch_loss_cnt += 1
                else: 
                    batch_draw_cnt += 1
                
                # 3.1 聚合 PPO 数据到红蓝全局 Buffer
                for k in blue_transition_dict:
                    blue_transition_dict[k].extend(b_tr[k])
                for k in red_transition_dict:
                    red_transition_dict[k].extend(r_tr[k])
            
            # 计算蓝方开火策略指标的批次平均值
            batch_blue_avg_fire_interval = float(np.mean(batch_blue_fire_intervals)) if batch_blue_fire_intervals else None
            batch_blue_avg_fire_delta_psi = float(np.mean(batch_blue_fire_delta_psis)) if batch_blue_fire_delta_psis else None
            batch_blue_avg_fire_distance = float(max(batch_blue_fire_distances)) if batch_blue_fire_distances else None
            batch_blue_avg_fire_AA_hor = float(np.mean(batch_blue_fire_AA_hors)) if batch_blue_fire_AA_hors else None
            batch_blue_avg_fire_altitude = float(np.mean(batch_blue_fire_alts)) if batch_blue_fire_alts else None
            batch_blue_avg_fire_theta = float(np.mean(batch_blue_fire_thetas)) if batch_blue_fire_thetas else None
            batch_blue_avg_ATA = float(np.mean(batch_blue_ATAs)) if batch_blue_ATAs else None
            batch_blue_avg_delta_psi_threat = float(np.mean(batch_blue_delta_psi_threats)) if batch_blue_delta_psi_threats else None
            batch_blue_avg_delta_theta = float(np.mean(batch_blue_delta_thetas)) if batch_blue_delta_thetas else None

            # [新增] 用 EMA(指数=0.2) 平滑批次均值
            ema_fire_interval = _ema_update(ema_fire_interval, batch_blue_avg_fire_interval if batch_blue_avg_fire_interval is not None else None)
            ema_fire_delta_psi = _ema_update(ema_fire_delta_psi, batch_blue_avg_fire_delta_psi*180/pi if batch_blue_avg_fire_delta_psi is not None else None)
            ema_fire_distance = _ema_update(ema_fire_distance, batch_blue_avg_fire_distance if batch_blue_avg_fire_distance is not None else None)
            ema_fire_AA_hor = _ema_update(ema_fire_AA_hor, batch_blue_avg_fire_AA_hor*180/pi if batch_blue_avg_fire_AA_hor is not None else None)
            ema_fire_altitude = _ema_update(ema_fire_altitude, batch_blue_avg_fire_altitude if batch_blue_avg_fire_altitude is not None else None)
            ema_fire_theta = _ema_update(ema_fire_theta, batch_blue_avg_fire_theta*180/pi if batch_blue_avg_fire_theta is not None else None)
            ema_ATA = _ema_update(ema_ATA, batch_blue_avg_ATA*180/pi if batch_blue_avg_ATA is not None else None)
            ema_delta_psi_threat = _ema_update(ema_delta_psi_threat, batch_blue_avg_delta_psi_threat*180/pi if batch_blue_avg_delta_psi_threat is not None else None)
            ema_delta_theta = _ema_update(ema_delta_theta, batch_blue_avg_delta_theta*180/pi if batch_blue_avg_delta_theta is not None else None)

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
            if use_RND and rnd_mse is not None:
                logger.add("train_plus/RND_mse", rnd_mse, total_steps)
            logger.add("train_plus/batch_score", batch_score, total_steps)
            logger.add("train_plus/filtered_score", filtered_score, total_steps)
            logger.add("train_plus/target_p1", target_p1, total_steps)

            # 记录开火策略指标 - 蓝方（本方），使用原始批次均值
            if batch_blue_avg_fire_interval is not None:
                logger.add("special/1 开火间隔时长", batch_blue_avg_fire_interval, total_steps)
            if batch_blue_avg_fire_delta_psi is not None:
                logger.add("special/2 开火abs(delta_psi)", batch_blue_avg_fire_delta_psi*180/pi, total_steps)
            if batch_blue_avg_fire_distance is not None:
                logger.add("special/3 开火距离", batch_blue_avg_fire_distance, total_steps)
            if batch_blue_avg_fire_AA_hor is not None:
                logger.add("special/4 开火abs(AA_hor)", batch_blue_avg_fire_AA_hor*180/pi, total_steps)
            if batch_blue_avg_fire_altitude is not None:
                logger.add("special/0 开火高度", batch_blue_avg_fire_altitude, total_steps)
            if batch_blue_avg_fire_theta is not None:
                logger.add("special/5 fire_theta", batch_blue_avg_fire_theta*180/pi, total_steps)
            if batch_blue_avg_ATA is not None:
                logger.add("special/6 ATA30", batch_blue_avg_ATA*180/pi, total_steps)
            if batch_blue_avg_delta_psi_threat is not None:
                logger.add("special/7 delta_psi_threat", batch_blue_avg_delta_psi_threat*180/pi, total_steps)
            if batch_blue_avg_delta_theta is not None:
                logger.add("special/8 delta_theta30", batch_blue_avg_delta_theta*180/pi, total_steps)
            
            # [新增] 保存 EMA 状态和控制器状态到 special.json
            special_data = {
                "ema_fire_interval": ema_fire_interval,
                "ema_fire_delta_psi": ema_fire_delta_psi,
                "ema_fire_distance": ema_fire_distance,
                "ema_fire_AA_hor": ema_fire_AA_hor,
                "ema_fire_altitude": ema_fire_altitude,
                "ema_fire_theta": ema_fire_theta,
                "ema_ATA": ema_ATA,
                "ema_delta_psi_threat": ema_delta_psi_threat,
                "ema_delta_theta": ema_delta_theta,
                "controller_state": RWController.state_dict(),  # [新增] 保存控制器状态
            }
            with open(os.path.join(log_dir, "special.json"), "w", encoding="utf-8") as f:
                json.dump(special_data, f, ensure_ascii=False, indent=2)
            
            # 记录平均回报与胜率
            logger.add("train/1 avg_episode_return", batch_total_return / num_workers, total_steps)
            logger.add("train_plus/Avg dense return", batch_total_dense_return / num_workers, total_steps)
            logger.add("train/2 win", batch_wins / num_workers, total_steps)
            logger.add("train/2 lose", batch_loss_cnt / num_workers, total_steps)
            logger.add("train/2 draw", batch_draw_cnt / num_workers, total_steps)
            logger.add("train/2 BVR perish together", batch_bvr_perish_together_cnt / num_workers, total_steps)
            logger.add("train/2 BVR not end", (batch_draw_cnt-batch_bvr_perish_together_cnt) / num_workers, total_steps)
            # 找最好的智能体
            logger.add("agent/ episode_step", batch_idx * num_workers, total_steps)
            logger.add("agent/ batch_step", batch_idx, total_steps)

            # --- 5. 更新，保存与维护 (Checkpoint & Pool) ---
            if batch_idx % save_interval == 0 and \
                len(blue_transition_dict['dones']) >= transition_dict_threshold:
                # 重构 Action 结构 (List[Dict] -> Dict[Array])
                blue_transition_dict['actions'] = restructure_actions(blue_transition_dict['actions'])
                red_transition_dict['actions'] = restructure_actions(red_transition_dict['actions'])
                
                # 学习率warm_up
                actor_lr = min(actor_lr0, actor_lr0 * total_steps/1e6)
                critic_lr = min(critic_lr0, critic_lr0 * total_steps/1e6)
                student_agent_b.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
                student_agent_r.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

                if batch_idx <= actor_freeze_until:
                    freeze_actor = 1
                else:
                    freeze_actor = 0
                # # critic先收敛
                # if total_steps < 5e3:
                #     freeze_actor = 1
                
                max_fire_logits = 4.0

                # 随机拜师法 (ADistill) 已随 PFSP/ELO 一并移除
                RDistill_kl = None

                if use_RND:
                    blue_transition_dict, rnd_mse_b = student_agent_b.RND_calc(blue_transition_dict, beta=beta_RND)
                    red_transition_dict, rnd_mse_r = student_agent_r.RND_calc(red_transition_dict, beta=beta_RND)
                    if rnd_mse_b is not None and rnd_mse_r is not None:
                        rnd_mse = (rnd_mse_b + rnd_mse_r) / 2.0
                    else:
                        rnd_mse = rnd_mse_b if rnd_mse_b is not None else rnd_mse_r
                else:
                    rnd_mse = None

                student_agent_b.update(blue_transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed, target_p1=target_p1,
                                     k_nonlinear=k_nonlinear, mask_on=fire_mask, actor_frozen=freeze_actor, bern_max_logits=max_fire_logits)
                student_agent_r.update(red_transition_dict, adv_normed=1, mini_batch_size=mini_batch_size_mixed, target_p1=target_p1,
                                     k_nonlinear=k_nonlinear, mask_on=fire_mask, actor_frozen=freeze_actor, bern_max_logits=max_fire_logits)

                # 开火概率保护，如果策略向满开火/不开一发坍缩，直接用有监督暴力修正开火概率
                if batch_idx % 10 == 0:
                    student_agent_b.fire_prob_protection(blue_transition_dict, protect_epochs=4)
                    student_agent_r.fire_prob_protection(red_transition_dict, protect_epochs=4)
                
                # # 机动概率保护，未被调好，无法区分告警状态的有无，加上进攻引导就不会躲，加上防御引导又不会进攻
                # if  \
                #     (batch_blue_avg_fire_delta_psi*180/pi > 60 or
                #         batch_blue_avg_delta_psi_threat*180/pi < 90):
                #     student_agent.maneuver_il_protection(transition_dict, alpha=3, epochs=12)

                # 计算/更新 PPO actor pre-clip 梯度的 EMA 值
                current_ppo_grad_b = student_agent_b.pre_clip_actor_grad
                current_ppo_grad_r = student_agent_r.pre_clip_actor_grad
                for g in [current_ppo_grad_b, current_ppo_grad_r]:
                    if g is not None and not np.isnan(g):
                        if ppo_grad_ema is None:
                            ppo_grad_ema = g
                        else:
                            ppo_grad_ema = 0.95 * ppo_grad_ema + 0.05 * g
                
                # 记录 Log
                for tag, agent in [('b', student_agent_b), ('r', student_agent_r)]:
                    logger.add(f"train/7 actor_loss_{tag}", agent.actor_loss, total_steps)
                    logger.add(f"train/8 critic_loss_{tag}", agent.critic_loss, total_steps)
                    logger.add(f"train/9 entropy_{tag}", agent.entropy_mean, total_steps)
                    logger.add(f"train/9 entropy_cat_{tag}", agent.entropy_cat, total_steps)
                    logger.add(f"train/9 entropy_bern_{tag}", agent.entropy_bern, total_steps)
                    logger.add(f"train_plus/max_fire_prob_{tag}", agent.max_fire_prob, total_steps)
                    logger.add(f"train_plus/min_fire_prob_{tag}", agent.min_fire_prob, total_steps)
                    logger.add(f"train_plus/td_error_var_{tag}", agent.td_error_var, total_steps)

                print(f"Step {total_steps}: Batch WinRate {batch_wins}/{num_workers}")

                # 原本是在这里清空Buffer的，但是现在要在搅拌之后清空，所以移到了后面
                
                # A. 保存模型
                
                if should_stir:
                    # 策略搅拌：计算目标熵并执行搅拌
                    # cat熵从0step的2到20Mstep的1.5线性退火
                    max_steps_for_stir = 20 * 1e6  # 20M steps
                    cat_entropy_start = 2.0
                    cat_entropy_end = 1.5
                    
                    # 线性插值计算当前目标cat熵
                    progress = min(total_steps / max_steps_for_stir, 1.0)
                    target_cat_entropy = cat_entropy_start + (cat_entropy_end - cat_entropy_start) * progress
                    
                    target_entropies = {
                        'cont': 0.0,  # 连续动作目标熵为0
                        'cat': target_cat_entropy,  # 离散动作线性退火
                        'bern': 0.0  # 伯努利动作目标熵为0
                    }
                    
                    print(f"  [should_stir] Target cat entropy: {target_cat_entropy:.3f} (progress: {progress:.3f})")
                    
                    # 执行策略搅拌
                    blue_stirred, blue_entropy_info = student_agent_b.Stir(blue_transition_dict, target_entropies, max_steps=50, lr=0.01)
                    red_stirred, red_entropy_info = student_agent_r.Stir(red_transition_dict, target_entropies, max_steps=50, lr=0.01)
                    
                    # 保存搅拌后的模型参数
                    torch.save(blue_stirred, os.path.join(log_dir, f"actor_rein{batch_idx}_b.pt"))
                    torch.save(student_agent_b.critic.state_dict(), os.path.join(log_dir, "critic_b.pt"))
                    torch.save(student_agent_b.actor.state_dict(), os.path.join(log_dir, "current_actor_b.pt"))
                    torch.save(red_stirred, os.path.join(log_dir, f"actor_rein{batch_idx}_r.pt"))
                    torch.save(student_agent_r.critic.state_dict(), os.path.join(log_dir, "critic_r.pt"))
                    torch.save(student_agent_r.actor.state_dict(), os.path.join(log_dir, "current_actor_r.pt"))
                    
                    # 记录搅拌后的熵值
                    logger.add("stir/cat_entropy_b", blue_entropy_info['cat_entropy'], total_steps)
                    logger.add("stir/bern_entropy_b", blue_entropy_info['bern_entropy'], total_steps)
                    logger.add("stir/cont_entropy_b", blue_entropy_info['cont_entropy'], total_steps)
                    logger.add("stir/cat_entropy_r", red_entropy_info['cat_entropy'], total_steps)
                    logger.add("stir/bern_entropy_r", red_entropy_info['bern_entropy'], total_steps)
                    logger.add("stir/cont_entropy_r", red_entropy_info['cont_entropy'], total_steps)
                    logger.add("stir/target_cat_entropy", target_cat_entropy, total_steps)
                    
                    print(f"  [should_stir] Blue cat entropy: {blue_entropy_info['cat_entropy']:.3f}, bern entropy: {blue_entropy_info['bern_entropy']:.3f}, cont entropy: {blue_entropy_info['cont_entropy']:.3f}")
                    print(f"  [should_stir] Red  cat entropy: {red_entropy_info['cat_entropy']:.3f}, bern entropy: {red_entropy_info['bern_entropy']:.3f}, cont entropy: {red_entropy_info['cont_entropy']:.3f}")
                    
                    print(f"Saved Stirred Checkpoints: actor_rein{batch_idx}_b.pt, actor_rein{batch_idx}_r.pt")
                    print(f"Saved Current Actors: current_actor_b.pt, current_actor_r.pt")
                else:
                    # 正常保存模型
                    torch.save(student_agent_b.actor.state_dict(), os.path.join(log_dir, f"actor_rein{batch_idx}_b.pt"))
                    torch.save(student_agent_b.critic.state_dict(), os.path.join(log_dir, "critic_b.pt"))
                    torch.save(student_agent_b.actor.state_dict(), os.path.join(log_dir, "current_actor_b.pt"))
                    torch.save(student_agent_r.actor.state_dict(), os.path.join(log_dir, f"actor_rein{batch_idx}_r.pt"))
                    torch.save(student_agent_r.critic.state_dict(), os.path.join(log_dir, "critic_r.pt"))
                    torch.save(student_agent_r.actor.state_dict(), os.path.join(log_dir, "current_actor_r.pt"))
                    print(f"Saved Checkpoints: actor_rein{batch_idx}_b.pt, actor_rein{batch_idx}_r.pt")
                    print(f"Saved Current Actors: current_actor_b.pt, current_actor_r.pt")
                
                # 清空 Buffer（在搅拌之后）
                blue_transition_dict = copy.deepcopy(empty_transition_dict)
                red_transition_dict = copy.deepcopy(empty_transition_dict)

                # B. 维护 Elite_Fire_Stats 中的历史版本记录
                if total_steps >= WARM_UP_STEPS:
                    # 复制当前蓝方/红方行为统计到该历史版本
                    if "__CURRENT_MAIN__" in Elite_Fire_Stats:
                        Elite_Fire_Stats[f"actor_rein{batch_idx}_b"] = copy.deepcopy(Elite_Fire_Stats["__CURRENT_MAIN__"])
                    else:
                        Elite_Fire_Stats[f"actor_rein{batch_idx}_b"] = [0.0, 0.0, 0.0, 0.0, 0.0]
                    if "__RED_MAIN__" in Elite_Fire_Stats:
                        Elite_Fire_Stats[f"actor_rein{batch_idx}_r"] = copy.deepcopy(Elite_Fire_Stats["__RED_MAIN__"])
                    else:
                        Elite_Fire_Stats[f"actor_rein{batch_idx}_r"] = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                # 保存 Elite_Fire_Stats
                with open(Elite_Fire_Stats_path, "w", encoding="utf-8") as f:
                    json.dump(Elite_Fire_Stats, f, ensure_ascii=False, indent=2)

                # --- 例行保存优化器状态和步数 ---
                opt_states_all = {}
                for tag, agent in [('b', student_agent_b), ('r', student_agent_r)]:
                    opt_states_all[tag] = {
                        'actor_optimizer': agent.actor_optimizer.state_dict(),
                        'critic_optimizer': agent.critic_optimizer.state_dict(),
                        'rnd_target': agent.rnd_target.state_dict() if agent.rnd_target is not None else None,
                        'rnd_prediction': agent.rnd_prediction.state_dict() if agent.rnd_prediction is not None else None,
                        'rnd_optimizer': agent.rnd_optimizer.state_dict() if agent.rnd_optimizer is not None else None,
                    }
                torch.save(opt_states_all, os.path.join(log_dir, "optimizers_state.pt"))

                # 保存训练状态
                train_state = {
                    "__LAST_UPDATE_STEP__": total_steps,
                    "__LAST_UPDATE_BATCH__": batch_idx,
                }
                with open(train_state_path, "w", encoding="utf-8") as f:
                    json.dump(train_state, f, ensure_ascii=False, indent=2)

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

    # 最终保存训练状态
    train_state = {
        "__LAST_UPDATE_STEP__": total_steps,
        "__LAST_UPDATE_BATCH__": batch_idx,
    }
    with open(train_state_path, "w", encoding="utf-8") as f:
        json.dump(train_state, f, ensure_ascii=False, indent=2)
    
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