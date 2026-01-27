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
    num_workers=20, # 新增参数
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
    transition_dict_capacity=1000,
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
    save_interval = 5,
    
):

    
    # 设置随机数种子
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # # 尽量开启确定性模式（可能影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception:
    #     pass
    
    # ... setup code ...
    
    # ----------------------------------------------------
    # 并行环境初始化 (Worker Setup)
    # ----------------------------------------------------
    workers = []
    pipes = []
    
    print(f"Initializing {num_workers} parallel workers...")
    for i in range(num_workers):
        parent_conn, child_conn = mp.Pipe()
        # 显式指定 Worker 使用 CPU，避免 20 个进程挤爆显存
        w_device = torch.device('cpu') 
        
        p = mp.Process(target=worker_process, args=(
            i, child_conn, args, state_dim, hidden_dim, action_dims_dict, w_device, dt_maneuver, seed
        ))
        p.start()
        workers.append(p)
        pipes.append(parent_conn)
    
    # ... Logger init ...
    
    # ----------------------------------------------------
    # 主循环重构
    # ----------------------------------------------------
    total_steps = 0
    i_episode = 0 # 这里 i_episode 将代表 "batch index" 或者累加的总回合数
    
    while True: # 外层控制
        while total_steps < current_max_steps:
            
            # 1. 准备本轮所有 Worker 的指令
            #    Master 可以在这里统一决定每个 Worker 的对手，或者把 ELO 表发过去让 Worker 自己选
            #    为了数据准确，建议 Master 选好发过去
            
            # 获取当前网络权重 (CPU)
            current_actor_weights = {k: v.cpu() for k, v in student_agent.actor.state_dict().items()}
            
            worker_commands = []
            
            # 为每个 Worker 准备参数
            for rank in range(num_workers):
                # A. 采样对手 (复用原本的 get_opponent_probabilities)
                # ... 你的对手采样逻辑 ...
                probs, opponent_keys = get_opponent_probabilities(elite_elo_ratings, target_elo=main_agent_elo, SP_type=self_play_type, sigma=sigma_elo)
                selected_opponent_name = np.random.choice(opponent_keys, p=probs)
                
                # B. 加载对手数据
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
                        # 加载到 CPU 发送
                        opp_data = torch.load(adv_path, map_location='cpu')
                    else:
                        opp_type = 'rule' # Fallback

                opp_info = (selected_opponent_name, opp_type, opp_data)
                
                # C. 其他参数
                settings = {
                    'randomized_birth': randomized_birth,
                    'action_cycle_multiplier': action_cycle_multiplier,
                    'weight_reward': weight_reward_0
                }
                
                worker_commands.append(('RUN_EPISODE', (current_actor_weights, opp_info, settings)))

            # 2. 发送指令 (Fan-out)
            for pipe, cmd_data in zip(pipes, worker_commands):
                pipe.send(cmd_data)
                
            # 3. 阻塞等待结果 (Barrier & Fan-in)
            #    这里实现了 "统一等待所有回合结束"
            batch_results = []
            for pipe in pipes:
                res = pipe.recv() # 阻塞直至该 Worker 完成
                batch_results.append(res)
                
            # 4. 数据聚合与处理
            #    此时所有 Worker 都已跑完一轮
            
            # 初始化聚合容器
            agg_transition_dict = {'obs': [], 'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'active_masks': []}
            batch_steps = []
            batch_returns = []
            
            for res in batch_results:
                # res 结构: {'trans': ..., 'metrics': ..., 'opp_name': ...}
                trans = res['trans']
                metrics = res['metrics']
                opp_name = res['opp_name']
                
                # A. 拼接 Transition
                for k in agg_transition_dict:
                    agg_transition_dict[k].extend(trans[k])
                    
                # B. SIL 数据收集 (同理处理 ego_trans / enm_trans)
                if use_sil:
                    if not metrics['lose']: # 赢了或平了
                         # 添加到 SIL buffer (需把 ego_trans 转为 Monte Carlo returns)
                         # 注意：这里需要先把 res['ego_trans'] 转成 dict list 形式再计算 returns
                         pass # 逻辑同原代码，只是数据源变了
                    if not metrics['win']: # 输了或平了
                         pass 

                # C. ELO 更新 (在 Master 端统一算)
                actual_score = 0.5
                if metrics['win']: actual_score = 1.0
                elif metrics['lose']: actual_score = 0.0
                
                # 复用你的 update_elo 逻辑
                if opp_name in elite_elo_ratings:
                    prev_main_elo = main_agent_elo
                    adv_elo = elite_elo_ratings[opp_name]
                    main_agent_elo = update_elo(prev_main_elo, adv_elo, actual_score)
                    # 更新对手...
                    new_adv_elo = update_elo(adv_elo, prev_main_elo, 1.0 - actual_score)
                    elite_elo_ratings[opp_name] = new_adv_elo
                    elo_ratings[opp_name] = new_adv_elo
                    # ... logging ...

                batch_steps.append(metrics['steps'])
                batch_returns.append(metrics['return'])
                i_episode += 1

            # 5. 更新全局步数 (取平均值)
            avg_steps = int(np.mean(batch_steps))
            total_steps += avg_steps
            print(f"Batch Finished. Avg Steps: {avg_steps}, Total: {total_steps}, Avg Return: {np.mean(batch_returns):.2f}")

            # 6. 训练更新 (PPO Update)
            #    现在 agg_transition_dict 包含了 num_workers 个回合的数据，相当于一个巨大的 Batch
            if len(agg_transition_dict['dones']) >= transition_dict_capacity:
                # 你的 mixed_update 或 update 逻辑
                # 注意：agg_transition_dict 中的 actions 列表需要 restructure_actions
                # 但你的 PPOHybrid 内部可能已经处理了，或者你需要在这里调用一次
                agg_transition_dict['actions'] = restructure_actions(agg_transition_dict['actions'])
                
                student_agent.mixed_update(agg_transition_dict, ...) 
                
                # Log ...

            # 7. Checkpoint 保存 (同原代码)
            if i_episode % save_interval == 0:
                # save ...
                pass
                
        # --- End of while total_steps ---
        # 交互式扩充 max_steps 逻辑 (同原代码)
        # ...

    # 清理
    for pipe in pipes: pipe.send(('EXIT', None))
    for p in workers: p.join()
    
    