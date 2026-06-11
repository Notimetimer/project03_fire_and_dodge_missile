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
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Algorithms.PPOHybrid23_0B import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from VsBaseline_while_training_hierarch_plus import test_worker
from RewardWeightController import FireRewardWeightController

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
        
        "反事实数据增强"
        # === 针对开火（'fire' / 'bern'）瞬间的数据集扩增 ===
        if 'states' in original_il_transition_dict and 'actions' in original_il_transition_dict:
            states_arr = original_il_transition_dict['states']
            returns_arr = original_il_transition_dict['returns'] if 'returns' in original_il_transition_dict else None
            actions_dict = original_il_transition_dict['actions']
            
            if 'bern' in actions_dict and 'cat' in actions_dict:
                bern_arr = actions_dict['bern']
                cat_arr = actions_dict['cat']
                
                # 找出 'fire'（即 'bern'）为 1 的索引
                fire_indices = np.where(bern_arr[:, 0] > 0.5)[0]
                print(f"[Data Augment] Found {len(fire_indices)} fire instances to augment.")
                
                expanded_states_list = []
                expanded_returns_list = []
                expanded_cats_list = []
                expanded_berns_list = []
                
                t_intervals = np.arange(0, 80/120, 0.3)
                # 索引位[6] cos_delta_psi：delta_psi 从 0 到 60度，步长5度
                delta_psi_vals = np.arange(0, np.pi/3, np.radians(20))
                cos_delta_psi_vals = np.cos(delta_psi_vals)
                sin_delta_psi_vals = np.sin(delta_psi_vals)
                # 索引位[8] delta_theta：从 -60度 到 60度，步长10度
                delta_theta_vals = np.arange(-np.pi/3, np.pi/3, np.radians(20))
                
                for idx in fire_indices:
                    orig_state = states_arr[idx]
                    orig_return = returns_arr[idx] if returns_arr is not None else 0.0
                    orig_cat = cat_arr[idx]
                    
                    for val_t in t_intervals:
                        t_since_launch_sec = val_t * 120.0
                        
                        for cos_dp, sin_dp in zip(cos_delta_psi_vals, sin_delta_psi_vals):
                            for delta_theta in delta_theta_vals:
                                # 复制并修改观测量
                                new_state = orig_state.copy()
                                new_state[21] = val_t # 索引位21（t_since_launch / 120）
                                new_state[6] = cos_dp  # 索引位6 cos_delta_psi
                                new_state[7] = sin_dp  # 索引位7 sin_delta_psi
                                new_state[8] = delta_theta      # 索引位8 delta_theta
                                
                                if t_since_launch_sec < 70.0:
                                    new_state[3] = 1.0  # 索引位3导弹中制导置为1
                                    new_bern = 0.0      # 动作开火置为0
                                else:
                                    new_state[3] = 0.0  # 索引位3导弹中制导置为0
                                    # 同时满足 delta_theta<0 且 cos_delta_psi > cos(20°) 才允许开火
                                    if delta_theta < 0 and cos_dp > np.cos(np.radians(20)):
                                        new_bern = 1.0
                                    else:
                                        new_bern = 0.0
                                    
                                expanded_states_list.append(new_state)
                                if returns_arr is not None:
                                    expanded_returns_list.append(orig_return)
                                expanded_cats_list.append(orig_cat)
                                expanded_berns_list.append([new_bern])
                
                if len(expanded_states_list) > 0:
                    expanded_states = np.array(expanded_states_list, dtype=np.float32)
                    expanded_cats = np.array(expanded_cats_list, dtype=np.int64)
                    expanded_berns = np.array(expanded_berns_list, dtype=np.float32)
                    
                    original_il_transition_dict['states'] = np.concatenate([states_arr, expanded_states], axis=0)
                    original_il_transition_dict['actions']['cat'] = np.concatenate([cat_arr, expanded_cats], axis=0)
                    original_il_transition_dict['actions']['bern'] = np.concatenate([bern_arr, expanded_berns], axis=0)
                    
                    if returns_arr is not None:
                        expanded_returns = np.array(expanded_returns_list, dtype=np.float32)
                        original_il_transition_dict['returns'] = np.concatenate([returns_arr, expanded_returns], axis=0)
                        
                    if 'obs' in original_il_transition_dict and original_il_transition_dict['obs'] is not None:
                        original_il_transition_dict['obs'] = np.concatenate([np.array(original_il_transition_dict['obs'], dtype=np.float32), expanded_states], axis=0)
                        
                    print(f"[Data Augment] Augmented dataset. Added {len(expanded_states_list)} samples. New total samples: {len(original_il_transition_dict['states'])}")
    
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

    # 中断续训
    if resume_dir is not None and os.path.exists(resume_dir):
        if collape_recover["collapsed"]:
            best_actor_name = collape_recover["best_actor_name"]
            current_actor_path = os.path.join(log_dir, f"{best_actor_name}.pt")
        else:
            # 优先加载current_actor（确保不加载搅拌后的参数）
            current_actor_path = os.path.join(log_dir, "current_actor.pt")
            
        if os.path.exists(current_actor_path):
            student_agent.actor.load_state_dict(torch.load(current_actor_path, map_location=device))
            print(f"Loaded current actor from: {current_actor_path}")
        else:
            # 如果current_actor不存在，回退到原来的逻辑
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
            no_bern = 0, # 0
        )
        
        # 为了临时中断marwil
        int_agent_name = "actor_rein0"
        torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{int_agent_name}.pt"))

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
                ("il_train/accuracy_cont", getattr(student_agent, "marwil_accuracy_cont", None)),
                ("il_train/accuracy_cat", getattr(student_agent, "marwil_accuracy_cat", None)),
                ("il_train/accuracy_bern", getattr(student_agent, "marwil_accuracy_bern", None)),
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
    
    # MARWIL 结束后恢复 bern bias，防止稀疏开火动作被拉向高熵中间态（无济于事）
    if hasattr(student_agent.actor.net, 'fc_bern'):
        with torch.no_grad():
            student_agent.actor.net.fc_bern[-1].bias.clamp_(max=-2.5)
    
    if IL_epoches > 0:
        print("IL Training Finished.")
    else:
        print("No IL")
    
    # 存储在线训练前的网络参数
    int_agent_name = "actor_rein0"
    torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{int_agent_name}.pt"))

