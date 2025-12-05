import os
import sys
import numpy as np
import torch
import argparse
import glob
import re
from math import pi

# --- 项目路径设置 ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# --- 模块导入 ---
from Envs.Tasks.ChooseStrategyEnv2 import ChooseStrategyEnv
from Algorithms.PPOHybrid2 import PolicyNetHybrid, HybridActorWrapper
from BasicRules import basic_rules
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
from ILRL_Combat_PFSP import *

use_tacview = 1  # 是否可视化


# 找出日期最大的目录
def get_latest_log_dir(pre_log_dir, mission_name=None):
    # 匹配 run-YYYYMMDD-HHMMSS 目录
    # pattern = re.compile(r"run-(\d{8})-(\d{6})")
    if mission_name:
        pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    else:
        pattern = re.compile(r"run-(\d{8})-(\d{6})")
    max_dt = None
    latest_dir = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if m:
            dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
            if max_dt is None or dt_str > max_dt:
                max_dt = dt_str
                latest_dir = d
    if latest_dir:
        return os.path.join(pre_log_dir, latest_dir)
    else:
        return None

logs_dir = os.path.join(project_root, "logs/combat")
mission_name = 'MARWIL_combat'
log_dir = get_latest_log_dir(logs_dir, mission_name=mission_name)

print("\nlog目录", log_dir,"\n")

if log_dir is None:
        raise ValueError("No valid log directory found. Please check the `pre_log_dir` or `mission_name`.")

def latest_actor_by_index(paths):
    best = None
    best_idx = -1
    for p in paths:
        m = re.search(r'actor_rein.*?(\d+)\.pt$', os.path.basename(p))
        if m:
            idx = int(m.group(1))
            if idx > best_idx:
                best_idx = idx
                best = p
    # fallback to most-recent-modified if no numeric match
    if best is None and paths:
        best = max(paths, key=os.path.getmtime)
    return best
rein_list = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
latest_actor_path = latest_actor_by_index(rein_list)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化b_agent
parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=10*60, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=55e3, help="")
args = parser.parse_args()
env = ChooseStrategyEnv(args, tacview_show=0)
state_dim = env.obs_dim
# 1. 创建神经网络
actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
critic_net = ValueNet(state_dim, hidden_dim).to(device)

# 2. Wrapper
actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

# 3. Agent
b_agent = PPOHybrid(
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
    max_std=0.3
)

if latest_actor_path:
    # 直接加载权重到现有的 agent
    sd = torch.load(latest_actor_path, map_location=device)
    b_agent.actor.load_state_dict(sd)  # , strict=False)  # 忽略缺失的键
    print(f"Loaded actor for test from: {latest_actor_path}")

env = ChooseStrategyEnv(args, tacview_show=1)
from BasicRules import *  # 可以直接读同一级目录
t_bias = 0
env.shielded = 1 # 不得不全程带上

def creat_initial_state():
    # 飞机出生状态指定
    # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
    blue_height = 9000
    red_height = 9000
    red_psi = -pi/2
    blue_psi = pi/2
    red_N = 0
    red_E = 45e3
    blue_N = red_N
    blue_E = -red_E
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                        'psi': red_psi
                        }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                'psi': blue_psi
                                }
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

transition_dict_capacity = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

steps_count = 0

total_steps = 0

training_start_time = time.time()
launch_time_count = 0

t_bias = 0

decide_steps_after_update = 0
try:
    r_action_list = []
    b_action_list = []
    
    
    # 示范数据采集
    for i_episode in range(1):

        last_r_action_label = 0
        last_b_action_label = 0

        episode_return = 0
        
        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        r_action_label=0
        b_action_label=0
        last_decision_state = None
        current_action = None
        b_reward = None

        done = False

        env.dt_maneuver = dt_maneuver
        
        episode_start_time = time.time()

        # 环境运行一轮的情况
        steps_of_this_eps = -1 # 没办法了
        for count in range(round(args.max_episode_len / dt_maneuver)):
            # print(f"time: {env.t}")  # 打印当前的 count 值
            # 回合结束判断
            # print(env.running)
            current_t = count * dt_maneuver
            steps_of_this_eps += 1
            if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                # print('回合结束，时间为：', env.t, 's')
                break
            # 获取观测信息
            r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
            b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_check_obs.copy()
            env.BUAV.obs_memory = b_check_obs.copy()

            # --- 智能体决策 ---
            # 判断是否到达了决策点（每 10 步）
            if steps_of_this_eps % action_cycle_multiplier == 0:
                # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了

                # **关键点 2: 开始【新的】一个动作周期**
                # 1. 记录新周期的起始状态
                last_decision_state = b_obs
                # 2. Agent 产生一个动作
                
                # 红方改变规则
                r_state_check = env.unscale_state(r_check_obs)
                r_action_label, r_fire = basic_rules(r_state_check, i_episode, last_action=last_r_action_label)
                last_r_action_label = r_action_label
                if r_fire:
                    launch_missile_immediately(env, 'r')

                # 蓝方使用智能体
                b_state_check = env.unscale_state(b_check_obs)
                
                b_action_exec, b_action_raw, _, b_action_check = student_agent.take_action(b_obs, explore=0)
                b_action_label = b_action_exec['cat'][0] # 返回可能是一个数组
                
                # _, b_fire = basic_rules(b_state_check, 1, last_action=last_b_action_label)
                b_fire = b_action_exec['bern'][0]
                if b_fire:
                    launch_missile_immediately(env, 'b')
                
                print("机动概率分布", b_action_check['cat'])
                print("开火概率", b_action_check['bern'][0])
                
                # print(type(b_action_exec['cat']))
                # print(type(b_action_exec['bern']))
                
                last_b_action_label = b_action_label

                decide_steps_after_update += 1
                
                b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                current_action = {'fly': b_action_label, 'fire': b_fire}
                # current_action = np.array([b_action_label, b_fire])

            r_action = env.maneuver14(env.RUAV, r_action_label)
            b_action = env.maneuver14(env.BUAV, b_action_label)

            _, _, _, _, fake_terminate = env.step(r_action, b_action) # Environment updates every dt_maneuver
            done, b_reward, b_reward_assisted = env.combat_terminate_and_reward('b', b_action_label, b_fire)
            done = done or fake_terminate

            # Accumulate rewards between agent decisions
            episode_return += b_reward * env.dt_maneuver

            next_b_check_obs = env.base_obs('b')
            next_b_obs = flatten_obs(next_b_check_obs, env.key_order)


            '''显示运行轨迹'''
            # 可视化
            env.render(t_bias=t_bias)
        
        # # --- 回合结束处理 ---
        # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
        # 循环结束后，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
        
        
        episode_end_time = time.time()  # 记录结束时间
        # print(f"回合时长: {episode_end_time - episode_start_time} 秒")
        
        # print(t_bias)
        env.clear_render(t_bias=t_bias)
        t_bias += env.t
        r_action_list = np.array(r_action_list)

    training_end_time = time.time()  # 记录结束时间
    
except KeyboardInterrupt:
    print("\n检测到 KeyboardInterrupt")
finally:
    env.end_render() # 停止发送

