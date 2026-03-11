import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# from TrainAndTests.Controls.FlightControl_Train_dual_a_out import *
from TrainAndTests.Controls.parallel_FlightControl_Train_dual_a_out import *

action_eps = 0 # np.array([0.5, 0.8, 0]) # 0.7 # 动作平滑度

from Utilities.LocateDirAndAgents import *

# 测试训练效果
action_dims_dict = {'cont': action_dim, 'cat': [], 'bern': 0}
policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
print(action_bound)
actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
# action_bounds 检查
print(actor.action_bounds)

# from Algorithms.MLP_heads import ValueNet
# critic = ValueNet(state_dim, hidden_dim).to(device)

# agent = PPOHybrid(actor, critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)



# pre_log_dir = os.path.join("./logs")
pre_log_dir = os.path.join(project_root, "logs/control")
mission_name = 'FlightControl_parallel无蒸馏'
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
# log_dir = os.path.join(pre_log_dir, "FlightControl-run-20260308-211329")

# 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
actor_path = load_actor_from_log(log_dir, number=None)
if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    # agent.actor.load_state_dict(sd)
    actor.load_state_dict(sd)
    print(f"Loaded actor for test from: {actor_path}")

# 舞龙测试
dragon_dance = 0

dt_decide = 0.2 # 0.05

out_range_count = 0

# action_bounds 检查
# print(agent.actor.action_bounds)
print(actor.action_bounds)

env = track_env(dt_move=0.025, tacview_show=1, time_limit=8*60)

t_bias = 0
# 强化学习测试
rl_steps = 0
i_episode = 0
while i_episode<=6:
    i_episode += 1
    episode_return = 0
    
    init_height = 6000 # np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

    birth_state={'position': np.array([0.0, init_height, 0.0]),
                        'psi': np.random.uniform(-pi/6, pi/6)
                        }
    height_req = np.clip(init_height-3000, 3000, 12000) # np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
    psi_req = sub_of_radian(birth_state['psi'] + pi + np.random.choice([-0.1, 0.1]), 0) # + np.random.uniform(-pi/6, pi/6), 0) # np.random.uniform(-pi, pi)
    v_req = 340 # np.random.uniform(0.8, 2.5)*340

    env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide) # 0.04
    obs, obs_check = env.get_obs()
    done = False

    psi_req_dot = 0
    height_req_dot = 0

    while not done:  # 每个训练回合
        # 舞龙测试
        if dragon_dance:
            psi_req_dot += np.random.uniform(-1, 1) *0.2*pi/180
            # psi_req_dot = np.clip(psi_req_dot, -10*pi/180, 10*pi/180)
            if abs(psi_req_dot)>=10*pi/180:
                psi_req_dot = 0
            env.psi_req += psi_req_dot * dt_decide
            height_req_dot += np.random.uniform(-1, 1) * 20
            height_req_dot = np.clip(height_req_dot, -100, 100)
            env.height_req += height_req_dot * dt_decide
            env.height_req = np.clip(env.height_req, 4000, 12000)
            
        # else:
        #     if round(env.t, 3) % 20 == 0:
        #         env.psi_req += pi
        #         env.psi_req = sub_of_radian(env.psi_req, 0)
        #         env.height_req -= 3000
        #         env.height_req = np.clip(env.height_req, 4000, 12000)
                

        # 1.执行动作得到环境反馈
        obs, obs_check = env.get_obs()
        # action, u, _, _ = agent.take_action(obs, explore=0)
        action, u, _, _ = actor.get_action(obs, explore=0)
        rl_steps += 1

        # if abs(env.t % 0.5) <= env.dt_move:
        #     print("action", action["cont"])
        #     print("tanh(u)", np.tanh(u["cont"]))
        #     print("--")
            # time.sleep(0.5)
        
        # action[2] = 1  # 强制油门推满
        next_obs, reward, done = env.step(action)

        # debug 用
        height_req_show = env.height_req/1000
        height_show = env.RUAV.alt/1000
        psi_req_show = env.psi_req*180/pi
        psi_show = env.RUAV.psi*180/pi
        v_req_show = env.v_req
        v_show = env.RUAV.speed

        env.render(t_bias)
        

    env.clear_render(t_bias)
    t_bias += env.t

    if env.fail==1:
        out_range_count+=1