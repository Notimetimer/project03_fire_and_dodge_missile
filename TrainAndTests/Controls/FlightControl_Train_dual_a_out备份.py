'''
当前做法的一个问题：
所有奖励是相加的，而且目标方向并不会随着姿态的变化而改变，当前训练的是什么？
是沿着一条安排好的轨迹动态跟踪，还是说只是照着指令里的速度、角速度和俯仰角去飞？
如果是后者，应该要区分“指令要你俯冲你俯冲撞地和指令没有要你俯冲你俯冲撞地的情况”
'''

use_tacview = 0

import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
from gym import spaces
import copy
import matplotlib.pyplot as plt
import json
import glob
import argparse

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from _context import *
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOHybrid23_0_distil2_one_step_KL import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Math_calculates.Calc_dist2border import calc_intern_dist2cylinder
from TrainAndTests.Controls.UPolicyWrapper import *



# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
max_steps = 10 * 65e4
hidden_dim = [128, 128] # [64, 64]
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
dt_decide = 0.2 # 0.2
pre_train_rate = 0 # 0.25 # 0.25

state_dim = 7+7+4  # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
action_dim = 4 # test
# action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
action_bound = np.array([[-1,1],[-1,1],[-1,1],[0,1]])
mission_name = 'FlightControl备份'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
log_dir = os.path.join(project_root, "./logs/control", mission_name + "-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__=='__main__':
    env = track_env(tacview_show=use_tacview)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=3*60, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=np.inf, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = track_env(args)

    teacher_agent = UnifiedPolicyWrapper(dummy_env)

    action_dims_dict = {'cont': action_dim, 'cat': [], 'bern': 0}
    policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
    from Algorithms.MLP_heads import ValueNet
    critic = ValueNet(state_dim, hidden_dim).to(device)
    
    agent = PPOHybrid(actor, critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        
    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    def save_meta_once(path, state_dict):
        if os.path.exists(path):
            return
        meta = {k: list(v.shape) for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(meta, f)

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Visualize.tensorboard_visualize import TensorBoardLogger

    out_range_count = 0
    return_list = []
    steps_count = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    try:
        t_bias = 0
        # 强化学习训练
        rl_steps = 0
        i_episode = 0
        while rl_steps < max_steps:
            i_episode += 1
            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
            
            init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

            birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)
                                }
            
            height_req = np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
            psi_req = np.random.uniform(-pi, pi) * np.clip(i_episode/1000, 0, 1)
            v_req = np.random.uniform(0.8, 2.5)*340

            env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)

            obs, obs_check = env.get_obs()
            done = False

            while not done:  # 每个训练回合
                # 1.执行动作得到环境反馈
                obs, obs_check = env.get_obs()
                action, u, _, _ = agent.take_action(obs, explore=True)
                rl_steps += 1

                # if abs(env.t % 0.5) <= env.dt_move:
                    # print("----")
                    # print("delta_psi", np.arctan2(obs_check["flight_cmd"][1], obs_check["flight_cmd"][0]) * 180 / pi)
                    # temp_state = env.unscale_state(obs_check)
                    # print("delta_height", temp_state["flight_cmd"][2])
                    # print("delta_speed", temp_state["flight_cmd"][3])
                    # print("--")
                    # print("aileron", action['cont'][0])
                    # print("elevator", action['cont'][1])
                    # print("rudder", action['cont'][2])
                    # print("throttle", action['cont'][3])
                    # print('--')
                    # print("obs_check", obs_check)
                    # print("----")
                    # print(f"Episode {i_episode}, Step {rl_steps}, time: {env.t}")

                
                next_obs, reward, done = env.step(action)

                # debug 用
                height_req_show = env.height_req/1000
                height_show = env.RUAV.alt/1000
                psi_req_show = env.psi_req*180/pi
                psi_show = env.RUAV.psi*180/pi
                v_req_show = env.v_req
                v_show = env.RUAV.speed

                transition_dict['states'].append(obs)
                transition_dict['actions'].append(u)
                transition_dict['next_states'].append(next_obs)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                transition_dict['action_bounds'].append(action_bound)
                obs = next_obs
                episode_return += reward * env.dt_report # 奖励按秒分析
                env.render(t_bias)

            env.clear_render(t_bias)
            t_bias += env.t

            if env.fail==1:
                out_range_count+=1
            return_list.append(episode_return)
            agent.update(transition_dict)
            agent.distil(transition_dict, teacher_agent=teacher_agent, epochs=1, alpha=1.0)

            # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
            if i_episode % 10 == 1:
                # critic overwrite
                critic_path = os.path.join(log_dir, "critic.pt")
                th.save(agent.critic.state_dict(), critic_path)
                # actor RL snapshot
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

            
            # tqdm 训练进度显示
            if (i_episode + 1) >= 10:
                print(f"episode {i_episode+1}, 进度: {rl_steps / max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")

            # tensorboard 训练进度显示
            logger.add("train/0 episode_return", episode_return, rl_steps)
            logger.add("train/0 survive", 1-env.fail, rl_steps)

            actor_grad_norm = model_grad_norm(agent.actor)
            critic_grad_norm = model_grad_norm(agent.critic)
            # 梯度监控
            logger.add("train/1 actor_grad_norm", actor_grad_norm, rl_steps)
            logger.add("train/2 critic_grad_norm", critic_grad_norm, rl_steps)
            # 损失函数监控
            logger.add("train/3 actor_loss", agent.actor_loss, rl_steps)
            logger.add("train/4 critic_loss", agent.critic_loss, rl_steps)
            # 强化学习actor特殊项监控
            logger.add("train/5 entropy", agent.entropy_mean, rl_steps)
            logger.add("train/6 ratio", agent.ratio_mean, rl_steps)
            logger.add("train/7 steps", i_episode + 1, rl_steps)
            if hasattr(agent, 'dis_actor_loss') and agent.dis_actor_loss != 0:
                logger.add("train/8 distil_loss", agent.dis_actor_loss, rl_steps)


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()


        print(f"日志已保存到：{logger.run_dir}")

