'''
训练进攻策略:

红方闲庭信步
蓝方追赶红方
'''

'''
训练事项：
1、所有动作缩放统一改在action_bound里面进行
2、随机初始化我机和目标机位置
3、禁止发射导弹，测试进攻相关的观测、结束判读和奖励能否完整运行 √
4、高度指令改为相对高度 √
5、tacview 文件头命令改在env init时候加入, √
    加入clear_render方法清空所有在可视化的实体, √
    在render方法中使用t_bias处理多次仿真可视化的起点问题 √
'''


import argparse
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Envs.battle6dof1v1_missile0919 import *
#   battle3dof1v1_proportion battle3dof1v1_missile0812 battle3dof1v1_missile0901
from math import pi
import numpy as np
import matplotlib
import json
import glob
import copy
import socket
import threading
from send2tacview import *
from Algorithms.Rules import decision_rule
from Math_calculates.CartesianOnEarth import *
from Math_calculates.sub_of_angles import *
from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from Envs.UAVmodel6d import UAVModel
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Visualize.tacview_visualize import *
from Visualize.tensorboard_visualize import *
from Algorithms.SquashedPPOcontinues_dual_a_out import *
from tqdm import tqdm

use_tacview = 1  # 是否可视化

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


parser = argparse.ArgumentParser("UAV swarm confrontation")
# Environment
parser.add_argument("--max-episode-len", type=float, default=120,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")

# parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
# parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
num_episodes = 1000  # 2000 400
hidden_dim = [128, 128, 128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
dt_decide = 2 # 2
pre_train_rate = 0 # 0.25 # 0.25


env = Battle(args, tacview_show=use_tacview)
r_obs_spaces = env.get_obs_spaces('r')
b_obs_spaces = env.get_obs_spaces('b')
# r_obs_spaces, b_obs_spaces = env.get_obs_spaces()
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces
action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])

r_action = []
b_action = []


def launch_missile_if_possible(env, side='r'):
    """
    根据条件判断是否发射导弹
    """
    if side == 'r':
        uav = env.RUAV
        ally_missiles = env.Rmissiles
        target = env.BUAV
    else:  # side == 'b'
        uav = env.BUAV
        ally_missiles = env.Bmissiles
        target = env.RUAV

    waite = False
    for missile in ally_missiles:
        if not missile.dead:
            waite = True
            break
        
    if not waite:
        # 判断是否可以发射导弹
        if uav.can_launch_missile(target, env.t):
            # 发射导弹
            new_missile = uav.launch_missile(target, env.t, missile_class)
            uav.ammo -= 1
            new_missile.side = 'red' if side == 'r' else 'blue'
            if side == 'r':
                env.Rmissiles.append(new_missile)
            else:
                env.Bmissiles.append(new_missile)
            env.missiles = env.Rmissiles + env.Bmissiles
            print(f"{'红方' if side == 'r' else '蓝方'}发射导弹")

start_time = time.time()
launch_time_count = 0

t_list = []
R_controll_type = []
B_controll_type = []
R_controll_check_switch1 = []
B_controll_check_switch1 = []
R_controll_check_switch2 = []
B_controll_check_switch2 = []
r_ammo = []
b_ammo = []

t_bias = 0

for i in range(10):
    print("轮次", i+1)
    episode_return = 0
    # 飞机出生状态指定
    red_R_ = random.uniform(20e3, 60e3)
    red_beta = random.uniform(0, 2*pi)
    red_psi = random.uniform(0, 2*pi)
    red_height = random.uniform(3e3, 10e3)
    red_N = red_R_*cos(red_beta)
    red_E = red_R_*sin(red_beta)
    blue_height = random.uniform(3e3, 10e3)

    DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                               'psi': red_psi
                               }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                'psi': pi
                                }
    env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
              red_init_ammo=0, blue_init_ammo=0)

    a1 = env.BUAV.pos_  # 58000,7750,20000
    a2 = env.RUAV.pos_  # 2000,7750,20000
    b1 = env.UAVs[0].pos_
    b2 = env.UAVs[1].pos_
    done = False
    r_action_list = []
    b_action_list = []
    Rtrajectory = []
    Btrajectory = []

    # 环境运行一轮的情况
    for count in range(round(args.max_episode_len / dt_maneuver)):
        # print(f"time: {env.t}")  # 打印当前的 count 值
        # 回合结束判断
        # print(env.running)
        current_t = count * dt_maneuver
        if env.running == False or count == round(args.max_episode_len / dt_maneuver) - 1:
            # print('回合结束，时间为：', env.t, 's')
            break
        # 获取观测信息
        r_obs_n = env.get_obs('r')
        b_obs_n = env.get_obs('b')

        # 在这里将观测信息压入记忆


        state = np.squeeze(r_obs_n)

        # todo 可发射判据添加解算，且更换if_possible和距离判据的顺序（发射时间间隔过了之后再解算攻击区会更有效率）
        distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
        # 发射导弹判决
        if distance <= 40e3 and distance >= 5e3 and count % 1 == 0:  # 在合适的距离范围内每0.2s判决一次导弹发射
            launch_time_count = 0
            launch_missile_if_possible(env, side='r')
            launch_missile_if_possible(env, side='b')

        # 机动决策
        r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                   enm_pos_=env.BUAV.pos_, distance=distance,
                                   ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                   o00=o00, R_cage=env.R_cage, wander=1
                                   )

        # test
        # b_action_n = [[0.3, 0.0, 1.0]]
        # L_ = env.RUAV.pos_ - env.BUAV.pos_
        # beta = atan2(L_[2], L_[0])
        # delta_psi = sub_of_radian(beta, env.BUAV.psi)
        # b_action_n[0][1] = delta_psi

        b_action_n = decision_rule(ego_pos_=env.BUAV.pos_, ego_psi=env.BUAV.psi,
                                   enm_pos_=env.RUAV.pos_, distance=distance,
                                   ally_missiles=env.Bmissiles, enm_missiles=env.Rmissiles,
                                   o00=o00, R_cage=env.R_cage, wander=0
                                   )
        
        r_action_list.append(r_action_n[0])
        b_action_list.append(b_action_n[0])

        r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n)  # 2、环境更新并反馈

        '''显示运行轨迹'''
        # print("红方位置：", env.RUAV.pos_)
        # print("蓝方位置：", env.BUAV.pos_)
        # # 如果导弹已发射
        # print(f"当前发射的导弹数量：{len(env.Rmissiles) + len(env.Bmissiles)}")
        # 遍历导弹列表
        for missile in env.Rmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            # print("红方导弹位置：", missile_pos)
        for missile in env.Bmissiles:
            if hasattr(missile, 'dead') and missile.dead:
                continue
            # 记录导弹的位置
            missile_pos = missile.pos_  # 假设导弹对象有 pos_ 属性表示位置
            # print("蓝方导弹位置：", missile_pos)
        
        # 可视化
        env.render(t_bias=t_bias)
        t_list.append(current_t)
        # 红
        position = env.RUAV.pos_
        angle = np.array([env.RUAV.gamma, env.RUAV.theta, env.RUAV.psi]) * 180 / pi
        Rtrajectory.append(np.hstack((position, angle)))
        R_controll_type.append(env.RUAV.PIDController.type)
        R_controll_check_switch1.append(env.RUAV.PIDController.check_switch1)
        R_controll_check_switch2.append(env.RUAV.PIDController.check_switch2)
        r_ammo.append(env.RUAV.ammo)

        # 蓝
        position = env.BUAV.pos_
        angle = np.array([env.BUAV.gamma, env.BUAV.theta, env.BUAV.psi]) * 180 / pi
        Btrajectory.append(np.hstack((position, angle)))
        B_controll_type.append(env.BUAV.PIDController.type)
        B_controll_check_switch1.append(env.BUAV.PIDController.check_switch1)
        B_controll_check_switch2.append(env.BUAV.PIDController.check_switch2)
        b_ammo.append(env.BUAV.ammo)

        if terminate == True:
            break
    
    print(t_bias)
    env.clear_render(t_bias=t_bias)
    t_bias += env.t

end_time = time.time()  # 记录结束时间
print(f"程序运行时间: {end_time - start_time} 秒")

r_action_list = np.array(r_action_list)
b_action_list = np.array(b_action_list)

R_controll_check_switch1 = np.array(R_controll_check_switch1)
R_controll_check_switch2 = np.array(R_controll_check_switch2)
B_controll_check_switch1 = np.array(B_controll_check_switch1)
B_controll_check_switch2 = np.array(B_controll_check_switch2)

# import matplotlib.pyplot as plt

# # Create a figure and 10 subplots in 2 columns
# fig, axs = plt.subplots(5, 2, figsize=(14, 24), sharex=True)

# axs[0, 0].plot(t_list, R_controll_type, label='R_controll_type', color='red')
# axs[0, 0].legend()
# axs[0, 0].grid()

# axs[0, 1].plot(t_list, B_controll_type, label='B_controll_type', color='blue')
# axs[0, 1].legend()
# axs[0, 1].grid()

# axs[1, 0].plot(t_list, r_action_list[:, 1] * 180 / pi, label='r_action_list[1]', color='green')
# axs[1, 0].legend()
# axs[1, 0].grid()

# axs[1, 1].plot(t_list, b_action_list[:, 1] * 180 / pi, label='b_action_list[1]', color='orange')
# axs[1, 1].set_xlabel('Time (s)')
# axs[1, 1].legend()
# axs[1, 1].grid()

# axs[2, 0].plot(t_list, R_controll_check_switch1, label='R_controll_check_switch1', color='purple')
# axs[2, 0].legend()
# axs[2, 0].grid()

# axs[2, 1].plot(t_list, R_controll_check_switch2, label='R_controll_check_switch2', color='brown')
# axs[2, 1].legend()
# axs[2, 1].grid()

# axs[3, 0].plot(t_list, B_controll_check_switch1, label='B_controll_check_switch1', color='pink')
# axs[3, 0].legend()
# axs[3, 0].grid()

# axs[3, 1].plot(t_list, B_controll_check_switch2, label='B_controll_check_switch2', color='cyan')
# axs[3, 1].set_xlabel('Time (s)')
# axs[3, 1].legend()
# axs[3, 1].grid()

# # 新增弹药数曲线
# axs[4, 0].plot(t_list, r_ammo, label='r_ammo', color='darkred')
# axs[4, 0].legend()
# axs[4, 0].grid()
# axs[4, 0].set_ylabel('Red Ammo')

# axs[4, 1].plot(t_list, b_ammo, label='b_ammo', color='navy')
# axs[4, 1].legend()
# axs[4, 1].grid()
# axs[4, 1].set_ylabel('Blue Ammo')
# axs[4, 1].set_xlabel('Time (s)')

# plt.tight_layout()
# plt.show()
