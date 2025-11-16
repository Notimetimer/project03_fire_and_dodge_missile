"""
对手也改成使用比例控制的模型，且回放池加入对手的经验进入
"""

import keyboard
import argparse
import time
from battle3dof1v1_proportion import *  # Battle
from math import pi
import numpy as np
import matplotlib
import copy
import os
import pickle

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import torch
from torch import nn
import torch.nn.functional as F
import collections
import rl_utils
from tqdm import tqdm
import pandas as pd
from torch.distributions import Normal
import random
import gym
from guid2contr_abs import guided_control

stop_flag = False  # 控制退出的全局变量


def stop_loop():
    global stop_flag
    stop_flag = True
    print("检测到 'esc' 键，准备退出所有循环！")


keyboard.add_hotkey('esc', stop_loop)  # 监听 'esc' 键，按下时执行 `stop_loop()`

# matplotlib.use('TkAgg')  # 'TkAgg' 或 'Qt5Agg'
if matplotlib.get_backend() != 'TkAgg':
    matplotlib.use('TkAgg')
    plt.switch_backend('TkAgg')


def get_args():
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=8 * 60,  # * 60.0,
                        help="maximum episode time length")  # 对局时长：真的中远距空战可能会持续20分钟那么长
    # parser.add_argument("--num-RUAVs", type=int, default=1, help="number of red UAVs")
    # parser.add_argument("--num-BUAVs", type=int, default=1, help="number of blue UAVs")
    args = parser.parse_args()
    return args


def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


# 将红蓝双方运行的速度和位置保存
red_pos_list = np.empty((0, 3))
blue_pos_list = np.empty((0, 3))
red_vel_list = np.empty((0, 3))
blue_vel_list = np.empty((0, 3))

args = get_args()
env = Battle(args)
env.reset()
r_obs_spaces, b_obs_spaces = env.r_obs_spaces, env.b_obs_spaces
r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

'超参数设置'
actor_lr0 = 3e-4  # 3e-4 # 归一化后学习率
critic_lr0 = actor_lr0 * 8  # 8,10 3e-3 critic 学习率
num_episodes = 1  # 1000  # 500
hidden_dim = [128]  # [128, 128, 128]  # 32  # 隐藏层维度
gamma = 0.95  # 0.98
alpha_lr = actor_lr0  # 3e-4  # * actor_lr0  # 3e-4 归一化后学习率
tau = 0.005  # 0.005  # 软更新参数
buffer_size = 4000  # 8000 对于0.2s的dt这个大小足够，但是对于0.02s的dt就不太够了
minimal_size = 240  # 480
batch_size = minimal_size  # 64
epsilon0 = 0.9  # 0.1  # 随机策略概率

best_actor_state_dict = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# todo 随机种子
replay_buffer = rl_utils.ReplayBuffer(buffer_size)  # 经验回放池
# actor网络输入层维度
r_obs_dim = np.shape(env.r_obs_spaces)[0]
# actor网络输出层维度
r_action_dim = 3  # np.shape(env.r_action_spaces[0])[0]
action_bound = 1  # env.r_action_spaces[0].high  # array([1., 1.], dtype=float32) 动作输出最大值

'actor-critic网络'


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


'''
chap4 训练红方智能体
'''
if __name__ == '__main__':
    replay_buffer = ReplayBuffer(buffer_size)  # 经验回放
    # action_bound = env.action_space.high[0]  # 动作最大值
    target_entropy = -env.r_action_spaces[0].shape[0]

    return_list = []
    start_time = time.time()

    # 在循环外部初始化PID控制所需的变量（重要！）
    prev_error_x = 0  # 切向控制上一次误差
    prev_error_y = 0  # 俯仰控制上一次误差
    prev_error_z = 0  # 滚转控制上一次误差
    integral_error_z = 0  # 滚转控制积分误差

    steps = 0
    last_action = np.zeros(3)

    with tqdm(total=int(num_episodes), desc='Iteration') as pbar:
        'num/10 start'
        for i_episode in range(num_episodes):  # range(2):

            steps_of_this_episode = 0
            epsilon = epsilon0 * 0.43 ** (-1 + (1 + i_episode) / num_episodes * 10)

            episode_return = 0
            env.reset()
            done = False
            # 环境运行一轮的情况
            'episode start'
            for count in range(round(args.max_episode_len / dt)):
                # 01
                steps += 1
                steps_of_this_episode += 1
                # 回合结束判断
                done = env.running == False or count == round(args.max_episode_len / dt) - 1
                if done:
                    # print('回合结束，时间为：', env.t, 's')
                    break

                # 03
                # 获取观测信息
                r_obs_n, b_obs_n = env.get_obs()
                mask = np.zeros(13)  # test 屏蔽暂时顾不了的观测信息
                mask[[7, 8]] = 1
                # mask[3] = 1 / 100  # 速率特征缩放
                b_obs = np.squeeze(b_obs_n * mask)
                r_obs = np.squeeze(r_obs_n * mask)

                '叠加当前角度和速度'
                # 红方引导式控制
                current_R_velocity = env.RUAV.vel_
                current_R_speed = np.linalg.norm(env.RUAV.vel_)
                current_R_psi = np.arctan2(env.RUAV.vel_[2], env.RUAV.vel_[0])
                current_R_theta = np.arctan2(env.RUAV.vel_[1], np.sqrt(env.RUAV.vel_[0] ** 2 + env.RUAV.vel_[2] ** 2))

                L_RB = env.BUAV.pos_ - env.RUAV.pos_
                L_RBh = np.array([L_RB[0], 0, L_RB[2]])
                q_betaR = np.arctan2(L_RBh[2], L_RBh[0])
                q_epsilonR = np.arctan2(L_RBh[1], np.sqrt(L_RBh[0] ** 2 + L_RBh[2] ** 2))

                thetaR_req = q_epsilonR
                psiR_req = q_betaR
                red_required_plus = [thetaR_req, psiR_req, 0]
                red_required_plus[0] = thetaR_req / (np.pi / 2)
                red_required_plus[1] = sub_of_radian(psiR_req, current_R_psi) / np.pi
                red_required_plus[2] = 0

                # print('red_required_plus', np.array(red_required_plus) * 180 / pi)

                vR_req = (0.5 + 0.5 * red_required_plus[2]) * (env.RUAV.speed_max - env.RUAV.speed_min) + \
                         env.RUAV.speed_min
                vR_req = np.clip(vR_req, env.RUAV.speed_min, env.RUAV.speed_max)

                # 蓝方引导式控制
                current_B_velocity = env.BUAV.vel_
                current_B_speed = np.linalg.norm(env.BUAV.vel_)
                current_B_psi = np.arctan2(env.BUAV.vel_[2], env.BUAV.vel_[0])
                current_B_theta = np.arctan2(env.BUAV.vel_[1], np.sqrt(env.BUAV.vel_[0] ** 2 + env.BUAV.vel_[2] ** 2))

                L_BR = env.RUAV.pos_ - env.BUAV.pos_
                L_BRh = np.array([L_BR[0], 0, L_BR[2]])
                q_betaB = np.arctan2(L_BRh[2], L_BRh[0])
                q_epsilonB = np.arctan2(L_BRh[1], np.sqrt(L_BRh[0] ** 2 + L_BRh[2] ** 2))

                thetaB_req = q_epsilonB
                psiB_req = q_betaB
                blue_required_plus = [thetaB_req, psiB_req, 0]

                # 示范动作计算
                blue_required_plus[0] = thetaB_req / (np.pi / 2)
                blue_required_plus[1] = sub_of_radian(psiB_req, current_B_psi) / np.pi
                blue_required_plus[2] = 0
                # print('blue_required_plus', np.array(blue_required_plus) * 180 / pi)
                vB_req = (0.5 + 0.5 * blue_required_plus[2]) * (env.BUAV.speed_max - env.BUAV.speed_min) + \
                         env.BUAV.speed_min
                vB_req = np.clip(vB_req, env.BUAV.speed_min, env.BUAV.speed_max)

                # 示范正确性检验
                thetaB_req = blue_required_plus[0] * np.pi / 2
                psiB_req = blue_required_plus[1] * np.pi + current_B_psi  # red_required_plus[1]*np.pi # + current_psi

                # 04 引导式控制

                r_action_n = guided_control(env.RUAV, thetaR_req, psiR_req, vR_req,prev_error_x ,prev_error_y,
                                            prev_error_z,integral_error_z,dt)
                # r_action_n = guided_control(env.RUAV, thetaB_req, psiB_req, vB_req, prev_error_x, prev_error_y,
                #                             prev_error_z, integral_error_z, dt)
                print(r_action_n)
                # 执行动作并获取环境反馈
                if i_episode + 1 < num_episodes:
                    b_action_n = guided_control(env.BUAV, thetaB_req, psiB_req, vB_req)
                    r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
                else:
                    b_action_n = guided_control(env.BUAV, thetaB_req, psiB_req, vB_req, prev_error_x, prev_error_y, prev_error_z, integral_error_z, dt)
                    r_reward_n, b_reward_n, r_dones, b_dones, terminate = env.step(r_action_n, b_action_n, record=True)
                    # print(theta_req * 180 / pi)
                    # 记录当前环境中飞机的运动状态
                    red_pos_list = np.vstack((red_pos_list, env.RUAV.pos_))
                    blue_pos_list = np.vstack((blue_pos_list, env.BUAV.pos_))
                done = r_dones or b_dones

                # 05
                r_obs_n, b_obs_n = env.get_obs()  # 后state
                # r_next_obs = np.squeeze(r_obs_n)
                # r_next_obs = r_next_obs * mask
                r_next_obs = np.squeeze(r_obs_n * mask)
                b_next_obs = np.squeeze(b_obs_n * mask)

                # 06
                # 运行记录添加回放池
                replay_buffer.add(r_obs, red_required_plus, r_reward_n, r_next_obs, done)
                replay_buffer.add(b_obs, blue_required_plus, b_reward_n, b_next_obs, done)

                episode_return += r_reward_n
                if stop_flag:
                    break
            'episode end'

            episode_return_per_step = episode_return / (count + 1)  # test
            return_list.append(episode_return_per_step)  # 平均回合奖励除以步数，以免存活越久看上去奖励越低

            # 显示训练进度
            # if (i_episode + 1) % 10 == 0:
            if (i_episode + 1) >= 10:
                pbar.set_postfix({'episode': '%d' % (i_episode + 1),
                                  'episode_return': '%.3f' % np.mean(return_list[-10:])})
            pbar.update(1)

            # print('steps_of_this_episode:', steps_of_this_episode)
            print('period:', env.t)
            # print('end_speed', env.RUAV.speed)
            if stop_flag:
                break

    episodes_list = list(range(len(return_list)))

    end_time = time.time()  # 记录结束时间
    print(f"程序运行时间: {end_time - start_time} 秒")

    if not stop_flag:
        # 最后一幕轨迹显示
        red_pos_list = env.RUAV.trajectory
        blue_pos_list = env.BUAV.trajectory
        # 可视化
        red_p_show_show = np.array(red_pos_list).T
        blue_p_show_show = np.array(blue_pos_list).T

        plt.figure(2)
        from show_trajectory import show_trajectory

        show_trajectory(red_pos_list, blue_pos_list, min_east, min_north, min_height, max_east, max_north, max_height,
                        r_show=1, b_show=1)

        # dataframe = pd.DataFrame({'N': red_pos_list[:, 0], 'U': red_pos_list[:, 1], 'E': red_pos_list[:, 2]})
        # dataframe.to_csv("test.csv", index=False, sep=',')
        print(steps)
