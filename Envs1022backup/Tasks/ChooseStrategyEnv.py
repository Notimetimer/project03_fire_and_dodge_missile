'''
出生点改在外面指定

子策略暂时使用规则智能体，留下使用神经网络的接口

'''

import numpy as np
from random import random
import random
from gym import spaces
import copy
import jsbsim
import sys
import os
import importlib
import copy
from math import *


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取project目录
def get_current_file_dir():
    # 判断是否在 Jupyter Notebook 环境
    try:
        shell = get_ipython().__class__.__name__  # ← 误报，不用管
        if shell == 'ZMQInteractiveShell':  # Jupyter Notebook 或 JupyterLab
            # 推荐用 os.getcwd()，指向启动 Jupyter 的目录
            return os.getcwd()
        else:  # 其他 shell
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # 普通 Python 脚本
        return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from Envs.battle6dof1v1_missile0919 import *
from Envs.Tasks.AttackManeuverEnv import *
from Envs.Tasks.CrankManeuverEnv import *
from Envs.Tasks.EscapeManeuverEnv import *
from Algorithms.Rules import *


# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

class ChooseStrategyEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None

    def attack_obs(self, side):
        full_obs = self.base_obs(side, pomdp=0)
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])
        full_obs["threat"] = copy.deepcopy(self.obs_init["threat"])
        full_obs["border"] = copy.deepcopy(self.obs_init["border"])

        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs

    def escape_obs(self, side):
        full_obs = self.base_obs(side, pomdp=0)  ###
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["target_alive"] = copy.deepcopy(self.obs_init["target_alive"])
        full_obs["missile_in_mid_term"] = copy.deepcopy(self.obs_init["missile_in_mid_term"])
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["weapon"] = copy.deepcopy(self.obs_init["weapon"])

        # 逃逸过程中会出现部分观测的情况，已在base_obs中写下规则
        # 只有在warning为TRUE的时候才能够获取威胁信息，已在get_state中写下规则
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs

    def crank_obs(self, side, pomdp=0):
        full_obs = self.base_obs(side)
        # 先对dict的元素mask
        # 只需要 target_information 和 ego_main
        full_obs["target_locked"] = copy.deepcopy(self.obs_init["target_locked"])
        full_obs["missile_in_mid_term"] = copy.deepcopy(self.obs_init["missile_in_mid_term"])
        full_obs["ego_control"] = copy.deepcopy(self.obs_init["ego_control"])
        full_obs["threat"] = copy.deepcopy(self.obs_init["threat"])

        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order)
        return flat_obs, full_obs

    def step(self, r_actions, b_actions):
        # 这一层的action是离散的动作类型

        # 对每个动作类型，调用子策略产生连续动作指令值
        report_move_time_rate = int(round(self.dt_maneuver / dt_move))
        # 输入动作（范围为[-1,1]
        self.t += self.dt_maneuver
        self.t = round(self.t, 2)  # 保留两位小数

        actions = [r_actions] + [b_actions]
        self.r_actions = r_actions  # .copy()
        self.b_actions = b_actions  # .copy()

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneuver中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                uav_obs = self.base_obs(UAV.side, pomdp=0)  ### test 部分观测的话用1
                delta_height_scaled = uav_obs["target_information"][0]
                delta_psi = uav_obs["target_information"][1]
                delta_theta = uav_obs["target_information"][2]
                distance = uav_obs['target_information'][3] * 10e3
                d_hor, leftright = uav_obs["border"]

                move_action = np.zeros(3)

                # 进攻机动
                if action == 0:
                    pass  # 调用进攻策略
                    # 规则智能体
                    move_action = self.track_behavior(UAV.alt, delta_psi)

                    # 训练出的智能体
                    pass

                    # 武器发射决策
                    pass

                if action == 1:
                    pass  # 调用escape策略
                    # 规则智能体
                    move_action = self.escape_behavior(UAV.alt, delta_psi, uav_obs["warning"], uav_obs["threat"][0])

                    # 训练出的智能体
                    pass

                if action == 2:
                    pass  # 调用Lcrank策略
                    # 规则智能体
                    move_action = self.left_crank_behavior(UAV.alt, delta_psi)

                    # 训练出的智能体
                    pass

                if action == 3:
                    pass  # 调用Rcrank策略
                    # 规则智能体
                    move_action = self.right_crank_behavior(UAV.alt, delta_psi)

                    # 训练出的智能体
                    pass

                if action == 4:
                    # 转圈搜索目标
                    # 该部分不训练智能体，改为使用规则智能体
                    move_action[0] = np.clip(delta_height_scaled, self.min_alt_save - UAV.alt,
                                             self.max_alt_save - UAV.alt) * 5000

                    if leftright >= 0:
                        move_action[1] = pi
                    else:
                        move_action[1] = -pi
                    move_action[2] = 1.0 * 340

                # 避让水平边界动作修正：
                # print(UAV.side)
                state = self.get_state(UAV.side)
                dist_2_hor = state["border"][0]
                move_action = self.back_in_cage(move_action, UAV.pos_, UAV.psi)

                # if UAV.side == 'r':
                #     print(move_action)
                #     print()

                pass

                UAV.move(move_action[0], move_action[1], move_action[2], relevant_height=True)

            # 导弹移动
            self.missiles = self.Rmissiles + self.Bmissiles
            for missile in self.missiles[:]:  # 使用切片创建副本以允许删除
                target = self.get_target_by_id(missile.target_id)
                if target is None:  # 目标不存在, 不更换目标而是击毁导弹
                    missile.dead = True
                    continue
                elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹, 在飞机1V1的时候可以节省一点计算量，不用费事处理多目标的问题
                    missile.dead = True
                    continue
                else:
                    missile.target = target
                # if not missile.dead:
                # print('目标位置', target.pos_)
                # 计算前导弹和目标位速
                last_pmt_ = missile.pos_
                last_vmt_ = missile.vel_
                last_ptt_ = target.pos_
                last_vtt_ = target.vel_
                if not missile.dead:
                    # 获取目标信息
                    target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, last_ptt_)
                    # 更新导弹制导阶段
                    has_datalink = False
                    for uav in self.UAVs:
                        # 找到载机，判断载机能否为导弹提供中制导
                        if uav.id == missile.launcher_id:
                            if uav.can_offer_guidance(missile, self.UAVs):
                                has_datalink = True
                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt=self.dt_move, datalink=has_datalink)

                vmt1 = norm(last_vmt_)
                if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                    missile.dead = True
                if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                    missile.dead = True
                if missile.t > missile.t_max:  # 超时自爆
                    missile.dead = True
                if missile.t >= 0 + self.dt_move and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                    hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, last_ptt_, last_vtt_,
                                                       dt=self.dt_move)
                    if hit:
                        print(target.side, 'is hit')
                        missile.dead = True
                        missile.hit = True
                        missile.pos_ = point_m
                        missile.vel_ = last_vmt_
                        target.pos_ = point_t
                        target.vel_ = last_vtt_
                        target.dead = True
                        target.got_hit = True
                        self.UAV_hit[self.UAV_ids.index(target.id)] = True

            # 毁伤判断
            for i, UAV in enumerate(self.UAVs):
                # 飞机被导弹命中判断
                if UAV.red:
                    adv = self.BUAV
                if UAV.blue:
                    adv = self.RUAV
                if self.UAV_hit[i]:
                    UAV.dead = True
                    UAV.got_hit = True
                # 其他毁伤判断
                adv = self.UAVs[1 - i]
                pt_ = adv.pos_
                L_ = pt_ - UAV.pos_
                distance = np.linalg.norm(L_)
                # 近距杀
                #     short_range_killed = UAV.short_range_kill(adv)
                #     if short_range_killed:
                #         # self.running = False
                #         adv.got_hit = True
                # 出界判别
                if self.out_range(UAV):
                    UAV.dead = True
                    # self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
                UAV.dead = True
                # self.running = False

        r_dones = False
        b_dones = False
        if self.RUAV.dead:
            r_dones = True
        if self.BUAV.dead:
            b_dones = True

        self.RUAV = self.UAVs[0]
        self.BUAV = self.UAVs[1]

        if terminate:
            self.running = False

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def combat_terminate_and_reward(self, side):
        terminate = self.get_terminate()
        done = terminate

        # todo 奖励函数调用或是重写都要在这实现
        self.get_missile_state()
        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            alive_enm_missiles = self.alive_b_missiles
            alive_ally_missiles = self.alive_r_missiles
        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_ally_missiles = self.alive_b_missiles

        # 如果对手和对手的所有导弹都没了，且己方飞机还在，判定为胜
        if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
            self.win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方飞机还在，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            self.lose = 1
            done = 1
        # 如果友方和敌方打光导弹且都存活，或双方飞机都没了，判定为平
        elif ego.ammo == 0 and enm.ammo == 0 and not ego.dead and not enm.dead or \
                ego.dead and enm.dead:
            self.draw = 1
            done = 1
        else:
            done = 0

        reward = 0
        event_reward = 0
        if self.win:
            reward += 20
        if self.lose:
            reward -= 20
        if self.draw:
            reward -= 10


        return done, reward, event_reward
