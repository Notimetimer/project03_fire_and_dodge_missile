'''
出生点改在外面指定

子策略暂时使用规则智能体，留下使用神经网络的接口

加入导弹发生相关奖励，区分主要奖励和辅助奖励
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

action_options = {
                    0: "track",
                    1: "30track",
                    2: "60track",
                    3: "-30track",
                    4: "-60track",
                    5: "+-30crank",
                    6: "+-60crank",
                    7: "snake",
                    8: "splitS",
                    9: "39",
                    10: "slowTurn",
                    11: "fastTurn",
                    12: "-30turn",
                    13: "-60turn",
                }

class ChooseStrategyEnv(Battle):
    def __init__(self, args, tacview_show=0):
        super().__init__(args, tacview_show)
        self.key_order_1v1 = [
            "target_alive", # 1
            "target_observable", # 1
            "target_locked", # 1
            "missile_in_mid_term",  # 1
            "locked_by_target",  # 1
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "weapon", # 1
            "threat",  # 4
            "border",  # 2
        ]
        self.obs_dim = 1*6+8+7+1+4+2
        self.fly_act_dim = [14]
        self.fire_dim = 1

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=0):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        self.last_obs = None
        self.pomdp = pomdp     
    
    def obs_1v1(self, side, pomdp=0):
        pre_full_obs = self.base_obs(side, pomdp)
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.key_order_1v1}
        
        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.key_order_1v1)
        return flat_obs, full_obs


    def maneuver14(self, UAV, action):
        # 输入动作与动力运动学状态
        uav_obs = self.base_obs(UAV.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = uav_obs["target_information"][2]
        distance = uav_obs["target_information"][3] * 10e3
        d_hor, leftright = uav_obs["border"]
        # state = self.get_state(UAV.side)
        speed = uav_obs["ego_main"][0]
        alt = uav_obs["ego_main"][1]
        cos_delta_psi = uav_obs["target_information"][0]
        sin_delta_psi = uav_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(uav_obs["threat"][1], uav_obs["threat"][0])

        move_action = np.zeros(3)

        # 水平跟踪
        if action == 0:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 0
            speed_cmd = 400

        # 30°爬升加速
        if action == 1:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 2500
            speed_cmd = 400

        # 60°爬升加速
        if action == 2:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = 5000
            speed_cmd = 400

        # -30°俯冲跟踪
        if action == 3:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # -60°俯冲跟踪
        if action == 4:
            delta_psi_cmd = np.clip(delta_psi, -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400

        # ±30°水平偏移
        if action == 5:
            delta_psi_cmd = delta_psi - np.sign(delta_psi)*pi/6
            delta_height_cmd = 0
            speed_cmd = 350

        # ±60°水平偏移
        if action == 6:
            delta_psi_cmd = delta_psi - np.sign(delta_psi) * 55 * pi/180
            delta_height_cmd = 0
            speed_cmd = 350

        # 水平蛇形机动
        if action == 7:
            if delta_psi > 50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            if delta_psi < -50 * pi/180:
                delta_psi_cmd = sub_of_radian(delta_psi, 0)
            elif UAV.phi>=0:
                delta_psi_cmd = sub_of_radian(delta_psi+pi/3, 0)
            else:
                delta_psi_cmd = sub_of_radian(delta_psi-pi/3, 0)
            delta_height_cmd = 0
            speed_cmd = 350

        # 破s
        if action == 8:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = sub_of_radian(delta_psi, pi)
            delta_height_cmd = max(-2000, self.min_alt_safe-UAV.alt)
            speed_cmd = 300

        # 水平三九线机动
        if action == 9:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = delta_psi_temp - np.sign(delta_psi)*pi/2
            delta_height_cmd = 0
            speed_cmd = 400

        # 水平慢置尾
        if action == 10:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = -np.sign(delta_psi)*np.clip((1-abs(delta_psi_temp)/pi) * 2, 0, 1) * 10*pi/180
            delta_height_cmd = 0
            speed_cmd = 600

        # 水平快置尾
        if action == 11:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -500 if abs(delta_psi_temp)<pi/2 else 0
            speed_cmd = 400

        # 水平快置尾后-30°俯冲
        if action == 12:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3
            speed_cmd = 400

        # 水平快置尾后-60°俯冲
        if action == 13:
            delta_psi_temp = delta_psi_threat if uav_obs["warning"] else delta_psi
            delta_psi_cmd = np.clip(sub_of_radian(delta_psi, pi), -pi/2, pi/2)
            delta_height_cmd = -5000/3*2
            speed_cmd = 400
        return np.array([delta_height_cmd, delta_psi_cmd, speed_cmd])
    

    def combat_terminate_and_reward(self, side, action_label, action_shoot):
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

        if self.t > self.game_time_limit:
            done = 1
            # 如果超时，我方打光导弹，导弹全自爆，对手导弹还有剩，且存活，判定为负
            if ego.ammo + len(alive_ally_missiles) == 0 and \
                enm.ammo + len(alive_enm_missiles) > 0 and not enm.dead:
                self.lose = 1

            # 如果超时，对手打光导弹，导弹全自爆，我方导弹还有剩，且存活，判定为胜
            elif enm.ammo + len(alive_enm_missiles) == 0 and \
                ego.ammo + len(alive_ally_missiles) > 0 and not ego.dead:
                self.win = 1                

            # 如果超时，双方均未打光导弹/仍有导弹在空中飞，且双方均存活，判定为平
            else:
                self.draw = 1

        ego_obs = self.base_obs(side, pomdp=self.pomdp)  ### test 部分观测的话用1
        enm_obs = self.base_obs(enm.side, pomdp=self.pomdp)  ### test 部分观测的话用1
        delta_theta = ego_obs["target_information"][2]
        distance = ego_obs["target_information"][3]
        speed = ego_obs["ego_main"][0]
        alt = ego_obs["ego_main"][1]
        cos_delta_psi = ego_obs["target_information"][0]
        sin_delta_psi = ego_obs["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = ego_obs["target_information"][4]
        warning = ego_obs["warning"]
        missile_in_mid_term = ego_obs["missile_in_mid_term"]
        missile_time_since_shoot = ego_obs["weapon"]
        AA_hor = ego_obs["target_information"][-2]
        
        cos_delta_psi_threat = ego_obs["threat"][0]
        sin_delta_psi_threat = ego_obs["threat"][1]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)

        ego_states = self.get_state(side)
        d_hor = ego_states["border"][0]


        # ---主要奖励---
        reward_main = 0
        if self.win:
            reward_main += 300
        if self.lose:
            reward_main -= 300
        if self.draw:
            reward_main -= 0

        # 为导弹提供制导
        if missile_in_mid_term:
            reward_main += 2

        # 锁定目标
        if ego_obs["target_locked"]:
            reward_main += 1.1

        # 被目标锁定
        if ego_obs["locked_by_target"]:
            reward_main -= 1

        # 收到导弹警告
        if warning:
            reward_main -= 3

        # 导弹锁定目标
        if enm_obs["warning"]:
            reward_main += 3

        # 逃脱导弹
        if ego.escape_once:
            reward_main += 10

        # 导弹被逃脱
        if enm.escape_once:
            reward_main -= 5

        # 发射导弹
        shoot = action_shoot
        # 发射惩罚
        if shoot == 1:
            reward_main -= 10

        # ---辅助奖励---
        reward_fire = 0
        # 没发射导弹时alpha越小越好
        if len(alive_ally_missiles) == 0 and not warning:
            reward_fire += 0.5 - alpha / pi
        
        # 导弹处于中制导阶段时alpha在±60*pi/180之间为奖励高台， 其余随alpha线性减少
        reward_lock = 0
        if missile_in_mid_term:
            # 提供制导信息奖励
            if abs(alpha) < np.pi / 3:
                reward_lock += 0.5
            else:
                reward_lock += 0.5 * (1 - (alpha - pi / 3) / pi)

        # # 有warning时alpha越大越好
        r_escape = 0
        # if warning:
        #     r_escape += 0.8 * abs(delta_psi_threat) / pi

        # 迎角惩罚
        r_angle = 0
        r_angle -= 0.01 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))

        # 高度限制奖励/惩罚
        r_alt = (alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)

        # 靠近边界惩罚
        r_border = 0
        if d_hor < 20e3:
            r_border = (d_hor/20e3-1) * 2

        # 导弹发射辅助奖励/惩罚
        reward_shoot = 0
        if shoot == 1:
            reward_shoot += np.clip((missile_time_since_shoot-30)/30, -1,1)  # 尽可能增大发射间隔
            reward_shoot += 1 * abs(AA_hor)/pi-1  # 要把敌人骗进来杀
            reward_shoot += 1 * np.clip(ego.theta/(pi/3), -1, 1)  # 鼓励抛射
            # reward_shoot -= np.clip(dist/40e3, 0, 1)
        if terminate and ego.ammo == ego.init_ammo:
            reward_shoot -= 300 # 一发都不打必须重罚 100
        if terminate and ego.ammo < ego.init_ammo:
            reward_shoot += 20 # 至少打了一枚
        # 重复发射导弹时惩罚, 否则有奖励
        if len(alive_ally_missiles)>1 and shoot==1:
            reward_shoot -= 30
        if len(alive_ally_missiles)>1 and shoot==0:
            reward_shoot += 5
        
        # deltapsi变化率奖励 todobedontinued

        # 态势优势度 tobecontinued

        reward_assisted = np.sum([
            1 * reward_fire,
            1 * reward_lock,
            1 * r_escape,
            1 * r_angle,
            1 * r_alt,
            1 * reward_shoot,
            1 * r_border
        ])

        return done, reward_main+reward_assisted, reward_assisted


        
        
