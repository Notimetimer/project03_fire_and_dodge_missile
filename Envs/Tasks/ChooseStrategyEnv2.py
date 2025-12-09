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
        
        # [新增] 初始化 last_obs 属性，用于记录上一帧状态以计算瞬时奖励
        self.last_obs = None

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, pomdp=0):
        # 1. 调用父类 Battle 的 reset 方法，执行所有通用初始化
        super().reset(red_birth_state, blue_birth_state, red_init_ammo, blue_init_ammo)
        # # 初始化红蓝远离速度
        # self.last_dist_dot = None
        # self.last_dhor = None
        
        # [确认存在/修改] 确保每个 Episode 开始时重置 last_obs
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
        # # debug
        # # 动作太多？那就砍掉一些
        # if 0<= action <=4:
        #     action = 0
        # if 5<=action<=7:
        #     action = 6
        # if 8<=action<=13:
        #     action = 11
        
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
        # [新增] 计算对手视角下的ATA，用于判定严格的“被锁定”条件
        # 注意：这里计算的是 enm 的速度矢量 与 enm指向ego的连线 之间的夹角
        L_enm2ego = ego.pos_ - enm.pos_
        dist_enm2ego = norm(L_enm2ego)
        cos_ATA_enm = np.dot(enm.vel_, L_enm2ego) / (norm(enm.vel_) * dist_enm2ego + 0.001)
        ATA_enm = np.arccos(np.clip(cos_ATA_enm, -1, 1))
        # [新增] 严格的被锁定判定：系统判定被锁 + 距离限制 + 角度限制
        strict_locked_by_target = ego_obs["locked_by_target"] and (dist_enm2ego <= 80e3) and (ATA_enm <= pi/3)
        # [新增] 初始化或获取上一时刻的状态字典
        if self.last_obs is None:
            self.last_obs = {'r': {}, 'b': {}}
        last_info = self.last_obs.get(side, {})
        # 获取上一时刻的关键状态，默认为0 (False)
        last_target_locked = last_info.get("target_locked", 0)
        last_strict_locked_by_target = last_info.get("strict_locked_by_target", 0)
        last_warning = last_info.get("warning", 0)
        last_enm_warning = last_info.get("enm_warning", 0)


        # ---主要奖励---
        reward_main = 0.1  # 防自杀奖励
        if self.win:
            reward_main += 300
        if self.lose:
            reward_main -= 300
        if self.draw:
            reward_main -= 0

        # [修改] 为导弹提供制导 (大幅降低持续奖励)
        # 假设制导过程持续较长，将每步奖励设为极小值，防止刷分。
        # 例如：0.05 * 100步 = 5分。
        if missile_in_mid_term:
            reward_main += 0.05 

        # [修改] 锁定目标 (改为瞬时奖励：当前锁定 且 上一时刻未锁定)
        if ego_obs["target_locked"] and not last_target_locked:
            reward_main += 3

        # [修改] 被目标锁定 (改为瞬时惩罚，且使用严格判定)
        if strict_locked_by_target and not last_strict_locked_by_target:
            reward_main -= 1

        # [修改] 收到导弹警告 (改为瞬时惩罚)
        if warning and not last_warning:
            reward_main -= 3

        # [修改] 导弹锁定目标 (对手收到警告) (改为瞬时奖励)
        if enm_obs["warning"] and not last_enm_warning:
            reward_main += 5

        # 逃脱导弹 (保持原逻辑，这通常由escape_once标志位控制，本身就是一次性的)
        if ego.escape_once:
            reward_main += 10

        # 导弹被逃脱 (保持原逻辑)
        if enm.escape_once:
            reward_main -= 10

        # 发射导弹
        shoot = action_shoot
        # 发射惩罚
        if shoot == 1:
            if alpha*180/pi > 10:
                reward_main -= 10
            else:
                reward_main -= 5

        
        if done and ego.ammo == ego.init_ammo:
            reward_main -= 300 # 一发都不打必须重罚 100
        # if done and ego.ammo < ego.init_ammo:
        #     reward_main += 20 # 至少打了一枚

        # 重复发射导弹时惩罚
        if len(alive_ally_missiles)>1 and shoot==1:
            reward_main -= 30

        # 高度限制奖励/惩罚
        reward_main += ((alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * 5

        # 靠近边界惩罚
        if d_hor < 20e3:
            reward_main += (d_hor/20e3-1) * 2

        # [新增] 在计算完奖励后，更新上一时刻的状态记录，供下一步使用
        self.last_obs[side] = {
            "target_locked": float(ego_obs["target_locked"]),
            "strict_locked_by_target": float(strict_locked_by_target),
            "warning": float(warning),
            "enm_warning": float(enm_obs["warning"])
        }
            
        # ---辅助奖励---
        reward_fire = 0
        # 没发射导弹时alpha越小越好
        if len(alive_ally_missiles) == 0 and not warning:
            reward_fire += 0.5 - alpha / pi
        
        # # 导弹处于中制导阶段时alpha在±60*pi/180之间为奖励高台， 其余随alpha线性减少
        # reward_lock = 0
        # if missile_in_mid_term:
        #     # 提供制导信息奖励
        #     if abs(alpha) < np.pi / 3:
        #         reward_lock += 0.5
        #     else:
        #         reward_lock += 0.5 * (1 - (alpha - pi / 3) / pi)

        # # 有warning时alpha越大越好
        r_escape = 0
        if warning:
            r_escape += 0.8 * abs(delta_psi_threat) / pi

        # 迎角惩罚
        r_angle = 0
        r_angle -= 0.1 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
                           (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))
        r_angle -= 0.1 * (abs(ego.theta)/pi*2)  # 俯仰角惩罚

        # 导弹发射辅助奖励/惩罚
        reward_shoot = 0
        # if shoot == 1:
        #     # reward_shoot += 1 * abs(AA_hor)/pi-1  # 要把敌人骗进来杀
        #     reward_shoot += -1 + 1 * np.clip(ego.theta/(pi/3), -1, 1)  # 鼓励抛射
        
        # deltapsi变化率奖励 todobedontinued

        # 态势优势度 tobecontinued

        reward_assisted = np.sum([
            1 * reward_fire,
            # 1 * reward_lock,
            1 * r_escape,
            1 * r_angle,
            1 * reward_shoot,
        ])

        return done, reward_main+reward_assisted, reward_assisted

    # # 新的奖励函数
    # def combat_terminate_and_reward(self, side, action_label, action_shoot):
    #     # -------------------------------------------------------------------------
    #     # 状态获取与解包 (保持原样)
    #     # -------------------------------------------------------------------------
    #     self.get_missile_state()
    #     if side == 'r':
    #         ego = self.RUAV
    #         enm = self.BUAV
    #         alive_enm_missiles = self.alive_b_missiles
    #         alive_ally_missiles = self.alive_r_missiles
    #     if side == 'b':
    #         ego = self.BUAV
    #         enm = self.RUAV
    #         alive_enm_missiles = self.alive_r_missiles
    #         alive_ally_missiles = self.alive_b_missiles

    #     # 胜负判定逻辑 (保持原样)
    #     if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
    #         self.win = 1
    #         done = 1
    #     elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
    #         self.lose = 1
    #         done = 1
    #     elif ego.ammo == 0 and enm.ammo == 0 and not ego.dead and not enm.dead or \
    #             ego.dead and enm.dead:
    #         self.draw = 1
    #         done = 1
    #     else:
    #         done = 0

    #     if self.t > self.game_time_limit:
    #         done = 1
    #         if ego.ammo + len(alive_ally_missiles) == 0 and \
    #             enm.ammo + len(alive_enm_missiles) > 0 and not enm.dead:
    #             self.lose = 1
    #         elif enm.ammo + len(alive_enm_missiles) == 0 and \
    #             ego.ammo + len(alive_ally_missiles) > 0 and not ego.dead:
    #             self.win = 1                
    #         else:
    #             self.draw = 1

    #     # 观测值解包 (保持原样)
    #     ego_obs = self.base_obs(side, pomdp=self.pomdp) 
    #     enm_obs = self.base_obs(enm.side, pomdp=self.pomdp)
    #     delta_theta = ego_obs["target_information"][2]
    #     distance = ego_obs["target_information"][3]
    #     speed = ego_obs["ego_main"][0]
    #     alt = ego_obs["ego_main"][1]
    #     cos_delta_psi = ego_obs["target_information"][0]
    #     sin_delta_psi = ego_obs["target_information"][1]
    #     delta_psi = atan2(sin_delta_psi, cos_delta_psi)
    #     alpha = ego_obs["target_information"][4]
    #     warning = ego_obs["warning"]
    #     missile_in_mid_term = ego_obs["missile_in_mid_term"]
    #     missile_time_since_shoot = ego_obs["weapon"]
    #     AA_hor = ego_obs["target_information"][-2]
        
    #     cos_delta_psi_threat = ego_obs["threat"][0]
    #     sin_delta_psi_threat = ego_obs["threat"][1]
    #     delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)

    #     ego_states = self.get_state(side)
    #     d_hor = ego_states["border"][0]

    #     L_enm2ego = ego.pos_ - enm.pos_
    #     dist_enm2ego = norm(L_enm2ego)
    #     cos_ATA_enm = np.dot(enm.vel_, L_enm2ego) / (norm(enm.vel_) * dist_enm2ego + 0.001)
    #     ATA_enm = np.arccos(np.clip(cos_ATA_enm, -1, 1))
    #     strict_locked_by_target = ego_obs["locked_by_target"] and (dist_enm2ego <= 80e3) and (ATA_enm <= pi/3)

    #     if self.last_obs is None:
    #         self.last_obs = {'r': {}, 'b': {}}
    #     last_info = self.last_obs.get(side, {})
    #     last_target_locked = last_info.get("target_locked", 0)
    #     last_strict_locked_by_target = last_info.get("strict_locked_by_target", 0)
    #     last_warning = last_info.get("warning", 0)
    #     last_enm_warning = last_info.get("enm_warning", 0)


    #     # =========================================================================
    #     # 核心修改区域：数值重平衡
    #     # 目标：Total Return 在 [-20, 20] 左右
    #     # =========================================================================

    #     reward_main = 0.0
        
    #     # 1. 生存/防自杀基础奖励 (Base Reward)
    #     # 不要给 +10 这种巨额数字。
    #     # 给一个极小的正数鼓励存活，或者给0。
    #     # 如果这里给 +0.02，100步就是 +2.0。如果死了是 -10.0。Agent 就会倾向于活着。
    #     reward_main += 0.02 

    #     # 2. 胜负结算 (Terminal Reward)
    #     # 之前的 ±300 太大，改为 ±15 ~ ±20 级别
    #     if self.win:
    #         reward_main += 20.0
    #     if self.lose:
    #         reward_main -= 20.0  # 死亡/失败重罚
    #     if self.draw:
    #         reward_main -= 5.0   # 平局轻微惩罚（鼓励进攻）

    #     # 3. 过程性状态奖励 (Continuous Step Reward) -> 降权到 0.05 级别
        
    #     # 为导弹提供制导
    #     if missile_in_mid_term:
    #         reward_main += 0.05  # 维持原样或微调，100步=5分，合理

    #     # 4. 瞬时事件奖励 (Instant Event Reward) -> 控制在 0.5 ~ 2.0 级别

    #     # 锁定目标 (瞬时)
    #     if ego_obs["target_locked"] and not last_target_locked:
    #         reward_main += 1.0  # 之前是3，太大

    #     # 被目标锁定 (瞬时惩罚)
    #     if strict_locked_by_target and not last_strict_locked_by_target:
    #         reward_main -= 0.5  # 之前是1，适中

    #     # 收到导弹警告 (瞬时惩罚)
    #     if warning and not last_warning:
    #         reward_main -= 1.0  # 之前是3，太大

    #     # 压迫对手：导弹让对手收到警告 (瞬时奖励)
    #     if enm_obs["warning"] and not last_enm_warning:
    #         reward_main += 1.5  # 进攻是最好的防守

    #     # 逃脱导弹 (One-off)
    #     if ego.escape_once:
    #         reward_main += 2.0  # 之前是10，太大

    #     # 导弹被逃脱 (One-off)
    #     if enm.escape_once:
    #         reward_main -= 2.0  # 之前是10，太大

    #     # 5. 发射动作惩罚 (Action Penalty)
    #     shoot = action_shoot
    #     if shoot == 1:
    #         # 基础弹药消耗：适度惩罚，防止乱射，但不要太重导致不敢射
    #         reward_main -= 1.0 
            
    #         # 发射条件差的额外惩罚
    #         if alpha * 180/pi > 10 or distance > 50e3 or abs(AA_hor) < pi/2:
    #             reward_main -= 1.0 # 额外罚
    #         else:
    #             pass # 好位置发射不额外罚

    #     # 弹药打光且无战果的最终惩罚
    #     if done and ego.ammo == ego.init_ammo and not self.win:
    #         pass 
    #         # 这里不需要额外罚300了，因为Lose已经是-20，
    #         # 如果需要，可以微调为 -5.0
    #         reward_main += -5

    #     # 重复发射导弹时惩罚
    #     if len(alive_ally_missiles) > 1 and shoot == 1:
    #         reward_main -= 2.0  # 之前是30，太大

    #     # 6. 边界与高度限制 (Constraints)
    #     if 1:  # not self.shielded:
    #         # 高度限制
    #         if alt <= self.min_alt_safe:
    #             reward_main += 0.1 * np.clip(ego.vu / 100, -1, 1) # 软约束
    #         if alt >= self.max_alt_safe:
    #             reward_main -= 0.1 * np.clip(ego.vu / 100, -1, 1)
        
    #     # 靠近边界惩罚
    #     if d_hor < 20e3:
    #         # d_hor/20e3 - 1 在 [0, -1] 之间
    #         # 系数设为 0.5，最大惩罚每步 -0.5，10步就是 -5，Agent 会非常害怕边界
    #         reward_main += (d_hor/20e3 - 1) * 0.5

    #     # 更新状态记录
    #     self.last_obs[side] = {
    #         "target_locked": float(ego_obs["target_locked"]),
    #         "strict_locked_by_target": float(strict_locked_by_target),
    #         "warning": float(warning),
    #         "enm_warning": float(enm_obs["warning"])
    #     }
            
    #     # =========================================================================
    #     # 辅助奖励 (Assisted Reward)
    #     # 目标：作为 Shaping，量级应该是 Main 的 1/10
    #     # =========================================================================
    #     reward_fire = 0
    #     if len(alive_ally_missiles) == 0 and not warning:
    #         # 范围约 [-0.5, 0.5] -> 乘系数 0.1 -> [-0.05, 0.05]
    #         reward_fire += 0.5 - alpha / pi 
        
    #     r_escape = 0
    #     if warning:
    #          # 范围 [0, 0.8] -> 乘系数 0.1 -> [0, 0.08]
    #         r_escape += 0.8 * abs(delta_psi_threat) / pi

    #     # 迎角与姿态惩罚 (通常很小)
    #     r_angle = 0
    #     r_angle -= 0.1 * ((ego.alpha_air*180/pi> 15)*(ego.alpha_air*180/pi-15)+\
    #                        (ego.alpha_air*180/pi< -5)*(-5 - ego.alpha_air*180/pi))
    #     r_angle -= 0.1 * (abs(ego.theta)/pi*2)

    #     reward_shoot = 0
    #     if shoot == 1:
    #         reward_shoot += -1 + 1 * np.clip(ego.theta/(pi/3), -1, 1)

    #     # 辅助奖励加权求和
    #     # 系数改为 0.1，让它们仅仅起到“微调引导”作用，不干扰主线任务
    #     weight_assist = 0.1 
        
    #     reward_assisted = weight_assist * np.sum([
    #         reward_fire,
    #         r_escape,
    #         r_angle,
    #         reward_shoot,
    #     ])

    #     # 返回总奖励
    #     return done, reward_main + reward_assisted, reward_assisted
        
        
