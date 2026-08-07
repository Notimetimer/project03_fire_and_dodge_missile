'''
论文名称：
深度强化学习驱动的超视距空战自主决策方法
An Autonomous Decision-making Method for Beyond Visual Range Air Combat Driven by Deep Reinforcement Learning
2026
'''

from Controller.Controller_function import sub_of_radian
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
    return os.path.dirname(os.path.abspath(__file__))


current_dir = get_current_file_dir()
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from Envs.battle6dof1v1_missile0309_hierarchical import *
from .ChooseStrategyEnv2_0_hierarchical import ChooseStrategyEnv as BaseChooseStrategyEnv
from .ChooseStrategyEnv2_0_hierarchical import action_optionsLR

# 通过继承构建观测空间、奖励函数和终止条件
# 通过类的组合获取各子策略的观测量裁剪

def sigmoid(x):
    return 1/(1+np.exp(-x))
def arcsigmoid(x):
    return -np.log(1/(x)-1)
def softplus(x):
    return np.log(1+np.exp(x))

class ChooseStrategyEnv(BaseChooseStrategyEnv):
    """
    Inherit everything from ChooseStrategyEnv2_0 except override
    combat_terminate_and_reward (keeps other definitions/vars from base).
    """
    
    def combat_terminate_and_reward(self, side, action_label, action_shoot, action_cycle_multiplier=30, 
        end_reward_weight=1.0, 
        fire_reward_weight=None,
        fire_inside_weight = None, ends_in_bvr=0):

        if fire_reward_weight is None:
            fire_reward_weight=1.0

        if fire_inside_weight is None:
            fire_inside_weight = np.array([
            1, # 0 time
            1, # 1 AA
            1, # 2 Δψ
            1, # 3 v
            1, # 4 θ
        ])

        # --- 1. 参数初始化与状态获取 ---
        # 权重在此仅作为内部计算比例，实际整体缩放由外部 lambda 控制
        reward_weights = {
            'missile_guidance': 0.04,
            'target_locked': 0.06,
            'locked_by_target': 0.05,
            'missile_warning': 0.06,
            'enemy_gets_warning': 0.05,
            'alt_limit_penalty': 1.0,
            'border_penalty_scale': 0.2,
            'border_reward': 0.2, # 旧的数值: 1.0, 新的数值：0.2
            'angle_advantage': 0.01, # 0.007, # 0.03
            'height_advantage': 0.01,
            'aoa_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.2
            'pitch_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.05
            'to_center_reward' : 0.005, # 0.02 占领中心点的价值
            'speed_penalty': 0.01, # 慢速惩罚
        }

        ego_win=0
        ego_lose=0
        ego_draw=0

        # --- [新增] 态势记录 ---
        # 计算是否到达记录周期
        cycle_time = self.dt_maneuver * action_cycle_multiplier
        step_idx = round(self.t / cycle_time)
        
        # 允许极小的误差，并通过 last_record_t 避免在一帧内被两个智能体调用时重复记录
        if abs(self.t - step_idx * cycle_time) < 1e-4 and (self.t - self.last_record_t) > (cycle_time * 0.5):
            r_dist = np.linalg.norm(np.array([self.RUAV.pos_[0] - self.horizontal_center[0], self.RUAV.pos_[2] - self.horizontal_center[1]]))
            b_dist = np.linalg.norm(np.array([self.BUAV.pos_[0] - self.horizontal_center[0], self.BUAV.pos_[2] - self.horizontal_center[1]]))
            self.r_dist_seq.append(r_dist)
            self.b_dist_seq.append(b_dist)
            self.last_record_t = self.t

        if len(self.alive_missiles)==0:
            self.close_range_kill() # 允许跑刀
        self.update_missile_state()
        
        if side == 'r':
            ego = self.RUAV
            enm = self.BUAV
            alive_enm_missiles = self.alive_b_missiles
            alive_ally_missiles = self.alive_r_missiles
            i_can_guide = self.r_can_guide
        if side == 'b':
            ego = self.BUAV
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_ally_missiles = self.alive_b_missiles
            i_can_guide = self.b_can_guide

        # --- 2. 终止判定 ---
        done = 0
        
        # 死亡时间戳
        if ego.dead and ego.dead_time == None:
            ego.dead_time = self.t
        if enm.dead and enm.dead_time == None:
            enm.dead_time = self.t

        # --简单判定法--
        # 严格回合时间限制
        if self.t > self.game_time_limit:
            done = 1
        # 双杀
        if enm.dead and ego.dead:
            done = 1

        # 如果敌方和敌方所有导弹都没了，且我方存活，判定为胜
        if len(alive_enm_missiles) == 0 and enm.dead and not ego.dead:
            ego_win = 1
            done = 1
        # 如果友方和友方的所有导弹都没了，且敌方存活，判定为负
        elif len(alive_ally_missiles) == 0 and ego.dead and not enm.dead:
            ego_lose = 1
            done = 1
        # 双杀双活时间到，就是平局
        elif done: 
            ego_draw = 1

        # 是否让回合在超视距阶段结束
        if ends_in_bvr:
            if len(alive_ally_missiles) == 0 and len(alive_enm_missiles) == 0 \
                and ego.ammo==0 and enm.ammo==0 \
                    and (not ego.dead) and (not enm.dead):
                    done = 1
                    ego_draw = 1

        # 回合的胜负取决于ego_side
        if ego.side == self.ego_side:
            self.win = ego_win
            self.lose = ego_lose
            self.draw = ego_draw
        else:
            self.win = ego_lose
            self.lose = ego_win
            self.draw = ego_draw
        
        # ego_states = self.get_state(side)
        # enm_states = self.get_state(enm.side)
        # --- 3. 基础变量计算 --- pomdp会导致观测污染奖励函数，必须重新获取一次状态变量
        ego_states = copy.deepcopy(self.get_state(side))
        enm_states = copy.deepcopy(self.get_state(enm.side))
        # ego_states = ego.current_state
        # enm_states = enm.current_state
        dist_enm2ego = norm(ego.pos_ - enm.pos_)
        
        cos_ATA_enm = np.dot(enm.vel_, (ego.pos_ - enm.pos_)) / (norm(enm.vel_) * dist_enm2ego + 1e-3)
        ATA_enm = np.arccos(np.clip(cos_ATA_enm, -1, 1))
        delta_theta = ego_states["target_information"][2]
        distance = ego_states["target_information"][3]
        speed = ego_states["ego_main"][0]
        alt = ego_states["ego_main"][1]
        enm_speed = enm_states["ego_main"][0]
        enm_alt = enm_states["ego_main"][1]
        cos_delta_psi = ego_states["target_information"][0]
        sin_delta_psi = ego_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = ego_states["target_information"][4]
        # 严格被锁判定
        locked_by_target_flag = ego_states["locked_by_target"]
        target_locked_flag = ego_states["target_locked"]
        AA_hor = ego_states["target_information"][-2]
        warning = ego_states["warning"]
        missile_in_mid_term = ego_states["missile_in_mid_term"]
        missile_time_since_shoot = ego_states["weapon"]
        cos_delta_psi_threat = ego_states["threat"][0]
        sin_delta_psi_threat = ego_states["threat"][1]
        threat_distance = ego_states["threat"][3]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)
        delta_theta_threat = ego_states["threat"][2]
        
        # 奖励项初始化
        r_event = 0.0      # 结果奖励 + 关键事件奖励
        r_shaping = 0.0    # 状态密集奖励

        # 关键事件奖励（只在 cycle_time 边界触发并更新记忆，以时间戳保护避免重复）
        at_cycle = abs(self.t - step_idx * cycle_time) < 1e-4
        current_cycle_t = step_idx * cycle_time

        if not hasattr(ego, '_last_target_locked'):
            ego._last_target_locked = False
            ego._last_locked_by_target = False

        if at_cycle:
            if target_locked_flag and not ego._last_target_locked:
                r_event += 10.0 * (not ego.dead) * (not enm.dead)
            if locked_by_target_flag and not ego._last_locked_by_target:
                r_event -= 10.0 * (not ego.dead) * (not enm.dead)

            ego._last_target_locked = bool(target_locked_flag)
            ego._last_locked_by_target = bool(locked_by_target_flag)

        # 开火：每步只要有射击动作就给 -5.0 * action_shoot
        if action_shoot >= 1:
            r_shaping -= 5.0 * action_shoot * (not ego.dead)

        # 状态密集奖励
        # 高度
        r_shaping += -0.003 * (6000.0 - alt) * (alt < 6000.0) - 0.002 * (alt - 9000.0) * (alt > 9000.0)
        # 速度奖励
        r_shaping += 0.1 * (speed / (enm_speed + 1e-3)) * exp((alt - enm_alt) / 1800.0)
        # 角度奖励
        abs_delta_psi = abs(delta_psi)
        if 0 <= abs_delta_psi <= pi / 3.0:
            r_shaping += 0.8 * exp(-(abs_delta_psi - pi / 6.0) ** 2 / 100.0)
        # 威胁惩罚
        if 0 <= threat_distance <= 25e3:
            r_shaping += -15.0 * 2 ** (-threat_distance / 5e3 + 1)
        # 时间惩罚
        r_shaping += -0.01 * self.t
        # 密集奖励仅在我方存活时生效
        r_shaping *= (not ego.dead)

        r_event1 = r_event
        r_event2 = r_event
        r_event3 = r_event

        # --- 6. 结果奖励计算 (r_event) - 核心稀疏奖励 ---
        if done:
            if enm.dead and not ego.dead:
                r_event += 50.0
            elif ego.dead and not enm.dead:
                r_event -= 50.0
            elif self.draw:
                r_event -= 20.0

            r_event1 = r_event
            r_event2 = r_event
            r_event3 = r_event
            
            # 打印详细奖励组成，方便调试
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if ego_win else 'Lose' if ego_lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Shaping: {r_shaping:.2f}")

        # 返回 done 和三个分项奖励
        return done, r_event1+r_shaping, r_event2+r_shaping, r_event3+r_shaping
