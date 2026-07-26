'''
带奖励时间回溯
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
        cos_delta_psi = ego_states["target_information"][0]
        sin_delta_psi = ego_states["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        alpha = ego_states["target_information"][4]
        # 严格被锁判定
        locked_by_target = ego_states["locked_by_target"] #  and (dist_enm2ego <= 80e3) and (ATA_enm <= self.RUAV.max_radar_angle_rad)
        target_locked = ego_states["target_locked"]
        AA_hor = ego_states["target_information"][-2]
        warning = ego_states["warning"]
        missile_in_mid_term = ego_states["missile_in_mid_term"]
        missile_time_since_shoot = ego_states["weapon"] 
        # ↑无法用在开火奖励函数里面，会随着执行顺序被覆盖掉
        # ↑但是可以用在机动奖励函数里面，不论如何，只要刚发射导弹就应该crank
        cos_delta_psi_threat = ego_states["threat"][0]
        sin_delta_psi_threat = ego_states["threat"][1]
        threat_distance = ego_states["threat"][3]
        delta_psi_threat = atan2(sin_delta_psi_threat, cos_delta_psi_threat)
        delta_theta_threat = ego_states["threat"][2]
        
        # 奖励项初始化
        r_event = 0.0
        r_dense = 0.0
        escape_missile_bool = 0.0

        # 关键事件奖励（只在 cycle_time 边界触发并更新时间戳，避免重复）
        at_cycle = abs(self.t - step_idx * cycle_time) < 1e-4
        current_cycle_t = step_idx * cycle_time

        if not hasattr(ego, '_last_target_locked'):
            ego._last_target_locked = False
            ego._last_locked_by_target = False
            ego._last_warning = False
            ego._last_lock_escape_t = -1.0
            ego._last_missile_escape_t = -1.0
            ego._last_seen_max_missile_id = 0

        if at_cycle:
            # 脱离对手雷达锁定瞬间 +5
            if not locked_by_target and ego._last_locked_by_target:
                if (current_cycle_t - ego._last_lock_escape_t) > (cycle_time * 0.5):
                    r_event += 5.0 * (not ego.dead)
                    ego._last_lock_escape_t = current_cycle_t
            # 逃脱来袭导弹瞬间 +10
            if not warning and ego._last_warning:
                if (current_cycle_t - ego._last_missile_escape_t) > (cycle_time * 0.5):
                    r_event += 10.0 * (not ego.dead)
                    escape_missile_bool = 1.0
                    ego._last_missile_escape_t = current_cycle_t

            ego._last_target_locked = bool(target_locked)
            ego._last_locked_by_target = bool(locked_by_target)
            ego._last_warning = bool(warning)

        # 密集状态奖励（每步计算，对应 KAERS event-based rewards 的密集化版本）
        enm_warning = enm_states["warning"]
        if not ego.dead:
            # 开火：每步只要有射击动作就给 -25
            if action_shoot >= 1:
                r_dense -= 25.0
            # Lock on / Be locked
            if target_locked:
                r_dense += 2.0
            if locked_by_target:
                r_dense -= 2.0
            # Missile lock（我方导弹锁定敌机） / Missile alert（敌弹锁定我机）
            if enm_warning:
                r_dense += 5.0
            if warning:
                r_dense -= 5.0
            # Out：靠近战场边界且速度朝外时惩罚
            ego_pos_h = np.array([ego.pos_[0], ego.pos_[2]])
            center_to_pos = ego_pos_h - np.array(self.horizontal_center)
            dist_center = np.linalg.norm(center_to_pos)
            if dist_center > (self.R_cage - 340 * 60):
                ego_vh = np.array([ego.vel_[0], ego.vel_[2]])
                # 从中心指向当前位置的向量与水平速度矢量夹角小于直角 -> 正飞向边界外
                if np.dot(center_to_pos, ego_vh) > 0:
                    r_dense -= 10.0

        # 结果奖励（对应 Episodic rewards）
        if done:
            if enm.dead and not ego.dead:
                r_event += 500.0
            elif ego.dead and not enm.dead:
                r_event -= 500.0
            # 平局保持 +0.0

            # 打印详细奖励组成，方便调试
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if ego_win else 'Lose' if ego_lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Dense: {r_dense:.2f}")

        # 返回 done 和三个分项奖励
        r_total = r_event + r_dense
        not_ego_dead = 0.0 if ego.dead else 1.0
        enm_dead = 1.0 if enm.dead else 0.0
        # 新增：记录本步新发射导弹的 ID，没有则为 0
        # 只在 action_shoot==1 且出现更高 ID 的导弹时记录，避免把“导弹在空中飞”的 ID 写进去
        current_max_id = max([m.id for m in alive_ally_missiles], default=0)
        new_missile_id = 0.0
        if action_shoot >= 1 and current_max_id > ego._last_seen_max_missile_id:
            new_missile_id = float(current_max_id)
        ego._last_seen_max_missile_id = current_max_id
        r_total_vec = np.array([r_total, not_ego_dead, escape_missile_bool, enm_dead, new_missile_id], dtype=np.float32)
        return done, r_total_vec, r_total_vec, r_total_vec
