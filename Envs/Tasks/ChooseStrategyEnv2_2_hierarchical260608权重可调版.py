'''
增加开火惩罚
三元组奖励
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
        fire_inside_weight = None,):

        if fire_reward_weight is None:
            fire_reward_weight=1.0

        if fire_inside_weight is None:
            fire_inside_weight = np.array([
            1, # 0 distance
            1, # 1 time
            1, # 2 AA
            1, # 3 Δψ
            1, # 4 v
            1, # 5 θ
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
            'aoa_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.2
            'pitch_penalty': 0.02, # 旧的数值: 0.02, 新的数值：0.05
        }

        ego_win=0
        ego_lose=0
        ego_draw=0

        # --- [新增] 态势记录 ---
        # 计算是否到达记录周期
        cycle_time = self.dt_maneuver * action_cycle_multiplier
        step_idx = round(self.t / cycle_time)
        
        # # 允许极小的误差，并通过 last_record_t 避免在一帧内被两个智能体调用时重复记录
        # if abs(self.t - step_idx * cycle_time) < 1e-4 and (self.t - self.last_record_t) > (cycle_time * 0.5):
        #     # 原有记录逻辑（距离序列）
        #     r_dist = np.linalg.norm(np.array([self.RUAV.pos_[0] - self.horizontal_center[0], self.RUAV.pos_[2] - self.horizontal_center[1]]))
        #     b_dist = np.linalg.norm(np.array([self.BUAV.pos_[0] - self.horizontal_center[0], self.BUAV.pos_[2] - self.horizontal_center[1]]))
        #     self.r_dist_seq.append(r_dist)
        #     self.b_dist_seq.append(b_dist)
        #     self.last_record_t = self.t

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
        ATA = ego_states["target_information"][4]
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
        out_locked_time = ego_states["out_locked"]
        
        # 奖励项初始化
        r_event = 0.0      # 结果奖励
        
        r_constraint = 0.0 # 约束与代价
        r_shaping = 0.0    # 战术引导

        # --- 4. 约束奖励计算 (r_constraint) - 固定权重 ---
        # # 高度限制奖励/惩罚
        # r_constraint += ((alt <= self.min_alt_safe) * np.clip(ego.vu / 100, -1, 1) + \
        #                 (alt >= self.max_alt_safe) * np.clip(-ego.vu / 100, -1, 1)) * reward_weights['alt_limit_penalty']
        
        reward_weights['border_penalty_scale']=0.2
        reward_weights['border_reward']=0.2
        reward_weights['to_center_reward']=0.005 # 0.02 占领中心点的价值
        
        
        # 靠近边界惩罚
        o002ego_ = np.array([ego.pos_[0], ego.pos_[2]]) # 北，东
        ego_vh_ = np.array([ego.vel_[0], ego.vel_[2]])
        d_hor = ego_states["border"][0]
        if d_hor <= 50e3:
            r_constraint -= (1-d_hor/50e3) * np.dot(ego_vh_, o002ego_)/norm(o002ego_ + 1e-3)/340 * reward_weights['border_penalty_scale']
        else:
            r_constraint += reward_weights['border_reward']
        
        # 补充占领中心奖励
        border_side = ego_states["border"][1]
        
        line2center = np.array([self.horizontal_center[0]-ego.pos_[0], self.horizontal_center[1]-ego.pos_[2]])
        psi2center = np.arctan2(line2center[1], line2center[0])
        dist2center = norm(line2center)
        delta_psi_to_center = sub_of_radian(psi2center, ego.psi)

        # 距离中心越远，指向中心的价值越高
        r_constraint += ((1 + dist2center/self.R_cage)/2) * \
                        ((1 - abs(delta_psi_to_center)/pi)) * \
                        reward_weights['to_center_reward']

        r_constraint *= (1-ego.dead) # 密集奖励只有在存活的时候有意义

        # 开火代价控制
        wasted = 0
        now_dead = ego.dead or self.out_cage(ego)
        if not getattr(ego, 'last_dead', False):
            if now_dead:
                shoot = ego.ammo # 死亡瞬间，记录清仓惩罚
                wasted = ego.ammo
                ego.last_dead = True
            else:
                shoot = action_shoot # 还在飞，正常记录
        else:
            shoot = 0 # 死后不再记录任何幻影开火
            wasted = 0


        # --- 态势辅助奖励 ---
        # threat_distance_threshold = 20e3
        # # 1. 进攻态势：我方导弹是否进入敌机周围20km，并迫使对手进入防御
        # enm_threat_dist = enm_states["threat"][3]
        # # 把导弹送得越近，分数就越高
        # if enm_threat_dist <= threat_distance_threshold:
        #     r_constraint += 1.0 * np.exp(-2*enm_threat_dist/threat_distance_threshold) * fire_reward_weight # * self.dt_maneuver * action_cycle_multiplier # 0.001
            
        #     if abs(self.t % 2) < 0.1:
        #         print("威胁奖励：", 1.0 * np.exp(-2*enm_threat_dist/threat_distance_threshold) * fire_reward_weight)
        #         print()

        if not hasattr(ego, '_last_enm_threat_dist'):
            ego._last_enm_threat_dist = enm_states["threat"][3]

        threat_distance_threshold = 13e3
        if ego._last_enm_threat_dist > threat_distance_threshold and enm_states["threat"][3] <= threat_distance_threshold:
            r_constraint += 10 * fire_reward_weight # 稀疏威胁奖励，导弹送进10km以内就给，便于跟开火惩罚换算


        # # 2. 防御态势：敌方导弹是否进入我机周围20km，并被迫进入防御，导弹离我越近，越指向导弹，惩罚越重
        # if threat_distance <= threat_distance_threshold:
        #     r_constraint -= 1.0 * (1.2-abs(delta_psi_threat)/pi)/1.2 * np.exp(-2*threat_distance/threat_distance_threshold) # * self.dt_maneuver * action_cycle_multiplier # 0.001

        # 战术引导奖励
        reward_weights['angle_advantage']= 10 # 0.2 # 0.08 # 0.05
        reward_weights['speed_penalty']= 4 # 0.1 # 0.03 # 0.005 # 慢速惩罚

        enm_threat_dist = enm_states["threat"][3]
        "所有引导奖励的除去权重，都缩放到-1~1之间，避免agent利用奖励差值刷分"

        # ==========================================
        # Delta 势能奖励计算（替代原引导奖励）
        # ==========================================
        # 1. 计算三个势能（无守卫，每次调用都执行当前状态）
        
        # 进攻引导, 如果没有存活导弹，或者导弹射错了方向，按瞄准误差给分
        # phi_attack = \
        #     9 * sigmoid((
        #         2-0.7*np.exp(1.2*abs(delta_psi)*2/pi) +
        #         1*np.clip(ego.vu/100, -1, 1) +
        #         1*(1-np.exp(-ego.theta/pi*3)) # (1-np.exp(-ego.theta/pi*3))  or (1-np.exp(delta_theta/pi*3))
        #     )/8)
        phi_attack = (
            2.0 * (1.0 - abs(delta_psi) / pi) +            # 偏角越小，势能越高 (线性)
            0.5 * np.clip(ego.alt / 100, -1, 1) +           # 爬升率势函数(线性)
            2.0 * np.clip(ego.theta / (pi/2), -1, 1)       # 俯仰角优势 (线性)
        )
        
        # crank引导，如果导弹飞在正确的方向上，做crank下高
        # i_can_guide = - np.tanh(8*(abs(ATA)-pi/3))
        # phi_crank = \
        #     4 * sigmoid((
        #         1.0 * i_can_guide +
        #         0.5 - 3*abs(abs(delta_psi)-pi/3) +
        #         1.5 * ((-ego.theta) / (pi/2))
        #     )/(3.4))
        i_can_guide = 1*(ATA<np.radians(55)) - (ATA>=np.radians(55)) * abs(ATA - np.radians(55)) / (pi/2 - np.radians(55)) # ATA偏离60度的线性惩罚
        phi_crank = (
            2.0 * i_can_guide +                            # 引导夹角
            -2.0 * abs(abs(delta_psi) - np.radians(55)) / (np.radians(55)) +   # 偏角逼近 60度
            0.5 * np.clip(-ego.alt / 100, -1, 1)            # Crank 必须伴随降高
        )

        
        # RWR防御引导，如果有告警，不论如何都置尾下高
        # phi_defense = \
        #     2 * sigmoid((
        #         -2 * np.exp(2*ego.theta/(pi/2)) * (delta_theta_threat>=0)+
        #         -5 * np.exp(1.2*(ego.theta*2/pi)**2) * (delta_theta_threat<0)+
        #         4 * (-1+(abs(delta_psi_threat)/(pi/2)))
        #     )/(5))
        phi_defense = (
            2.0 * (abs(delta_psi_threat) / pi) +           # 导弹在后半球势能最高
            2.0 * np.clip(-ego.theta / (pi/2), 0, 1) +     # 严厉逼迫俯冲下高
            0.8 * np.clip(-ego.alt / 100, -1, 1)           # 降高度增加阻力
        )
        
        # 速度势函数（替代原速度惩罚）
        target_mach = 1.0
        if ego.speed < target_mach*340:
            phi_speed = -((target_mach-ego.speed/340)/(target_mach - 0.8) + max(ego.theta, 0)/(pi/2))
        else:
            phi_speed = 0.0

        # 2. 初始化上一时刻势能（首次进入）
        if not hasattr(ego, '_last_phi_t'):
            ego._last_phi_attack = phi_attack
            ego._last_phi_crank = phi_crank
            ego._last_phi_defense = phi_defense
            ego._last_phi_speed = phi_speed
            ego._last_phi_t = self.t
        
        # 3. 计算 Delta 奖励（无守卫，每次调用都执行）
        dt_phi = cycle_time # self.t - ego._last_phi_t
        gamma = 1.0 # 0.997
        
        if self.t % 10 < 0.1 and not ego.dead:
            print(ego.side)
            print("进攻势变化率", (gamma*phi_attack - ego._last_phi_attack)/(dt_phi + 1e-6) * reward_weights['angle_advantage'])
            print("crank势变化率", (gamma*phi_crank - ego._last_phi_crank)/(dt_phi + 1e-6) * reward_weights['angle_advantage'])
            print("防御势变化率", (gamma * phi_defense - ego._last_phi_defense)/(dt_phi + 1e-6) * reward_weights['angle_advantage'])
            print()

        if not warning:
            # 进攻期：如果没有存活导弹，按瞄准误差给分
            # if len(alive_ally_missiles) == 0:
            if not missile_in_mid_term:
                delta_phi = gamma*phi_attack - ego._last_phi_attack
                r_constraint += np.clip((delta_phi / (dt_phi + 1e-6)) * reward_weights['angle_advantage'], -3, 3) * (1-ego.dead)
            
            # crank期：如果导弹飞在正确的方向上，做crank下高
            else:
                delta_phi = gamma*phi_crank - ego._last_phi_crank
                r_constraint += np.clip((delta_phi / (dt_phi + 1e-6)) * reward_weights['angle_advantage'], -3, 3) * (1-ego.dead)
        
        # 防御期：如果有告警，不论如何都置尾下高
        elif warning:
            delta_phi = gamma * phi_defense - ego._last_phi_defense
            r_constraint += np.clip((delta_phi / (dt_phi + 1e-6)) * reward_weights['angle_advantage'], -3, 3) * (1-ego.dead)
        
        # 速度奖励（速度势能变化）
        delta_phi_speed = gamma * phi_speed - ego._last_phi_speed
        r_constraint += np.clip((delta_phi_speed / (dt_phi + 1e-6)) * reward_weights['speed_penalty'], -3, 3) * (1-ego.dead)
        
        # 4. 时间戳保护：旧势能更新（守卫内）
        # 注意：时间戳必须挂在 ego 上，而非 self，否则红蓝双方共享同一个时间戳，
        # 导致第二个调用方（如蓝方）的 _last_phi_* 永远不被更新，产生巨大累积误差
        if not hasattr(ego, '_last_phi_t'):
            ego._last_phi_t = -cycle_time
        if abs(self.t - step_idx * cycle_time) < 1e-4 and (self.t - ego._last_phi_t) > (cycle_time * 0.5):
            # 更新时间戳保护的"上一次"状态（守卫内）
            ego._last_phi_attack = phi_attack
            ego._last_phi_crank = phi_crank
            ego._last_phi_defense = phi_defense
            ego._last_phi_speed = phi_speed
            ego._last_phi_t = self.t

            # 上一步敌方受到的威胁距离
            ego._last_enm_threat_dist = enm_states["threat"][3]


        # --- 6. 事件奖励计算 (r_event) - 核心稀疏奖励 ---
        if shoot >= 1:
            # r_event -= 20*np.tanh( \
            #     (
            #         (distance/100e3)**2 +\
            #         max(0, 1-time_since_last_shoot/100)**2 +\
            #         (1 - abs(AA_hor)/pi)**2 +\
            #         (abs(delta_psi)/pi)**2 +\
            #         (max(1.0-ego.speed/340, 0)/(target_mach - 0.7))**2 +\
            #         sigmoid(-2*ego.theta/pi*2)
            #     )/6
            # )

            # r_event -= 20 *(
            #     3 * (distance/100e3)**2 +
            #     5 * max(0, 1-time_since_last_shoot/100)**2 +
            #     5 * (1 - abs(AA_hor)/pi) +
            #     5 * (abs(delta_psi)/pi) +
            #     3 * (max(1.0-ego.speed/340, 0)/(target_mach - 0.7)) +
            #     3 * (max(-1 + np.exp(ego.theta/pi*2), -1))
            # ) / 24

            # 从 ego.launch_times 读取上次开火间隔（而非 states["weapon"]，避免能观性问题）
            launch_times = getattr(ego, 'launch_times', [])
            if len(launch_times) <= 1:
                time_since_last_shoot = 120.0
            else:
                time_since_last_shoot = np.clip(self.t - launch_times[-2], 0, 120)

            r_event -= 10 * fire_reward_weight * (1.0 + \
            np.tanh(
                sum(
                    fire_inside_weight * \
                    np.array([
                        1 * 1, # (distance/100e3),
                        3 * (-1 + np.exp(2*np.maximum(0, 1 - (time_since_last_shoot) / 120))),
                        1 * (-1 + np.exp(1-np.abs(AA_hor) / np.pi)),
                        3 * (-1 + np.exp(1*np.abs(delta_psi) / np.pi)), # * (len(alive_ally_missiles)<=1) +
                            # (-1+np.e)*(len(alive_ally_missiles)>1)), # 敢重复开火，砍掉所有瞄准收益
                        2 * 1, # np.exp((max(1.0-ego.speed/340, 0)/(target_mach - 0.6))),
                        3 * max(-1 + np.exp(-2 * ego.theta/pi*2), -50), #  * (len(alive_ally_missiles)<=1) +
                            # 4 * (len(alive_ally_missiles)>1)), # 敢重复开火，砍掉所有高抛收益
                    ])
                )/15 # 10 # 20
            ))

            # if len(alive_ally_missiles) > 1:
            #     r_event -= 5 # 重复开火有额外惩罚


            # r_event -= 5 * shoot
            
            # # if len(alive_ally_missiles) > 1: # 重复开火惩罚
            # #     r_event -= 10 * max(1-time_since_last_shoot/60, 0) # 20
            
            # # # 发射时的态势惩罚/奖励（归类为资源使用的约束，防止乱射）
            # if not ego.dead:
            #     # elif ego.alt <= 5000:
            #     r_event += 5.0 * (abs(AA_hor)/(np.radians(180)) - 1) # 对头射击不额外罚，低空且在绝杀距离外追尾射击重罚

            #     r_event += - 1.0 * abs(delta_psi)/(pi/3) # 瞄准了再打
            #     # r_event += 1.0 * (abs(AA_hor)/pi - 1) # 鼓励对头射击，惩罚追尾射击
            #     r_event += 1.3 * (np.clip(ego.theta/(pi/3), -1, 1) - 1)  # 鼓励抛射 # 1.0 # 高抛项太多了，都忽视速度了
            #     # r_event -= 0.7 * max(1.0-ego.speed/340, 0)  # 开火时候的速度不能太低
            
            # print("状态空间里记录到的间隔时间", ego_states["weapon"])
            # print("新记录方式记录到的间隔时间", time_since_last_shoot)
            # print("在途导弹数量", len(alive_ally_missiles))
            # print()


        # 逃脱导弹，做三九线和置尾机动才算是逃脱而非对手打偏
        if ego.escape_once:
            r_event += 5 * (abs(delta_psi_threat)>pi/2) * (1-ego.dead) # 20 活着才算逃脱，否则只是游戏机制 

        # 导弹脱靶
        if enm.escape_once:
            r_event -= 5 * (1-enm.dead) * fire_reward_weight # 20


        # # 死了也当剩下导弹全被逃脱处理 (自杀代价追加)
        # if wasted > 0:
        #     r_event -= 20 * wasted

        "not done" # 胜负未分，所有偏好的奖励都一样
        r_event1 = r_event
        r_event2 = r_event
        r_event3 = r_event

        if done: # 胜负已分，所有类型各自有结果奖励
            time_left = self.game_time_limit - self.t
            steps_left = time_left / (action_cycle_multiplier * self.dt_maneuver/0.2)
            total_shaping_sum = sum(reward_weights.values())

            if ego_win:
                r_event += 180 * end_reward_weight # 150 + 0.2 * steps_left * total_shaping_sum # 旧 150 新 145
                r_event1 = r_event
                r_event2 = r_event
                r_event3 = r_event
            elif ego_lose:
                r_event -= 180 * end_reward_weight # 125 + steps_left * total_shaping_sum # 旧 100 新 125
                # if self.out_cage(ego) or ego.alt < self.min_alt:
                #     r_event -= 50
                r_event1 = r_event
                r_event2 = r_event
                r_event3 = r_event
            elif ego_draw:
                # # [修改] 不再使用常数-50奖励，而是根据平均态势分来结算
                # if len(self.r_dist_seq) > 0 and len(self.b_dist_seq) > 0:
                #     r_avg_dist = sum(self.r_dist_seq) / len(self.r_dist_seq)
                #     b_avg_dist = sum(self.b_dist_seq) / len(self.b_dist_seq)
                    
                #     if side == 'r':
                #         ego_avg_dist = r_avg_dist
                #         enm_avg_dist = b_avg_dist
                #     else:
                #         ego_avg_dist = b_avg_dist
                #         enm_avg_dist = r_avg_dist
                    
                    # # 赢不了，也要占据中心，并把对手逼到边上，如果赢了或者输了，都禁止加这个奖励
                    # r_event1 += -30 - 20 * (ego_avg_dist-enm_avg_dist)/self.R_cage0
                    # self.middle_hold_score = (ego_avg_dist-enm_avg_dist)/self.R_cage0
                
                if enm.dead: # 平局，对面还死了，那就是双杀了
                    r_event1 = r_event + 0 * end_reward_weight
                    r_event2 = r_event + 180 * end_reward_weight # 双杀当做赢
                    r_event3 = r_event - 180 * end_reward_weight # 双杀当做输
                else:
                    r_event1 = r_event + 0 * end_reward_weight
                    r_event2 = r_event - 180 * end_reward_weight # 双杀策略
                    r_event3 = r_event + 180 * end_reward_weight # 求生者可以把双存活作为胜利
            
            # 打印详细奖励组成，方便调试
            print(f"--- Episode Done ---")
            print(f"Side: {side} | Result: {'Win' if ego_win else 'Lose' if ego_lose else 'Draw'}")
            print(f"R_Event: {r_event:.2f} | R_Constraint: {r_constraint:.2f} | R_Shaping: {r_shaping:.2f}")

        # 返回 done 和三个分项奖励
        return done, \
                    r_event1+r_constraint+r_shaping, \
                        r_event2+r_constraint+r_shaping, \
                            r_event3+r_constraint+r_shaping
