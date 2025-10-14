'''出生点改在外面指定'''
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

# 通过继承构建观测空间、奖励函数和终止条件

class EscapeTrainEnv(Battle):
    # # 任务：敌机一开始就发射导弹，规避的时候我机需要做置尾下高机动尽力逃脱追击
    def escape_obs(self, side):
        full_obs = self.base_obs(side)
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
        return flat_obs
    
    def escape_terminate_and_reward(self, side): # 逃逸策略训练与奖励
        # copy了进攻的，还没改
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        target_alt = alt+state["target_information"][0]
        delta_psi = state["target_information"][1]
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        # alpha = state["target_information"][4]
        alpha = abs(delta_psi) # 实际上是把alpha换掉
        threat_delta_psi, threat_delta_theta, threat_distance =\
            state["threat"]

        RWR = state["warning"]
        obs = self.base_obs(side)
        d_hor = obs["border"][0]

        if side == 'r':
            ego = self.RUAV
            ego_missile = self.Rmissiles[0] if self.Rmissiles else None
            enm = self.BUAV
            alive_own_missiles = self.alive_r_missiles
            alive_enm_missiles = self.alive_b_missiles
        if side == 'b':
            ego = self.BUAV
            ego_missile = self.Bmissiles[0] if self.Bmissiles else None
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_own_missiles = self.alive_b_missiles
        
        '''
        逃逸机动训练
        目标机从不可逃逸区外~40km向本机发射一枚导弹并对本机做纯追踪，
        本机被导弹命中有惩罚，除此之外根据和导弹的ATA和提供密集奖励
        '''
        self.close_range_kill() # 加入近距杀

        # 被命中判为失败
        if ego.got_hit:
            terminate = True
            self.lose = 1

        # 高度出界失败
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.lose = 1

        # 飞出水平边界失败
        if self.out_range(ego):
            terminate = True
            self.lose = 1
        
        # 导弹规避成功
        if self.t > self.game_time_limit \
            and not ego.dead and enm.ammo==0 and \
                len(alive_enm_missiles)==0:
            self.win = 1
            terminate = True
        
        # 密集奖励
        if RWR: # 存在雷达告警时, 规避雷达
            # 水平角度奖励， 奖励置尾机动， 因为状态信息被投影到了另一边，现在变成追踪训练了
            r_angle_h = 1-abs(threat_delta_psi)/pi

            # 垂直角度奖励，导弹相对飞机俯仰角>-30°时越低越好，否则应该水平规避
            sin_theta = state["ego_main"][2]
            if -threat_delta_theta>-pi/6:
                sin_theta_opt = -1*np.clip((alt-self.min_alt_save)/5000, -0.99, 0.99)
            else:
                sin_theta_opt = 0
            r_angle_v = (sin_theta<=sin_theta_opt)*(sin_theta-(-1))/(sin_theta_opt-(-1))+\
                        (sin_theta>sin_theta_opt)*(1-(sin_theta-sin_theta_opt)/(1-sin_theta_opt))
            
            # 距离奖励，和导弹之间的距离
            
        else:
            # 不存在雷达告警时，对敌机做三九
            # 水平角度奖励， 奖励和敌机在同一高度层的 三九机动(×) 置尾机动(√)
            # r_angle_h = abs(abs(threat_delta_psi)-pi/2)/pi*2
            r_angle_h = 1-abs(threat_delta_psi)/pi
            r_angle_v = abs(ego.theta + obs["target_information"][2])/pi
            # r_angle_h = abs(threat_delta_psi)/pi
            # r_angle_v = abs(ego.theta - obs["target_information"][2])/pi
            r_v = 1 - np.abs(speed-0.95*340)/(2*340)
            pre_alt_opt = 5e3*obs["target_information"][0] # 和目标相同高度
            alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)
            r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
                        (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
            # 距离奖励，和目标机之间的距离
            r_dist = -1+np.clip(dist/30e3, -1, 1)

        # 速度奖励
        temp = abs(threat_delta_psi)/pi # 远离度,对头时候最好是0.8Ma，置尾的时候越快越好
        v_opt = (0.8+(2-0.8)*temp)*340
        r_v = 1 - np.abs(speed-v_opt)/(2*340)

        # 高度奖励
        pre_alt_opt = self.min_alt_save + 1e3 # 比最小安全高度高1000m
        alt_opt = np.clip(pre_alt_opt, self.min_alt_save, self.max_alt_save)
        r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt)+\
                    (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
        
        # 距离奖励，和目标机之间的距离
        r_dist = -1+np.clip(dist/30e3, -1, 1)

        # 水平边界奖励
        r_border = d_hor

        # 稀疏奖励
        # 失败惩罚
        if self.lose:
            r_event = -20
        # 取胜奖励
        elif self.win:
            r_event = 20
        else:
            r_event = 0

        reward = np.sum(\
            np.array([1, 0.5, 1, 1, 1, 2, 0.5])*\
            np.array([r_angle_h, r_angle_v, r_v, r_alt, r_event, r_border, r_dist]))

        if terminate:
            self.running = False
        
        return terminate, reward, r_event