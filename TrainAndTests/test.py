import sys
import os
import numpy as np
import torch as th
from math import *
from gym import spaces
import copy

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
sys.path.append(os.path.dirname(current_dir))
from Envs.UAVmodel6d import UAVModel
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE

class height_track_env():
    def __init__(self, dt_move=0.02, o00=None):
        super(height_track_env, self).__init__()
        self.UAV_ids = None
        self.dt_report = None
        self.dt_move = dt_move
        self.t = None
        self.done = None
        self.success = None # 胜
        self.fail = None # 负
        self.draw = None # 平
        self.action_space = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, 0.0]),
                               'psi': 0
                               }
        if o00 == None:
            o00 = np.array([118, 30])  # 地理原点的经纬
        self.o00=o00
        # 高于升限会导致动作无法实施，影响
        self.time_limit = 180
        self.min_alt = 1e3
        self.min_alt_save = 3e3
        self.max_alt_save = 14e3
        self.max_alt = 15e3

        # △h动作输出有效性测试
        self.height_req = None
        
    
    def reset(self, o00=None, birth_state=None, height_req=8e3, dt_report = 0.2, t0=0):
        self.t = t0
        self.success = 0
        self.done = 0
        self.fail = 0
        self.draw = 0

        if birth_state == None:
            birth_state = self.DEFAULT_RED_BIRTH_STATE
        self.dt_report = dt_report
        UAV = UAVModel(dt=self.dt_move)
        UAV.ammo = 0
        UAV.id = 1
        UAV.red = True
        UAV.blue = False
        UAV.label = "red"
        UAV.color = np.array([1, 0, 0])
        # 红方出生点
        UAV.pos_ = DEFAULT_RED_BIRTH_STATE['position']
        UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
        speed = UAV.speed
        UAV.psi = DEFAULT_RED_BIRTH_STATE['psi']
        UAV.theta = 0 * pi / 180
        UAV.gamma = 0 * pi / 180
        UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                            sin(UAV.theta),
                                            cos(UAV.theta) * sin(UAV.psi)])
        lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
        UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                    theta0=UAV.theta, o00=o00)
        self.UAV = UAV
        
        # △h动作输出有效性测试
        self.height_req = height_req


    def get_obs(self):
        pass
        '''
        0 h abs /5e3 m
        1 h_dot /340 m/s
        2 sin θ_v
        3 cos θ_v
        4 sin φ
        5 cos φ
        6 v /340 m/s
        '''
        obs = np.zeros(4)
        obs[0] = UAV.alt / 5e3
        obs[1] = UAV.climb_rate /340
        v_hor = abs(UAV.vel_[0]**2+UAV.vel_[2]**2)
        theta_v = np.arctan2(UAV.vel_[1], v_hor)
        obs[2] = sin(theta_v)
        obs[3] = cos(theta_v)
        obs[4] = sin(UAV.phi)
        obs[5] = cos(UAV.phi)
        obs[6] = UAV.speed /340
        
        return obs


    def get_obs_spaces(self):
        self.reset()
        obs = self.get_obs()
        self.obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs1.shape, dtype=np.float32) for obs1 in obs]
        return self.obs_spaces

    def step(self):
        pass
        # 
        current_t = i*dt_move
        self.UAV.move(target_height, delta_heading, target_speed)
        


    def get_done(self):
        done = 0
        # 高度追踪失败条件：跑出h_min~h_max的范围立即失败
        h_current = UAV.alt
        if h_current<self.min_alt or h_current>self.max_alt:
            done = 1
            self.fail = 1
            return done
        
        # 高度保持成功条件：到时间结束为止没有超出距离
        if self.t>=self.time_limit:
            done = 1
            return done
        
        # 立即成功条件(暂时不做）：距离h_req小于100m，且爬升率绝对值小于10m/s
        pass
        return done


    def get_reward(self, ):
        # 高度奖励
        h_current = UAV.alt
        h_req = self.height_req
        r_h_norm = (h_current<=h_req)*(h_current-self.min_alt)/(h_req-self.min_alt)+\
                    (h_current>h_req)*(1-(h_current-h_req)/(self.max_alt-h_req))
        r_h_norm = 1 * r_h_norm
        # 高度出界惩罚
        if self.fail:
            r_h_norm -= 10
        if self.success:
            r_h_norm += 3
        
        # 其他奖励待续
        return r_h_norm
        

    def reder(self, tacview_show=0):
        pass



o00 = np.array([118, 30])  # 地理原点的经纬
DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, 0.0]),
                               'psi': 0
                               }
dt_move=0.02
UAV = UAVModel(dt=dt_move)
UAV.ammo = 0
UAV.id = 1
UAV.red = True
UAV.blue = False
UAV.label = "red"
UAV.color = np.array([1, 0, 0])
# 红方出生点
UAV.pos_ = DEFAULT_RED_BIRTH_STATE['position']
UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
speed = UAV.speed
UAV.psi = DEFAULT_RED_BIRTH_STATE['psi']
UAV.theta = 0 * pi / 180
UAV.gamma = 0 * pi / 180
UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                    sin(UAV.theta),
                                    cos(UAV.theta) * sin(UAV.psi)])
lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
            theta0=UAV.theta, o00=o00)

action = [0,0,0]

target_height = 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
delta_heading = action[1]  # 相对方位(弧度)
target_speed = 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
# print('target_height',target_height)
# for i in range(int(self.dt // dt_move)):
t_last = 60

tacview_show = 1

if tacview_show:
    from Visualize.tacview_visualize import *
    tacview = Tacview()

for i in range(int(t_last//dt_move)):
    current_t = i*dt_move
    UAV.move(target_height, delta_heading, target_speed)
    loc_r = [UAV.lon, UAV.lat, UAV.alt]
    if tacview_show:
        data_to_send = ''
        data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
                float(current_t), UAV.id, loc_r[0], loc_r[1], loc_r[2], UAV.phi * 180 / pi, UAV.theta * 180 / pi,
                UAV.psi * 180 / pi)
    if tacview_show:
            tacview.send_data_to_client(data_to_send)

