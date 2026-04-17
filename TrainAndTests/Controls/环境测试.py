'''
调试飞控训练环境和PID teacher的合并使用
'''

use_tacview = 1

import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
from gym import spaces
import copy
import matplotlib.pyplot as plt
import json
import glob
import argparse

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from _context import *
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize2 import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOHybrid23_0_distil2_one_step_KL import *
from Utilities.FlattenDictObs import flatten_obs2 as flatten_obs
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from TrainAndTests.Controls.UPolicyWrapper import *

class track_env():
    def __init__(self, dt_move=0.02, tacview_show=0, time_limit=3*60):
        super(track_env, self).__init__()
        self.RUAV_ids = None
        self.dt_report = None
        self.dt_move = dt_move
        self.t = None
        # self.done = None
        self.success = None # 胜
        self.fail = None # 负
        self.draw = None # 平
        self.action_space = [spaces.Box(low=-1, high=+1, shape=(4,), dtype=np.float32)]
        self.DEFAULT_RED_BIRTH_STATE = {'position': np.array([-38000.0, 8000.0, 0.0]),
                               'psi': 0
                               }
        
        self.time_limit = time_limit
        self.min_alt = 1e3
        self.min_alt_safe = 3e3

        self.max_alt_safe = 13e3
        self.max_alt = 15e3

        self.flight_key_order = [
            "ego_main",  # 7
            "ego_control",  # 7
            "flight_cmd", # 4
        ]
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()
            self.tacview.handshake()
    
    def reset(self, o00=None, birth_state=None, height_req=8e3, psi_req=0, v_req=340, dt_report=0.2, t0=0):
        self.t = t0
        self.success = 0
        # self.done = 0
        self.fail = 0
        self.draw = 0
        if o00 == None:
            o00 = np.array([118, 30])  # 地理原点的经纬
            self.o00 = o00
        if birth_state == None:
            birth_state = self.DEFAULT_RED_BIRTH_STATE
        self.dt_report = dt_report
        UAV = UAVModel(dt=self.dt_move)
        UAV.ammo = 0
        UAV.id = 1
        UAV.red = True
        UAV.blue = False
        UAV.side = "r"
        UAV.dead = 0
        UAV.color = np.array([1, 0, 0])
        # 红方出生点
        UAV.pos_ = birth_state['position']
        UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
        speed = UAV.speed
        UAV.psi = birth_state['psi']
        UAV.last_psi_v = UAV.psi
        UAV.theta = 0 * pi / 180
        UAV.gamma = 0 * pi / 180
        UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                            sin(UAV.theta),
                                            cos(UAV.theta) * sin(UAV.psi)])
        lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
        UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                    theta0=UAV.theta, o00=o00)
        self.RUAV = UAV
        
        # △h动作输出有效性测试
        self.height_req = height_req
        self.psi_req = psi_req
        self.v_req = v_req


    def get_state(self, side='r'):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        own = self.RUAV

        # 夹角
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        
        # 速度观测量
        v_own = v
        # 本机高度
        h_own = own.alt
        # 本机俯仰角
        sin_theta = sin(own.theta)
        cos_theta = cos(own.theta)
        # 本机滚转角
        sin_phi = sin(own.phi)
        cos_phi = cos(own.phi)

        # 剩余导弹量
        ammo = 0

        p = own.p
        q = own.q
        r = own.r

        theta_v = own.theta_v
        psi_v = own.psi_v
        delta_psi_v = sub_of_radian(self.psi_req, psi_v)  # 水平速度分量和目标航向之间的差角(弧度)

        alpha_air = own.alpha_air
        beta_air = own.beta_air

        delta_psi_control = sub_of_radian(self.psi_req, self.RUAV.psi)
        cos_delta_psi = cos(delta_psi_control)
        sin_delta_psi = sin(delta_psi_control)
        delta_height_control = np.clip((self.height_req-self.RUAV.alt), -5000, 5000)

        one_side_states = {
            "ego_main": np.array([
                float(v_own),  # 0本机速度 m/s
                float(h_own),  # 1本机高度 m
                float(cos_theta),  # 2
                float(sin_theta),  # 3
                float(cos_phi),  # 4
                float(sin_phi),  # 5
                int(ammo)  # 6剩余导弹数量
            ]),

            "ego_control": np.array([
                float(p),  # 0 p rad/s act1_last
                float(q),  # 1 q rad/s act2_last
                float(r),  # 2 r rad/s act3_last
                float(theta_v),  # 3
                float(delta_psi_v),  # 4
                float(alpha_air),  # 5 rad
                float(beta_air)  # 6 rad
            ]),

            "flight_cmd":  np.array([
                cos_delta_psi,
                sin_delta_psi,
                delta_height_control,
                self.v_req - self.RUAV.speed,
            ])
        }
        return one_side_states
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340

        s["flight_cmd"][2] /= 5000
        s["flight_cmd"][3] /= 340
        return s

    def unscale_state(self, obs_input):
        # 使用 deepcopy 避免修改传入对象
        o = copy.deepcopy(obs_input)
        o["ego_main"][0] *= 340
        o["ego_main"][1] *= 5e3
        o["ego_control"][0] *= (2 * pi)  # (2 * pi) 5000
        o["ego_control"][1] *= (2 * pi)  # (2 * pi) pi
        o["ego_control"][2] *= (2 * pi)  # (2 * pi) 340

        o["flight_cmd"][2] *= 5000
        o["flight_cmd"][3] *= 340
        return o
    
    def obs2obs_check(self, obs):
        """
        将扁平化的 obs (numpy array) 还原为 check_obs (dict)。
        该 check_obs 处于 scale 后的状态 (即可以直接输入 unscale_state)。
        
        Args:
            obs: (Dim, ) 或 (Batch, Dim) 的 numpy 数组或 tensor
        
        Returns:
            check_obs: 字典形式的状态
        """
        # 1. 确保 obs 是 numpy 数组
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().detach().numpy()
            
        key_dims = {
            "ego_main": 7,        # bool -> 1
            "ego_control": 7,   # int -> 1
            "flight_cmd": 4,       # bool -> 1
        }
        
        is_batch = (obs.ndim > 1)
        
        obs_dict = {}
        ptr = 0
        
        # 4. 按顺序切分重构
        for key in key_dims:
            if key not in key_dims:
                # 理论上 key_order_1v1 里的 key 都应该有定义，防御性编程
                continue
                
            dim = key_dims[key]
            
            # 切片
            if is_batch:
                val = obs[:, ptr : ptr + dim]
            else:
                val = obs[ptr : ptr + dim]
            
            ptr += dim
            
            if dim == 1:
                if not is_batch:
                    val = val[0] # 还原为标量
                # Batch 模式下通常保留 (B, 1) 维度以便后续处理
            
            obs_dict[key] = val

        # 注意：ego_control 不在 key_order_1v1 中，因此无法恢复。
        # unscale_state 函数中有 `if "ego_control" in s` 的检查，所以这是安全的。
            
        return obs_dict

    def base_obs(self, side='r', pomdp=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        state = self.get_state(side)  # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
        self.state_init["flight_cmd"] = np.array([0, 0, 0, 0])

        observation = self.scale_state(state)
        self.obs_init = self.scale_state(self.state_init)
        return observation

    def get_obs(self, side='r'):
        pre_full_obs = self.base_obs()
        full_obs = {k: (pre_full_obs[k].copy() if hasattr(pre_full_obs[k], "copy") else pre_full_obs[k]) \
                    for k in self.flight_key_order}
        
        # 弹药量不能被看到
        full_obs["ego_main"][6] = 0
        # full_obs["ego_control"][4] = 0

        # 将观测按顺序拉成一维数组
        flat_obs = flatten_obs(full_obs, self.flight_key_order)
        return flat_obs, full_obs

    def step(self, action):
        self.action = action
        # action['cont'] 由 PID 输出，顺序为 [aileron, elevator, rudder, throttle]
        aileron, elevator, rudder, throttle = action['cont']
        self.t += self.dt_report
        self.t = round(self.t, 2) # 保留两位小数
        time_rate = int(round(self.dt_report/self.dt_move))
        self.RUAV.last_psi_v = self.RUAV.psi_v
        for _ in range(time_rate):
            # UAVModel.move(p2p=True) 期望第一个参数对应 elevator, 第二个参数对应 aileron, 
            # 第四个参数对应 throttle, rudder 参数单独传递
            self.RUAV.move(target_height=elevator, delta_heading=aileron, target_speed=throttle, \
                relevant_height=True, relevant_speed=False, with_theta_req=False, p2p=True, rudder=rudder)
            done = self.get_done()
            if done:
                break
        next_obs, _ = self.get_obs()
        reward = self.get_reward()
        
        return next_obs, reward, done

    def get_done(self,):
        done = 0
        # 超时结束
        if self.t > self.time_limit:
            done = 1
        ruav_state = self.get_state()
        alt = ruav_state["ego_main"][1]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        # 失败条件：失速、高度过低
        self.fail = 0
        if alt < self.min_alt or alpha_air < -20 or alpha_air > 45 or abs(beta_air) > 15:
            self.fail = 1
            done = 1
            self.RUAV.dead = 1
        return done


    def get_reward(self, ):
        ruav_state = self.get_state()
        speed = ruav_state["ego_main"][0]
        alt = ruav_state["ego_main"][1]
        sin_theta = ruav_state["ego_main"][3]
        cos_theta = ruav_state["ego_main"][2]
        theta = np.arctan2(sin_theta, cos_theta)
        sin_phi = ruav_state["ego_main"][5]
        cos_phi = ruav_state["ego_main"][4]
        phi = atan2(sin_phi, cos_phi)
        p = ruav_state["ego_control"][0]
        q = ruav_state["ego_control"][1]
        r = ruav_state["ego_control"][2]
        theta_v = ruav_state["ego_control"][3]
        delta_psi_v = ruav_state["ego_control"][4]
        alpha_air = ruav_state["ego_control"][5]*180/pi
        beta_air = ruav_state["ego_control"][6]*180/pi
        cos_delta_psi = ruav_state["flight_cmd"][0]
        sin_delta_psi = ruav_state["flight_cmd"][1]
        height2req = ruav_state["flight_cmd"][2]
        speed2req = ruav_state["flight_cmd"][3]
        climb_rate = self.RUAV.vu

        delta_psi = np.arctan2(sin_delta_psi, cos_delta_psi)

        self.get_done()

        # 存活奖励
        reward_alive = 0.01 # 10

        # 失败惩罚
        reward_end = 0
        if self.fail:
            reward_end -= 400

        # 误差计算
        psi2req = delta_psi_v

        # 和奖励无关，方便画图
        self.theta_req = height2req/5000*pi/2
        
        # L_ = np.array([cos(self.theta_req)*cos(self.psi_req), sin(self.theta_req), cos(self.theta_req)*sin(self.psi_req)])
        # ATA = np.arccos(np.dot(L_, self.RUAV.point_) / (1*1 + 0.0001))  # 防止计算误差导致分子>分母
        # r_angle = 1 - ATA / (pi / 3)  # 超出雷达范围就惩罚狠一点

        # 高度误差惩罚
        r_alt = -abs(height2req)/5000
        # r_alt += np.clip(self.RUAV.vu / 100, -1, 1) * height2req * np.sign(height2req)
                
        # 高度限制奖励/惩罚
        r_alt += (alt <= self.min_alt_safe) * np.clip(self.RUAV.vu / 100, -1, 1) + \
                (alt >= self.max_alt_safe) * np.clip(-self.RUAV.vu / 100, -1, 1)

        # 航向误差惩罚
        r_angle = 1
        r_angle += -abs(delta_psi)/pi  # 航向的水平误差psi2req，或者头部的水平误差 delta_psi

        # 俯仰角惩罚
        r_angle += -0.05 * abs(np.arcsin(sin_theta))
        # 滚转角惩罚
        r_angle += -0.05 * abs(phi)
        # 滚转角速度惩罚
        r_angle += -0.01 * abs(p)

        # 速度误差惩罚
        r_speed = -abs(speed2req) / 340

        # 迎角过载惩罚(惩罚负迎角和过大的正迎角)
        reward_alpha = 0.5
        if alpha_air >= 15:
            reward_alpha -= alpha_air/15
        if alpha_air < 0:
            reward_alpha += alpha_air/2       
        ny = self.RUAV.Ny
        if ny<=-1 or ny > 9:
            reward_alpha -= 2
            
        # 侧滑角惩罚（尽量少侧滑）
        reward_beta = - abs(beta_air/5)

        reward = np.sum([
            1 * reward_alive,
            1 * reward_end,
            1 * r_angle,
            1 * r_alt,
            1 * r_speed,
            1 * reward_alpha,
            1 * reward_beta,
        ])

        # 其他奖励待续
        return reward
        

    def render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            loc_LLH = self.RUAV.lon, self.RUAV.lat, self.RUAV.alt
            if not self.RUAV.dead:
                pilot = 'Dragon'
                color = 'Red'
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                            f"{self.RUAV.phi * 180 / pi:.6f}|{self.RUAV.theta * 180 / pi:.6f}|{self.RUAV.psi * 180 / pi:.6f},"
                            f"Name=F16,Pilot={pilot},Color={color}\n"
                        )
                # 绘制目标
                delta_N = 5e3*cos(self.theta_req)*cos(self.psi_req)
                delta_U = 5e3*sin(self.theta_req)
                delta_E = 5e3*cos(self.theta_req)*sin(self.psi_req)
                N, U, E = LLH2NUE(loc_LLH[0], loc_LLH[1], loc_LLH[2], lon_o=self.o00[0], lat_o=self.o00[1])
                delta_H = self.height_req
                lon_T, lat_T, _ = NUE2LLH(N+delta_N,U+delta_U,E+delta_E,lon_o=self.o00[0], lat_o=self.o00[1])
                
                data_to_send += (
                            f"#{send_t:.2f}\n"
                            f"{self.RUAV.id+1},T={(lon_T):.6f}|{(lat_T):.6f}|{delta_H:.6f},"
                            f"Name=DragonBall,Color=Blue\n"
                        )

            else:
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
                data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"

            # loc_r = [self.RUAV.lon, self.RUAV.lat, self.RUAV.alt]
            # if self.tacview_show:
            #     data_to_send = ''
            #     data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            #             float(self.t + t_bias), self.RUAV.id, loc_r[0], loc_r[1], loc_r[2], self.RUAV.phi * 180 / pi, self.RUAV.theta * 180 / pi,
            #             self.RUAV.psi * 180 / pi)

            self.tacview.send_data_to_client(data_to_send)

    def clear_render(self, t_bias=0):
        if self.tacview_show:
            send_t = self.t + t_bias
            data_to_send = ''
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id}\n"
            data_to_send += f"#{send_t:.2f}\n-{self.RUAV.id+1}\n"
            self.tacview.send_data_to_client(data_to_send)


# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
max_steps = 10 * 65e4
hidden_dim = [128, 128]  # 128, 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
dt_decide = 0.2 # 0.2 原先的0.2会不好飞，现在缩小dt_decide会不好训
pre_train_rate = 0 # 0.25 # 0.25

state_dim = 7+7+4  # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
action_dim = 4 # test
# action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
action_bound = np.array([[-1.1,1.1],[-1.1,1.1],[-1.1,1.1],[-0.2,1.2]])  # aileron, elevator, rudder, throttle
mission_name = 'FlightControl'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
log_dir = os.path.join(project_root, "./logs/control", mission_name + "-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

if __name__=='__main__':
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=8*60, help="maximum episode time length")
    args = parser.parse_args()
    
    env = track_env(tacview_show=use_tacview, time_limit=args.max_episode_len)

    # 创建一个 dummy env 获取维度
    dummy_env = track_env(tacview_show=0, time_limit=args.max_episode_len)

    teacher_agent = UnifiedPolicyWrapper(dummy_env)

    from Visualize.tensorboard_visualize import TensorBoardLogger

    out_range_count = 0
    return_list = []
    steps_count = 0

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    try:
        t_bias = 0
        # 强化学习训练
        rl_steps = 0
        i_episode = 0
        while i_episode==0: # debug rl_steps < max_steps:
            i_episode += 1
            episode_return = 0
            # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
            
            init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

            birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)
                                }
            
            height_req = init_height-2000  # np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
            psi_req = birth_state['psi']+-180*pi/180  # np.random.uniform(-pi, pi) * np.clip(i_episode/1000, 0, 1)
            v_req = 200 # np.random.uniform(0.8, 2.5)*340

            env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)

            obs, obs_check = env.get_obs()
            done = False

            psi_req_dot = 0
            height_req_dot = 0

            while not done:  # 每个训练回合
                # # 舞龙
                # if round(env.t, 3) % 10 == 0:
                #     env.psi_req += np.random.uniform(-1, 1) *60*pi/180
                #     env.height_req += np.random.uniform(-1, 1) * 2000
                #     env.height_req = np.clip(env.height_req, 4000, 13000)
                if 1: # round(env.t, 3) % 10 == 0:
                    psi_req_dot += np.random.uniform(-1, 1) *0.1*pi/180
                    psi_req_dot = np.clip(psi_req_dot, -8*pi/180, 8*pi/180)
                    env.psi_req += psi_req_dot * dt_decide
                    height_req_dot += np.random.uniform(-1, 1) * 20
                    height_req_dot = np.clip(height_req_dot, -100, 100)
                    env.height_req += height_req_dot * dt_decide
                    env.height_req = np.clip(env.height_req, 4000, 12000)

                # 1.执行动作得到环境反馈
                obs, obs_check = env.get_obs()
                # action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
                action = teacher_agent.get_action(obs)
                rl_steps += 1

                if round(env.t, 3) % 0.5 == 0:
                    print("----")
                    print("delta_psi", np.arctan2(obs_check["flight_cmd"][1], obs_check["flight_cmd"][0]) * 180 / pi)
                    temp_state = env.unscale_state(obs_check)
                    print("delta_height", temp_state["flight_cmd"][2])
                    print("delta_speed", temp_state["flight_cmd"][3])
                    print("--")
                    print("aileron", action['cont'][0])
                    print("elevator", action['cont'][1])
                    print("rudder", action['cont'][2])
                    print("throttle", action['cont'][3])
                    print('--')
                    print("obs_check", obs_check)
                    print("----")
                    print(f"Episode {i_episode}, Step {rl_steps}, time: {env.t}")

                
                next_state, reward, done = env.step(action)

                # debug 用
                height_req_show = env.height_req/1000
                height_show = env.RUAV.alt/1000
                psi_req_show = env.psi_req*180/pi
                psi_show = env.RUAV.psi*180/pi
                v_req_show = env.v_req
                v_show = env.RUAV.speed

                # transition_dict['states'].append(obs)
                # transition_dict['actions'].append(u)
                # transition_dict['next_states'].append(next_state)
                # transition_dict['rewards'].append(reward)
                # transition_dict['dones'].append(done)
                # transition_dict['action_bounds'].append(action_bound)
                state = next_state
                episode_return += reward
                env.render(t_bias)

            env.clear_render(t_bias)
            t_bias += env.t

            if env.fail==1:
                out_range_count+=1
           

    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        pass

