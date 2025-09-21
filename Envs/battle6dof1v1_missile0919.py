'''出生点改在外面指定'''

from random import random
import random
from gym import spaces
import copy
import jsbsim
import sys
import os
import importlib

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
sys.path.append(os.path.dirname(current_dir))

from Envs.MissileModel0910 import *  # test
from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
from Math_calculates.sub_of_angles import *
from Math_calculates.coord_rotations import *
from Math_calculates.SimpleAeroDynamics import *
from Envs.UAVmodel6d import UAVModel
from Visualize.tacview_visualize import *

g = 9.81
dt_maneu_dec = 0.5  # 0.02 0.8 0.2
dt_move = 0.02
report_move_time_rate = int(round(dt_maneu_dec / dt_move))

o00 = np.array([118, 30])  # 地理原点的经纬
# t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180

# battle_field = np.array([10000, 1e4, 10000])  # 东西、南北方向各10km，高度范围1000m,以0,0,500为中心

d1 = 200e3
d2 = d1  # 100e3

min_height = 0
max_height = 15e3

R_cage = d1
R_birth = 40e3

horizontal_center = np.array([0, 0])

min_north = -d1 / 2
max_north = d1 / 2

min_east = -d2 / 2
max_east = d2 / 2

battle_field = np.array([[min_north, max_north], [min_height, max_height], [min_east, max_east]])
birthpointr = np.array([min_north + 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])
birthpointb = np.array([max_north - 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])

# random_array = np.clip(np.random.randn(3),-1,1)
# "笼子"的范围：x、y方向+-各5000m, z方向0~1000m

def out_range(UAV):
    position = UAV.pos_
    pos_h = np.array([position[0], position[2]])
    R_uav = norm(pos_h - horizontal_center)
    out = True
    if min_height <= position[1] <= max_height:
        if R_uav <= R_cage:
            out = False
    # if min_north <= position[0] <= max_north and \
    #         min_height <= position[1] <= max_height and \
    #         min_east <= position[2] <= max_east:
    #     out = False
    return out


class Battle(object):
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([-R_birth * cos(0), 9000.0, -R_birth * sin(0)]),
                               'psi': 0
                               }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([-R_birth * cos(pi), 8000.0, -R_birth * sin(pi)]),
                                'psi': pi
                                }

    def __init__(self, args):
        super(Battle, self).__init__()
        self.BUAV = None
        self.RUAV = None
        self.dt_maneu_dec = dt_maneu_dec
        self.dt_move = dt_move
        self.UAV_ids = None
        self.UAV_hit = None
        self.Bmissiles = None
        self.Rmissiles = None
        self.missiles = None
        self.args = args
        self.RUAVs = None
        self.BUAVs = None
        self.UAVs = None
        self.RUAVsTable = None
        self.BUAVsTable = None
        self.UAVsTable = None
        self.RmissilesTable = None
        self.BmissilesTable = None
        self.missilesTable = None
        self.t = None
        self.game_time_limit = self.args.max_episode_len  # None
        self.running = None
        self.action_space = []
        # self.reset()  # 重置位置和状态
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.train_side_win = None
        self.train_side_loss = None
        self.train_side_draw = None
        self.max_alt = 15e3
        self.max_alt_save = 13e3
        self.min_alt_save= 3e3
        self.min_alt = 1e3
        self.tacview_show = None
        # # 智能体的观察空间
        # r_obs_n, b_obs_n = self.get_obs()
        # self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      r_obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]
        

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6, tacview_show=0):  # 重置位置和状态
        self.tacview_show = tacview_show
        if tacview_show:
            self.tacview = Tacview()

        if red_birth_state is None:
            red_birth_state = self.DEFAULT_RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.DEFAULT_BLUE_BIRTH_STATE

        self.Rmissiles = []
        self.Bmissiles = []
        self.missiles = []
        self.dt_maneu_dec = dt_maneu_dec  # simulation interval，1 second
        self.t = 0
        # self.game_time_limit = self.args.max_episode_len
        # 初始化无人机
        self.RUAVs = []
        self.BUAVs = []
        self.Rnum = 1
        self.Bnum = 1
        self.RUAVsTable = {}
        self.BUAVsTable = {}
        self.UAVsTable = {}
        self.RmissilesTable = {}
        self.BmissilesTable = {}
        self.missilesTable = {}
        # 红方初始化
        for i in range(self.Rnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = red_init_ammo
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.label = "Red"
            UAV.color = np.array([1, 0, 0])
            # 红方出生点
            # UAV.pos_ = birthpointr.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = red_birth_state['position']  # np.array([-38841.96119795, 9290.02131746, -1686.95469864])
            UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
            speed = UAV.speed
            mach, _ = calc_mach(speed, UAV.pos_[1])
            UAV.mach = mach
            UAV.psi = red_birth_state[
                'psi']  # (0 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            self.RUAVs.append(UAV)
            self.RUAVsTable[UAV.id]=(UAV, UAV.label, UAV.dead)
        # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel(dt=dt_move)
            UAV.ammo = blue_init_ammo
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.label = "Blue"
            UAV.color = np.array([0, 0, 1])
            # 蓝方出生点
            # UAV.pos_ = birthpointb.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = blue_birth_state['position']  # np.array([38005.14540582, 6373.80721704, -1734.42509136])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
            UAV.psi = blue_birth_state['psi']
            # (180 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.psi = sub_of_radian(UAV.psi, 0)
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            lon_uav, lat_uav, h_uav = NUE2LLH(UAV.pos_[0], UAV.pos_[1], UAV.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
            UAV.reset(lon0=lon_uav, lat0=lat_uav, h0=h_uav, v0=UAV.speed, psi0=UAV.psi, phi0=UAV.gamma,
                      theta0=UAV.theta, o00=o00)
            self.BUAVs.append(UAV)
            self.BUAVsTable[UAV.id]=(UAV, UAV.label, UAV.dead)
        self.running = True
        self.UAVs = self.RUAVs + self.BUAVs
        self.UAVsTable = {**self.RUAVsTable, **self.BUAVsTable}
        self.UAV_ids = [UAV.id for UAV in self.UAVs]
        self.UAV_hit = [False for _ in range(len(self.UAVs))]

        # todo 1v1的残留
        self.RUAV = self.RUAVs[0]
        self.BUAV = self.BUAVs[0]

        print(red_birth_state)
        print(blue_birth_state)

    def get_obs_spaces(self, side):
        self.reset()
        # 智能体的观察空间
        # r_obs_n, b_obs_n = self.get_obs()
        obs_n = self.get_obs(side)
        # b_obs_n = self.get_obs('b')
        self.obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]
        return self.obs_spaces

    def step(self, r_actions, b_actions):
        # 输入动作（范围为[-1,1]
        self.t += dt_maneu_dec
        self.t = round(self.t, 2)  # 保留两位小数

        actions = r_actions + b_actions

        # 导弹发射不在这里执行，这里只处理运动解算，且发射在step之前
        # 运动按照dt_move更新，结果合并到dt_maneu_dec中

        for j1 in range(int(report_move_time_rate)):
            # 飞机移动
            for UAV, action in zip(self.UAVs, actions):
                if UAV.dead:
                    continue
                # 输入动作与动力运动学状态
                action = np.clip(action, -1, 1)
                # print(action)
                target_height = 3000 + (action[0] + 1) / 2 * (10000 - 3000)  # 高度使用绝对数值
                delta_heading = action[1]  # 相对方位(弧度)
                target_speed = 170 + (action[2] + 1) / 2 * (544 - 170)  # 速度使用绝对数值
                # print('target_height',target_height)
                # for i in range(int(self.dt_maneu_dec // dt_move)):
                UAV.move(target_height, delta_heading, target_speed)
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
                        print(target.label, 'is hit')
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
                if out_range(UAV):
                    UAV.dead = True
                    # self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or out_range(UAV):
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

    def get_state(self, side):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        更正，UAVmodel具有一部分判断目标可见性的功能，需要管理

        输入给1v1智能体的状态空间包含以下结构：
        0  目标可见性标志 bool
        1  目标相对高度/5e3， 如果看不到，在get_obs置为0
        2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        4  目标相对距离 / 10e3, 如果看不到，在get_obs置为最大
        5  夹角 (rad)
        6  本机速度 /340
        7  本机高度 /5e3
        8  sinθ
        9  cosθ
        10 sinφ
        11 cosφ
        12 目标可跟踪标志 bool
        13 导弹中制导状态 bool
        14 导弹预计碰撞时间 / 30s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        15 目标雷达跟踪标志 bool
        16 告警标志 bool
        17 最近来袭导弹相对方位角(rad), 如果没有导弹，在get_obs置为pi
        18 最近来袭导弹相对俯仰角(rad), 如果没有导弹，在get_obs置为0
        19 最近来袭导弹距离模糊值(0:<8km, 1: <20km, 2:>20km), 如果没有导弹，在get_obs置为最大
        
        （废弃不用）    目标相对方位角速度 (rad/s) / 0.35
        （废弃不用）    目标相对俯仰角速度 (rad/s) / 0.35
        '''
        side = self.UAV_ids
        
        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
            own_missiles = self.Rmissiles
            enm_missiles = self.Bmissiles
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
            own_missiles = self.Bmissiles
            enm_missiles = self.Rmissiles
        # 目标可见性标志 bool
        target_observable = 1
        pass # 目标可见性判断逻辑
        # 目标相对高度
        delta_alt_obs = (adv.alt-own.alt)/5e3
        # 目标相对方位角
        L_ = adv.pos_ - own.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        delta_psi = sub_of_radian(q_beta, own.psi)
        # 目标相对俯仰角
        delta_theta = sub_of_radian(q_epsilon, own.theta)
        # 目标相对距离
        dist = norm(L_)
        dist_obs = dist/10e3
        # 夹角
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        v = norm(v_)
        alpha = np.arccos(np.dot(L_,v_)/(v*dist))
        # 速度观测量
        v_obs = v/340
        # 本机高度
        h_abs_obs = own.alt / 5000
        # 本机俯仰角
        sin_theta = sin(own.theta)
        cos_theta = cos(own.theta)
        # 本机滚转角
        sin_phi = sin(own.phi)
        cos_phi = cos(own.phi)
        # 雷达可跟踪标志
        target_tracked = 1
        if not target_observable:
            target_tracked = 0
        pass # todo
        # 目标相对方位角速度 (rad/s) / 0.35 与 目标相对俯仰角速度 (rad/s) / 0.35
        # vT_ = adv.vel_
        # vr_ = vT_ - v_
        # vr_radial = np.dot(vr_, L_) / dist  # 径向速度
        # vr_tangent_ = np.cross(L_, vr_) / dist # 目标周向速度矢量
        # omega_ = vr_tangent_/dist # 目标相对角速度矢量
        # down_ = np.array([0,-1,0])
        # L_left_ = np.cross(L_, down_)/dist
        # L_left_ = L_left_/norm(L_left_)
        # omega_vert_ = np.dot(omega_, down_)*1
        # omega_hor_ = omega_-omega_vert_
        # delta_psi_dot = np.dot(omega_vert_, down_) # 目标相对方位角速度
        # delta_theta_dot = -np.dot(omega_hor_, L_left_) # 目标相对俯仰角速度

        # 导弹中制导状态 bool
        missile_in_mid_term = 0
        pass # todo 需要由导弹的类实例报告

        # 导弹预计碰撞时间 / 30s, 如果没有在飞行导弹，在get_obs中置为4(120s)
        missile_time_to_hit_obs = 4
        pass # todo 需要由导弹的类实例报告

        # 目标雷达跟踪标志 bool
        locked_by_target = 0
        pass # 和雷达跟踪目标的逻辑一起做

        # 告警标志 bool
        warning = 0
        pass # 取自目标发射导弹的雷达状态

        # 最近来袭导弹判断
        for missile in enm_missiles:
            pass

        # 最近来袭导弹相对方位角(rad), 如果没有导弹，在get_obs置为pi
        threat_delta_psi = pi
        pass

        # 最近来袭导弹相对俯仰角(rad), 如果没有导弹，在get_obs置为0
        threat_delta_theta = 0
        pass

        # 最近来袭导弹距离模糊值(0:<8km, 1: <20km, 2:>20km), 如果没有导弹，在get_obs置为最大
        threat_distance = 2
        pass
        
        one_side_obs = np.array([
            target_observable, 
            delta_alt_obs, 
            delta_psi, 
            delta_theta,
            dist_obs,
            alpha,
            v_obs,
            h_abs_obs,
            sin_theta,
            cos_theta,
            sin_phi,
            cos_phi,
            target_tracked,
            missile_in_mid_term,
            missile_time_to_hit_obs,
            locked_by_target,
            warning,
            threat_delta_psi,
            threat_delta_theta,
            threat_distance,
        ])

        return one_side_obs

    def get_obs(self, side):
        return self.get_state(side) # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

    # 进攻策略观测量
    def attack_obs(self, side):
        # 0  目标可见性标志 bool
        # 1  目标相对高度/5e3， 如果看不到，在get_obs置为0
        # 2  目标相对方位 (rad), 如果看不到，在get_obs置为pi
        # 3  目标相对俯仰角 (rad), 如果看不到，在get_obs置为0
        # 4  目标相对距离 / 10e3, 如果看不到，在get_obs置为最大
        # 5  夹角 (rad)
        # 6  本机速度 /340
        # 7  本机高度 /5e3
        # 8  sinθ
        # 9  cosθ
        # 10 sinφ
        # 11 cosφ
        # 12 目标可跟踪标志 bool
        mask = np.ones(20)
        mask[12:] = 0
        return self.get_state(side)*mask
        
    # crank策略观测量

    # 规避策略观测量

    



    def get_reward(self, missiled_combat='Flase'):
        if missiled_combat == True:
            # 添加导弹命中相关的奖励和惩罚
            pass
        '结果奖励部分'
        RUAV = self.RUAV
        BUAV = self.BUAV
        UAVs = [RUAV, BUAV]
        A = [0, 0]  # R, B
        rewards = [0, 0]  # R, B

        for i, UAV in enumerate(UAVs):  # UAVs[0]为红方，UAVs[1]为蓝方
            # adv = UAVs[1 - i]
            # # # 出界剩余时间惩罚：
            # t_last_max = 60
            # t_last = np.ones(6)
            # # 出界惩罚
            # if out_range(UAV):
            #     rewards[i] -= 100
            # # 超出限高惩罚
            # if UAV.pos_[1] >= max_height:
            #     rewards[i] -= 80
            # # 对手出界
            # if out_range(adv):
            #     rewards[i] += 10
            # # 被命中(不管是导弹还是"扫描枪")
            # if UAV.got_hit:
            #     rewards[i] -= 100
            # # 命中对手
            # if adv.got_hit:
            #     rewards[i] += 200  # 增加命中奖励
            # # 撞机
            # if UAV.crash:
            #     rewards[i] -= 70
            # # 平局
            # if self.t >= self.game_time_limit and not any([UAV.dead, adv.dead]):
            #     rewards[i] -= 10

            # test 目标追赶
            r_obs_n = self.get_obs('r')
            b_obs_n = self.get_obs('b')
            # r_obs_n, b_obs_n = self.get_obs()
            # rewards[0] = (1 - np.linalg.norm(r_obs_n[7] / pi * 2)) * 100
            rewards[0] = (1 - np.linalg.norm((r_obs_n[7] - RUAV.theta) / pi * 2 * 2)) * 100  # test
            rewards[1] = (1 - np.linalg.norm((b_obs_n[7] - BUAV.theta) / pi * 2 * 2)) * 100  # test
            # rewards[1] = (1-np.linalg.norm(BUAV.theta/pi*2)) * 100
            rewards[0] += (1 - np.linalg.norm(r_obs_n[8] / pi)) * 100
            rewards[1] += (1 - np.linalg.norm(b_obs_n[8] / pi)) * 100

            # 稀疏奖励

        # return rewards[0], rewards[1]
        # todo 奖励改成元组形式，第一项喂给经验池，第二项用作episode_return
        return (rewards[0], rewards[0]), (rewards[1], rewards[1])

    def get_target_by_id(self, target_id):
        for uav in self.UAVs:
            if uav.id == target_id:
                return uav
        return None

    def get_terminate(self):
        if all(self.UAV_hit):
            return True
        missile_dead_list = []
        uav_dead_list = []
        # battle和uav各自所属的missile没有同步，判断起来不方便，现改为所有发射的导弹都挂了,且无人机有一方坠落，
        # 则仿真就结束
        for missile in self.missiles:
            missile_dead_list.append(missile.dead)
        for uav in self.UAVs:
            uav_dead_list.append(uav.dead)

        if all(missile_dead_list) and any(uav_dead_list):
            return True

        # r_dead = [self.RUAV.got_hit]
        # b_dead = [self.BUAV.got_hit]
        # if self.running == False:
        #     return True
        # if all(r_dead) or all(b_dead):
        #     return True
        return False
    
    def render(self,):
        if self.tacview_show:
            data_to_send = ''
            # 传输飞机信息
            for UAV in self.UAVs:
                loc_LLH = UAV.lon, UAV.lat, UAV.alt
                if not UAV.dead:
                    # data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color={UAV.label}\n" % (
                    #     float(self.t), UAV.id, loc_LLH[0], loc_LLH[1], loc_LLH[2], 
                    #     UAV.phi * 180 / pi, UAV.theta * 180 / pi, UAV.psi * 180 / pi)
                    data_to_send += (
                        f"#{self.t:.2f}\n"
                        f"{UAV.id},T={loc_LLH[0]:.6f}|{loc_LLH[1]:.6f}|{loc_LLH[2]:.6f}|"
                        f"{UAV.phi * 180 / pi:.6f}|{UAV.theta * 180 / pi:.6f}|{UAV.psi * 180 / pi:.6f},"
                        f"Name=F16,Color={UAV.label}\n"
                    )
                else:
                    data_to_send += f"#{self.t:.2f}\n-{UAV.id}\n"

            # 传输导弹信息
            for missile in self.missiles:
                if hasattr(missile, 'dead') and missile.dead:
                    data_to_send += f"#{self.t:.2f}\n-{missile.id}\n"
                else:
                    # 记录导弹的位置
                    loc_m = NUE2LLH(missile.pos_[0], missile.pos_[1], missile.pos_[2], lon_o=o00[0], lat_o=o00[1], h_o=0)
                    if missile.side == 'red':
                        color = 'Orange'
                    else:
                        color = 'Green'
                    data_to_send += f"#{self.t:.2f}\n{missile.id},T={loc_m[0]:.6f}|{loc_m[1]:.6f}|{loc_m[2]:.6f}," \
                                    f"Name=AIM-120C,Color={color}\n"

            self.tacview.send_data_to_client(data_to_send)