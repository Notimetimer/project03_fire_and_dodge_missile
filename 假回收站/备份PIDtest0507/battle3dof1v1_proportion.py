import numpy as np
from math import cos, sin, pi, atan2, acos
from random import random
import random
from gym import spaces
import copy

'''
坐标系：
统一用北天东，前上右顺序
主要改动：
    观测结构
    奖励函数
动作输入结构未改
*暂时不想改的：动力学，补充垂直机动
'''

g = 9.81
dt = 1  # 0.02
# t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180

# battle_field = np.array([10000, 1e4, 10000])  # 东西、南北方向各10km，高度范围1000m,以0,0,500为中心

d1 = 80e3
d2 = d1  # 100e3
d3 = 15e3

min_north = -d1 / 2
max_north = d1 / 2

min_east = -d2 / 2
max_east = d2 / 2

min_height = 0
max_height = d3

battle_field = np.array([[min_north, max_north], [min_height, max_height], [min_east, max_east]])

birthpointr = np.array([min_north + 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])
birthpointb = np.array([max_north - 2e3, (min_height + max_height) / 2, (min_east + max_east) / 2])


# random_array = np.clip(np.random.randn(3),-1,1)
# “笼子”的范围：x、y方向+-各5000m, z方向0~1000m

def sub_of_radian(input1, input2):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


# 定义大气计算等通用部分
# 大气密度计算
def rho(h):
    # 输入高度(m)
    answer = 1.225 * np.exp(-h / 9300)
    return answer


# 马赫数计算：
def calc_mach(v, height):
    sound_speed = 20.0463 * np.sqrt(288.15 - 0.00651122 * height) if height <= 11000 else 295.069
    return v / sound_speed, sound_speed


# 坐标转换
def passive_rotation(vector, psi, theta, gamma):
    # vector是行向量，根据psi，theta，gamma的顺序旋转坐标系，最后输出行向量
    # 注意：北天东坐标
    R1 = np.array([
        [cos(-psi), 0, -sin(-psi)],
        [0, 1, 0],
        [sin(-psi), 0, cos(-psi)]
    ])
    R2 = np.array([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cos(gamma), sin(gamma)],
        [0, -sin(gamma), cos(gamma)]
    ])
    # (R3*R2*R1*vector).T=vector.T*R1.T*R2.T*R3.T
    return vector @ R1.T @ R2.T @ R3.T


# 大气模型
# 声速
def soundspeed(height):
    if height <= 11000:
        soundspeed1 = 20.05 * np.sqrt(288.15 - 0.00651122 * height)
    else:
        soundspeed1 = 295.069
    return soundspeed1


# 密度
def air_density(height):
    return 1.225 * np.exp(height / 9300)


# 弧度压缩
def radian_lim(input1):
    if input1 > pi:
        return input1 - 2 * pi
    if input1 <= -pi:
        return input1 + 2 * pi
    else:
        return input1


# 导弹模型


# 无人机模型
class UAVModel(object):
    def __init__(self):
        super(UAVModel, self).__init__()
        # 无人机的标识
        self.id = None
        self.red = False
        self.blue = False
        self.side = None  # 阵营
        self.size = 5  # meter
        self.area = 27.87  # F-16机翼面积
        self.m = 12000  # F-16 正常起飞重量
        self.color = None
        # 无人机的状态
        # # 无人机的初始位置
        # self.pos_ = birthpointr if self.blue == False else birthpointb
        # self.vel_ = np.zeros(3)
        self.pos_ = np.zeros(3)
        self.speed = None  # 速率
        self.vel_ = None  # 速度矢量
        self.mach = None
        self.psi = None  # 航向角
        self.theta = None  # 俯仰角
        self.gamma = None  # 滚转角
        self.nx = None
        self.ny = None
        self.nz = None
        # 无人机飞行约束
        self.speed_max = 2 * 340
        self.speed_min = 120
        # 无人机轨迹
        self.trajectory = np.empty((0, 3))  # 新增轨迹列表
        self.vellist = np.empty((0, 3))

        # self.speed_max = args.uav_speed_max
        # self.speed_min = args.uav_speed_min

        # 过载量控制
        self.nx_limit = [-1, 1.5]
        self.ny_limit = [-1.5, 6]
        # self.nx_max = args.uav_nx_max
        # self.nx_min = args.uav_nx_min
        # self.ny_max = args.uav_ny_max
        # self.ny_min = args.uav_ny_min

        # 滚转角度限制
        self.gamma_max = 170 * pi / 180  # args.uav_gamma_max

        # 无人机对抗相关
        self.got_hit = False
        self.crash = False
        self.attacking = False
        self.dead = False

        # 无人机的感知范围和攻击范围
        # self.detect_range = args.uav_detect_range
        # self.detect_angle = args.uav_detect_angle
        # self.attack_range = args.uav_attack_range
        # self.attack_angle = args.uav_attack_angle  # 忽略上下角度和左右角度的区别

    # todo 阻力系数：应该是和马赫数和迎角有关的，但是先借用下导弹的阻力系数函数了
    def Cd(self, mach):
        if 0 < mach <= 0.9:
            cd = 0.16
        if 0.9 < mach <= 1.1:
            cd = 0.16 + 0.29 * (mach - 0.9) / 0.2
        if 1.1 < mach <= 3:
            cd = 0.45 - 0.25 * (mach - 1.1) / 1.9
        else:
            cd = 0.2
        return cd / 10

    # def reset(self):
    #     # self.got_hit = False
    #     # self.crash = False
    #     # self.attacking = False
    #     # 随机生成位置
    #     if self.blue == False:
    #         self.pos_ = birthpointr
    #     else:
    #         self.pos_ = birthpointb

    # todo 补充无人机的运动方程和动作逻辑
    def move(self, nx, ny, gamma, record=False):
        # 动力学运动学输出在上述基础上进行更新
        height = self.pos_[1]
        rho = air_density(height)
        speed = self.speed
        mach, _ = calc_mach(speed, height)
        Drag = -1 / 2 * rho * speed ** 2 * self.area * self.Cd(mach)
        # 加速度更新速度
        self.speed += g * (nx - sin(self.theta)) + Drag / self.m

        v = np.clip(self.speed, self.speed_min, self.speed_max)
        self.speed = v

        self.theta += g / v * ny * cos(gamma) - g / v * cos(self.theta)
        self.theta = np.clip(self.theta, -85 * pi / 180, 85 * pi / 180)  # theta限幅
        self.psi += g * ny * sin(gamma) / (v * cos(self.theta))

        if self.psi > pi:
            self.psi -= 2 * pi
        if self.psi <= -pi:
            self.psi += 2 * pi

        v_ = np.array([v * cos(self.theta) * cos(self.psi),
                       v * sin(self.theta),
                       v * cos(self.theta) * sin(self.psi)])

        self.vel_ = v_ * v / np.linalg.norm(v_)
        # 速度更新位置
        self.pos_ += self.vel_ * dt
        if record:
            self.trajectory = np.vstack((self.trajectory, self.pos_))
            self.vellist = np.vstack((self.vellist, self.vel_))

    def short_range_kill(self, target):
        # 近距杀，不需要导弹的模型
        pt_ = target.pos_
        L_ = pt_ - self.pos_
        distance = np.linalg.norm(L_)
        speed = np.linalg.norm(self.vel_)
        attack_angle = acos(np.dot(self.vel_, L_) / (distance * speed))
        if distance < 8e3 and attack_angle * 180 / pi < 30:
            if target.side == "red":  # test 红方开无敌
                return False
            else:
                return True  # “绝对杀伤锥”


class Battle(object):
    def __init__(self, args):
        super(Battle, self).__init__()
        self.args = args
        self.UAVs = None
        self.dt = None
        self.t = None
        self.game_time_limit = None
        self.running = None
        self.action_space = []
        self.reset()  # 重置位置和状态
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        # 智能体的观察空间
        r_obs_n, b_obs_n = self.get_obs()
        self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             r_obs_n]
        self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             b_obs_n]

    def reset(self):  # 重置位置和状态
        self.dt = dt  # simulation interval，1 second
        self.t = 0
        self.game_time_limit = self.args.max_episode_len
        self.UAVs = [UAVModel() for _ in range(2)]
        for i, UAV in enumerate(self.UAVs):
            UAV.id = i
            if i < 1:
                UAV.red = True
                UAV.blue = False
                UAV.side = "red"
                UAV.color = np.array([1, 0, 0])
                self.RUAV = UAV
            else:
                UAV.red = False
                UAV.blue = True
                UAV.side = "blue"
                UAV.color = np.array([0, 0, 1])
                self.BUAV = UAV

        self.running = True
        for i, UAV in enumerate(self.UAVs):
            # UAV.trajectory = np.empty((0, 3))  # 新增轨迹列表
            # UAV.vellist = np.empty((0, 3))
            # outrange
            UAV.got_hit = False
            UAV.dead = False
            # UAV.reset()
            # 出生点
            if UAV.red:
                # 红方初始化
                UAV.pos_ = birthpointr.copy() + \
                           np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
                # UAV.pos_ = birthpointr.copy() + np.array([0, 0, 10e3])
                UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
                UAV.psi = (0 + random.uniform(-30, 30)) * pi / 180
                UAV.theta = 0 * pi / 180
                UAV.gamma = 0 * pi / 180
                UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                                 sin(UAV.theta),
                                                 cos(UAV.theta) * sin(UAV.psi)])
                # UAV.vel_ = np.array([59.41381164,  0.34337249, -0.22103594])/dt
                self.RUAV = UAV  # 同步名称（reset）
            if UAV.blue:
                # 蓝方初始化
                UAV.pos_ = birthpointb.copy() + \
                           np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
                # UAV.pos_ = birthpointb.copy() + np.array([0, 0, -10e3])
                UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
                UAV.psi = (180 + random.uniform(-30, 30)) * pi / 180
                UAV.theta = 0 * pi / 180
                UAV.gamma = 0 * pi / 180
                UAV.psi = sub_of_radian(UAV.psi, 0)
                UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                                 sin(UAV.theta),
                                                 cos(UAV.theta) * sin(UAV.psi)])
                # UAV.vel_ = np.array([-56.67453876, 0., 12.5974088]) / dt
                self.BUAV = UAV
        # 不考虑弹药数量，只考虑发射间隔

    def step(self, r_actions, b_actions, assume=False, record=False):
        # 输入动作（范围为[-1,1],含义依次为nx,ny,gamma）
        if not assume:
            # if self.running == True:
            #     print('self.running = True')
            # if self.running == False:
            #     print('self.running = False')
            self.t += dt
            self.t = round(self.t, 2)  # 保留两位小数
            # if self.t % 30 == 0:
            # print(self.t)
        else:  # if assume
            self.t = round(self.t, 2)  # 保留两位小数
            UAVs_ = copy.deepcopy(self.UAVs)
            # print('pos', UAVs_[1].pos_)
            # print('traj', UAVs_[1].trajectory)
            pass
        actions = r_actions + b_actions  # 这里是把列表并起来，不是加
        # todo 导弹发射判断(待续)
        # 导弹运动更新
        # 无人机运动更新
        for UAV, action in zip(self.UAVs, actions):
            if UAV.dead:
                continue
            # 输入动作与动力运动学状态
            action = np.clip(action, -1, 1)
            vel_ = UAV.vel_
            pos_ = UAV.pos_
            vel_hor = np.linalg.norm([vel_[0], vel_[2]])
            theta = np.arctan2(vel_[1], vel_hor)
            gamma = action[2] * pi / 1
            UAV.gamma = gamma
            g_x = -g * sin(theta)
            g_y = -g * cos(theta) / cos(gamma) if abs(
                gamma) != pi / 2 else 0  # * cos(gamma)  # test 如果是*，就是抵消重力的分量，如果是/，就是拿分量来抵消重力
            height = pos_[1]
            rho = air_density(height)
            speed = UAV.speed
            mach, _ = calc_mach(speed, height)
            Drag = -1 / 2 * rho * speed ** 2 * UAV.area * UAV.Cd(mach)

            # 控制输出分为神经网络输出的反归一化输出和规则输出两部分组成
            # 规则输出
            # 重力与阻力补偿
            nx_compensate = -(g_x + Drag / UAV.m) / g
            ny_compensate = -g_y / g

            # 动作输出反归一化
            nx_min = UAV.nx_limit[0]  # 减速推重比，瞎写的
            nx_max = UAV.nx_limit[1]  # 推重比，采用1.8作为上限
            ny_min = UAV.ny_limit[0]  # 最小法向过载为
            ny_max = UAV.ny_limit[1]  # 最大法向过载

            # nx = (nx_min + nx_max) / 2 + action[0] * (nx_max - nx_min) / 2 + nx_compensate
            # ny = (ny_min + ny_max) / 2 + action[1] * (ny_max - ny_min) / 2 + ny_compensate
            nx = action[0] * nx_max + nx_compensate if action[0] >= 0 else -action[0] * nx_min + nx_compensate
            ny = action[1] * ny_max + ny_compensate if action[1] >= 0 else -action[1] * ny_min + ny_compensate

            gamma = 180 * (pi / 180) * action[2]
            # 限制输出
            nx = np.clip(nx, nx_min, nx_max)
            ny = np.clip(ny, ny_min, ny_max)

            gamma = radian_lim(gamma)

            # 动力学运动学输出在上述基础上进行更新
            # if not assume:
            #     print(UAV.theta*180/pi)
            #     print('nx,ny,gamma=', nx, ny, gamma)
            UAV.move(nx, ny, gamma, record)

        # 毁伤判断
        for index, UAV in enumerate(self.UAVs):  # self.UAVs 应该是一个二阶列表
            adv = self.UAVs[1 - index]
            # 近距杀
            short_range_killed = UAV.short_range_kill(adv)
            if short_range_killed:
                self.running = False
                adv.got_hit = True
                # if not assume:
                #     print(adv.side, "被", UAV.side, "近距杀")

        # 中远距导弹命中判断
        missile_hit = False  # missile.hit_target(UAV)

        # 撞击判断
        for index, UAV in enumerate(self.UAVs):  # self.UAVs 应该是一个二阶列表
            adv = self.UAVs[1 - index]
            # 相撞判别
            pt_ = adv.pos_
            L_ = pt_ - UAV.pos_
            distance = np.linalg.norm(L_)
            if distance <= 100:
                self.running = False
                UAV.crash = True
                adv.crash = True
            # 出界判别
            if self.out_range(UAV):
                self.running = False
                if not assume:
                    pass
                    # print(self.t, UAV.side, "出界")

        r_reward_n, b_reward_n = self.get_reward(actions)
        # r_get_shot, b_get_shot = self.take_hit()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
                UAV.dead = True
                self.running = False
        r_dones = False
        b_dones = False
        if self.RUAV.dead:
            r_dones = True
        if self.BUAV.dead:
            b_dones = True

        if assume:
            # r_dones = False
            # b_dones = False
            terminate = False
            self.running = True
            self.UAVs = UAVs_
        self.RUAV = self.UAVs[0]  # 同步名称（step)
        self.BUAV = self.UAVs[1]

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def get_state(self, side):
        '''
        每个智能体动作输入为nx, ny, gamma
        观测结构:
        自身状态信息：
            012 自身位置（东北天）
            345 v, θ, ψ
        敌方目标共7个位，分别为：
            0、可观测标志位
            1、相对本机速度矢量的俯仰角
            2、相对本机速度矢量的航向角
            3、同本机的距离
            4、视径向速度
            5、视周向角速率
            6、空置
        '''
        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
        own_situations = np.zeros((6,), dtype=np.float32)
        adv_situations = np.zeros((7,), dtype=np.float32)
        # todo 观测空间欠缺归一化！暂不考虑局部观测的问题，
        own_situations[0:3] = own.pos_
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        # print(vv_)
        theta = atan2(vv_, np.linalg.norm(vh_))
        psi = atan2(vh_[2], vh_[0])
        own_situations[3] = np.linalg.norm(v_)
        own_situations[4] = theta
        own_situations[5] = psi
        # 目标观测信息
        # 补充:能否观测标记
        L_ = adv.pos_ - own.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)

        adv_situations[0] = 1
        adv_situations[1] = q_epsilon # sub_of_radian(q_epsilon, theta)
        adv_situations[2] = sub_of_radian(q_beta, psi)
        adv_situations[3] = np.linalg.norm(L_)
        # 相对速度
        vT_ = adv.vel_
        vr_ = vT_ - v_
        vr_radial = np.dot(vr_, v_) / np.linalg.norm(v_)  # 径向速度
        temp = np.cross(v_, vr_) / np.linalg.norm(v_)
        vr_tangent = np.linalg.norm(temp)  # 周向速度
        omega = vr_tangent / np.linalg.norm(L_)
        adv_situations[4] = vr_radial
        adv_situations[5] = omega
        adv_situations[6] = 0  # 空置，我也不知道能写啥

        one_side_obs_one_drone = np.concatenate([
            own_situations.flatten(),
            adv_situations.flatten(),
        ])
        one_side_obs = one_side_obs_one_drone
        return one_side_obs

    def get_obs(self):
        r_obs_n = self.get_state('r')
        # v_ = self.RUAV.vel_
        # L_ = self.BUAV.pos_ - self.RUAV.pos_
        # off_axis = acos(np.dot(L_, v_) / (np.linalg.norm(v_) * np.linalg.norm(L_))) * 180 / pi
        # if off_axis > 100:  # 部分观测 test
        #     r_obs_n[6:] = 0
        b_obs_n = self.get_state('b')
        # v_ = self.BUAV.vel_
        # L_ = self.RUAV.pos_ - self.BUAV.pos_
        # off_axis = acos(np.dot(L_, v_) / (np.linalg.norm(v_) * np.linalg.norm(L_))) * 180 / pi
        # if off_axis > 100:  # 部分观测 test
        #     b_obs_n[6:] = 0

        return np.stack(r_obs_n), np.stack(b_obs_n)

    # 碰边或撞地
    def out_range(self, UAV):
        position = UAV.pos_
        out = True
        if min_north <= position[0] <= max_north and \
                min_height <= position[1] <= max_height and \
                min_east <= position[2] <= max_east:
            out = False
        # if min_north <= position[0] <= max_north and \
        #         min_height <= position[1] and \
        #         min_east <= position[2] <= max_east:
        #     out = False
        # # 对升限的处理有限，目前只能强制按平
        # if position[1] >= max_height:
        #     UAV.vel_[1] = 0
        #     UAV.pos_[1] = max_height
        #     UAV.theta = 0
        return out

    def get_reward(self, actions, missiled_combat='Flase'):
        if missiled_combat == True:
            # 添加导弹命中相关的奖励和惩罚
            pass
        '针对状态的奖励'
        RUAV = self.RUAV
        BUAV = self.BUAV
        UAVs = [RUAV, BUAV]
        A = [0, 0]  # R, B
        rewards = [0, 0]  # R, B

        for i, UAV in enumerate(UAVs):  # UAVs[0]为红方，UAVs[1]为蓝方
            adv = UAVs[1 - i]
            # 存活奖励
            rewards[i] += 100 if self.t < 0.1 * self.game_time_limit else 400

            # # # 出界剩余时间惩罚：
            # t_last_max = 60
            # t_last=np.ones(6)
            # t_last[0]=(max_height-UAV.pos_[1])/(UAV.vel_[1]) if UAV.vel_[1]>0 else t_last_max
            # t_last[1]=(UAV.pos_[1]-min_height)/(-UAV.vel_[1]) if UAV.vel_[1]<0 else t_last_max
            # t_last[2]=(max_east-UAV.pos_[2])/(UAV.vel_[2]) if UAV.vel_[2]>0 else t_last_max
            # t_last[3]=(UAV.pos_[2]-min_east)/(-UAV.vel_[2]) if UAV.vel_[2]<0 else t_last_max
            # t_last[4]=(max_north-UAV.pos_[0])/(UAV.vel_[0]) if UAV.vel_[0]>0 else t_last_max
            # t_last[5]=(UAV.pos_[0]-min_north)/(-UAV.vel_[0]) if UAV.vel_[0]<0 else t_last_max
            # #
            # min_t_last=min(t_last)
            # rewards[i] -= (1-min_t_last/t_last_max)*10

            # 猛打舵面惩罚：(需要重构加入控制量参数，暂时不加)
            # 靠近边界惩罚
            if UAV.pos_[1] < 800:
                rewards[i] -= 8 * (1 - UAV.pos_[1] / 800) * -UAV.theta / (pi / 2)
            if max_height - UAV.pos_[1] < 800:
                rewards[i] -= 8 * (1 - (max_height - UAV.pos_[1]) / 800) * UAV.theta / (
                            pi / 2)
            ns_range = max_north - min_north
            ew_range = max_east - min_east

            trigger = 2e3
            if min(max_north - UAV.pos_[0], UAV.pos_[0] - min_north) <= trigger:
                rewards[i] -= 3 * (trigger - min(max_north - UAV.pos_[0],
                                                  UAV.pos_[0] - min_north)) / trigger
            if min(max_east - UAV.pos_[2], UAV.pos_[2] - min_east) <= trigger:
                rewards[i] -= 3 * (trigger - min(max_east - UAV.pos_[2],
                                                  UAV.pos_[2] - min_east)) / trigger

            # 出界惩罚
            if self.out_range(UAV):
                rewards[i] -= 10 * self.game_time_limit / self.dt
            # 超出限高惩罚
            if UAV.pos_[1] >= max_height:
                rewards[i] -= 8 * self.game_time_limit / self.dt
            # 对手出界
            if self.out_range(adv):
                rewards[i] += 10
            # 被命中(不管是导弹还是“扫描枪”)
            # if UAV.got_hit:
            #     rewards[i] -= 100
            # # 命中对手
            # if adv.got_hit:
            #     rewards[i] += 100
            # 撞机
            if UAV.crash:
                rewards[i] -= 70
            # 平局
            if self.t >= self.game_time_limit and not any([UAV.dead, adv.dead]):
                rewards[i] -= 10

        dist = np.linalg.norm(RUAV.pos_ - BUAV.pos_)  # 敌我距离

        '优势度函数'
        # 距离+角度优势函数

        ang_R = np.arccos(np.dot(BUAV.pos_ - RUAV.pos_, RUAV.vel_) /
                          (dist * np.linalg.norm(RUAV.vel_))) * 180 / pi  # 进攻角度

        ang_esc_R = np.arccos(np.dot(RUAV.pos_ - BUAV.pos_, RUAV.vel_) /
                              (dist * np.linalg.norm(RUAV.vel_))) * 180 / pi  # 逃逸角度

        ang_B = np.arccos(np.dot(RUAV.pos_ - BUAV.pos_, BUAV.vel_) /
                          (dist * np.linalg.norm(BUAV.vel_))) * 180 / pi  # 进攻角度

        ang_esc_B = np.arccos(np.dot(BUAV.pos_ - RUAV.pos_, BUAV.vel_) /
                              (dist * np.linalg.norm(BUAV.vel_))) * 180 / pi  # 逃逸角度

        H_R = RUAV.pos_[1]
        H_B = BUAV.pos_[1]

        '''这里要加上逃逸角度和进攻角度的判断逻辑，还需要把所有密集奖励都调成正的'''

        if dist < 3e3:
            A_R = max(1 - ang_R / 90, 0) * dist / 3e3
            A_B = max(1 - ang_B / 90, 0) * dist / 3e3
        elif dist < 30e3:
            A_R = max(1 - ang_R / 90, 0) * (1 - (dist - 3e3) / (30e3 - 3e3))
            A_B = max(1 - ang_B / 90, 0) * (1 - (dist - 3e3) / (30e3 - 3e3))
        elif dist < 50e3:
            A_R = max(1 - ang_R / 90, 0) * (1 - (dist - 30e3) / (50e3 - 30e3))
            A_B = max(1 - ang_B / 90, 0) * (1 - (dist - 30e3) / (50e3 - 30e3))
        else:
            A_R = 0
            A_B = 0

        A_R = 3 * A_R
        A_B = 3 * A_B

        # 高度奖励函数
        h_R = RUAV.pos_[1]
        h_B = BUAV.pos_[1]
        h_ = [h_R, h_B]
        H = [0, 0]
        for i in range(len(h_)):
            h = h_[i]
            if h < 13e3:
                H[i] = (h / 13e3)
            else:
                H[i] = 1 - (h - 13e3) / (max_height - 13e3)
        H_R = 0.5 * H[0]
        H_B = 0.5 * H[1]

        # 速度奖励函数，todo 未完待续
        # 如果处于进攻态势，且剩余到达时间比较长或者为负，应该向max加速，如果剩余到达时间比较短，应该向min减速
        # 如果处于劣势，且剩余到达时间到达时间比较短，应该向max加速
        # v_R = np.linalg.norm(RUAV.vel_)  # 红方速度
        # v_B = np.linalg.norm(BUAV.vel_)  # 蓝方速度
        # dist_dot = np.dot(BUAV.pos_-RUAV.pos_, BUAV.vel_-RUAV.vel_)/dist
        # t_last = -dist/dist_dot
        # if t_last < 0 or t_last>30:
        #     V_R = v_R
        V_R = 0
        V_B = 0

        '过程奖励部分'
        # rewards[0] += A_R + H_R + V_R - 1*(A_B+H_B+V_B) + 10  # test 红方开无敌
        # rewards[1] += A_B + H_B + V_B - 1*(A_R+H_R+V_R) + 10

        # test 目标追赶
        r_obs_n, b_obs_n = self.get_obs()
        # rewards[0] = (1 - np.linalg.norm(r_obs_n[7] / pi * 2)) * 100
        rewards[0] = (1 - np.linalg.norm((r_obs_n[7] - RUAV.theta) / pi * 2)) * 100 # test
        rewards[1] = (1-np.linalg.norm(BUAV.theta/pi*2)) * 100
        rewards[0] += (1 - np.linalg.norm(r_obs_n[8] / pi)) * 100
        rewards[1] += (1 - np.linalg.norm(sub_of_radian(BUAV.psi, pi) / pi)) * 100
        # rewards[0] += (r_obs_n[3]-self.RUAV.speed_min) / (self.RUAV.speed_max - self.RUAV.speed_min)*50  # 试试能不能越跑越快
        # rewards[1] += BUAV.speed

        # # test theta保持
        # rewards[0] = (1-np.linalg.norm((RUAV.theta+60*pi/1800)/pi*2)) * 100
        # rewards[1] = (1-np.linalg.norm(BUAV.theta/pi*2)) * 100
        #
        # # test psi保持
        # rewards[0] += (1-np.linalg.norm(sub_of_radian(RUAV.psi, 30*pi/180)/pi)) * 100
        # rewards[1] += (1-np.linalg.norm(sub_of_radian(BUAV.psi, pi)/pi)) * 100

        # # test 第二个出圈惩罚：
        # for UAV in self.UAVs:
        #     if self.out_range(UAV):
        #         rewards[i] -= 50000
        #     else:
        #         rewards[i] += 100

        # # test 预判 奖励
        # future_UAVs = copy.deepcopy(UAVs)
        # for uav in future_UAVs:
        #     uav.pos_ += uav.vel_ * 5 *dt
        #     if self.out_range(uav):
        #         rewards[i] -= 5000
        #     else:
        #         rewards[i] += 100

        return rewards[0], rewards[1]


    # def take_hit(self):
    #     # r_alive = np.array([UAV.got_hit for UAV in self.RUAVs], dtype=np.float32)
    #     # b_alive = np.array([UAV.got_hit for UAV in self.BUAVs], dtype=np.float32)
    #     r_alive = np.array([self.RUAV.got_hit], dtype=np.float32)
    #     b_alive = np.array([self.BUAV.got_hit], dtype=np.float32)
    #
    #     return r_alive, b_alive

    def get_terminate(self):  # 历史遗留，别动
        r_dead = [self.RUAV.got_hit]
        b_dead = [self.BUAV.got_hit]
        if self.running == False:
            return True
        if all(r_dead) or all(b_dead):
            return True
        return False

    # def clear_death(self):
    #     for UAV in self.UAVs:
    #         UAV.dead = UAV.got_hit
