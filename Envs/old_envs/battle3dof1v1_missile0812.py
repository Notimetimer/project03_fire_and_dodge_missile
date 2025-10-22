'''出生点改在外面指定'''

from random import random
import random
from gym import spaces
import copy
# from MissileModel2_2 import *  # from MissileModel2 import missile_class
from Envs.MissileModel1 import *  # test

g = 9.81
dt = 0.2  # 0.02 0.8 0.2
dt_refer = dt
# t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180

# battle_field = np.array([10000, 1e4, 10000])  # 东西、南北方向各10km，高度范围1000m,以0,0,500为中心

d1 = 200e3
d2 = d1  # 100e3
d3 = 15e3

min_height = 0
max_height = d3

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


# def calculate_attack_angle(self):
#     """计算攻击角度"""
#     target_vector = self.BUAV.pos_ - self.RUAV.pos_
#     velocity_vector = self.RUAV.vel_
#
#     # 计算夹角
#     dot_product = np.dot(target_vector, velocity_vector)
#     norms = np.linalg.norm(target_vector) * np.linalg.norm(velocity_vector)
#
#     if norms == 0:
#         return 180
#
#     cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
#     angle = np.arccos(cos_angle) * 180 / np.pi
#
#     return angle


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

        # 导弹相关属性
        self.ammo = 6
        self.missile_count = 0  # 已发射导弹数量
        self.missile_launch_interval = 10  # 发射间隔
        self.last_launch_time = -10  # 上次发射时间
        self.missile_detect_range = 20e3  # 探测范围
        self.missile_launch_angle = 45  # 发射视角范围
        self.missile_min_range = 5e3  # 最小发射距离
        self.missile_optimal_range = 25e3  # 最佳发射距离
        self.missile_max_range = 40e3  # 最大发射距离 50
        self.missile_launch_speed_threshold = 100  # 300  # 最小发射速度要求

        # 过载量控制
        self.nx_limit = [-1, 1.5]
        self.ny_limit = [-1.5, 6]
        self.gamma_max = 170 * pi / 180

        # 无人机对抗相关
        self.got_hit = False
        self.crash = False
        self.attacking = False
        self.dead = False

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

    # todo 补充无人机的运动方程和动作逻辑
    def move(self, nx, ny, gamma, record=False):
        # 动力学运动学输出在上述基础上进行更新
        height = self.pos_[1]
        rho = air_density(height)
        speed = self.speed
        mach, _ = calc_mach(speed, height)
        self.mach = mach
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
        self.pos_ += self.vel_ * dt_refer
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
            return True  # "杀伤锥"

    def can_launch_missile(self, target, current_time):
        can_shoot = True
        """判断是否可以发射导弹"""
        if self.dead or target.dead:  # 如果自己或者目标死了，不能发射导弹
            return False

        # 检查导弹数量限制 fixme
        if self.missile_count >= self.ammo:
            # print('导弹射完')
            can_shoot = False

        # 检查发射间隔
        if current_time - self.last_launch_time < self.missile_launch_interval:
            can_shoot = False

        # 检查目标是否在探测范围内
        distance = np.linalg.norm(target.pos_ - self.pos_)
        if distance > self.missile_max_range or distance < self.missile_min_range:
            can_shoot = False

        # 检查目标是否在视角范围内
        target_vector = target.pos_ - self.pos_
        velocity_vector = self.vel_
        dot_product = np.dot(target_vector, velocity_vector)
        norms = np.linalg.norm(target_vector) * np.linalg.norm(velocity_vector)
        if norms == 0:
            can_shoot = False
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180 / pi
        if angle > self.missile_launch_angle:
            can_shoot = False

        # todo 可发射区解算
        if can_shoot:
            pass

        # todo 导弹发射概率输出
        return can_shoot

    def launch_missile(self, target, current_time):
        """发射导弹"""
        # 创建新导弹
        # new_missile = missile_class(self.pos_, self.vel_, current_time)
        new_missile = missile_class(self.pos_, self.vel_, target.pos_, target.vel_, current_time)  # test
        if self.blue:  # target.red:
            new_missile.side = 'blue'
            new_missile.id = 301 + self.missile_count
        if self.red:  # target.blue:
            new_missile.side = 'red'
            new_missile.id = 101 + self.missile_count
        # 根据距离调整导弹参数
        distance = np.linalg.norm(target.pos_ - self.pos_)
        new_missile.max_g = 40  # 最大过载
        new_missile.sight_angle_max = pi / 2  # 导引头视角
        new_missile.launcher_id = self.id  # 发射机id
        new_missile.target_id = target.id  # 目标机id
        print('导弹已发射')
        print(current_time)
        self.missile_count += 1
        print(self.missile_count)
        self.last_launch_time = current_time
        return new_missile

    # 是否可探测到目标
    def can_detect_target(self, target):
        can = False
        if self.dead:  # 本机已死，拒绝探测目标
            return False
        L_ego_enm_ = target.pos_ - self.pos_
        dist = norm(L_ego_enm_)
        if dist <= 160e3:
            angle = np.arccos(np.dot(L_ego_enm_, self.vel_) / (dist * self.speed))
            if angle * 180 / pi <= 60:
                can = True
        return can

    # 是否可跟踪目标
    def can_track_target(self, target):
        can = False
        if self.dead:  # 本机已死，拒绝追踪目标
            return False
        L_ego_enm_ = target.pos_ - self.pos_
        dist = norm(L_ego_enm_)
        if dist <= 80e3:
            angle = np.arccos(np.dot(L_ego_enm_, self.vel_) / (dist * self.speed))
            if angle * 180 / pi <= 55:
                can = True
        return can

    # 是否可为导弹提供数据链中制导
    def can_offer_guidance(self, missile, UAVs):
        can = False
        if self.dead:  # 本机已死，拒绝提供中制导
            return False
        L_ego_m_ = missile.pos_ - self.pos_
        dist = norm(L_ego_m_)
        angle = np.arccos(np.dot(L_ego_m_, self.vel_) / (dist * self.speed))
        if angle * 180 / pi < 60 and dist <= 50e3:  # 假设飞机雷达和导弹的通信距离在50km
            target_uav = None
            # 根据导弹目标id查找目标
            for uav in UAVs:
                if uav.id == missile.target_id:
                    target_uav = uav
                    break
            # 检查该目标是否可被本机追踪
            if target_uav and self.can_track_target(target_uav):
                can = True
        return can


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
    DEFAULT_RED_BIRTH_STATE = {'position': np.array([-R_birth*cos(0), 9000.0, -R_birth*sin(0)]),
                               'psi': 0
                               }
    DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([-R_birth*cos(pi), 8000.0, -R_birth*sin(pi)]),
                                'psi': pi
                                }

    def __init__(self, args):
        super(Battle, self).__init__()
        self.UAV_ids = None
        self.UAV_hit = None
        self.Bmissiles = None
        self.Rmissiles = None
        self.missiles = None
        self.args = args
        self.RUAVs = None
        self.BUAVs = None
        self.UAVs = None
        self.dt = None
        self.t = None
        self.game_time_limit = None
        self.running = None
        self.action_space = []
        # self.reset()  # 重置位置和状态
        self.r_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        self.b_action_spaces = [spaces.Box(low=-1, high=+1, shape=(3,), dtype=np.float32)]
        # # 智能体的观察空间
        # r_obs_n, b_obs_n = self.get_obs()
        # self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      r_obs_n]
        # self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
        #                      b_obs_n]

    def reset(self, red_birth_state=None, blue_birth_state=None, red_init_ammo=6, blue_init_ammo=6):  # 重置位置和状态
        if red_birth_state is None:
            red_birth_state = self.DEFAULT_RED_BIRTH_STATE
        if blue_birth_state is None:
            blue_birth_state = self.DEFAULT_BLUE_BIRTH_STATE

        print(red_birth_state)
        print(blue_birth_state)

        self.Rmissiles = []
        self.Bmissiles = []
        self.missiles = []
        self.dt = dt_refer  # simulation interval，1 second
        self.t = 0
        self.game_time_limit = self.args.max_episode_len
        # 初始化无人机
        self.RUAVs = []
        self.BUAVs = []
        self.Rnum = 1
        self.Bnum = 1
        # 红方初始化
        for i in range(self.Rnum):
            UAV = UAVModel()
            UAV.ammo = red_init_ammo
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.side = "red"
            UAV.color = np.array([1, 0, 0])
            # 红方出生点
            # UAV.pos_ = birthpointr.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = red_birth_state['position']  # np.array([-38841.96119795, 9290.02131746, -1686.95469864])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
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
            self.RUAVs.append(UAV)
        # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel()
            UAV.ammo = blue_init_ammo
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.side = "blue"
            UAV.color = np.array([0, 0, 1])
            # 蓝方出生点
            # UAV.pos_ = birthpointb.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = blue_birth_state['position']  # np.array([38005.14540582, 6373.80721704, -1734.42509136])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
            UAV.psi = blue_birth_state[
                'psi']  # (180 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.psi = sub_of_radian(UAV.psi, 0)
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            self.BUAVs.append(UAV)
        self.running = True
        self.UAVs = self.RUAVs + self.BUAVs
        self.UAV_ids = [UAV.id for UAV in self.UAVs]
        self.UAV_hit = [False for _ in range(len(self.UAVs))]

        # todo 1v1的残留
        self.RUAV = self.RUAVs[0]
        self.BUAV = self.BUAVs[0]

    def get_obs_spaces(self):
        self.reset()
        # 智能体的观察空间
        r_obs_n, b_obs_n = self.get_obs()
        self.r_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             r_obs_n]
        self.b_obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs.shape, dtype=np.float32) for obs in
                             b_obs_n]
        return self.r_obs_spaces, self.b_obs_spaces


    def step(self, r_actions, b_actions, assume=False, record=False):
        # 输入动作（范围为[-1,1],含义依次为nx,ny,gamma）
        if not assume:
            self.t += dt_refer
            self.t = round(self.t, 2)  # 保留两位小数
        else:
            self.t = round(self.t, 2)
            UAVs_ = copy.deepcopy(self.UAVs)
            pass

        actions = r_actions + b_actions
        if not assume:
            # 更新导弹状态
            UAV_hit = self.update_missiles(dt_refer)

            # 更新飞机状态
            for i, UAV in enumerate(self.UAVs):
                if UAV.red:
                    adv = self.BUAV
                if UAV.blue:
                    adv = self.RUAV
                if UAV_hit[i] == True:
                    UAV.dead = True
                    UAV.got_hit = True

            self.missiles = self.Rmissiles + self.Bmissiles

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
                gamma) != pi / 2 else 0
            height = pos_[1]
            rho = air_density(height)
            speed = UAV.speed
            mach, _ = calc_mach(speed, height)
            Drag = -1 / 2 * rho * speed ** 2 * UAV.area * UAV.Cd(mach)

            nx_compensate = -(g_x + Drag / UAV.m) / g
            ny_compensate = -g_y / g

            nx_min = UAV.nx_limit[0]
            nx_max = UAV.nx_limit[1]
            ny_min = UAV.ny_limit[0]
            ny_max = UAV.ny_limit[1]

            nx = action[0] * nx_max + nx_compensate if action[0] >= 0 else -action[0] * nx_min + nx_compensate
            ny = action[1] * ny_max + ny_compensate if action[1] >= 0 else -action[1] * ny_min + ny_compensate

            gamma = 180 * (pi / 180) * action[2]

            nx = np.clip(nx, nx_min, nx_max)
            ny = np.clip(ny, ny_min, ny_max)
            gamma = radian_lim(gamma)

            UAV.move(nx, ny, gamma, record)

        # 毁伤判断
        for index, UAV in enumerate(self.UAVs):
            adv = self.UAVs[1 - index]
            # 近距杀
            short_range_killed = UAV.short_range_kill(adv)
            if short_range_killed:
                # self.running = False
                adv.got_hit = True

        # 撞击判断
        for index, UAV in enumerate(self.UAVs):
            adv = self.UAVs[1 - index]
            # 相撞判别
            pt_ = adv.pos_
            L_ = pt_ - UAV.pos_
            distance = np.linalg.norm(L_)
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

        if assume:
            terminate = False
            self.running = True
            self.UAVs = UAVs_

        self.RUAV = self.UAVs[0]
        self.BUAV = self.UAVs[1]

        if terminate:
            self.running = False

        return r_reward_n, b_reward_n, r_dones, b_dones, terminate

    def get_state(self, side):
        '''
        特征缩放方式：
        距离/10e3
        高度/1e3
        速率/300
        弧度不变
        距离变化率/100

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
        own_situations[0:3] = own.pos_ / np.array([10e3, 1e3, 10e3])
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        # print(vv_)
        theta = atan2(vv_, np.linalg.norm(vh_))
        psi = atan2(vh_[2], vh_[0])
        own_situations[3] = np.linalg.norm(v_) / 300
        own_situations[4] = theta
        own_situations[5] = psi
        # 目标观测信息
        # 补充:能否观测标记
        L_ = adv.pos_ - own.pos_
        q_beta = atan2(L_[2], L_[0])
        L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
        L_v = L_[1]
        q_epsilon = atan2(L_v, L_h)
        # 6↓
        adv_situations[0] = 1
        # 7↓
        adv_situations[1] = q_epsilon  # sub_of_radian(q_epsilon, theta)
        # 8↓
        adv_situations[2] = sub_of_radian(q_beta, psi)
        # 9↓
        adv_situations[3] = np.linalg.norm(L_) / 10e3
        # 相对速度
        vT_ = adv.vel_
        vr_ = vT_ - v_
        vr_radial = np.dot(vr_, v_) / np.linalg.norm(v_)  # 径向速度
        temp = np.cross(v_, vr_) / np.linalg.norm(v_)
        vr_tangent = np.linalg.norm(temp)  # 周向速度
        omega = vr_tangent / np.linalg.norm(L_)
        # 10↓
        adv_situations[4] = vr_radial / 300
        # 11↓
        adv_situations[5] = omega
        # 12↓
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
            r_obs_n, b_obs_n = self.get_obs()
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

    def update_missiles(self, dt):
        # self.UAV_ids = [UAV.id for UAV in self.UAVs]
        # self.UAV_hit = [False for _ in range(len(self.UAVs))]
        """更新所有已发射导弹的状态"""
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
            # 对每一枚导弹小步走
            if not missile.dead:
                for j1 in range(int(plane_missile_time_rate)):
                    # 插值计算目标位置
                    ptt1_ = last_ptt_ + last_vtt_ * dt / plane_missile_time_rate * j1
                    # 获取目标信息
                    target_info = missile.observe(last_vmt_, last_vtt_, last_pmt_, ptt1_)
                    # 更新导弹状态
                    has_datalink = False
                    for uav in self.UAVs:
                        # 找到载机，判断载机能否为导弹提供中制导
                        if uav.id == missile.launcher_id:
                            if uav.can_offer_guidance(missile, self.UAVs):
                                has_datalink = True

                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt1=dt / plane_missile_time_rate, datalink=has_datalink)

                    # print('导弹', missile.id, '有数据链制导?', has_datalink)

                    # 毁伤判定
                    # 判断命中情况并终止运行
                    vmt1 = norm(last_vmt_)
                    if vmt1 < missile.speed_min and missile.t > 0.5 + missile.stage1_time + missile.stage2_time:
                        missile.dead = True
                    if last_pmt_[1] < missile.minH_m:  # 高度小于限高自爆
                        missile.dead = True
                    if missile.t > missile.t_max:  # 超时自爆
                        missile.dead = True
                    if missile.t >= 0 + dt and not target.dead:  # 只允许目标被命中一次, 在同一个判定时间区间内可能命中多次
                        hit, point_m, point_t = hit_target(last_pmt_, last_vmt_, ptt1_, last_vtt_,
                                                           dt1=dt / plane_missile_time_rate)
                        if hit:
                            print(target.side, 'is hit')
                            missile.dead = True
                            missile.hit = True
                            missile.pos_ = point_m
                            missile.vel_ = last_vmt_
                            target.pos_ = point_t
                            target.vel_ = last_vtt_
                            target.dead = True
                            target.got_hit = True
                            self.UAV_hit[self.UAV_ids.index(target.id)] = True
        return self.UAV_hit

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
