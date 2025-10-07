'''缺少按阵营归属飞机和导弹的逻辑，需要补充'''

from random import random
import random
from gym import spaces
import copy
# from MissileModel2_2 import *  # from MissileModel2 import missile_class
from Envs.MissileModel1 import *  # test

'''
状态空间：（未生效）
己方：
[
0：高度/5km （同时也是场地）
1：倾斜角度sin
2：倾斜角cos
3：俯仰角sin值
4：俯仰角cos值
5：速率/600(m/s)
]

友方（不考虑）

敌方(嵌套列表，数量n_adv*7)：
n_adv*[
0：相对距离/10km，但我机离轴角超过90度就假设为10km(无记忆或是GRU)
1：敌机看我机的离轴角余弦值，作为离散值，>90度为0，60~90为1，30~60为2，0~30为3
2：我机看敌机的离轴角余弦值（clip到[0,1]）
3：敌机相对我机方位角弧度
4：敌机相对我机俯仰角弧度
5：敌机相对我机高度/5km
6：敌机速率/300(m/s)
]

导弹：（嵌套列表，如果有多枚导弹就做平均，n_missiles * 7
n_missiles * [
0：导弹速率/300(m/s)
1：导弹相对我机方位角弧度
2：导弹相对我机俯仰角弧度
3：导弹相对我机距离阶梯函数(5km内为4，10km内为3，15km内为2，20km内为1, 20km外为0)
4：距离变化率绝对值/300
5：视线角变化率绝对值/10
6、累计被锁定时间/30s
]

观测空间：
0~5:己方部分同状态空间
6~12:敌方部分只留下一组，通过加权或者选取最近的一架处理，感受域最多1组。
    如果没有目标，会输出全0向量
    如果有多个目标，通过外部导入的规则或是注意力网络处理
13~19:导弹部分也只留下一组，通过加权或者选取最近的一枚处理，感受域最多1组
    如果没有导弹，会输出全0向量，右边作废：{输出action的时候绕过这个神经网络如果没有目标，输出action的时候绕过这个神经网络}
    如果有多枚导弹，通过外部导入的规则或是注意力网络处理
'''

g = 9.81
dt = 0.2  # 0.02 0.8
dt_refer = dt
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
        self.missiles = []  # 已发射的导弹列表
        self.missile_count = 0  # 已发射导弹数量
        self.max_missiles = 6  # 增加最大可携带导弹数量 6
        self.missile_launch_interval = 9.1 * 2  # 发射间隔
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
        # 检查导弹数量限制 fixme
        if self.missile_count > self.max_missiles:
            print('导弹射完')
            can_shoot = False
        # print(type(self.missiles))

        # if len(self.missiles) >= 2:  # 一次最多打1枚 导弹单次可发射数量
        #     can_shoot = False

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

        return can_shoot

    def launch_missile(self, target, current_time):
        """发射导弹"""
        # 创建新导弹
        # new_missile = missile_class(self.pos_, self.vel_, current_time)
        new_missile = missile_class(self.pos_, self.vel_, target.pos_, target.vel_, current_time)  # test
        if target.red:
            new_missile.side = 'blue'
            new_missile.id = 301 + self.missile_count  # len(self.missiles)
        if target.blue:
            new_missile.side = 'red'
            new_missile.id = 101 + self.missile_count  # len(self.missiles)
        # 根据距离调整导弹参数
        distance = np.linalg.norm(target.pos_ - self.pos_)
        new_missile.max_g = 40  # 最大过载
        new_missile.sight_angle_max = pi / 2  # 导引头视角
        # new_missile.target = target
        new_missile.target_id = target.id
        self.missiles.append(new_missile)
        print('导弹已发射')
        print(current_time)
        self.missile_count += 1
        print(self.missile_count)
        self.last_launch_time = current_time
        return new_missile


class Battle(object):
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
            UAV.id = i + 1
            UAV.red = True
            UAV.blue = False
            UAV.side = "red"
            UAV.color = np.array([1, 0, 0])
            # 红方出生点
            # UAV.pos_ = birthpointr.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = np.array([-38841.96119795, 9290.02131746, -1686.95469864])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
            UAV.psi = (0 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
            UAV.theta = 0 * pi / 180
            UAV.gamma = 0 * pi / 180
            UAV.vel_ = UAV.speed * np.array([cos(UAV.theta) * cos(UAV.psi),
                                             sin(UAV.theta),
                                             cos(UAV.theta) * sin(UAV.psi)])
            self.RUAVs.append(UAV)
            # 蓝方初始化
        for i in range(self.Bnum):
            UAV = UAVModel()
            UAV.id = i + 201
            UAV.red = False
            UAV.blue = True
            UAV.side = "blue"
            UAV.color = np.array([0, 0, 1])
            # 蓝方出生点
            # UAV.pos_ = birthpointb.copy() + \
            #            np.random.uniform(-1, 1, 3) * np.array([1e3, 2e3, 10e3])
            UAV.pos_ = np.array([38005.14540582, 6373.80721704, -1734.42509136])
            UAV.speed = (UAV.speed_max - UAV.speed_min) / 2
            UAV.psi = (180 + (random.randint(0, 1) - 0.5) * 2 * random.uniform(50, 60)) * pi / 180
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
                    # self.running = False
                    # break

                if UAV.red:  # 红方无人机
                    if not (UAV.got_hit or adv.dead):
                        # 发射新导弹
                        if UAV.can_launch_missile(self.BUAV, self.t):
                            new_missile = UAV.launch_missile(self.BUAV, self.t)
                            self.Rmissiles.append(new_missile)
                if UAV.blue:  # 蓝方无人机
                    if not (UAV.got_hit or adv.dead):
                        # 发射新导弹 test
                        if UAV.can_launch_missile(self.RUAV, self.t):
                            new_missile = UAV.launch_missile(self.RUAV, self.t)
                            self.Rmissiles.append(new_missile)
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
            if distance <= 100:
                # self.running = False
                UAV.crash = True
                adv.crash = True
            # 出界判别
            if self.out_range(UAV):
                UAV.dead = True
                # self.running = False

        r_reward_n, b_reward_n = self.get_reward()
        terminate = self.get_terminate()

        for UAV in self.UAVs:
            if UAV.got_hit or UAV.crash or self.out_range(UAV):
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
        if side == 'r':
            own = self.RUAV
            adv = self.BUAV
        else:  # if side=='b':
            own = self.BUAV
            adv = self.RUAV
        own_states = np.zeros((1,), dtype=np.float32)

        # 0 高度
        own_states[0] = own.pos_[1]
        v_ = own.vel_
        vh_ = own.vel_ * np.array([1, 0, 1])  # 掩模 取水平速度
        vv_ = own.vel_[1]  # 掩模 取垂直速度
        theta = atan2(vv_, np.linalg.norm(vh_))
        psi = atan2(vh_[2], vh_[0])
        # 1 倾斜角度sin
        own_states = np.append(own_states, sin(own.gamma))
        # 2 倾斜角cos
        own_states = np.append(own_states, cos(own.gamma))
        # 3 俯仰角sin值
        own_states = np.append(own_states, sin(own.theta))
        # 4 俯仰角cos值
        own_states = np.append(own_states, cos(own.theta))
        # 5 速率/600(m/s)
        own_states = np.append(own_states, norm(v_) / 600)

        # 敌机情况
        all_adv_states = []
        advs = [self.UAVs[1 - own.id]]  # 可单可复
        for index, adv in enumerate(advs):
            adv_states = np.zeros((1,), dtype=np.float32)
            L_ = adv.pos_ - own.pos_
            q_beta = atan2(L_[2], L_[0])
            L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
            L_v = L_[1]
            q_epsilon = atan2(L_v, L_h)
            # 0 相对距离
            adv_states = norm(L_) / 1e4
            # 1 敌机看我机的离轴角余弦值
            offaxis_adv_cos = np.dot(adv.vel_, -L_) / (norm(adv.vel_) * norm(L_))
            if offaxis_adv_cos < 0:
                offaxis_adv_cos = 0.0
            elif offaxis_adv_cos < cos(pi / 3):
                offaxis_adv_cos = 1.0
            elif offaxis_adv_cos < cos(pi / 6):
                offaxis_adv_cos = 2.0
            else:
                offaxis_adv_cos = 3.0
            offaxis_adv_cos /= 3.0
            adv_states = np.append(adv_states, offaxis_adv_cos)
            # 2 我机看敌机的离轴角余弦值函数
            offaxis_ego_cos = np.dot(v_, L_) / (norm(v_) * norm(L_))
            adv_states = np.append(adv_states, np.clip(offaxis_ego_cos, 0, 1))
            # 3 敌机相对我机方位角弧度
            adv_states = np.append(adv_states, sub_of_radian(q_beta, psi))
            # 4 敌机相对我机俯仰角弧度
            adv_states = np.append(adv_states, sub_of_radian(q_epsilon, theta))
            # 5 敌机相对我机高度/5km
            adv_states = np.append(adv_states, L_[1] / 5e3)
            # 6 敌机速率/300
            adv_states = np.append(adv_states, norm(v_) / 300)
            all_adv_states.append(adv_states)

        # 导弹情况
        all_missile_states = []
        # 判断对面的导弹是否已经发射
        enm_missiles = []
        if own.blue and self.Rmissiles:
            enm_missiles = self.Rmissiles
        if own.red and self.Bmissiles:
            enm_missiles = self.Bmissiles
        if not enm_missiles:
            all_missile_states = [np.zeros(7)]
        else:
            for i, missile in enumerate(enm_missiles):
                missile_states = np.zeros(7)
                # 0：导弹速率/300(m/s)
                missile_states[0] = norm(missile.vel_) / 300
                # 1：导弹相对我机方位角弧度
                L_ = missile.pos_ - own.pos_
                q_beta = atan2(L_[2], L_[0])
                L_h = np.sqrt(L_[0] ** 2 + L_[2] ** 2)
                L_v = L_[1]
                q_epsilon = atan2(L_v, L_h)
                missile_states[1] = np.append(missile_states, sub_of_radian(q_beta, psi))
                # 2：导弹相对我机俯仰角弧度
                missile_states[2] = np.append(missile_states, sub_of_radian(q_epsilon, theta))
                # 3：导弹相对我机距离阶梯函数
                if norm(L_) < 5e3:
                    missile_states[3] = 4
                elif norm(L_) < 10e3:
                    missile_states[3] = 3
                elif norm(L_) < 15e3:
                    missile_states[3] = 2
                elif norm(L_) < 20e3:
                    missile_states[3] = 1
                else:
                    missile_states[3] = 0
                missile_states[3] /= 4

                # 4：距离变化率绝对值
                missile_states[4] = -np.dot(missile.vel, -L_) / norm(L_) / 300
                missile_states[4] = max(0, missile_states[4])
                # 5：视线角变化率绝对值/10
                missile_states[5] = norm(np.cross(missile.vel, -L_)) / norm(L_) ** 2 / 10
                # 6、累计被锁定时间 / 30s
                missile_states[6] = 0
                if missile.lock_time:
                    missile_states[6] = (missile.t - missile.lock_time) / 30
                missile_states[6] = min(1, missile_states[6])

                all_missile_states.append(missile_states)

        one_side_obs_one_drone = {'own_states': own_states,
                                 'all_adv_states': all_adv_states,
                                 'all_missile_states': all_missile_states}

        one_side_obs = one_side_obs_one_drone
        return one_side_obs

    def get_obs(self, multi_input_method='rule'):
        if multi_input_method=='rule':
            pass


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
        #         min_height <= position[1] <= max_height and \
        #         min_east <= position[2] <= max_east:
        # if min_north <= position[0] <= max_north and \
        #         min_height <= position[1] and \
        #         min_east <= position[2] <= max_east:
        #     out = False
        # # 对升限的处理有限，目前只能强制按平
        # if position[1] >= max_height + 2000:
        #     UAV.vel_[1] = 0
        #     UAV.pos_[1] = max_height + 2000
        #     UAV.theta = 0
        return out

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
            adv = UAVs[1 - i]
            # # 出界剩余时间惩罚：
            t_last_max = 60
            t_last = np.ones(6)
            t_last[0] = (max_height - UAV.pos_[1]) / (UAV.vel_[1]) if UAV.vel_[1] > 0 else t_last_max
            t_last[1] = (UAV.pos_[1] - min_height) / (-UAV.vel_[1]) if UAV.vel_[1] < 0 else t_last_max
            t_last[2] = (max_east - UAV.pos_[2]) / (UAV.vel_[2]) if UAV.vel_[2] > 0 else t_last_max
            t_last[3] = (UAV.pos_[2] - min_east) / (-UAV.vel_[2]) if UAV.vel_[2] < 0 else t_last_max
            t_last[4] = (max_north - UAV.pos_[0]) / (UAV.vel_[0]) if UAV.vel_[0] > 0 else t_last_max
            t_last[5] = (UAV.pos_[0] - min_north) / (-UAV.vel_[0]) if UAV.vel_[0] < 0 else t_last_max
            #
            min_t_last = min(t_last)
            rewards[i] -= (1 - min_t_last / t_last_max) * 100

            # 猛打舵面惩罚：(需要重构加入控制量参数，暂时不加)
            # 靠近边界惩罚
            if UAV.pos_[1] < 800:
                rewards[i] -= 80 * (1 - UAV.pos_[1] / 800) * -UAV.theta / (pi / 2)
            if max_height - UAV.pos_[1] < 800:
                rewards[i] -= 80 * (1 - (max_height - UAV.pos_[1]) / 800) * UAV.theta / (pi / 2)
            ns_range = max_north - min_north
            ew_range = max_east - min_east
            trigger = 2e3
            if min(max_north - UAV.pos_[0], UAV.pos_[0] - min_north) <= trigger:
                rewards[i] -= 30 * (trigger - min(max_north - UAV.pos_[0], UAV.pos_[0] - min_north)) / trigger
            if min(max_east - UAV.pos_[2], UAV.pos_[2] - min_east) <= trigger:
                rewards[i] -= 30 * (trigger - min(max_east - UAV.pos_[2], UAV.pos_[2] - min_east)) / trigger

            # 出界惩罚
            if self.out_range(UAV):
                rewards[i] -= 100
            # 超出限高惩罚
            if UAV.pos_[1] >= max_height:
                rewards[i] -= 80
            # 对手出界
            if self.out_range(adv):
                rewards[i] += 10
            # 被命中(不管是导弹还是"扫描枪")
            if UAV.got_hit:
                rewards[i] -= 100
            # 命中对手
            if adv.got_hit:
                rewards[i] += 200  # 增加命中奖励
            # 撞机
            if UAV.crash:
                rewards[i] -= 70
            # 平局
            if self.t >= self.game_time_limit and not any([UAV.dead, adv.dead]):
                rewards[i] -= 10

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
            if target is None:  # test 目标不存在, 不更换目标而是击毁导弹
                missile.dead = True
                continue
            elif target.dead:  # test 目标死亡, 不更换目标而是击毁导弹, 在飞机1V1的时候可以节省一点计算量，不用费事处理多目标的问题
                missile.dead = True
                continue
            else:
                missile.target = target
            if not missile.dead:
                print('目标位置', target.pos_)
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
                    last_vmt_, last_pmt_, v_dot, nyt, nzt, line_t_, q_beta_t, q_epsilon_t, theta_mt, psi_mt = \
                        missile.step(target_info, dt1=dt / plane_missile_time_rate)
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
        # r_dead = [self.RUAV.got_hit]
        # b_dead = [self.BUAV.got_hit]
        # if self.running == False:
        #     return True
        # if all(r_dead) or all(b_dead):
        #     return True
        return False
