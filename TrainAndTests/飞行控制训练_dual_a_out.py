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

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
from Visualize.tacview_visualize import *
from Visualize.tensorboard_visualize import *
from Algorithms.PPOcontinues_dual_a_out import *


class track_env():
    def __init__(self, dt_move=0.02):
        super(track_env, self).__init__()
        self.UAV_ids = None
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
        
        # 高于升限会导致动作无法实施，影响
        self.time_limit = 180 # 300 t_last
        self.min_alt = 1e3
        self.min_alt_safe = 3e3
        self.max_alt_safe = 14e3
        self.max_alt = 15e3

        # △h动作输出有效性测试
        self.height_req = None
        
        self.tacview_show = None
    
    def reset(self, o00=None, birth_state=None, height_req=8e3, psi_req=0 , v_req= 340, dt_report=0.2, t0=0, tacview_show=0):
        self.tacview_show = tacview_show
        self.t = t0
        self.success = 0
        # self.done = 0
        self.fail = 0
        self.draw = 0
        if o00 == None:
            o00 = np.array([118, 30])  # 地理原点的经纬
        if birth_state == None:
            birth_state = self.DEFAULT_RED_BIRTH_STATE
        self.dt_report = dt_report
        UAV = UAVModel(dt=self.dt_move)
        UAV.ammo = 0
        UAV.id = 1
        UAV.red = True
        UAV.blue = False
        UAV.side = "red"
        UAV.color = np.array([1, 0, 0])
        # 红方出生点
        UAV.pos_ = self.DEFAULT_RED_BIRTH_STATE['position']
        UAV.speed = 300  # (UAV.speed_max - UAV.speed_min) / 2
        speed = UAV.speed
        UAV.psi = self.DEFAULT_RED_BIRTH_STATE['psi']
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

        if tacview_show:
            self.tacview = Tacview()


    def get_state(self, side):
        '''
        在这里统一汇总所有用得到的状态量，计算状态量可见性并分配各各个子策略的观测
        这里不缩放，统一在get_obs缩放（因为有些会直接输入到规则里面）
        默认值在这里设定
        '''

        own = self.UAV

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

        alpha_air = own.alpha_air
        beta_air = own.beta_air


        # 原先将所有量打包成一个 numpy array，这里改为 dict 结构
        self.key_order = [
            "target_alive",  # 1 暂未使用
            "target_observable",  # 1 仅用于动作切换
            "target_locked",  # 1
            "missile_in_mid_term",  # 1 仅用于动作切换
            "locked_by_target",  # 1 仅用于动作切换
            "warning",  # 1
            "target_information",  # 8
            "ego_main",  # 7
            "ego_control",  # 7
            "weapon",  # 1 仅用于动作切换
            "threat",  # 4
            "border",  # 2
        ]

        one_side_states = {
            # 单独键（标量或布尔）
            "target_alive": bool(1),
            "target_observable": int(0),  # 0 完全不可见 1 角度信息可见 2 完全可见
            "target_locked": bool(0),  # 已锁定敌机
            "missile_in_mid_term": bool(0),
            "locked_by_target": bool(0),  # 敌锁定
            "warning": bool(0),

            # 打包的向量 / 子组
            "target_information": np.array([
                # float(delta_alt_obs),  # 0相对高度 m
                # float(delta_psi),  # 1相对方位 rad
                0,  # 0相对方位 cos
                0,  # 1相对方位 sin
                0,  # 2相对俯仰角 rad
                0,  # 3距离 m
                0,  # 4夹角 rad
                0,  # 5目标速度 m/s
                0,  # 6水平进入角 rad
                0  # 7垂直进入角 rad
            ]),

            "ego_main": np.array([
                float(v_own),  # 0本机速度 m/s
                float(h_own),  # 1本机高度 m
                float(sin_theta),  # 2
                float(cos_theta),  # 3
                float(sin_phi),  # 4
                float(cos_phi),  # 5
                int(ammo)  # 6剩余导弹数量
            ]),

            "ego_control": np.array([
                float(p),  # 0 p rad/s act1_last
                float(q),  # 1 q rad/s act2_last
                float(r),  # 2 r rad/s act3_last
                float(theta_v),  # 3
                float(psi_v),  # 4
                float(alpha_air),  # 5 rad
                float(beta_air)  # 6 rad
            ]),

            "weapon": float(0),

            "threat": np.array([
                float(cos(0)),  # 0
                float(sin(0)),  # 1
                float(0),  # 2
                float(0),  # 3
            ]),

            "border": np.array([
                float(0),  # 0
                int(0),  # 1
            ])

        }

        return one_side_states
    
    # 尺度缩放
    def scale_state(self, state_input):
        # 使用 deepcopy 避免修改传入对象
        s = copy.deepcopy(state_input)
        s["target_information"][3] /= 10e3
        s["target_information"][5] /= 340
        s["ego_main"][0] /= 340
        s["ego_main"][1] /= 5e3
        s["ego_control"][0] /= (2 * pi)  # (2 * pi) 5000
        s["ego_control"][1] /= (2 * pi)  # (2 * pi) pi
        s["ego_control"][2] /= (2 * pi)  # (2 * pi) 340
        s["weapon"] /= 120
        s["threat"][3] /= 10e3
        s["border"][0] = min(1, s["border"][0] / 50e3)
        s["border"][1] = 0 if s["border"][0] == 1 else s["border"][1]
        return s
        
    def base_obs(self, side, pomdp=0):
        # 处理部分可观测、默认值问题、并尺度缩放
        # 输出保持字典的形式
        if side == 'r':
            uav = self.RUAV
        if side == 'b':
            uav = self.BUAV

        state = self.get_state(side)  # np.stack(self.get_state(side)) stack适用于多架无人机观测拼接为np数组

        # 默认值设定
        self.state_init = self.get_state(side)
        self.state_init["target_alive"] = 1  # 默认目标存活
        self.state_init["target_observable"] = 2  # 默认完全可见
        self.state_init["target_locked"] = 0
        self.state_init["missile_in_mid_term"] = 0
        self.state_init["locked_by_target"] = 0
        self.state_init["warning"] = 0
        # self.state_init["target_information"] = np.array([0, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["target_information"] = np.array([1, 0, 0, 100e3, 0, 0, 0, 0])
        self.state_init["ego_main"] = np.array([300, 5000, 0, 1, 0, 1, 0])
        self.state_init["ego_control"] = np.array(
            [0, 0, 0, 0, 0, 0, 0])  # pqr[0, 0, 0, 0, 0, 0, 0] 历史动作[0, 0, 340, 0, 0, 0, 0]
        self.state_init["weapon"] = 120
        # self.state_init["threat"] = np.array([pi, 0, 30e3])  # [pi,0,30e3]  [0,0,30e3]
        self.state_init["threat"] = np.array([1, 0, 0, 30e3])
        self.state_init["border"] = np.array([50e3, 0])

        # todo 重构pomdp的代码实现，尤其是state[x]的部分，state已经被改成字典了
        if pomdp:  # 只有在部分观测情况下需要添加屏蔽
            if uav.obs_memory is None:  # 假如不存在记忆，给默认值
                memory = copy.deepcopy(self.state_init)
            else:
                memory = uav.obs_memory

            ATA = state["target_information"][4]
            dist = state["target_information"][3]

            # 超出探测距离
            if dist > 160e3:  # 啥也看不到
                state["target_observable"] = 0
                state["target_information"] = memory["target_information"].copy()
            # 探测距离到近距
            elif dist > 10e3:
                if ATA > pi / 3 and state["locked_by_target"] == 0:  # 夹角>3/pi时观测不到目标
                    state["target_observable"] = 0
                    state["target_information"] = memory["target_information"].copy()
                elif ATA > pi / 3 and state["locked_by_target"] == 1:  # 被目标探测后有对目标的角度信息
                    state["target_observable"] = 1
                    # for idx in (0, 3, 5, 6, 7):
                    for idx in (3, 5, 6, 7):
                        state["target_information"][idx] = memory["target_information"][idx]
                else:
                    state["target_observable"] = 2  # 否则除了不能发射导弹，都是可见的
            else:
                state["target_observable"] = 2  # 10km类信息完全可见
            # if not state["target_alive"]:
            #     state["target_observable"] = 0

        uav.obs_memory = copy.deepcopy(state)  # 更新残留值

        observation = self.scale_state(state)
        self.obs_init = self.scale_state(self.state_init)
        return observation

    def get_obs(self, ):
        pass


    def get_obs_spaces(self):
        self.reset()
        obs = self.get_obs()
        self.obs_spaces = [spaces.Box(low=-np.inf, high=+np.inf, shape=obs1.shape, dtype=np.float32) for obs1 in obs]
        return self.obs_spaces

    def step(self, action):
        self.action = action
        target_height, delta_heading, target_speed, rudder = action
        self.t += self.dt_report
        time_rate = int(round(self.dt_report/self.dt_move))
        for _ in range(time_rate):
            self.UAV.move(target_height, delta_heading, target_speed, relevant_height=True, relevant_speed=False, with_theta_req=False, p2p=True, rudder=rudder)
            done = self.get_done()
            # 单智能体特例
            if self.fail:
                break
        next_obs = self.get_obs()
        # done = self.get_done()
        reward = self.get_reward()
        
        return next_obs, reward, done


    def get_done(self):
        done = 0
        # 高度追踪失败条件：跑出h_min~h_max的范围立即失败
        h_current = self.UAV.alt
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
        # 高度奖励（高度误差惩罚）
        h_current = self.UAV.alt
        h_req = self.height_req
        r_h_norm = (h_current<=h_req)*(h_current-self.min_alt)/(h_req-self.min_alt)+\
                    (h_current>h_req)*(1-(h_current-h_req)/(self.max_alt-h_req))
        r_h_norm = 1 * r_h_norm

        # 航向奖励（误差惩罚）

        # 速度奖励（速度误差惩罚，失速惩罚）

        # 滚转角奖励(快到位时abs越小越好)

        # 迎角奖励(惩罚负迎角和过大的正迎角)

        # 侧滑角奖励（尽量少侧滑）

        # pqr奖励(快到位时候pqr越小越好)



        # 操作直接奖励
        delta_height, delta_heading, target_speed = self.action
        r_delta_height_instruction = 1-abs(h_req-(h_current+delta_height))/5000
        r_h_norm += 0.8 * r_delta_height_instruction
        

        # 高度出界惩罚
        if self.fail:
            r_h_norm -= 10
        if self.success:
            r_h_norm += 3
        
        # 其他奖励待续
        return r_h_norm
        

    def render(self,):
        loc_r = [self.UAV.lon, self.UAV.lat, self.UAV.alt]
        if self.tacview_show:
            data_to_send = ''
            data_to_send += "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
                    float(self.t), self.UAV.id, loc_r[0], loc_r[1], loc_r[2], self.UAV.phi * 180 / pi, self.UAV.theta * 180 / pi,
                    self.UAV.psi * 180 / pi)
            self.tacview.send_data_to_client(data_to_send)


env = track_env()
obs_space = env.get_obs_spaces()
action_space = env.action_space

# dof = 3
# 超参数
actor_lr = 1e-4 # 1e-4 1e-6  # 2e-5 警告，学习率过大会出现"nan"
critic_lr = actor_lr * 5  # *10 为什么critic学习率大于一都不会梯度爆炸？ 为什么设置成1e-5 也会爆炸？ chatgpt说要actor的2~10倍
max_steps = 65e4
hidden_dim = [128, 128]  # 128
gamma = 0.9
lmbda = 0.9
epochs = 10  # 10
eps = 0.2
dt_decide = 2 # 2
pre_train_rate = 0 # 0.25 # 0.25

state_dim = len(obs_space) # obs_space[0].shape[0]  # env.observation_space.shape[0] # test
action_dim = 4 # test
action_bound = np.array([[-1,1]]*action_dim)  # 动作幅度限制, 必须使用双方括号，否则不能将不同维度分离
env_name = '高低测试'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                      lmbda, epochs, eps, gamma, device)

# --- 仅保存一次网络形状（meta json），如果已存在则跳过
# log_dir = "./logs"
from datetime import datetime
log_dir = os.path.join("./logs", "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))

os.makedirs(log_dir, exist_ok=True)
actor_meta_path = os.path.join(log_dir, "actor.meta.json")
critic_meta_path = os.path.join(log_dir, "critic.meta.json")

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

save_meta_once(actor_meta_path, agent.actor.state_dict())
save_meta_once(critic_meta_path, agent.critic.state_dict())

from Math_calculates.ScaleLearningRate import scale_learning_rate
# 根据参数数量缩放学习率
actor_lr = scale_learning_rate(actor_lr, agent.actor)
critic_lr = scale_learning_rate(critic_lr, agent.critic)

from Visualize.tensorboard_visualize import TensorBoardLogger

out_range_count = 0
return_list = []
steps_count = 0

logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
try:
    # 强化学习训练
    rl_steps = 0
    i_episode = 0
    while rl_steps < max_steps:
        i_episode += 1
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
        
        init_height = np.random.uniform(4000, 10000)  # 生成一个介于 4000 和 10000 的均匀分布值

        birth_state={'position': np.array([-38000.0, init_height, 0.0]),
                            'psi': 0
                            }
        height_req = np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
        psi_req = np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*pi
        v_req = np.random.uniform(0.8, 2.5)*340


        env.reset(birth_state=birth_state, height_req=height_req, tacview_show=0, dt_report=dt_decide) # 打乱顺序也行啊
        state = env.get_obs()
        done = False

        actor_grad_list = []
        critc_grad_list = []
        actor_loss_list = []
        critic_loss_list = []
        entropy_list = []
        ratio_list = []

        while not done:  # 每个训练回合
            # 1.执行动作得到环境反馈
            action, u = agent.take_action(state, action_bounds=action_bound, explore=True)
            rl_steps += 1
            total_action = np.array([5000 * action[0], 0, 300]) # 1000 * 

            next_state, reward, done = env.step(total_action)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(u)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            transition_dict['action_bounds'].append(action_bound)
            state = next_state
            episode_return += reward * env.dt_report # 奖励按秒分析

        if env.fail==1:
            out_range_count+=1
        return_list.append(episode_return)
        agent.update(transition_dict)

        # --- 保存模型（强化学习阶段：actor_rein + i_episode，critic 每次覆盖）
        os.makedirs(log_dir, exist_ok=True)
        # critic overwrite
        critic_path = os.path.join(log_dir, "critic.pt")
        th.save(agent.critic.state_dict(), critic_path)
        # actor RL snapshot
        actor_name = f"actor_rein{i_episode}.pt"
        actor_path = os.path.join(log_dir, actor_name)
        th.save(agent.actor.state_dict(), actor_path)

        
        # tqdm 训练进度显示
        if (i_episode + 1) >= 10:
            print({'episode': '%d' % (i_episode + 1),
                            'return': '%.3f' % np.mean(return_list[-10:])})

        # tensorboard 训练进度显示
        logger.add("train/episode_return", episode_return, i_episode + 1)

        actor_grad_norm = model_grad_norm(agent.actor)
        critic_grad_norm = model_grad_norm(agent.critic)
        # 梯度监控
        logger.add("train/actor_grad_norm", actor_grad_norm, i_episode + 1)
        logger.add("train/critic_grad_norm", critic_grad_norm, i_episode + 1)
        # 损失函数监控
        logger.add("train/actor_loss", agent.actor_loss, i_episode + 1)
        logger.add("train/critic_loss", agent.critic_loss, i_episode + 1)
        # 强化学习actor特殊项监控
        logger.add("train/entropy", agent.entropy_mean, i_episode + 1)
        logger.add("train/ratio", agent.ratio_mean, i_episode + 1)

    # # 在训练结束后，——但仍在 try 范围内——使用最新保存的 actor 权重进行测试
    # # 优先加载最新 RL 快照，其次 supervised 快照
    # agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
    #                 lmbda, epochs, eps, gamma, device)
    # if os.path.exists(actor_meta_path):
    #     rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
    #     sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
    #     latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
    #     if latest_actor_path:
    #         # 直接加载权重到现有的 agent
    #         sd = th.load(latest_actor_path, map_location=device)
    #         agent.actor.load_state_dict(sd) # , strict=False)  # 忽略缺失的键
    #         print(f"Loaded actor for test from: {latest_actor_path}")

    # # 测试回合（在 try 内，位于 except KeyboardInterrupt 之前）
    # env.reset(height_req=5e3, dt_report = dt_decide, tacview_show=1)
    # step = 0
    # state = env.get_obs()
    # done = False
    # while not env.get_done():
    #     action, _ = agent.take_action(state, action_bounds=action_bound, explore=False)

    #     total_action = np.array([5000 * action[0], 0, 300]) # 1000 * 

    #     next_state, reward, done = env.step(total_action)
    #     state = next_state
    #     step += 1
    #     env.render()
    #     time.sleep(0.01)

except KeyboardInterrupt:
    print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
finally:
    logger.close()
    print(f"日志已保存到：{logger.run_dir}")
