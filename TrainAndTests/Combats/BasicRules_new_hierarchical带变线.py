import numpy as np
from math import *
import torch
import argparse
from numpy.linalg import norm
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import * # 三元组奖励
from Math_calculates.sub_of_angles import *
import re

use_tacview = 1  # 是否可视化
action_cycle_multiplier = 10

class BasicRules:
    '''
    rules_num = 0: 保持和目标相同高度只打进攻(action_number = 0,1,3)，0平飞追踪 1爬升追踪 3下降追踪，攻击区内发射导弹，上一枚导弹发射后如果还在中制导，不发射新导弹
    rules_num = 1: 保持和目标相同高度进攻(0,1,3), 发射完导弹立马crank(5,6), 受到威胁立刻回转至5000m高度以下(11水平回转, 12俯冲回转), 威胁结束后回归进攻
    rules_num = 2: 保持和目标相同高度打首轮进攻(0,1,3), 在距离40km以外先爬升60°(2), 一个决策回合后射击, 否则直接射击，设计后立刻crank(6)，
        收到威胁立刻splitS(8), 威胁解除后回转进攻(0,1,3)
    rules_num = 5: 基于rule3，crank阶段每隔6s变换方向（左右交替）
    '''

    def __init__(self, rules_num, p_random=0):
        self.rules_num = rules_num
        self.p_random = p_random
        self.reset()

    def reset(self):
        self.last_t = 0.0
        self.last_action_h = None  # 上一步水平动作
        self.crank_timer = 0.0  # 距离上一次crank方向切换的累计时间
        self.crank_direction = None  # 当前crank方向: 1=左crank, 5=右crank

    def decide(self, state_check, t=0.0):
        dt = t - self.last_t
        self.last_t = t

        rules_num = self.rules_num
        p_random = self.p_random

        delta_theta = state_check["target_information"][2]
        distance = state_check["target_information"][3]
        delta_alt = distance*sin(delta_theta)
        d_hor, leftright = state_check["border"]
        speed = state_check["ego_main"][0]
        alt = state_check["ego_main"][1]
        cos_delta_psi = state_check["target_information"][0]
        sin_delta_psi = state_check["target_information"][1]
        delta_psi = atan2(sin_delta_psi, cos_delta_psi)
        delta_psi_threat = atan2(state_check["threat"][1], state_check["threat"][0])
        RWR = state_check["warning"]
        threat_distance = state_check["threat"][3]
        on_guiding = state_check["missile_in_mid_term"]
        t_fired = state_check["weapon"]
        ATA = state_check["target_information"][4]
        AA_hor = state_check["target_information"][6]
        sin_theta = state_check["ego_main"][3]

        threat_distance_list = np.array([
                                        0, # 0
                                        45, # 1
                                        45, # 2
                                        45, # 3
                                        45, # 4
                                        45, # 5
                                        ]) * 1e3

        # 1. 计算初始的开火意图
        fire_missile = False
        case1 = distance < 95e3 and (sin_theta >= sin(30*pi/180) or alt >= 9e3)
        case2 = distance < 80e3
        if (case1 or case2) and ATA < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
            if t_fired >= 40 and not on_guiding and not (distance>12e3 and abs(AA_hor) < 30*pi/180):
                fire_missile = True

        action_v = 2
        action_h = 0
        action_number = [action_v, action_h]
        base_offensive_action = action_number
        action_number = base_offensive_action

        # 3. 根据规则编号(rules_num)决定最终机动和开火决策
        if rules_num == 0:
            action_number = base_offensive_action
            fire_missile_affirmative = fire_missile

        elif rules_num == 1:
            if RWR and threat_distance < threat_distance_list[rules_num]:
                if alt > 5000:
                    action_v = 4
                else:
                    action_v = 2
                action_h = 3
            elif on_guiding:
                if delta_psi < 0:
                    action_v = 2
                    action_h = 5
                else:
                    action_v = 2
                    action_h = 1
            else:
                action_number = base_offensive_action
            fire_missile_affirmative = fire_missile
            action_number = [action_v, action_h]

        elif rules_num == 2:
            if RWR and threat_distance < threat_distance_list[rules_num]:
                action_v = 4
                action_h = 3
                action_number = [action_v, action_h]
                fire_missile = False
            elif on_guiding:
                if delta_psi < 0:
                    action_v = 2
                    action_h = 5
                    action_number = [action_v, action_h]
                else:
                    action_v = 2
                    action_h = 1
                    action_number = [action_v, action_h]
            elif fire_missile and distance > 40e3:
                if sin_theta < sin(30*pi/180) and alt < 9500:
                    action_v = 0
                    action_h = 0
                    fire_missile = False
                    action_number = [action_v, action_h]
                else:
                    fire_missile = True
            else:
                action_number = base_offensive_action
            fire_missile_affirmative = fire_missile

        elif rules_num == 3:
            if RWR:
                action_v = 4
                action_h = 3
                action_number = [action_v, action_h]
                fire_missile = False
            elif on_guiding:
                if delta_psi < 0:
                    action_v = 3
                    action_h = 5
                    action_number = [action_v, action_h]
                else:
                    action_v = 3
                    action_h = 1
                    action_number = [action_v, action_h]
            elif fire_missile and distance > 40e3:
                if sin_theta < sin(30*pi/180) and alt < 9500:
                    action_v = 0
                    action_h = 0
                    fire_missile = False
                    action_number = [action_v, action_h]
                else:
                    fire_missile = True
            else:
                action_number = base_offensive_action
            fire_missile_affirmative = fire_missile

        elif rules_num == 4:
            if RWR and threat_distance < threat_distance_list[rules_num]:
                action_v = 4
                action_h = 3
                action_number = [action_v, action_h]
                fire_missile = False
            elif on_guiding:
                action_v = 2
                action_h = 0
                action_number = [action_v, action_h]
            elif fire_missile and distance > 40e3:
                if sin_theta < sin(30*pi/180) and alt < 9500:
                    action_v = 0
                    action_h = 0
                    fire_missile = False
                    action_number = [action_v, action_h]
                else:
                    fire_missile = True
            else:
                action_number = base_offensive_action
            fire_missile_affirmative = fire_missile

        elif rules_num == 5:
            # 规则5: 基于rule3，crank阶段每隔6s变线
            if RWR:
                action_v = 4
                action_h = 3
                action_number = [action_v, action_h]
                fire_missile = False
            elif on_guiding:
                # 初始crank方向基于delta_psi
                if delta_psi < 0:
                    desired_crank = 5  # Rcrank
                else:
                    desired_crank = 1  # Lcrank

                # 变线逻辑：如果上一步也是crank，累计时间并判断是否换向
                is_crank = self.last_action_h in (1, 5)
                if is_crank:
                    self.crank_timer += dt
                    if self.crank_timer >= 24.0:
                        # 换方向
                        if self.crank_direction == 1:
                            self.crank_direction = 5
                        else:
                            self.crank_direction = 1
                        self.crank_timer = 0.0
                    action_h = self.crank_direction
                else:
                    # 首次进入crank
                    self.crank_direction = desired_crank
                    self.crank_timer = 0.0
                    action_h = desired_crank

                action_v = 3
                action_number = [action_v, action_h]
            elif fire_missile and distance > 40e3:
                if sin_theta < sin(30*pi/180) and alt < 9500:
                    action_v = 0
                    action_h = 0
                    fire_missile = False
                    action_number = [action_v, action_h]
                else:
                    fire_missile = True
            else:
                action_number = base_offensive_action
            fire_missile_affirmative = fire_missile

        if np.random.rand() <= p_random:
            v_action = np.random.randint(0, 5)
            h_action = np.random.randint(0, 6)
            action_number = [v_action, h_action]

        # 更新记忆
        self.last_action_h = action_number[1]

        return np.array(action_number), fire_missile_affirmative


def basic_rules(state_check, rules_num, last_action=0, p_random=0, t=0):
    '''兼容旧接口的无状态版本（不含变线逻辑）'''
    agent = BasicRules(rules_num=rules_num, p_random=p_random)
    agent.last_t = t
    return agent.decide(state_check, t=t)


if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    # 在这里调用规则(编号)下的策略
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    # Environment
    parser.add_argument("--max-episode-len", type=float, default=15*60,  # 8 * 60,
                        help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
    parser.add_argument("--R-cage", type=float, default=55.0e3, # 69
                        help="")
    args = parser.parse_args()

    # 构建场地边界
    vertices = None # 默认圆形边界
    # 南北长54km，东西宽100km的长方形边界
    # vertices = [[15e3, 50e3], [-15e3, 50e3], [-15e3, -50e3], [15e3, -50e3]]
    env = ChooseStrategyEnv(args, tacview_show=use_tacview, vertices=vertices)
    # test
    env.dt_move = 0.04 # 4 # 0.025 0.02

    env.shielded = 1 # 0 # 有防撞地就可以不要这个

    r_action_spaces, b_action_spaces = env.r_action_spaces, env.b_action_spaces

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    def creat_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 55e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier # Agent takes action every dt_action_cycle seconds

    transition_dict_threshold = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    steps_count = 0

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    decide_steps_after_update = 0
    try:
        r_action_list = []
        b_action_list = []
        r_guide_list = []
        b_guide_list = []
        
        # 采集不同轨迹的动作
        for i_episode in range(6): # 5

            last_r_action_label = 0
            last_b_action_label = 0

            episode_return = 0
            transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [],}

            DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

            env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                    red_init_ammo=6, blue_init_ammo=6)
            r_action_label=0
            b_action_label=0
            last_decision_state = None
            
            b_reward = None

            done = False

            env.dt_maneuver = dt_maneuver
            
            episode_start_time = time.time()

            # 实例化规则对象（每回合重置）
            r_rule_agent = BasicRules(rules_num=5)
            b_rule_agent = BasicRules(rules_num=i_episode)

            # 环境运行一轮的情况
            steps_of_this_eps = -1 # 没办法了
            for count in range(round(args.max_episode_len / dt_maneuver)):
                # print(f"time: {env.t}")  # 打印当前的 count 值
                # 回合结束判断
                # print(env.running)
                current_t = count * dt_maneuver
                steps_of_this_eps += 1
                if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                    # print('回合结束，时间为：', env.t, 's')
                    break
                # 获取观测信息
                r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
                b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
                
                # 在这里将观测信息压入记忆
                # env.RUAV.obs_memory = r_check_obs.copy()
                # env.BUAV.obs_memory = b_check_obs.copy()

                # --- 智能体决策 ---
                # 判断是否到达了决策点（每 10 步）
                if steps_of_this_eps % action_cycle_multiplier == 0:
                    # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                    # # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                    # if steps_of_this_eps > 0:
                    #     transition_dict['states'].append(last_decision_state)

                    # **关键点 2: 开始【新的】一个动作周期**
                    # 1. 记录新周期的起始状态
                    last_decision_state = b_obs
                    # 2. Agent 产生一个动作

                    # 红方根据规则活动
                    r_state_check = env.unscale_state(env.obs2obs_check(r_obs))  # r_check_obs)
                    r_action_label, r_fire = r_rule_agent.decide(r_state_check, t=current_t)
                    last_r_action_label = r_action_label
                    if r_fire:
                        env.RUAV.about_to_fire = 1

                    # 蓝方根据规则活动
                    b_state_check = env.unscale_state(env.obs2obs_check(b_obs))  # b_check_obs)
                    b_action_label, b_fire = b_rule_agent.decide(b_state_check, t=current_t)
                    last_b_action_label = b_action_label
                    if b_fire:
                        env.BUAV.about_to_fire = 1

                    decide_steps_after_update += 1
                    
                    # r_action_list.append(np.array([env.t + t_bias, r_action_label]))
                    # b_action_list.append(np.array([env.t + t_bias, b_action_label]))


                    # # debug
                    # if env.t >= 60+54 and i_episode==0:
                    #     print("r_action", r_action_label)
                    #     print("r_state_check", r_state_check["warning"])
                    #     print()
                        
                    # if env.t > 40:
                    #     print("r_state_check", r_state_check["warning"])
                    #     print("r_action_label", r_action_label)
                    #     print()
                    #     print("b_state_check", b_state_check["warning"])
                    #     print("b_action_label", b_action_label)
                    #     print()
                
                # action_label 设置为 r_action_label 或者 b_action_label 适合测试，完全禁止在动作没到位时开火
                # 设置为 None 适合采样，试错，必须错才能学会
                if getattr(env.RUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'r', action_label=None) # r_action_label)
                if getattr(env.BUAV, 'about_to_fire', 0):
                    launch_missile_immediately(env, 'b', action_label=None) # b_action_label)

                r_action = env.maneuver14LR(env.RUAV, r_action_label)
                b_action = env.maneuver14LR(env.BUAV, b_action_label)

                env.step(r_action, b_action) # Environment updates every dt_maneuver
                r_guide_list.append(np.array([env.t + t_bias, env.r_can_guide]))
                b_guide_list.append(np.array([env.t + t_bias, env.b_can_guide]))
                done, b_reward_event, b_reward_constraint, b_reward_shaping = env.combat_terminate_and_reward('b', b_action_label, b_fire)
                b_reward = b_reward_event + b_reward_constraint + b_reward_shaping

                # Accumulate rewards between agent decisions
                episode_return += b_reward * env.dt_maneuver

                next_b_check_obs = env.base_obs('b')
                next_b_obs = flatten_obs(next_b_check_obs, env.key_order)


                '''显示运行轨迹'''
                # 可视化
                env.render(t_bias=t_bias)
            
            episode_end_time = time.time()  # 记录结束时间
            # print(f"回合时长: {episode_end_time - episode_start_time} 秒")

            
            # print(t_bias)
            env.clear_render(t_bias=t_bias)
            t_bias += env.t


        training_end_time = time.time()  # 记录结束时间
        


    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt")
    finally:
        env.end_render() # 停止发送

        # import matplotlib.pyplot as plt
        # r_action_arrays = np.array(r_action_list)
        # b_action_arrays = np.array(b_action_list)
        # r_guide_arrays = np.array(r_guide_list)
        # b_guide_arrays = np.array(b_guide_list)

        # # 绘制红方和蓝方的动作
        # x_b = b_action_arrays[:, 0].astype(float)
        # y_b = b_action_arrays[:, 1].astype(float)
        
        # x_r = r_action_arrays[:, 0].astype(float)
        # y_r = r_action_arrays[:, 1].astype(float)

        # x_bg = b_guide_arrays[:, 0].astype(float)
        # y_bg = b_guide_arrays[:, 1].astype(float)
        # x_rg = r_guide_arrays[:, 0].astype(float)
        # y_rg = r_guide_arrays[:, 1].astype(float)

        # fig, (ax, ax_guide) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # ax.plot(x_b, y_b, marker='o', linestyle='-', label='Blue (Action Type)', color='blue', alpha=0.7)
        # ax.plot(x_r, y_r, marker='x', linestyle='--', label='Red (Action Type)', color='red', alpha=0.7)
        
        # ax.set_ylabel('Action Label')
        # ax.set_title('Red & Blue Action Type over time')
        # ax.legend()
        # ax.grid(True)

        # # 绘制制导状态
        # ax_guide.plot(x_bg, y_bg, label='Blue Guidance Status', color='blue', alpha=0.7)
        # ax_guide.plot(x_rg, y_rg, label='Red Guidance Status', color='red', alpha=0.7)
        # ax_guide.set_xlabel('time (s)')
        # ax_guide.set_ylabel('Guidance Status')
        # ax_guide.set_title('Red & Blue Guidance Status (can_guide) over time')
        # ax_guide.legend()
        # ax_guide.grid(True)

        # # 自定义 x 轴刻度：每 10s 一个刻度；若刻度能被60整除，额外在刻度下方显示整除后的结果（分钟数），
        # # 否则显示该刻度除以60后的余数（秒）
        # step = 10
        # xmin = min(x_b.min(), x_r.min(), x_rg.min(), x_bg.min())
        # xmax = max(x_b.max(), x_r.max(), x_rg.max(), x_bg.max())
        # ticks = np.arange(np.floor(xmin / step) * step, np.ceil(xmax / step) * step + 1, step)
        # labels = []
        # for t in ticks:
        #     ti = int(round(t))
        #     if ti % 60 == 0:
        #         labels.append(f"{ti}\n{ti//60}")
        #     else:
        #         labels.append(str(ti % 60))
        # ax_guide.set_xticks(ticks)
        # ax_guide.set_xticklabels(labels)

        # plt.tight_layout()
        # plt.show()
        # print()
