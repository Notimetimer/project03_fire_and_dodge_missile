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


action_cycle_multiplier = 10

def basic_rules(state_check, rules_num, last_action=0, p_random=0):
    '''
    rules_num = 0: 保持和目标相同高度只打进攻(action_number = 0,1,3)，0平飞追踪 1爬升追踪 3下降追踪，攻击区内发射导弹，上一枚导弹发射后如果还在中制导，不发射新导弹
    rules_num = 1: 保持和目标相同高度进攻(0,1,3), 发射完导弹立马crank(5,6), 受到威胁立刻回转至5000m高度以下(11水平回转, 12俯冲回转), 威胁结束后回归进攻
    rules_num = 2: 保持和目标相同高度打首轮进攻(0,1,3), 在距离40km以外先爬升60°(2), 一个决策回合后射击, 否则直接射击，设计后立刻crank(6)，
        收到威胁立刻splitS(8), 威胁解除后回转进攻(0,1,3)
    '''

    delta_theta = state_check["target_information"][2] # 目标相对俯仰角
    distance = state_check["target_information"][3] # 距离
    delta_alt = distance*sin(delta_theta)  # 目标相对高度
    d_hor, leftright = state_check["border"]
    speed = state_check["ego_main"][0]
    alt = state_check["ego_main"][1] # 我机高度
    cos_delta_psi = state_check["target_information"][0]
    sin_delta_psi = state_check["target_information"][1]
    delta_psi = atan2(sin_delta_psi, cos_delta_psi)
    delta_psi_threat = atan2(state_check["threat"][1], state_check["threat"][0])
    RWR = state_check["warning"] # 受到威胁标志
    threat_distance = state_check["threat"][3] # 威胁距离
    on_guiding = state_check["missile_in_mid_term"] # 中制导状态标志
    t_fired = state_check["weapon"] # 导弹发射后计时，<12s不允许发射新导弹
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
    case1 = distance < 95e3 and (sin_theta >= sin(30*pi/180) or alt >= 9e3) # 105, 30
    case2 = distance < 80e3 # 75e3
    if (case1 or case2) and ATA < 60 * pi/180 and abs(delta_psi) < 30*pi/180:
        if t_fired >= 40 and not on_guiding and not (distance>12e3 and abs(AA_hor) < 30*pi/180):
            fire_missile = True

    # # # 2. 根据目标相对高度选择基础进攻机动
    # if delta_theta < -15 * pi/180:
    #     action_v = 3 # 下降
    #     action_h = 0 # 追踪
    # elif delta_theta >= 15 * pi/180:
    #     action_v = 1 # 爬升
    #     action_h = 0 # 追踪
    # else:
    action_v = 2 # 追踪
    action_h = 0 # 追踪
    action_number = [action_v, action_h] # 默认执行基础进攻
    base_offensive_action = action_number

    # # # 2. 根据目标相对高度选择基础进攻机动
    # if abs(delta_theta) < 30:
    #     base_offensive_action = 0  # 平飞追踪
    # elif delta_theta >= 30:
    #     base_offensive_action = 1  # 爬升追踪
    # else: # delta_theta < -pi/6
    #     base_offensive_action = 3  # 下降追踪

    action_number = base_offensive_action # 默认执行基础进攻

    # 3. 根据规则编号(rules_num)决定最终机动和开火决策
    if rules_num == 0:
        # 规则0: 纯进攻
        action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    elif rules_num == 1:
        # 规则1: 带防御机动
        if RWR and threat_distance < threat_distance_list[rules_num]: # 受到威胁
            # 优先俯冲回转至5000m以下
            if alt > 5000:
                action_v = 4 # 快速下高
            else:
                action_v = 2 # 平飞
            # 置尾机动
            action_h = 3 # 水平方向背离
            # # 优先俯冲回转至5000m以下
            # if alt > 5000:
            #     action_number = 12 # 俯冲回转
            # else:
            #     action_number = 11 # 水平回转
            # # fire_missile = False # 防御时不发射
        elif on_guiding: # 如果本回合决定发射导弹
            if delta_psi < 0:
                action_v = 2 # 平飞
                action_h = 5 # Rcrank
            else:
                action_v = 2 # 平飞
                action_h = 1 # Lcrank
            # action_number = 6 if delta_psi < 0 else 5 # random.choice([5,6]) # 立刻crank
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile
        action_number = [action_v, action_h]

    elif rules_num == 2:
        # 规则2: Loft爬升射击序列
        if RWR and threat_distance < threat_distance_list[rules_num]: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            action_number = [action_v, action_h]
            # action_number = 8 # 立刻 split-S
            fire_missile = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离，或上一回合是爬升
            if delta_psi < 0:
                action_v = 2 # 平飞
                action_h = 5 # Rcrank
                action_number = [action_v, action_h]
                # action_number = 6 # 右crank
            else:
                action_v = 2 # 平飞
                action_h = 1 # Lcrank
                action_number = [action_v, action_h]
                # action_number = 5 # 左crank
            # action_number = 6 if delta_psi < 0 else 5 # random.choice([5,6]) # 立刻crank
        elif fire_missile and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:  # last_action != 2: # 如果上一动作为非爬升
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                # action_number = 2 # 则本回合执行爬升
                fire_missile = False
                action_number = [action_v, action_h]
            else:
                fire_missile = True
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    elif rules_num == 3:
        if RWR: #  and threat_distance < threat_distance_list[rules_num]: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            action_number = [action_v, action_h]
            fire_missile = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离，或上一回合是爬升
            if delta_psi < 0:
                action_v = 3 # 下降
                action_h = 5 # Rcrank
                action_number = [action_v, action_h]
            else:
                action_v = 3 # 下降
                action_h = 1 # Lcrank
                action_number = [action_v, action_h]
        elif fire_missile and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:  # last_action != 2: # 如果上一动作为非爬升
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_missile = False
                action_number = [action_v, action_h]
            else:
                fire_missile = True
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile
    
    elif rules_num == 4:
        # 规则4: 只知道进攻和防御，不知道crank
        if RWR and threat_distance < threat_distance_list[rules_num]: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            action_number = [action_v, action_h]
            fire_missile = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离
            action_v = 2 # 平飞
            action_h = 0 # 追踪，不crank
            action_number = [action_v, action_h]
        elif fire_missile and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_missile = False
                action_number = [action_v, action_h]
            else:
                fire_missile = True
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    elif rules_num == 5:
        # 规则5: 在rule2基础上，leftright>0时左crank，否则右crank
        if RWR and threat_distance < threat_distance_list[rules_num]: # 受到威胁
            action_v = 4 # 下降高度
            action_h = 3 # 置尾机动
            action_number = [action_v, action_h]
            fire_missile = False # 防御时不发射
        elif on_guiding: # 满足开火条件但在中近距离
            if delta_psi < 0:
                action_v = 3 # 下降
                action_h = 5 # Rcrank
                action_number = [action_v, action_h]
            else:
                action_v = 3 # 下降
                action_h = 1 # Lcrank
                action_number = [action_v, action_h]
            # 保留战略纵深
            if d_hor < 50e3:
                if leftright > 0:
                    action_v = 2 # 平飞
                    action_h = 1 # Lcrank
                    action_number = [action_v, action_h]
                if leftright < 0:
                    action_v = 2 # 平飞
                    action_h = 5 # Rcrank
                action_number = [action_v, action_h]
        elif fire_missile and distance > 40e3: # 满足开火条件且在远距离
            if sin_theta < sin(30*pi/180) and alt < 9500:
                action_v = 0 # 爬升
                action_h = 0 # 追踪
                fire_missile = False
                action_number = [action_v, action_h]
            else:
                fire_missile = True
        else:
            action_number = base_offensive_action
        fire_missile_affirmative = fire_missile

    if np.random.rand() <= p_random:
        # 再出现动作数值越界改这里
        v_action = np.random.randint(0, 5)
        h_action = np.random.randint(0, 6) # 7

        action_number = [v_action, h_action]
        # action_number = np.random.randint(0, 13+1)
        # action_number = np.clip(action_number, 0, 13)
    
    # if rules_num in [3, 4]:
    #     # 不准出界
    #     if d_hor < 8e3:
    #         action_number = base_offensive_action
    
    # # 防撞地规则
    # if alt < 3000:
    #     if action_number in [0, 3, 4]:
    #         action_number = 1
    #     if action_number in [8,11,12,13]:
    #         action_number = 11
    # # 防破升限
    # if alt > 12000:
    #     if action_number in [1,2]:
    #         action_number = 0

    # print(f"[basic_rules] rules_num={rules_num}, RWR={int(RWR) if 'RWR' in locals() else 'N/A'}, on_guiding={int(on_guiding) if 'on_guiding' in locals() else 'N/A'}, fire_missile={int(fire_missile) if 'fire_missile' in locals() else 'N/A'}, action_v={action_v}, action_h={action_h}, action_number={action_number}")
    return np.array(action_number), fire_missile_affirmative


def run_single_test(red_height, blue_height, horizontal_distance, AA_hor=0, env=None, use_tacview=0):
    """
    单次测试：红方rule2(1枚导弹，进场立即开火) vs 蓝方rule2(无导弹)
    初始态势：双机对头，红方在[0, h, 0]向东飞，蓝方在[0, h2, d]向西飞(可偏AA_hor)
    
    Args:
        red_height: 红方初始高度 (m)
        blue_height: 蓝方初始高度 (m)
        horizontal_distance: 初始水平距离 (m)
        AA_hor: 水平进入角偏移 (rad)，默认0表示正对头，正值表示蓝方航向偏转
        env: 可选传入环境实例，若为None则新建
        use_tacview: 是否使用tacview可视化
        
    Returns:
        hit: 1表示蓝方被击中，0表示蓝方未被击中
        t_end: 仿真结束时间
    """
    from Envs.battle6dof1v1_missile0309_hierarchical import dt_maneuver, launch_missile_immediately
    
    # 创建初始状态
    # 红方在西边[0, h, 0]面向东，蓝方在东边[0, h2, d]面向西，双机对头
    DEFAULT_RED_BIRTH_STATE = {
        'position': np.array([0.0, red_height, 0.0]),  # [0, h, 0] 西边
        'psi': pi/2,  # 面向东 (z轴正方向)
        'e2e': False
    }
    DEFAULT_BLUE_BIRTH_STATE = {
        'position': np.array([0.0, blue_height, horizontal_distance]),  # [0, h2, d] 东边
        'psi': sub_of_radian(pi/2 + AA_hor,0),  # 面向西，加上AA_hor偏移
        'e2e': False
    }
    
    # 创建环境
    if env is None:
        import argparse
        parser = argparse.ArgumentParser("UAV swarm confrontation")
        parser.add_argument("--max-episode-len", type=float, default=120.0, help="maximum episode time length")
        args = parser.parse_args([])
        args.R_cage = 100e3  # 100km边界
        env = ChooseStrategyEnv(args, tacview_show=use_tacview, vertices=None)
        env.dt_move = 0.04
        env.shielded = 1
    
    # 重置环境：红方1枚导弹，蓝方0枚导弹
    env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, 
              blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
              red_init_ammo=1, blue_init_ammo=0)
    
    action_cycle_multiplier = 10
    dt_action_cycle = dt_maneuver * action_cycle_multiplier
    
    r_action_label = np.array([2, 0])  # 默认动作
    b_action_label = np.array([2, 0])
    
    done = False
    steps_of_this_eps = -1
    
    # 红方立即开火标志（仅在开始时执行一次）
    red_first_fire = True
    
    while env.t < 120.0 and not done and env.running:
        steps_of_this_eps += 1
        
        # 获取观测信息
        r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
        b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
        
        # 决策周期判断
        if steps_of_this_eps % action_cycle_multiplier == 0:
            # 红方：rule2
            r_state_check = env.unscale_state(env.obs2obs_check(r_obs))
            r_action_label, r_fire = basic_rules(r_state_check, rules_num=2)
            
            # 蓝方：rule2
            b_state_check = env.unscale_state(env.obs2obs_check(b_obs))
            b_action_label, b_fire = basic_rules(b_state_check, rules_num=2)
        
        # 红方第一次立即开火
        if red_first_fire and env.RUAV.ammo > 0 and not env.RUAV.dead and env.t > 3:
            launch_missile_immediately(env, 'r', tabu=0, action_label=None)
            red_first_fire = False
        
        # 执行动作
        r_action = env.maneuver14LR(env.RUAV, r_action_label)
        b_action = env.maneuver14LR(env.BUAV, b_action_label)
        
        env.step(r_action, b_action)
        
        # 检查终止条件
        done, _, _, _ = env.combat_terminate_and_reward('b', b_action_label, False)
        
        # 可视化
        if use_tacview:
            env.render(t_bias=0)
        
        # 检查时间限制
        if env.t >= 120.0:
            break
    
    # 判定结果：蓝方是否被击中
    hit = 1 if env.BUAV.got_hit else 0
    t_end = env.t
    
    if use_tacview:
        env.clear_render(t_bias=0)
    
    return hit, t_end


if __name__=='__main__':
    print("\n根目录为：", project_root, "\n")
    import csv
    from itertools import product
    import time
    
    # 参数范围（红蓝高度相同）
    heights = np.arange(2000, 10001, 1000)          # 2000到10000，间隔1000
    distances = np.arange(20e3, 81e3, 10e3)         # 20km到80km，间隔10km
    enter_angle = np.radians(180) # 0
    use_tacview = 0  # 批量仿真关闭可视化
    
    # 创建结果列表
    results = []
    total_cases = len(heights) * len(distances)
    case_num = 0
    
    print(f"开始蒙特卡洛仿真，共 {total_cases} 组参数...")
    start_time = time.time()
    
    # 遍历所有参数组合（红蓝高度相同）
    for h, distance in product(heights, distances):
        case_num += 1
        
        # 执行仿真（红蓝高度相同）
        hit, t_end = run_single_test(h, h, distance, AA_hor=enter_angle, env=None, use_tacview=use_tacview)
        
        # 记录结果
        results.append({
            'height_m': h,
            'distance_km': distance / 1000,
            'hit': hit,
            't_end': t_end
        })
        
        # 进度打印
        if case_num % 10 == 0 or case_num == total_cases:
            elapsed = time.time() - start_time
            print(f"  进度: {case_num}/{total_cases} ({100*case_num/total_cases:.1f}%), 用时: {elapsed:.1f}s")
    
    # 保存到CSV
    csv_filename = 'monte_carlo_results.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['height_m', 'distance_km', 'hit', 't_end'])
        writer.writeheader()
        writer.writerows(results)
    
    total_time = time.time() - start_time
    print(f"\n仿真完成！")
    print(f"  总用时: {total_time:.1f}s")
    print(f"  结果已保存至: {csv_filename}")
    print(f"  总命中次数: {sum(r['hit'] for r in results)}/{total_cases}")
    
    # 简单统计
    hits_by_distance = {}
    hits_by_height = {}
    for r in results:
        d = r['distance_km']
        h = r['height_m']
        if d not in hits_by_distance:
            hits_by_distance[d] = [0, 0]
        hits_by_distance[d][0] += r['hit']
        hits_by_distance[d][1] += 1
        if h not in hits_by_height:
            hits_by_height[h] = [0, 0]
        hits_by_height[h][0] += r['hit']
        hits_by_height[h][1] += 1
    
    print(f"\n不同距离命中率统计:")
    for d in sorted(hits_by_distance.keys()):
        h, total = hits_by_distance[d]
        print(f"  {d}km: {h}/{total} = {100*h/total:.1f}%")
    
    print(f"\n不同高度命中率统计:")
    for h in sorted(hits_by_height.keys()):
        hit_count, total = hits_by_height[h]
        print(f"  {h}m: {hit_count}/{total} = {100*hit_count/total:.1f}%")
