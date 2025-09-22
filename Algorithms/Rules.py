import numpy as np
from math import *
from numpy.linalg import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Math_calculates.sub_of_angles import *


def decision_rule(ego_pos_, ego_psi, enm_pos_, distance, ally_missiles, enm_missiles, o00, R_cage, wander=0):
    # 输出为所需的绝对高度、相对方位和绝对速度
    # 是否有导弹可用
    has_missile_in_the_air = 0
    for missile in ally_missiles:
        if not missile.dead:
            has_missile_in_the_air = 1
            break
    # 是否被敌机导弹锁定（存在敌机导弹而且距离在20km内，此时导弹雷达可以锁定）
    should_escape = 0
    rel_psi_m = None
    for missile in enm_missiles:
        if not missile.dead:
            line_m_ = missile.pos_ - ego_pos_
            dist_mt = norm(line_m_)
            if dist_mt <= 20e3:  # 假设为导弹雷达的锁定距离
                should_escape = 1
                rel_psi_m = sub_of_radian(atan2(line_m_[2], line_m_[0]), ego_psi)
                break  # trick：先发射的导弹被当做距离最近的，没有在所有导弹中判断距离 todo 判断距离

    action_n = [[0.0, 0.0, 400]]
    ego_height = ego_pos_[1]

    action_n[0][0] = 7e3-ego_height # 固定高度为7000m

    L_ = enm_pos_ - ego_pos_
    beta = atan2(L_[2], L_[0])
    delta_psi = sub_of_radian(beta, ego_psi)

    # 1 追踪
    if distance > 40e3:  # test
        action_n[0][1] = delta_psi

    # 2 crank
    elif not should_escape and has_missile_in_the_air:
        # print('crank')
        if 0 <= delta_psi:
            temp = 0.4 * (delta_psi - pi / 4) / (pi / 4) * 2
            action_n[0][1] = np.clip(temp, -0.4, 0.4)
        if delta_psi < 0:
            temp = 0.4 * (delta_psi + pi / 4) / (pi / 4) * 2
            action_n[0][1] = np.clip(temp, -0.4, 0.4)
        action_n[0][2] = 1.1*340  # 速度往1.1Ma去

    # 3 escape
    elif should_escape:
        # print('逃')
        # 水平置尾机动
        temp = 0.4 * sub_of_radian(rel_psi_m, pi) / pi * 4
        action_n[0][1] = np.clip(temp, -0.4, 0.4)
        action_n[0][2] = 1.5*340  # 速度往1.5Ma去

    # 4 attack again
    else:
        # print('回击')
        action_n[0][1] = delta_psi
        action_n[0][2] = 1.2*340

    # 追踪任务的目标在散步
    if wander:
        action_n[0][0] = 3000 * np.random.uniform(-1,1)
        action_n[0][1] = np.random.normal(0, 25*pi/180)

    # 最高优先级：不许出圈
    R_to_o00 = np.linalg.norm([ego_pos_[0], ego_pos_[2]])
    if R_cage-R_to_o00 < 8e3:
        beta_of_o00 = atan2(-ego_pos_[2], -ego_pos_[0])
        action_n[0][1] = sub_of_radian(beta_of_o00, ego_psi)
    if ego_height>13e3:
        action_n[0][0] = -5000
    elif ego_height<3e3:
        action_n[0][0] = 5000
    
    return action_n
