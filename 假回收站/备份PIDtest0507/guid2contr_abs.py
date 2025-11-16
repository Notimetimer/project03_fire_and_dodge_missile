def guided_control(selected_uav, theta_req, psi_req, v_req, prev_error_x, prev_error_y, prev_error_z, integral_error_z,
                   dt: object) -> object:
    from math import pi
    import numpy as np
    theta_req = np.clip(theta_req, -pi / 2, pi / 2)

    # 调整控制参数（需要根据实际响应调整）
    k_thrust_p = 4  # 切向比例增益
    k_thrust_d = 2  # 切向微分增益
    k_lift_p = 0.33  # 俯仰比例增益
    k_lift_d = 0.04  # 俯仰微分增益
    k_roll_p = 0.44  # 滚转比例增益
    k_roll_i = 0.05  # 滚转积分增益
    k_roll_d = 0.06  # 滚转微分增益

    m = selected_uav.m
    v_ = selected_uav.vel_
    # # 航向飞行
    L_ = 10e3 * np.array(
        [np.cos(theta_req) * np.cos(psi_req), np.sin(theta_req), np.cos(theta_req) * np.sin(psi_req)])

    # 切向控制
    error_x = v_req - selected_uav.speed
    derivative_x = (error_x - prev_error_x) / dt  # 微分项
    ax_required = k_thrust_p * error_x + k_thrust_d * derivative_x
    # 将计算得到的切向加速度需求裁剪到 [-1, 1] 范围内，避免加速度值过大
    ax_required = np.clip(ax_required, -1, 1)
    # 对裁剪后的切向加速度需求进行归一化处理，得到切向控制动作，归一化是为了符合环境对动作输入的要求
    action_x_required = ax_required / selected_uav.nx_limit[1]
    prev_error_x = error_x  # 保存当前误差

    # 俯仰控制
    # 计算目标方向向量 L_ 和当前速度向量 v_ 之间的偏轴角
    offaxis_angle = np.arccos(np.dot(L_, v_) / (np.linalg.norm(L_) * np.linalg.norm(v_)))
    # 根据偏轴角计算俯仰控制动作，使用升力控制增益 k_lift 进行缩放
    error_y = offaxis_angle / pi
    derivative_y = (error_y - prev_error_y) / dt  # 微分项
    action_y_required = k_lift_p * error_y + k_lift_d * derivative_y
    # 将俯仰控制动作裁剪到 [-1, 1] 范围内
    action_y_required = np.clip(action_y_required, -1, 1)
    prev_error_y = error_y  # 保存当前误差

    # 滚转控制
    # y轴位于速度和目标线矢量围成的平面内并垂直于速度
    temp = np.cross(v_, L_)
    if np.linalg.norm(temp) == 0:
        action_z_required = 0
    else:
        # 计算速度向量 v_ 的单位向量，代表无人机当前速度方向
        x_b_ = v_ / np.linalg.norm(v_)
        # 计算 temp 和 v_ 的叉积，得到一个垂直于 temp 和 v_ 所构成平面的向量
        y_b_ = np.cross(temp, v_)
        # 将 y_b_ 转换为单位向量，用于后续的角度和方向计算
        y_b_ = y_b_ / np.linalg.norm(y_b_)

        # 把铅垂线往z轴的方向旋转theta从而和y_b_计算phi
        temp_ = temp / np.linalg.norm(temp) * selected_uav.theta  # 旋转矢量
        alpha = np.linalg.norm(temp_)
        up_ = np.array([0, 1, 0])
        up_temp_ = (1 - np.cos(alpha)) * np.dot(temp_, up_) * temp_ + np.cos(alpha) * up_ + np.sin(
            alpha) * np.cross(temp_, up_)  # 罗德里格斯旋转公式
        up_temp_ = up_temp_ / np.linalg.norm(up_temp_)
        abs_phi_req = np.arccos(np.dot(up_temp_, y_b_))
        if np.dot(np.cross(up_temp_, y_b_), x_b_) > 0:
            phi_req = abs_phi_req
        else:
            phi_req = -abs_phi_req
        current_gamma = selected_uav.gamma  # 获取当前实际滚转角
        error_z = (phi_req - current_gamma) / pi  # 误差计算改进
        integral_error_z += error_z * dt
        integral_error_z = np.clip(integral_error_z, -1.0, 1.0)  # 积分抗饱和
        derivative_z = (error_z - prev_error_z) / dt
        action_z_required = (
                k_roll_p * error_z +
                k_roll_i * integral_error_z +
                k_roll_d * derivative_z
        )
        action_z_required = np.clip(action_z_required, -1, 1)  # 输出限幅
        prev_error_z = error_z  # 保存当前误差

    r_action_n = [[action_x_required, action_y_required, action_z_required]]
    return r_action_n
