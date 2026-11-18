import numpy as np
from math import *
import sys
import os
import time
import jsbsim


class PositionPID(object):
    """位置式PID算法实现"""

    # output_list = []
    def __init__(self, max, min, p, i, d) -> None:
        self._max = max  # 最大输出限制，规避过冲
        self._min = min  # 最小输出限制
        self.k_p = p  # 比例系数
        self.k_i = i  # 积分系数
        self.k_d = d  # 微分系数
        self._pre_error = 0  # t-1 时刻误差值
        self._integral = 0  # 误差积分值

    def calculate(self, error, dt, d_error=None, i_error=None):
        """
        计算t时刻PID输出值cur_val
        """
        # 比例项
        p_out = self.k_p * error
        # 积分项
        if i_error is None:  # 不可直接获取，再做积分
            self._integral += (error * dt)
        else:
            self._integral = i_error

        # 仿照simple_pid，将积分项整项预先限幅
        self._integral = np.clip(self._integral, self._min / self.k_i, self._max / self.k_i) \
            if self.k_i != 0 else self._integral

        i_out = self.k_i * self._integral
        # 微分项
        if d_error is None:  # 不可直接获取，再做差分
            derivative = (error - self._pre_error) / dt
        else:
            derivative = d_error
        d_out = self.k_d * derivative
        # t 时刻pid输出
        output = p_out + i_out + d_out
        # 限制输出值
        if output > self._max:
            output = self._max
        elif output < self._min:
            output = self._min
        self._pre_error = error
        return output

    def clear_integral(self):
        self._integral = 0


def sub_of_radian(input1, input2=0):
    # 计算两个弧度的差值，范围为[-pi, pi]
    diff = input1 - input2
    diff = (diff + pi) % (2 * pi) - pi
    return diff


def sub_of_degree(input1, input2=0):
    # 计算两个角度的差值，范围为[-180, 180]
    diff = input1 - input2
    diff = (diff + 180) % 360 - 180
    return diff


class F16ChaiLQR:
    """
    F-16 控制器（Chai 等 2023）：
    外环由高度误差解算期望俯仰角、由航向误差通过 PD 解算期望滚转角；
    内环采用论文式(19)的线性状态反馈（LQR）。
    """

    def __init__(self):
        # 油门：沿用原控制器的速度 PID
        self.t_pid = PositionPID(max=1, min=0, p=1, i=0.3, d=0.2)
        # 航向角误差 -> 期望滚转角：PD 转换
        # p = 85/90：|Δψ| = 90° 时 φ_req = 85°，90° 以内线性，输出限幅即 ±85° 饱和
        self.phi_pid = PositionPID(max=np.deg2rad(85), min=-np.deg2rad(85),
                                   p=85 / 90, i=0.0, d=0.2)

        # 式(19) 俯仰通道：xθ = [eθ, vt, α, Q] -> δe
        # 原文升降舵行写成 xφ（下标交叉），这里按物理通道取俯仰状态；
        # JSBSim 中正 elevator-cmd-norm 为低头，故整体取反，并把 Q 反馈调成阻尼
        # 整定值：整行按 cmd-norm 尺度重标，并加入小配平偏置抵消巡航低头力矩
        self.k_elevator = np.array([-13.028, 0.00973, -7.789, 2.444])
        self.trim_elev = -0.0058
        # 式(19) 滚转/偏航通道：xφ = [β, eφ, P, R] -> [δa, δr]
        # 原文副翼/方向舵行写成 xθ（下标交叉），这里按物理通道取滚转状态；
        # 副翼的 P 反馈按 JSBSim 正舵偏=右滚调成阻尼，其余增益保持原文
        # 整定值：原文行增益分别按 0.140、0.812 缩放，匹配 F-16 舵面灵敏度；
        # 速率/侧滑反馈在 flight_output 中按航向误差调度
        self.k_aileron_hi = np.array([-3.929, 1.367, -1.176, 0.857])
        self.k_rudder_hi = np.array([0.586, -1.021, -1.113, 7.463])
        self.k_aileron = self.k_aileron_hi.copy()
        self.k_rudder = self.k_rudder_hi.copy()
        # 调度门限：|Δψ| >= 10° 时满增益，接近目标时关掉速率/侧滑反馈以抑制抖振
        self.a_switch = np.deg2rad(10.0)
        # 俯仰指令限幅与高度环增益：原式直接用 ±90°，会让升降舵长期饱和
        self.theta_max = np.deg2rad(75.0)
        self.k_h = 1.60
        # 航迹角阻尼：theta_max 放大到 75° 后，单靠比例高度环会给出过大俯仰指令
        self.k_gamma = 1.60
        # 俯仰指令速率限幅：限制 Nz 峰值，避免大俯仰指令阶跃直接拉出十几 g
        self.theta_rate_max = np.deg2rad(6.0)
        self._theta_cmd = 0.0
        # 过载保护：Nz 估计超过该值后按比例收回拉杆指令
        self.nz_limit = 8.5
        self.nz_soft = 1.0

    def flight_output(self, state_input, dt=0.02):
        # 外环：高度误差 -> 期望俯仰角（沿用原控制器）
        error_h = np.clip(state_input[0] - state_input[13], -1, 1)
        # 限幅到 ±theta_max，避免 ±90° 的俯仰指令使升降舵整段饱和
        theta_cmd = error_h * self.k_h
        theta_damp = theta_cmd - self.k_gamma * state_input[11]
        # 阻尼只允许减小俯仰指令幅值，不允许由俯冲反向拉杆，避免过载尖峰
        theta_req = np.clip(theta_damp, min(0.0, theta_cmd), max(0.0, theta_cmd))
        theta_req = np.clip(theta_req, -self.theta_max, self.theta_max)
        # 俯仰指令速率限幅，平滑初始/目标切换时的大指令阶跃
        d_theta = np.clip(theta_req - self._theta_cmd,
                          -self.theta_rate_max * dt, self.theta_rate_max * dt)
        self._theta_cmd += d_theta
        theta_req = self._theta_cmd

        # 外环：航向角误差 -> 期望滚转角，另用一个 PD 控制器转换
        delta_heading_req = state_input[1]
        yaw_rate = state_input[10]
        # 按 |航向误差| 线性调度滚转通道的速率/侧滑反馈，保留 eφ 比例反馈
        g = min(1.0, abs(delta_heading_req) / self.a_switch)
        self.k_aileron = np.array([self.k_aileron_hi[0] * g, self.k_aileron_hi[1],
                                   self.k_aileron_hi[2] * g, self.k_aileron_hi[3] * g])
        self.k_rudder = np.array([self.k_rudder_hi[0] * g, self.k_rudder_hi[1],
                                  self.k_rudder_hi[2] * g, self.k_rudder_hi[3] * g])
        phi_req = self.phi_pid.calculate(delta_heading_req, dt=dt, d_error=-yaw_rate)

        return self.att_calculate(theta_req, phi_req, state_input, dt=dt)

    def att_calculate(self, theta_req, phi_req, state_input, dt):
        theta = state_input[3]
        phi = state_input[5]
        v = state_input[4]
        alpha = state_input[6]
        beta = state_input[7]
        p_rate = state_input[8]
        q_rate = state_input[9]
        r_rate = state_input[10]
        v_req = state_input[2]

        # 油门控制
        throttle = self.t_pid.calculate(v_req - v, dt)

        # 式(19) 升降舵
        x_pitch = np.array([theta_req - theta, v, alpha, q_rate])
        elevator = np.clip(self.k_elevator.dot(x_pitch) + self.trim_elev, -1, 1)

        # 过载保护：Nz ≈ cosθ + V·q/g，超限时线性收回拉杆（负舵偏为抬头）
        nz_est = np.cos(theta) + v * 340 * q_rate / 9.81
        if nz_est > self.nz_limit and elevator < 0:
            elevator *= max(0.0, 1.0 - (nz_est - self.nz_limit) / self.nz_soft)

        # 式(19) 副翼与方向舵
        x_roll = np.array([beta, phi_req - phi, p_rate, r_rate])
        aileron = np.clip(self.k_aileron.dot(x_roll), -1, 1)
        rudder = np.clip(self.k_rudder.dot(x_roll), -1, 1)

        return np.array([aileron, elevator, rudder, throttle])

    def visualize(self, tacview, current_time, lon, lat, alt, phi, theta, psi):
        """统一可视化入口：tacview_show==1 发送给 Tacview 软件，==2 由 Tacview_s 静默写入 ACMI 文件"""
        send_t = f"{current_time:.2f}"
        name_R = '001'
        loc_r = [float(lon), float(lat), float(alt)]
        # 画飞机
        data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=F16,Color=Red\n" % (
            float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], phi, theta, psi)
        tacview.send_data_to_client(data_to_send)


if __name__ == '__main__':

    tacview_show = 2  # 0: 不显示, 1: 连接 Tacview, 2: 静默写入本地 ACMI

    dt = 0.02  # 0.02

    if tacview_show:
        # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Visualize.tacview_visualize2 import Tacview, Tacview_s
        from Math_calculates.CartesianOnEarth import NUE2LLH, LLH2NUE
        if tacview_show == 2:
            # 静默写入本地 ACMI 文件，不连接 Tacview 软件
            tacview = Tacview_s()
            tacview.handshake()
        else:
            print('please prepare tacview')
            tacview = Tacview()
            tacview.handshake()

    # 记录轨迹和状态数据
    psis = []
    thetas = []
    phis = []
    heights = []
    alphas = []
    betas = []
    speeds = []
    overloads = []
    thrusts = []  # 添加推力数据记录
    fuel_flows = []
    fuels = []

    time_steps = []
    # 记录控制量
    aileron_cmd = []
    elevator_cmd = []
    rudder_cmd = []
    throttle_cmd = []

    # 启动 JSBSim
    sim = jsbsim.FGFDMExec(None, None)
    sim.set_debug_level(0)
    sim.set_dt(dt)  # 解算步长 dt 秒

    # 设置模型路径（一般 JSBSim pip 包自动包含）
    sim.load_model("f16")  # f15, p51d, ball 等模型可选

    # 初始姿态与目标姿态
    # 连续输出并tacview中可视化
    start_time = time.time()
    # target_theta = 1 # 测试姿态控制
    target_height = 2e3  # m # 测试飞行控制器
    target_heading = 270  # 度 to rad
    target_speed = 340 * 1.13  # m/s
    t_last = 60 * 5

    # 设置初始状态（单位：英尺、节、角度）
    sim["ic/h-sl-ft"] = 4e3 * 3.2808  # 高度：m -> ft
    sim["ic/vt-kts"] = 340 * 1.13 * 1.9438  # 空速： m/s-> 节
    sim["ic/psi-true-deg"] = 90  # 航向角: °
    sim["ic/phi-deg"] = 0
    sim["ic/theta-deg"] = 0
    sim["ic/alpha-deg"] = 0
    sim["ic/beta-deg"] = 0
    sim["ic/long-gc-deg"] = -118  # 设置初始经度（单位：度）
    sim["ic/lat-gc-deg"] = 34  # 设置初始纬度（单位：度）

    # 设置初始油量(实际燃油会按照F-16的最大容量提供，会小于这个设定值)
    sim["propulsion/tank[0]/contents-lbs"] = 5000.0  # 设置0号油箱油量
    sim["propulsion/tank[1]/contents-lbs"] = 5000.0  # 设置1号油箱油量（如果有）

    # 初始化状态
    sim.run_ic()
    sim.set_property_value('propulsion/set-running', -1)
    # 直接舵偏指令：式(19)按舵偏角设计，绕过模型自带的滚转/俯仰速率回路
    sim.set_property_value('fcs/fbw-override', 1)

    # 指数滑动平均控制量
    aileron_last = 0
    elevator_last = 0
    rudder_last = 0
    throttle_last = 0.5

    f16ChaiLQR = F16ChaiLQR()

    # hist_act=np.array([0,0,0,1])
    for step in range(int(t_last / dt)):
        current_t = step * dt

        # 无限燃油
        sim["propulsion/tank[0]/contents-lbs"] = 5000.0  # 设置0号油箱油量
        sim["propulsion/tank[1]/contents-lbs"] = 5000.0  # 设置1号油箱油量（如果有）

        # target_heading = np.random.rand()*10

        # # 舞狮
        # if current_t < 15:
        #     target_height = 5000  # m
        #     target_heading = 90  # 度 to rad
        #     target_speed = 300
        # elif current_t < 1 * 60:
        #     target_height = 10000  # m
        #     target_heading = -120  # 度 to rad
        # elif current_t < 1 * 60 + 27:
        #     target_height = 7000  # m
        #     target_heading = 0  # 度 to rad
        # elif current_t < 2 * 60 + 10:
        #     target_height = 8000  # m
        #     target_heading = sub_of_degree(sim["attitude/psi-deg"], 60)  # 度 to rad
        # else:
        #     target_heading = sub_of_degree(sim["attitude/psi-deg"], -10)

        sim.run()
        current_time = step * dt
        time_steps.append(current_time)

        # delta_height = (target_height - sim["position/h-sl-ft"]) / 3.2808
        delta_heading = sub_of_degree(target_heading, sim["attitude/psi-deg"])
        # # delta_heading = 19 * pi / 180 # test
        # delta_speed = (target_velocity - sim["velocities/vt-fps"] * 0.3048) / 1.9438

        # 取姿态角度
        phi = sim["attitude/phi-deg"]  # 滚转角 (roll)
        theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
        psi = sim["attitude/psi-deg"]  # 航向角 (yaw)
        alpha = sim["aero/alpha-deg"]  # 迎角
        beta = sim["aero/beta-deg"]  # 侧滑角
        # 过载量
        nz_g = sim["accelerations/Nz"]  # 垂直过载
        ny_g = sim["accelerations/Ny"]  # 侧向过载
        nx_g = sim["accelerations/Nx"]  # 纵向过载

        # 角速度
        p = sim["velocities/p-rad_sec"]  # 横滚角速度（弧度/秒）
        q = sim["velocities/q-rad_sec"]  # 俯仰角速度（弧度/秒）
        r = sim["velocities/r-rad_sec"]  # 偏航角速度（弧度/秒）

        # 速度矢量关于地面的角度
        vn = sim["velocities/v-north-fps"]  # 向北分量
        ve = sim["velocities/v-east-fps"]  # 向东分量
        vu = -sim["velocities/v-down-fps"]  # 向下分量（正表示下降）

        gamma_angle = atan2(vu, sqrt(vn ** 2 + ve ** 2)) * 180 / pi  # 爬升角（度）
        course_angle = atan2(ve, vn) * 180 / pi  # 航迹角 地面航向（度）速度矢量在地面投影与北方向的夹角

        # 构建观测向量
        obs_jsbsim = np.zeros(15)
        # obs_jsbsim[0] = target_theta * pi / 180  # 期望俯仰角 # 测试姿态控制器
        obs_jsbsim[0] = target_height / 5000  # 期望高度 # 测试飞行控制器
        obs_jsbsim[1] = delta_heading * pi / 180  # 期望相对航向角
        obs_jsbsim[2] = target_speed / 340  # 期望速度
        obs_jsbsim[3] = sim["attitude/theta-deg"] * pi / 180  # 当前俯仰角
        obs_jsbsim[4] = sim["velocities/vt-fps"] * 0.3048 / 340  # 当前速度
        obs_jsbsim[5] = sim["attitude/phi-deg"] * pi / 180  # 当前滚转角
        obs_jsbsim[6] = sim["aero/alpha-deg"] * pi / 180  # 当前迎角
        obs_jsbsim[7] = sim["aero/beta-deg"] * pi / 180  # 当前侧滑角
        obs_jsbsim[8] = p
        obs_jsbsim[9] = q
        obs_jsbsim[10] = r
        obs_jsbsim[11] = gamma_angle * pi / 180  # 爬升角
        obs_jsbsim[12] = sub_of_degree(target_heading, course_angle) * pi / 180  # 相对航迹角
        obs_jsbsim[13] = sim["position/h-sl-ft"] * 0.3048 / 5000  # 高度/5000（英尺转米）
        obs_jsbsim[14] = ny_g
        # obs_jsbsim[15] = psi * pi / 180  # test

        # 输出姿态控制指令
        norm_act = f16ChaiLQR.flight_output(obs_jsbsim, dt=dt)

        # # 指数滑动平均控制量
        # last_control = np.array([aileron_last, elevator_last, rudder_last, throttle_last])
        # norm_act=last_control*0.1+0.9*np.array(norm_act)

        sim["fcs/aileron-cmd-norm"], \
            sim["fcs/elevator-cmd-norm"], \
            sim["fcs/rudder-cmd-norm"], \
            sim["fcs/throttle-cmd-norm"] = norm_act  # .tolist()  # 设置控制量

        # # 记录控制量
        aileron_cmd.append(sim["fcs/aileron-cmd-norm"])
        elevator_cmd.append(sim["fcs/elevator-cmd-norm"])
        rudder_cmd.append(sim["fcs/rudder-cmd-norm"])
        throttle_cmd.append(sim["fcs/throttle-cmd-norm"])

        # 取当前位置
        lon = sim["position/long-gc-deg"]  # 经度
        lat = sim["position/lat-gc-deg"]  # 纬度
        alt = sim["position/h-sl-ft"] * 0.3048  # 高度（英尺转米）

        # 取速度分量
        u = sim["velocities/u-fps"] * 0.3048  # X轴速度 (fps转m/s)
        v = sim["velocities/v-fps"] * 0.3048  # Y轴速度 (fps转m/s)
        w = sim["velocities/w-fps"] * 0.3048  # Z轴速度 (fps转m/s)

        # 记录状态量
        phis.append(phi)
        psis.append(psi)
        thetas.append(theta)
        heights.append(alt)
        alphas.append(alpha)
        betas.append(beta)
        speeds.append(sim["velocities/vt-fps"] * 0.3048)
        overloads.append(sim["forces/load-factor"])
        thrust = sim.get_property_value('propulsion/engine/thrust-lbs')
        fuel_flow = sim["propulsion/engine/fuel-flow-rate-pps"]  # 燃油流量
        thrusts.append(thrust)
        fuel_flows.append(fuel_flow)

        fuels.append(sim["propulsion/total-fuel-lbs"])

        # 通过tacview可视化
        if tacview_show:  # and step % np.round(0.4 / dt) == 0:
            # 统一调用可视化方法，tacview_show==1/2 均走同一入口
            f16ChaiLQR.visualize(tacview, current_time, lon, lat, alt, phi, theta, psi)
            # # 画绣球
            # data_to_send += (
            #                 # f"#{send_t:.2f}\n"
            #                 f"{'002'},T={(lon_T):.6f}|{(lat_T):.6f}|{delta_H:.6f},"
            #                 f"Name=Target,Color=Orange\n"
            #             )
            # time.sleep(0.001)

        mach = sim["velocities/mach"]
        # # 可以记录或打印
        # print(f"Time: {current_time:.1f}s, Mach: {mach:.3f}")

        # time.sleep(0.01)

    # 结束可视化：连接 Tacview 时断开，静默写入时关闭并 flush ACMI 文件
    if tacview_show:
        tacview.end_render()

    end_time = time.time()
    print(f"程序运行时间: {end_time - start_time:.2f} 秒")
