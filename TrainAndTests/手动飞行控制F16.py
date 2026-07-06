"手操飞机，查看控制量"

import jsbsim
import matplotlib.pyplot as plt
import math
import numpy as np
import socket
import threading
import time
import keyboard  # 需要安装: pip install keyboard
import pygame  # 需要安装: pip install pygame，用于Xbox手柄
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

tacview_show = True  # 是否显示Tacview
model_name = "f16"  # JSBSim模型名称, f15, f16等可用

class Tacview(object):
    def __init__(self):
        host = "localhost"
        port = 42674
        # host = input("请输入服务器IP地址：")
        # port = int(input("请输入服务器端口："))
        # 提示用户打开tacview软件高级版，点击"记录"-"实时遥测"
        print("请打开tacview软件高级版，点击\"记录\"-\"实时遥测\"，并使用以下设置：")
        print(f"IP地址：{host}")
        print(f"端口：{port}")

        # 创建套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host, port))

        # 启动监听
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")

        # 等待客户端连接
        client_socket, address = server_socket.accept()
        print(f"Accepted connection from {address}")

        self.client_socket = client_socket
        self.address = address

        # 构建握手数据
        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        # 发送握手数据
        client_socket.send(handshake_data.encode())


        # 接收客户端发送的数据
        data = client_socket.recv(1024)
        print(f"Received data from {address}: {data.decode()}")
        print("已建立连接")

        # 向客户端发送头部格式数据

        data_to_send = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                        "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
                        )
        client_socket.send(data_to_send.encode())

    def send_data_to_client(self, data):

        self.client_socket.send(data.encode())

if tacview_show:
    tacview = Tacview()

# 初始化pygame手柄
pygame.init()
pygame.joystick.init()
joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"检测到手柄: {joystick.get_name()}")
else:
    print("未检测到手柄，使用键盘控制")

t_total = 90  # 总时间（秒）
dt = 0.02

try:
    # 启动 JSBSim
    sim = jsbsim.FGFDMExec(None, None)
    sim.set_debug_level(0)
    sim.set_dt(dt)  # 解算步长 dt 秒

    # 设置模型路径（一般 JSBSim pip 包自动包含）
    sim.load_model(model_name) # f15, p51d, ball 等模型可选

    # 设置初始速率（单位：英尺、节、角度）
    sim[ "ic/vt-kts"     ] = 0 * 1.94384     # 空速m/s转换为节

    # 设置初始位置（单位：经度、纬度）
    # 注意：经度和纬度的单位是度，JSBSim使用的是地球坐标系
    sim[ "ic/long-gc-deg" ] = 116.0          # 经度
    sim[ "ic/lat-gc-deg"  ] = 39.0        # 纬度
    sim["ic/h-sl-ft"] = 0 * 3.28084  # 高度转换为英尺

    # 设置初始姿态（单位：度）
    sim[ "ic/psi-true-deg" ] = -60             # 航向角
    sim[ "ic/phi-deg"    ] = 0
    sim[ "ic/theta-deg"  ] = 10
    sim[ "ic/alpha-deg"  ] = 0
    sim[ "ic/beta-deg"   ] = 0
    # sim[ "ic/num-engines"] = 1
    # sim[ "propulsion/stationary-thrust-lbs" ] = 5000  # optional

    # 初始化状态
    sim.run_ic()

    # 设置引擎为开启
    # self.fdm["propulsion/active_engine"] = True
    # self.fdm["propulsion/starter_cmd"] = 1 似乎无效，换成下面这行
    sim.set_property_value('propulsion/set-running', -1)

    # # 或者尝试这些属性
    # sim["propulsion/engine[0]/starter"] = 1.0
    # sim["propulsion/engine[0]/cutoff"] = 0.0
    # sim["propulsion/tank[0]/contents-lbs"] = 10000  # 设置燃油量

    # 设置四轴舵量为 0（范围[-1,1]）
    sim["fcs/aileron-cmd-norm"] = 0.0 # 副翼 -左+右
    sim["fcs/elevator-cmd-norm"] = 0.0 # 升降舵,-拉杆+推杆
    sim["fcs/rudder-cmd-norm"] = 0.0 # 方向舵，-左+右
    sim["fcs/throttle-cmd-norm"] = 1.0 # 先设置小油门启动

    # 减速板，F16的效果看起来更像是襟翼， F15看着无效
    # sim["fcs/speedbrake-cmd-norm"] = 0.0   # 减速板收起
    # sim["fcs/speedbrake-cmd-norm"] = 1.0   # 减速板完全展开

    # 扰流板，无效
    # sim["fcs/spoiler-cmd-norm"] = 0.0      # 扰流板收起
    # sim["fcs/spoiler-cmd-norm"] = 1.0      # 扰流板展开

    # # 襟翼 (Flaps) - F16看起来无效
    # sim["fcs/flap-cmd-norm"] = 0.0         # 襟翼收起
    # sim["fcs/flap-cmd-norm"] = 1.0         # 襟翼完全展开

    # # 起落架 (Landing Gear) - 未测试
    # sim["gear/gear-cmd-norm"] = 0.0        # 起落架收起
    # sim["gear/gear-cmd-norm"] = 1.0        # 起落架放下

    # # 加力燃烧室 (Afterburner)，似乎不需要 
    # sim["fcs/ab-cmd-norm"] = 0.0           # 加力关闭
    # sim["fcs/ab-cmd-norm"] = 1.0           # 加力全开

    # 记录轨迹和状态数据
    positions = []
    attitudes = []
    velocities = []
    thrust_data = []  # 添加推力数据记录
    time_steps = []

    # 记录控制量
    aileron_cmd = []
    elevator_cmd = []
    rudder_cmd = []
    throttle_cmd_list = []

    # 记录过载量
    load_factors = []

    # 记录过载量
    load_factors = []

    flap_state = 0  # 襟翼状态
    break_state = 0  # 减速板状态

    for step in range(int(t_total / dt)):
        # 处理pygame事件
        pygame.event.pump()
        
        result = sim.run()
        if not result:
            print(f"模拟在第{step}步失败，时间{current_time:.1f}s")
            break
            
        current_time = step * dt
        
        time_steps.append(current_time)
        
        if step == 0:
            print(f"模拟开始，总时间{t_total}s，步长{dt}s")

        # 默认控制量
        sim["fcs/aileron-cmd-norm"] = 0.0 # 副翼 -左+右
        sim["fcs/elevator-cmd-norm"] = 0.0 # 升降舵,-拉杆+推杆
        sim["fcs/rudder-cmd-norm"] = 0.0 # 方向舵，-左+右
        throttle_cmd = 0.5  # 默认油门
        sim["fcs/throttle-cmd-norm"] = throttle_cmd
        sim["fcs/ab-cmd-norm"] = 0.0 # 加力关闭
        sim["fcs/speedbrake-cmd-norm"] = 0.0   # 减速板收起
        sim["fcs/flap-cmd-norm"] = 0.0 # 襟翼收起

        # 手柄控制（优先）
        if joystick:
            # 右摇杆控制升降舵和副翼
            right_stick_x = joystick.get_axis(2)  # 右摇杆X轴 -> 副翼
            right_stick_y = joystick.get_axis(3)  # 右摇杆Y轴 -> 升降舵
            # 左摇杆控制方向舵和油门
            left_stick_x = joystick.get_axis(0)   # 左摇杆X轴 -> 方向舵
            left_stick_y = joystick.get_axis(1)   # 左摇杆Y轴 -> 油门
            
            # 设置舵面（Xbox手柄摇杆范围[-1,1]）
            sim["fcs/aileron-cmd-norm"] = right_stick_x  # 右摇杆左右 -> 副翼
            sim["fcs/elevator-cmd-norm"] = -right_stick_y  # 右摇杆上下 -> 升降舵（反向）
            sim["fcs/rudder-cmd-norm"] = -left_stick_x   # 左摇杆左右 -> 方向舵
            # 左摇杆上下 -> 油门（上推增加油门，需要反向）
            throttle_cmd = 0.5 - left_stick_y * 0.5  # 映射到[0,1]范围
            throttle_cmd = max(0.0, min(1.0, throttle_cmd))  # 限制在[0,1]
            sim["fcs/throttle-cmd-norm"] = throttle_cmd
            
            # 手柄按钮控制
            if joystick.get_button(0):  # A键 -> 减速板
                sim["fcs/speedbrake-cmd-norm"] = 1.0
            if joystick.get_button(1):  # B键 -> 襟翼
                sim["fcs/flap-cmd-norm"] = 1.0
            if joystick.get_button(2):  # X键 -> 加力
                throttle_cmd = 1.5  # 油门大于1启用加力
                sim["fcs/throttle-cmd-norm"] = throttle_cmd

        # 键盘检测和控制量设置（备用）
        if keyboard.is_pressed('w'):
            sim["fcs/elevator-cmd-norm"] = 0.8  # 推杆
        if keyboard.is_pressed('s'):
            sim["fcs/elevator-cmd-norm"] = -0.8  # 拉杆
        if keyboard.is_pressed('a'):
            sim["fcs/aileron-cmd-norm"] = -1.0  # 左滚转
        if keyboard.is_pressed('d'):
            sim["fcs/aileron-cmd-norm"] = 1.0   # 右滚转
        if keyboard.is_pressed('j'):
            sim["fcs/rudder-cmd-norm"] = 1.0   # 左偏航
        if keyboard.is_pressed('l'):
            sim["fcs/rudder-cmd-norm"] = -1.0    # 右偏航
        if keyboard.is_pressed("i"): # ('shift'):
            throttle_cmd = 1.0  # 最大油门
            sim["fcs/throttle-cmd-norm"] = throttle_cmd
        if keyboard.is_pressed("k"): # ('ctrl'):
            throttle_cmd = 0.3  # 低油门
            sim["fcs/throttle-cmd-norm"] = throttle_cmd
        if keyboard.is_pressed('b'):
            sim["fcs/speedbrake-cmd-norm"] = 1.0  # 减速板展开，作用比起减速板更像是襟翼
        if keyboard.is_pressed('9'):
            throttle_cmd = 1.5  # 油门大于1启用加力
            sim["fcs/throttle-cmd-norm"] = throttle_cmd
        if keyboard.is_pressed('f'):
            sim["fcs/flap-cmd-norm"] = 1.0  # 襟翼完全展开

        # 记录控制量
        aileron_cmd.append(sim["fcs/aileron-cmd-norm"])
        elevator_cmd.append(sim["fcs/elevator-cmd-norm"])
        rudder_cmd.append(sim["fcs/rudder-cmd-norm"])
        throttle_cmd_list.append(throttle_cmd)
        
        # 调试输出 - 每0.5秒打印一次
        if step % np.round(0.5/dt) == 0:
            print(f"Cmd: ail={sim['fcs/aileron-cmd-norm']:.2f}, ele={sim['fcs/elevator-cmd-norm']:.2f}, rud={sim['fcs/rudder-cmd-norm']:.2f}")
            print(f"Pos: ail_l={sim['fcs/left-aileron-pos-norm']:.2f}, ele={sim['fcs/elevator-pos-norm']:.2f}, rud={sim['fcs/rudder-pos-norm']:.2f}")

        # 取当前位置
        lon = sim["position/long-gc-deg"]  # 经度
        lat = sim["position/lat-gc-deg"]   # 纬度
        alt = sim["position/h-sl-ft"] * 0.3048  # 高度（英尺转米）
        
        # 简单的相对位置计算
        if step == 0:
            start_lon, start_lat = lon, lat
        
        x = (lon - start_lon) * 111320  # 经度差转米（近似）
        y = (lat - start_lat) * 110540  # 纬度差转米（近似）
        z = alt
        positions.append((x, y, z))
        
        # 取姿态角度
        phi = sim["attitude/phi-deg"]      # 滚转角 (roll)
        theta = sim["attitude/theta-deg"]  # 俯仰角 (pitch)
        psi = sim["attitude/psi-deg"]      # 航向角 (yaw)
        alpha = sim["aero/alpha-deg"]      # 迎角
        beta = sim["aero/beta-deg"]        # 侧滑角
        attitudes.append((phi, theta, psi, alpha, beta))
        
        # 取速度分量
        u = sim["velocities/u-fps"] * 0.3048  # X轴速度 (fps转m/s)
        v = sim["velocities/v-fps"] * 0.3048  # Y轴速度 (fps转m/s)
        w = sim["velocities/w-fps"] * 0.3048  # Z轴速度 (fps转m/s)
        velocities.append((u, v, w))
        
        # 记录推力和发动机参数
        try:
            thrust=sim.get_property_value('propulsion/engine/thrust-lbs')
            fuel_flow = sim["propulsion/engine/fuel-flow-rate-pps"]  # 燃油流量
            total_speed = sim["velocities/vt-fps"] * 0.3048  # 总速度 (m/s)
            thrust_data.append((thrust, fuel_flow, total_speed))
            
            # 记录过载量
            N_front = sim["accelerations/Nx"]
            N_side = sim["accelerations/Ny"]
            N_normal = sim["accelerations/Nz"]
            load_factors.append((N_front, N_side, N_normal))
            
            # 打印关键参数
            if step % np.round(1/dt) == 0:  # 每1秒打印一次
                weight = sim["inertia/weight-lbs"]  # 飞机总重
                thrust_to_weight = thrust / weight  # 推重比
                print(f"Time: {current_time:.1f}s, Throttle: {throttle_cmd:.2f}, "
                    f"Thrust: {thrust:.0f} lbs (Eng0: {thrust:.0f}, Weight: {weight:.0f} lbs, T/W: {thrust_to_weight:.2f}, Speed: {total_speed:.1f} m/s, N_normal: {N_normal:.2f}, N_front: {N_front:.2f}, N_side: {N_side:.2f}")
                if np.linalg.norm(alpha)>15:
                    print(f"Warning: High angle of attack detected: {alpha:.2f} degrees")
                # 升力和阻力系数不能像下面这样获取
                # lift_coeff = sim["aero/CL"]
                # drag_coeff = sim["aero/CD"]
                # print(f"Lift Coefficient: {lift_coeff}, Drag Coefficient: {drag_coeff}")
        except Exception as e:
            print(f"推力数据记录错误: {e}")
            thrust_data.append((0, 0, 0))

        # 通过tacview可视化
        if tacview_show:
            send_t = f"{current_time:.2f}"
            name_R = '001'
            loc_r = [float(lon), float(lat), float(alt)]
            # data_to_send = f"#{send_t:.2f}\n{name_R},T={loc_r[0]:.6f}|{loc_r[1]:.6f}|{loc_r[2]:.6f},Name=F16,Color=Red\n"
            data_to_send = "#%.2f\n%s,T=%.6f|%.6f|%.6f|%.6f|%.6f|%.6f,Name=%s,Color=Red\n" % (float(send_t), name_R, loc_r[0], loc_r[1], loc_r[2], phi, theta, psi, model_name)
            tacview.send_data_to_client(data_to_send)
            time.sleep(0.02)
except Exception as e:
    print(f"模拟循环异常: {e}")
    import traceback
    traceback.print_exc()
finally:


    # 拆分数据
    if len(positions) == 0:
        print("警告：没有记录到任何数据，可能模拟运行时间太短")
        exit()
    
    x_vals, y_vals, z_vals = zip(*positions)
    phi_vals, theta_vals, psi_vals, alpha_vals, beta_vals = zip(*attitudes)
    u_vals, v_vals, w_vals = zip(*velocities)
    thrust_vals, fuel_vals, speed_vals = zip(*thrust_data)
    N_front_vals, N_side_vals, N_normal_vals = zip(*load_factors)
    throttle_vals = throttle_cmd_list



    pass

    # 创建一个大的figure，包含所有子图
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))

    # 第一行：位置、速度、迎角和侧滑角
    axes[0, 0].plot(time_steps, x_vals, label='East', color='tab:blue')
    axes[0, 0].plot(time_steps, y_vals, label='North', color='tab:green')
    axes[0, 0].plot(time_steps, z_vals, label='Height', color='tab:red')
    axes[0, 0].set_title('Position vs Time')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(time_steps, u_vals, label='East', color='tab:blue')
    axes[0, 1].plot(time_steps, v_vals, label='North', color='tab:green')
    axes[0, 1].plot(time_steps, w_vals, label='Height', color='tab:red')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[0, 2].plot(time_steps, alpha_vals, label='Alpha (AoA)', color='tab:orange')
    axes[0, 2].plot(time_steps, beta_vals, label='Beta (Sideslip)', color='tab:purple')
    axes[0, 2].set_title('Alpha & Beta vs Time')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Angle (deg)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 第二行：推力、速度、油耗
    axes[1, 0].plot(time_steps, thrust_vals, label='Thrust', color='tab:gray')
    axes[1, 0].set_title('Engine Thrust vs Time')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Thrust (lbs)')
    axes[1, 0].grid(True)

    axes[1, 1].plot(time_steps, speed_vals, label='Speed', color='tab:cyan')
    axes[1, 1].set_title('Total Speed vs Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Speed (m/s)')
    axes[1, 1].grid(True)

    axes[1, 2].plot(time_steps, fuel_vals, label='Fuel Flow', color='tab:brown')
    axes[1, 2].set_title('Fuel Flow vs Time')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Fuel Flow (lbs/s)')
    axes[1, 2].grid(True)

    # 第三行：控制量、姿态角度、空白（或其他图表）
    axes[2, 0].plot(time_steps, aileron_cmd, label='Aileron', color='tab:blue')
    axes[2, 0].plot(time_steps, elevator_cmd, label='Elevator', color='tab:green')
    axes[2, 0].plot(time_steps, rudder_cmd, label='Rudder', color='tab:red')
    axes[2, 0].plot(time_steps, throttle_vals, label='Throttle', color='tab:orange')
    axes[2, 0].set_title('Control Inputs vs Time')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Normalized Command')
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(time_steps, phi_vals, label='Roll (Phi)', color='tab:blue')
    axes[2, 1].plot(time_steps, theta_vals, label='Pitch (Theta)', color='tab:green')
    axes[2, 1].plot(time_steps, psi_vals, label='Yaw (Psi)', color='tab:red')
    axes[2, 1].set_title('Attitude Angles vs Time')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Angle (deg)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    axes[2, 2].plot(time_steps, N_front_vals, label='N_front (Front)', color='tab:blue')
    axes[2, 2].plot(time_steps, N_side_vals, label='N_side (Side)', color='tab:green')
    axes[2, 2].plot(time_steps, N_normal_vals, label='N_normal (Normal)', color='tab:red')
    axes[2, 2].set_title('Load Factors vs Time')
    axes[2, 2].set_xlabel('Time (s)')
    axes[2, 2].set_ylabel('Load Factor (g)')
    axes[2, 2].legend()
    axes[2, 2].grid(True)

    plt.tight_layout()
    plt.show()

    pass
