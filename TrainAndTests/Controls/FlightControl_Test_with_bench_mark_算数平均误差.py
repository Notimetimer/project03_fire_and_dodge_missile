import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
import time
import torch
import csv

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Controls.FlightControl_Train_dual_a_out2 import *
# from TrainAndTests.Controls.parallel_FlightControl_Train_dual_a_out import *
from Utilities.LocateDirAndAgents import *
from TrainAndTests.Controls.UPolicyWrapper import *

# 设备设置
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 网络初始化
action_dims_dict = {'cont': 4, 'cat': [], 'bern': 0}
policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
print(action_bound)
actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)

# 模型加载逻辑
pre_log_dir = os.path.join(project_root, "logs/control")
mission_name = "PID" # "FlightControl_parallel无课程无蒸馏_有过载限制_动态lr"
# 可选其它控制器
"PID"
"FlightControl_parallel无课程无蒸馏_有过载限制"
"FlightControl_parallel无课程无蒸馏半高度误差惩罚"

if mission_name != "PID":
    log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)
    # log_dir = os.path.join(pre_log_dir, "FlightControl-run-20260308-211329")

    # 用新函数加载 actor：若想强制加载编号为 990 的模型，传入 number=990
    actor_path = load_actor_from_log(log_dir, number=15800)
    if not actor_path:
        print(f"No actor checkpoint found in {log_dir}")
        sys.exit()
    else:
        sd = th.load(actor_path, map_location=device, weights_only=True)
        actor.load_state_dict(sd)
        print(f"Loaded actor for test from: {actor_path}")

# Benchmark 参数
height_list = [8000]
speed_list = [300]
dt_decide = 0.02
dt_move = 0.01

# 是否可视化
visualize = 0

# 是否跟踪动目标（会导致超调量记录失效）
chasing_wave = 0
realistic = 1

delta_height = -4000 # -5000

test_name1 = "wave" if chasing_wave else "static"
test_name2 = "delta_h" + str(delta_height) if not chasing_wave else ""

if chasing_wave:
    time_limit = 8 * 60  # 每组测试限时 8 分钟
    height_list = [8000]
    speed_list = [340]
else:
    time_limit = 4 * 60  # 每组测试限时 3 分钟

avg_height_overshoot = 0
max_h_overshoot = 0
avg_heading_overshoot = 0
max_heading_overshoot = 0
avg_v_error = 0
avg_ao = 0
avg_psi_error = 0
avg_theta_error = 0
survive_rate = 0

max_ny = -float('inf')
min_ny = float('inf')
max_alpha = 0
min_alpha = float('inf')
max_beta = 0

env = track_env(dt_move=dt_move, tacview_show=visualize, time_limit=time_limit)
env.realistic = realistic

# PID 策略初始化
pidcontroller = UnifiedPolicyWrapper(env, dt_decide=dt_decide) # 

# 目标变化的波动参数（正弦波轨迹，在不同测试中保持一致）
A_psi_dot = 10 * (pi / 180)  # deg/s 振幅
w_psi = 2 * pi / 120         # s 一个周期
A_h_dot = 100                # m/s 振幅
w_h = 2 * pi / 200           # s 一个周期

total_cases = len(height_list) * len(speed_list)
success_count = 0

t_bias = 0 # 用于 Tacview 时间偏移，防止轨道重叠

t_list = []
# 姿态、速高
theta_list = []
psi_list = []
phi_list = []
v_list = []
h_list = []
# 空气动力学
alpha_air_list = []
beta_air_list = []
# 过载
Ny_list = []
# 目标
theta_req_list = []
psi_req_list = []
v_req_list = []
height_req_list = []
# 控制量
aileron_list = []
elevetor_list = []
rudder_list =[]
throttle_list = []

# NUE 坐标系下的轨迹
traj_t_list = []
uav_n_list = []
uav_u_list = []
uav_e_list = []
target_n_list = []
target_u_list = []
target_e_list = []
round_list = []


print(f"\nBenchmark 开始，当前测试配置 [{mission_name}]，共 {total_cases} 组测试案例...")
i=0
for init_h in height_list:
    for target_v in speed_list:
        i+=1
        print(f"\n>>> 正在测试: 初始高度 {init_h}m, 目标速度 {target_v}m/s (t_bias: {t_bias:.1f}s)")
        
        # 固定初始化
        birth_state = {'position': np.array([0.0, init_h, 0.0]), 'psi': 3*pi/180}
        env.reset(birth_state=birth_state, height_req=init_h, psi_req=0, v_req=target_v, dt_report=dt_decide)
        
        obs, obs_check = env.get_obs()
        done = False
        
        ao_sum_episode = 0
        v_error_sum_episode = 0
        psi_error_sum_episode = 0.0
        theta_error_sum_episode = 0.0
        steps_in_episode = 0

        while not done:
            # 更新动态目标 (按照预设的正弦曲线变化)
            current_t = env.t
            
            if chasing_wave:
                # 航向角速波动
                psi_dot_t = A_psi_dot * sin(w_psi * current_t)
                env.psi_req += psi_dot_t * dt_decide
                env.psi_req = sub_of_radian(env.psi_req, 0)
                
                # 高度变化率波动
                h_dot_t = A_h_dot * - sin(w_h * current_t)
                env.height_req += h_dot_t * dt_decide
                env.height_req = np.clip(env.height_req, 3000, 13000)
            else:
                env.height_req = np.clip(init_h + delta_height, 3000, 13000)
                env.psi_req = sub_of_radian(birth_state['psi'] + pi*5/6 ) #, pi + 2*pi/180*(i%2-0.5)*2)
                env.v_req = target_v
            
            # 决策
            obs, obs_check = env.get_obs()
            if mission_name != "PID":
                # NN
                action, u, _, _ = actor.get_action(obs, explore=0)
            else:
                # PID
                action = pidcontroller.get_action(obs, explore=0)
            
            # 推进环境
            next_obs, reward, done = env.step(action)
            
            # 累积采样误差 (算术平均用)
            ao_sum_episode += env.AO
            v_error_sum_episode += abs(env.v_error)
            psi_error_sum_episode += abs(env.psi_error)
            theta_error_sum_episode += abs(env.theta_error)
            steps_in_episode += 1
            
            # 记录飞行包线极限值
            max_ny = max(max_ny, env.RUAV.Ny)
            min_ny = min(min_ny, env.RUAV.Ny)
            max_alpha = max(max_alpha, abs(env.RUAV.alpha_air) * 180 / pi)
            min_alpha = min(min_alpha, env.RUAV.alpha_air * 180 / pi)
            max_beta = max(max_beta, abs(env.RUAV.beta_air) * 180 / pi)

            # 可视化输出 (使用 t_bias)
            env.render(t_bias)

            # 记录飞行数据
            t_list.append(env.t)
            theta_list.append(env.RUAV.theta * 180/pi)
            psi_list.append(env.RUAV.psi * 180/pi)
            phi_list.append(env.RUAV.phi * 180/pi)
            v_list.append(env.RUAV.speed)
            h_list.append(env.RUAV.alt)
            alpha_air_list.append(env.RUAV.alpha_air * 180/pi)
            beta_air_list.append(env.RUAV.beta_air * 180/pi)
            Ny_list.append(env.RUAV.Ny)
            theta_req_list.append(env.theta_v_req * 180/pi)
            psi_req_list.append(env.psi_req * 180/pi)
            v_req_list.append(env.v_req)
            height_req_list.append(env.height_req)

            aileron, elevetor, rudder, throttle = action['cont']
            aileron_list.append(aileron)
            elevetor_list.append(elevetor)
            rudder_list.append(rudder)
            throttle_list.append(throttle)

            # 记录 NUE 轨迹 (使用偏置后的时间以衔接 Tacview)
            traj_t_list.append(env.t + t_bias)
            uav_n_list.append(env.uav_pos_[0])
            uav_u_list.append(env.uav_pos_[1])
            uav_e_list.append(env.uav_pos_[2])
            target_n_list.append(env.target_pos_[0])
            target_u_list.append(env.target_pos_[1])
            target_e_list.append(env.target_pos_[2])
            round_list.append(i)
        
        env.clear_render(t_bias)
        t_bias += env.t # 累加偏置，使下一条轨迹衔接在后面
        
        if steps_in_episode > 0:
            avg_height_overshoot += abs(env.height_overshoot)/total_cases
            max_h_overshoot = max(abs(env.height_overshoot), max_h_overshoot)
            avg_heading_overshoot += abs(env.heading_overshoot)*180/pi/total_cases
            max_heading_overshoot = max(abs(env.heading_overshoot)*180/pi, max_heading_overshoot)
            avg_ao += (ao_sum_episode / steps_in_episode) / total_cases
            avg_v_error += (v_error_sum_episode / steps_in_episode) / total_cases
            avg_psi_error += (psi_error_sum_episode / steps_in_episode) / total_cases
            avg_theta_error += (theta_error_sum_episode / steps_in_episode) / total_cases
        
        # 判断本轮是否成功
        if not env.fail and env.t >= time_limit:
            success_count += 1
            print(f"结果: 成功. 耗时 {env.t:.1f}s")
            survive_rate += 1/total_cases
        else:
            print(f"结果: 失败. 因 {'失速/坠毁' if env.fail else '提前终止'}. 耗时 {env.t:.1f}s")

# 统计输出
print("\n" + "="*40)
print("             BENCHMARK SUMMARY")
print("="*40)
print(f"当前测试模型: {mission_name}")
print(f"测试总例数: {total_cases}")
print(f"成功例数:   {success_count}")
print(f"总成功率:   {success_count/total_cases*100:.1f}%")

# 全部回合平均误差指标 (算术平均)
print(f"\n全部回合平均算术误差指标 (Arithmetic Mean Error):")
print(f" - 速度误差 (Speed Err):   {avg_v_error:.3f} m/s")
print(f" - 航向误差 (Heading Err): {avg_psi_error:.3f} deg")
print(f" - 高度误差 (Altitude Err): {avg_height_overshoot:.3f} m")
print(f" - 俯仰角误差 (Pitch Err): {avg_theta_error:.3f} deg")
print(f" - 指向误差 (AO):         {avg_ao:.3f} deg")
print(" - 平均高度超调", avg_height_overshoot, "m")
print(" - 最大高度超调", max_h_overshoot, "m")
print(" - 平均航向超调", avg_heading_overshoot, "°")
print(" - 最大航向超调", max_heading_overshoot, "°")
print("="*40)


print("\n飞行包线极限统计:")
print(f" - 最大正过载 (Max Ny): {max_ny:.3f}")
print(f" - 最大负过载 (Min Ny): {min_ny:.3f}")
print(f" - 最大迎角 (Max Alpha): {max_alpha:.3f} deg")
print(f" - 最小迎角 (Min Alpha): {min_alpha:.3f} deg")
print(f" - 最大侧滑角 (Max Beta): {max_beta:.3f} deg")

# --- 保存数据到 CSV ---
try:
    save_dir = os.path.join(os.path.dirname(__file__), "test_result")
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{mission_name}_{test_name1}_{test_name2}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    csv_path = os.path.join(save_dir, file_name)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'time', 'theta', 'psi', 'phi', 'v', 'h', 'alpha', 'beta', 'Ny',
            'theta_req', 'psi_req', 'v_req', 'h_req', 'aileron', 'elevator', 'rudder', 'throttle', 'round'
        ])
        # 写入数据 (使用 zip 聚合序列)
        writer.writerows(zip(
            t_list, theta_list, psi_list, phi_list, v_list, h_list, 
            alpha_air_list, beta_air_list, Ny_list,
            theta_req_list, psi_req_list, v_req_list, height_req_list,
            aileron_list, elevetor_list, rudder_list, throttle_list, round_list
        ))
    print(f"\n[数据导出] 飞行记录已存至: {csv_path} (共 {len(t_list)} 条记录)")

    # --- 保存 NUE 轨迹到另一个 CSV ---
    traj_file_name = f"{mission_name}_{test_name1}_{test_name2}_trajectory_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    traj_csv_path = os.path.join(save_dir, traj_file_name)
    with open(traj_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'uav_N', 'uav_U', 'uav_E', 'target_N', 'target_U', 'target_E', 'round'])
        writer.writerows(zip(
            traj_t_list, uav_n_list, uav_u_list, uav_e_list,
            target_n_list, target_u_list, target_e_list, round_list
        ))
    print(f"[数据导出] NUE 轨迹已存至: {traj_csv_path}")
except Exception as e:
    print(f"\n[错误] 保存 CSV 失败: {e}")