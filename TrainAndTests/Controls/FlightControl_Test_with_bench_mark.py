import sys
import os
import numpy as np
from numpy.linalg import norm
import torch as th
from math import *
import time
import torch

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
mission_name = "FlightControl_parallel无课程无蒸馏_有过载限制"
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
height_list = [3000, 5000, 7000, 9000, 11000]
speed_list = [340, 250]
dt_decide = 0.05
dt_move = 0.05
time_limit = 5 * 60  # 每组测试限时 5 分钟

# 是否跟踪动目标（会导致超调量记录失效）
chasing_wave = 0

avg_height_overshoot = 0
avg_heading_overshoot = 0
avg_v_error = 0
avg_ao = 0
survive_rate = 0

beta_ao = 0.01 ** (dt_decide / 10.0) # 超出最后10s以前的误差忽略不计
max_ny = -float('inf')
min_ny = float('inf')
max_alpha = 0
min_alpha = float('inf')
max_beta = 0

env = track_env(dt_move=dt_move, tacview_show=1, time_limit=time_limit)

# PID 策略初始化
pidcontroller = UnifiedPolicyWrapper(env, dt_decide=dt_decide) # 

# 目标变化的波动参数（正弦波轨迹，在不同测试中保持一致）
A_psi_dot = 6 * (pi / 180)  # deg/s 振幅
w_psi = 2 * pi / 120         # s 一个周期
A_h_dot = 100                # m/s 振幅
w_h = 2 * pi / 150           # s 一个周期

# 数据汇总
total_cases = len(height_list) * len(speed_list)
success_count = 0
global_total_speed_err = 0
global_total_psi_err = 0
global_total_theta_err = 0
global_total_alt_err = 0
global_success_seconds = 0

t_bias = 0 # 用于 Tacview 时间偏移，防止轨道重叠

print(f"\nBenchmark 开始，当前测试配置 [{mission_name}]，共 {total_cases} 组测试案例...")
i=0
for init_h in height_list:
    for target_v in speed_list:
        i+=1
        print(f"\n>>> 正在测试: 初始高度 {init_h}m, 目标速度 {target_v}m/s (t_bias: {t_bias:.1f}s)")
        
        # 固定初始化
        birth_state = {'position': np.array([0.0, init_h, 0.0]), 'psi': 0}
        env.reset(birth_state=birth_state, height_req=init_h, psi_req=0, v_req=target_v, dt_report=dt_decide)
        
        obs, obs_check = env.get_obs()
        done = False
        
        case_speed_err_sum = 0
        case_psi_err_sum = 0
        case_theta_err_sum = 0
        case_alt_err_sum = 0
        case_steps = 0
        ao_ema_episode = 0
        v_error_ema_episode = 0

        while not done:
            # 更新动态目标 (按照预设的正弦曲线变化)
            current_t = env.t
            
            if chasing_wave:
                # 航向角速波动
                psi_dot_t = A_psi_dot * sin(w_psi * current_t)
                env.psi_req += psi_dot_t * dt_decide
                env.psi_req = sub_of_radian(env.psi_req, 0)
                
                # 高度变化率波动
                h_dot_t = A_h_dot * sin(w_h * current_t)
                env.height_req += h_dot_t * dt_decide
                env.height_req = np.clip(env.height_req, 1500, 14000)
            else:
                env.height_req = np.clip(init_h + 5000, 3000, 13000)
                env.psi_req = sub_of_radian(birth_state['psi'], pi+2*pi/180*(i%2-0.5)*2)
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
            
            # 累加采样误差
            ao_ema_episode = beta_ao * ao_ema_episode + (1 - beta_ao) * (env.AO)
            v_error_ema_episode = beta_ao * v_error_ema_episode + (1 - beta_ao) * (env.v_error)

            case_steps += 1
            case_speed_err_sum += abs(env.RUAV.speed - env.v_req)
            case_psi_err_sum += abs(sub_of_radian(env.RUAV.psi, env.psi_req))
            case_alt_err_sum += abs(env.RUAV.alt - env.height_req)
            
            # 俯仰角误差 (基于训练时的理想 desired_theta)
            h2req = np.clip((env.height_req - env.RUAV.alt), -5000, 5000)
            desired_theta = (h2req >= 0) * h2req / 5000 * pi / 3 + (h2req < 0) * h2req / 5000 * pi / 2
            case_theta_err_sum += abs(env.RUAV.theta - desired_theta)
            
            # 记录飞行包线极限值
            max_ny = max(max_ny, env.RUAV.Ny)
            min_ny = min(min_ny, env.RUAV.Ny)
            max_alpha = max(max_alpha, abs(env.RUAV.alpha_air) * 180 / pi)
            min_alpha = min(min_alpha, env.RUAV.alpha_air * 180 / pi)
            max_beta = max(max_beta, abs(env.RUAV.beta_air) * 180 / pi)

            # 可视化输出 (使用 t_bias)
            env.render(t_bias)
        
        env.clear_render(t_bias)
        t_bias += env.t # 累加偏置，使下一条轨迹衔接在后面
        
        steps_run = int(env.t/dt_decide)
        avg_height_overshoot += abs(env.height_overshoot)/total_cases
        avg_heading_overshoot += abs(env.heading_overshoot)*180/pi/total_cases
        avg_ao += ao_ema_episode/(1 - beta_ao**max(1, steps_run))
        avg_v_error += v_error_ema_episode/(1 - beta_ao**max(1, steps_run))
        
        # 判断本轮是否成功
        if not env.fail and env.t >= time_limit:
            success_count += 1
            global_total_speed_err += case_speed_err_sum
            global_total_psi_err += case_psi_err_sum
            global_total_theta_err += case_theta_err_sum
            global_total_alt_err += case_alt_err_sum
            global_success_seconds += env.t
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

if success_count > 0:
    # 平均每秒误差指标
    final_avg_speed = (global_total_speed_err * dt_decide) / global_success_seconds
    final_avg_psi   = (global_total_psi_err * dt_decide) / global_success_seconds
    final_avg_alt   = (global_total_alt_err * dt_decide) / global_success_seconds
    final_avg_theta = (global_total_theta_err * dt_decide) / global_success_seconds
    
    print(f"\n成功回合平均每秒误差指标:")
    print(f" - 速度误差 (Speed Err):   {final_avg_speed:.3f} m/s")
    print(f" - 航向误差 (Heading Err): {final_avg_psi*180/pi:.3f} deg")
    print(f" - 高度误差 (Altitude Err): {final_avg_alt:.3f} m")
    print(f" - 俯仰角误差 (Pitch Err): {final_avg_theta*180/pi:.3f} deg")
else:
    print("\n无成功回合，无法计算平均误差指标。")
print("="*40)

print("survive_rate", round(survive_rate,2))
print("avg_height_overshoot", avg_height_overshoot)
print("avg_heading_overshoot", avg_heading_overshoot)
print("avg_v_error", avg_v_error)
print("avg_ao", avg_ao)

print("\n飞行包线极限统计:")
print(f" - 最大正过载 (Max Ny): {max_ny:.3f}")
print(f" - 最大负过载 (Min Ny): {min_ny:.3f}")
print(f" - 最大迎角 (Max Alpha): {max_alpha:.3f} deg")
print(f" - 最小迎角 (Min Alpha): {min_alpha:.3f} deg")
print(f" - 最大侧滑角 (Max Beta): {max_beta:.3f} deg")