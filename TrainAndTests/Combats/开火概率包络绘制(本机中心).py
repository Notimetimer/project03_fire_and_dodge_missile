import os
import sys
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from math import pi, sin, cos
from itertools import product
from _context import *

from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Math_calculates.sub_of_angles import sub_of_radian
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def create_initial_states(red_height, blue_height, distance, delta_psi, AA_hor=0):
    """
    创建初始状态配置，借用BasicRules_new_hierarchical找攻击区.py的初始化方法
    
    Args:
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m) 
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad) - 这是我机-目标视线方位角和初始我机航向角之间的差值
        AA_hor: 水平进入角偏移 (rad)
    
    说明：
    - delta_psi = 目标视线方位角 - 我机航向角
    - 红方(我机)初始位置：[0, h, 0]，航向角0 (向北)
    - 蓝方(目标)初始位置：[distance*cos(delta_psi), h, distance*sin(delta_psi)]
      航向角 = 0 + AA_hor (基础向东 + 进入角偏移)
    - 通过调整蓝方的横向位置来实现不同的delta_psi，而不是调整航向角
    """
    # 红方在西边[0, h, 0]面向东
    DEFAULT_RED_BIRTH_STATE = {
        'position': np.array([0.0, red_height, 0.0]),
        'psi': 0,  # 面向北
        'e2e': False
    }
    
    # 蓝方在东边，航向角 = 基础向东(0) + AA_hor(进入角偏移)
    # delta_psi通过调整蓝方位置来实现，而不是航向角
    blue_psi = sub_of_radian(delta_psi + AA_hor, 0)
    
    # 根据delta_psi调整蓝方的横向位置，实现不同的视线角度
    blue_N = distance * cos(delta_psi)  # 横向偏移
    blue_E = distance * sin(delta_psi)  # 纵向距离
    
    DEFAULT_BLUE_BIRTH_STATE = {
        'position': np.array([blue_N, blue_height, blue_E]),
        'psi': blue_psi,
        'e2e': False
    }
    
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

def load_trained_actor(model_path, device='cpu'):
    """加载训练好的actor模型"""
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建网络结构（需要与训练时一致）
    state_dim = 22  # 根据环境观测维度
    hidden_dims = [256, 128]
    action_dims_dict = {'cont': 0, 'cat': [5, 6], 'bern': 1}  # 根据你的动作空间
    
    # 创建网络
    policy_net = PolicyNetHybrid(state_dim, hidden_dims, action_dims_dict)
    actor = HybridActorWrapper(policy_net, action_dims_dict, device=device)
    
    # 加载权重
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    return actor

def run_single_step_firing_probability(actor, red_height, blue_height, distance, delta_psi, AA_hor=0, device='cpu'):
    """
    运行单步并获取开火概率
    
    Args:
        actor: 策略网络
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m)
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad)
        AA_hor: 水平进入角偏移 (rad)
        device: 计算设备
    """
    # 创建环境
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=120.0)
    args = parser.parse_args([])
    args.R_cage = 100e3
    
    env = ChooseStrategyEnv(args, tacview_show=False, vertices=None)
    env.dt_move = 0.04
    env.shielded = 1
    
    # 创建初始状态
    red_state, blue_state = create_initial_states(
        red_height=red_height, 
        blue_height=blue_height, 
        distance=distance, 
        delta_psi=delta_psi, 
        AA_hor=AA_hor
    )
    
    # 重置环境
    env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
              red_init_ammo=6, blue_init_ammo=6)
    
    # 获取观测
    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)

    # 验证开火等待时间和中制导标志位是否起到效果
    t_since_launch = r_obs[21]
    missile_in_mid_term = r_obs[3]
    
    # 修改数值
    r_obs[21] = 35 /120 # 35.0  # 设置等待时间
    r_obs[3] = 1.0    # 设置中制导标志位
    
    # 转换为tensor
    if isinstance(r_obs, np.ndarray):
        r_obs_tensor = torch.tensor(r_obs, dtype=torch.float).unsqueeze(0).to(device)
    else:
        r_obs_tensor = r_obs.unsqueeze(0).to(device)
    
    # 获取动作和开火概率
    with torch.no_grad():
        actions_exec, actions_raw, _, actions_dist_check = actor.get_action(
            r_obs_tensor, explore=False, check_obs=None, temperature=1.0
        ) # check_obs=r_check_obs, None
    
    # 提取开火概率
    firing_probability = actions_dist_check['bern'][0] if 'bern' in actions_dist_check else 0.0
    
    return firing_probability

def plot_firing_probability_heatmap_polar(delta_psis, distances, probabilities):
    """
    在极坐标中绘制开火概率热图（扇形）
    
    Args:
        delta_psis: 角度数组 (rad)
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (delta_psi_count, distance_count)
    """
    # 创建扇形极坐标网格
    Theta, R = np.meshgrid(delta_psis, distances/1000)
    
    # 转置probabilities以匹配网格形状
    # probabilities形状: (delta_psi_count, distance_count)
    # 网格形状: (distance_count, delta_psi_count)
    probabilities_plot = probabilities.T
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(12, 10))
    
    # 绘制热图
    # c = ax.contourf(Theta, R, probabilities_plot, levels=20, cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=0.5)) # RdYlBu_r, levels是颜色层数
    c = ax.contourf(Theta, R, probabilities_plot, levels=18, cmap='Blues_r') # RdYlBu_r, levels是颜色层数
    # 添加颜色条
    cbar = plt.colorbar(c, ax=ax, pad=0.1)
    cbar.set_label('Firing Probability', rotation=270, labelpad=20)
    
    # 设置标签
    ax.set_theta_zero_location('N')  # 0度在北边（正前方）
    ax.set_theta_direction(-1)  # 顺时针方向
    ax.set_title('Firing Probability Heatmap (Sector View)', pad=20)
    ax.set_rlabel_position(45)  # 径向标签位置
    
    # 设置角度范围只显示数据范围
    ax.set_thetamin(-60)  # 最小角度 -60°
    ax.set_thetamax(60)   # 最大角度 +60°
    
    # 设置角度标签
    theta_ticks = np.linspace(-pi/3, pi/3, 30)  # -60°到+60°，每20度一个标签
    ax.set_thetagrids(np.degrees(theta_ticks), [f'{int(np.degrees(t))}°' for t in theta_ticks])
    
    # 添加中心线标记
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    return fig, ax

def plot_firing_probability_heatmap_cartesian(delta_psis, distances, probabilities):
    """
    在直角坐标系中绘制开火概率热图
    
    Args:
        delta_psis: 角度数组 (rad)
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (delta_psi_count, distance_count)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建网格 - 注意顺序要与probabilities数组匹配
    Delta_Psi, Distances = np.meshgrid(delta_psis, distances/1000)
    
    # 转置probabilities以匹配网格形状
    # probabilities形状: (delta_psi_count, distance_count)
    # 网格形状: (distance_count, delta_psi_count)
    probabilities_plot = probabilities.T
    
    # 绘制热图
    c = ax.contourf(Delta_Psi, Distances, probabilities_plot, levels=30, cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=0.5)) # RdYlBu_r, levels是颜色层数
    # c = ax.contourf(Delta_Psi, Distances, probabilities_plot, levels=30, cmap='Blues_r') # RdYlBu_r, levels是颜色层数
    
    # 添加颜色条
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label('Firing Probability')
    
    # 设置标签
    ax.set_xlabel('Delta Psi (rad)')
    ax.set_ylabel('Distance (km)')
    ax.set_title('Firing Probability Heatmap (Cartesian Coordinates)')
    
    # 设置角度标签（转换为度数）
    ax.set_xticks(np.linspace(-pi/3, pi/3, 7))
    ax.set_xticklabels([f'{int(np.degrees(t))}°' for t in np.linspace(-pi/3, pi/3, 7)])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def main():
    # 固定参数
    red_height = 8000  # 红方高度 8km
    blue_height = 8000  # 蓝方高度 8km
    AA_hor = np.radians(180)  # 水平进入角
    device = 'cpu'
    
    # 网格搜索参数
    distances = np.arange(8e3, 101e3, 15e3)  # 8km到100km，间隔5km
    delta_psis = np.arange(-pi/3, pi/3 + np.radians(5), np.radians(5))  # ±π/3，间隔2度
    
    print(f"开始计算开火概率...")
    print(f"距离范围: {distances[0]/1000:.0f}km - {distances[-1]/1000:.0f}km, 间隔: 5km")
    print(f"角度范围: {np.degrees(delta_psis[0]):.0f}° - {np.degrees(delta_psis[-1]):.0f}°, 间隔: 2°")
    print(f"总计算点数: {len(distances) * len(delta_psis)}")
    
    # 查找并加载训练好的模型
    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = None
    dir_name = "PurePFSP_分阶段_SAC-run-20260621-193555"
    
    "PurePFSP_分阶段_混规则对手_挑战_并行_训练满熵项-run-20260616-171415"
    
    "PurePFSP_分阶段_混规则对手_挑战_并行_训练满熵项-run-20260616-171415"
    "PurePFSP_分阶段_混规则对手_挑战_并行_低熵模仿-run-20260616-130304"
    "预训练评估-run-20260614-205839"


    experiment_name = 'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项'
    # experiment_name = 'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项'

    'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项_对照奖励函数'
    
    'NoILPFSP_分阶段_混规则对手_挑战_并行_训练满熵项_旧版奖励函数'
    
    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, experiment_name)
    
    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{experiment_name}'")
    
    agent_path = find_latest_agent_path(latest_log_dir, None)
    if not agent_path:
        raise FileNotFoundError(f"No agent file found in '{latest_log_dir}'")
    
    print(f"找到模型: {agent_path}")
    
    try:
        # 加载模型
        env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3)
        env = ChooseStrategyEnv(env_args, tacview_show=False, vertices=None)
        state_dim = env.obs_dim
        action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
        hidden_dim = [128, 128, 128]
        
        actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        actor = HybridActorWrapper(actor_net, action_dims_dict, device=device).to(device)
        actor.load_state_dict(torch.load(agent_path, map_location=device, weights_only=True), strict=False)
        actor.eval()
        
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 计算开火概率网格
    probabilities = np.zeros((len(delta_psis), len(distances)))
    
    total_points = len(distances) * len(delta_psis)
    current_point = 0
    
    for i, delta_psi in enumerate(delta_psis):
        for j, distance in enumerate(distances):
            current_point += 1
            print(f"进度: {current_point}/{total_points} ({100*current_point/total_points:.1f}%)", end='\r')
            
            try:
                prob = run_single_step_firing_probability(
                    actor=actor, 
                    red_height=red_height, 
                    blue_height=blue_height, 
                    distance=distance, 
                    delta_psi=delta_psi, 
                    AA_hor=AA_hor, 
                    device=device
                )
                probabilities[i, j] = prob
            except Exception as e:
                print(f"\n计算错误 (距离={distance/1000:.0f}km, 角度={np.degrees(delta_psi):.0f}°): {e}")
                probabilities[i, j] = 0.0
    
    print(f"\n计算完成！")
    
    # 保存结果
    results = {
        'distances_km': distances / 1000,
        'delta_psis_rad': delta_psis,
        'delta_psis_deg': np.degrees(delta_psis),
        'probabilities': probabilities
    }
    
    np.savez('firing_probability_results.npz', **results)
    print("结果已保存至: firing_probability_results.npz")
    
    # 绘制图形
    print("绘制极坐标热图...")
    fig1, ax1 = plot_firing_probability_heatmap_polar(
        delta_psis=delta_psis, 
        distances=distances, 
        probabilities=1-np.power(1-probabilities, 5)
    )
    plt.savefig('firing_probability_polar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # print("绘制直角坐标热图...")
    # fig2, ax2 = plot_firing_probability_heatmap_cartesian(
    #     delta_psis=delta_psis, 
    #     distances=distances, 
    #     probabilities=probabilities
    # )
    # plt.savefig('firing_probability_cartesian.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    print("图形已保存！")

if __name__ == '__main__':
    main()

