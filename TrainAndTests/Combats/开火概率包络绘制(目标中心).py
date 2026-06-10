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

def create_initial_states(red_height, blue_height, distance, AA_hor):
    """
    目标中心视角：以蓝方(目标)为中心，红方(我机)绕转一圈
    
    Args:
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m)
        distance: 水平距离 (m)
        AA_hor: 红方相对于目标的水平方位角 (rad), 0=北, 顺时针为正
    
    说明：
    - 蓝方(目标)固定在原点[0, h, 0]，航向角=0 (朝北)
    - 红方(我机)位于 [distance*cos(-AA_hor), h, distance*sin(-AA_hor)]，
      面向目标，即航向角 = AA_hor + pi (朝向目标)
    - delta_psi固定为0 (我机始终正对目标)
    """
    # 蓝方(目标)固定在原点，朝北
    DEFAULT_BLUE_BIRTH_STATE = {
        'position': np.array([0.0, blue_height, 0.0]),
        'psi': 0,  # 朝北
        'e2e': False
    }
    
    # 红方(我机)在目标周围，面向目标
    red_N = distance * cos(-AA_hor)
    red_E = distance * sin(-AA_hor)
    red_psi = sub_of_radian(-AA_hor + pi, 0)  # 朝向目标
    
    DEFAULT_RED_BIRTH_STATE = {
        'position': np.array([red_N, red_height, red_E]),
        'psi': red_psi,
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

def run_single_step_firing_probability(actor, red_height, blue_height, distance, AA_hor, device='cpu'):
    """
    运行单步并获取开火概率（目标中心视角）
    
    Args:
        actor: 策略网络
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m)
        distance: 水平距离 (m)
        AA_hor: 红方相对目标的水平方位角 (rad)
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
        AA_hor=AA_hor
    )
    
    # 重置环境
    env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
              red_init_ammo=6, blue_init_ammo=6)
    
    # 获取观测
    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
    
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

def plot_firing_probability_heatmap_polar(AA_hors, distances, probabilities):
    """
    在极坐标中绘制开火概率热图（完整圆盘，目标中心视角）
    
    Args:
        AA_hors: 方位角数组 (rad), 0到2π
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (AA_hor_count, distance_count)
    """
    # 创建完整圆盘极坐标网格
    Theta, R = np.meshgrid(AA_hors, distances/1000)
    
    # 转置probabilities以匹配网格形状
    # probabilities形状: (AA_hor_count, distance_count)
    # 网格形状: (distance_count, AA_hor_count)
    probabilities_plot = probabilities.T
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10))
    
    # 绘制热图
    c = ax.contourf(Theta, R, probabilities_plot, levels=30, cmap='Blues_r')
    
    # 添加颜色条
    cbar = plt.colorbar(c, ax=ax, pad=0.1)
    cbar.set_label('Firing Probability', rotation=270, labelpad=20)
    
    # 设置标签
    ax.set_theta_zero_location('N')  # 0度在北边
    ax.set_theta_direction(-1)       # 顺时针方向
    ax.set_title('Firing Probability Heatmap\n(Target-Centered, delta_psi=0)', pad=20)
    ax.set_rlabel_position(45)
    
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
    c = ax.contourf(Delta_Psi, Distances, probabilities_plot, levels=30, cmap='Blues_r') # RdYlBu_r, levels是颜色层数
    
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
    red_height = 8000   # 红方高度 8km
    blue_height = 8000  # 蓝方高度 8km
    device = 'cpu'
    
    # 网格搜索参数
    distances = np.arange(8e3, 101e3, 15e3)  # 8km到100km，间隔15km
    AA_hors = np.arange(0, 2*pi + np.radians(20), np.radians(20))  # 0到360°，间隔5度
    
    print(f"开始计算开火概率（目标中心视角）...")
    print(f"距离范围: {distances[0]/1000:.0f}km - {distances[-1]/1000:.0f}km")
    print(f"方位角范围: 0° - 360°, 间隔: 5°")
    print(f"总计算点数: {len(distances) * len(AA_hors)}")
    
    # 查找并加载训练好的模型
    # 优先使用dir_name，如果没有则使用experiment_name
    dir_name = None
    # dir_name = "IL_and_Mixed经典PFSP_挑战_并行_分层_rule3_无mask-run-20260531-233800"
    experiment_name = 'PFSP_分阶段_混规则对手_挑战_并行_训练满熵项_对照奖励函数'

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
    probabilities = np.zeros((len(AA_hors), len(distances)))
    
    total_points = len(distances) * len(AA_hors)
    current_point = 0
    
    for i, AA_hor in enumerate(AA_hors):
        for j, distance in enumerate(distances):
            current_point += 1
            print(f"进度: {current_point}/{total_points} ({100*current_point/total_points:.1f}%)", end='\r')
            
            try:
                prob = run_single_step_firing_probability(
                    actor=actor, 
                    red_height=red_height, 
                    blue_height=blue_height, 
                    distance=distance, 
                    AA_hor=AA_hor, 
                    device=device
                )
                probabilities[i, j] = prob
            except Exception as e:
                print(f"\n计算错误 (距离={distance/1000:.0f}km, 方位角={np.degrees(AA_hor):.0f}°): {e}")
                probabilities[i, j] = 0.0
    
    print(f"\n计算完成！")
    
    # 保存结果
    results = {
        'distances_km': distances / 1000,
        'AA_hors_rad': AA_hors,
        'AA_hors_deg': np.degrees(AA_hors),
        'probabilities': probabilities
    }
    
    np.savez('firing_probability_target_center.npz', **results)
    print("结果已保存至: firing_probability_target_center.npz")
    
    # 绘制图形
    print("绘制极坐标热图（圆盘）...")
    fig1, ax1 = plot_firing_probability_heatmap_polar(
        AA_hors=AA_hors, 
        distances=distances, 
        probabilities=probabilities
    )
    plt.savefig('firing_probability_target_center.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("图形已保存！")

if __name__ == '__main__':
    main()

