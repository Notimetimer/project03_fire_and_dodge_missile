import os
import re
import time
import torch
import numpy as np
import argparse
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from _context import *
from Envs.Tasks.ChooseStrategyEnv2_2 import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Envs.battle6dof1v1_missile0919 import launch_missile_immediately
# --- 2. 辅助函数 ---
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path


# --- 1. 参数配置 ---
action_cycle_multiplier = 30
dt_maneuver = 0.2
TOTAL_ROUNDS = 100 # 每两队之间打100场
TEAM_SIZE = 10     # 每队成员数

def get_agent_teams(log_dir):
    """根据文件编号划分三个进度的队伍"""
    files = [f for f in os.listdir(log_dir) if f.startswith('actor_rein') and f.endswith('.pt')]
    # 提取编号并排序
    agents = []
    for f in files:
        match = re.search(r'actor_rein(\d+)', f)
        if match:
            agents.append({'id': int(match.group(1)), 'path': os.path.join(log_dir, f)})
    
    agents.sort(key=lambda x: x['id'])
    total_count = len(agents)
    
    # 定义进度索引 (1/3, 2/3, 1.0)
    indices = [total_count // 3, (2 * total_count) // 3, total_count - 1]
    teams = []
    
    for idx in indices:
        # 在该进度点附近进行间隔采样，确保覆盖一个小范围
        # 取idx附近的20个，每隔1个取1个，凑够10个
        start = max(0, idx - 10)
        end = min(total_count, idx + 10)
        sample_pool = agents[start:end]
        team = sample_pool[::max(1, len(sample_pool)//TEAM_SIZE)][:TEAM_SIZE]
        teams.append(team)
    
    return teams

def run_battle(env, red_wrapper, blue_wrapper, device):
    """单场仿真逻辑"""
    env.reset(red_init_ammo=6, blue_init_ammo=6) # 此处可加入固定初始位置逻辑
    done = False
    
    # [修复] 初始化动作标签，防止第一次循环未定义
    r_label, b_label = 0, 0
    
    for count in range(3000): # 最大步数
        if done: break
        
        if count % action_cycle_multiplier == 0:
            r_obs, r_check = env.obs_1v1('r', pomdp=1)
            b_obs, b_check = env.obs_1v1('b', pomdp=1)
            
            with torch.no_grad():
                # 红方决策
                r_act, _, _, _ = red_wrapper.get_action(r_obs, explore={'cat':0,'bern':1}, check_obs=r_check)
                # 蓝方决策
                b_act, _, _, _ = blue_wrapper.get_action(b_obs, explore={'cat':0,'bern':1}, check_obs=b_check)
            
            if r_act['bern'][0]: launch_missile_immediately(env, 'r')
            if b_act['bern'][0]: launch_missile_immediately(env, 'b')
            
            r_label, b_label = r_act['cat'][0], b_act['cat'][0]

        r_maneuver = env.maneuver14LR(env.RUAV, r_label)
        b_maneuver = env.maneuver14LR(env.BUAV, b_label)
        env.step(r_maneuver, b_maneuver)
        done, _, _, _ = env.combat_terminate_and_reward('b', b_label, b_act['bern'][0], action_cycle_multiplier)
    
    if env.win: return 1.0   # 蓝胜
    if env.lose: return 0.0  # 红胜
    return 0.5               # 平局

# --- 主程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission-name", type=str, default='RL_combat_PFSP_简单熵_区分左右_无淘汰机制_有模仿学习') # 测试路径
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = get_latest_log_dir(logs_root_dir, args.mission_name)
    
    log_dir = latest_log_dir # f"./logs/combat/{args.mission_name}" # 根据实际路径修改
    
    # 1. 准备队伍
    teams = get_agent_teams(log_dir)
    team_labels = ['Progress 33%', 'Progress 67%', 'Final 100%']
    results_matrix = np.zeros((3, 3))
    np.fill_diagonal(results_matrix, 0.5)

    # 2. 初始化模型容器
    env = ChooseStrategyEnv(argparse.Namespace(max_episode_len=600, R_cage=55e3), tacview_show=0)
    state_dim, action_dims = env.obs_dim, {'cont':0, 'cat':env.fly_act_dim, 'bern':env.fire_dim}
    
    actor_red = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    actor_blue = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)

    # 3. 对抗实验
    # [新增] 时间统计
    total_start_time = time.time()
    
    # 我们只跑上三角：(0,1), (0,2), (1,2)，其中 i 是红方，j 是蓝方
    for i in range(3):
        for j in range(i + 1, 3):
            # [新增] 每队对战开始计时
            team_start_time = time.time()
            
            print(f"\nTesting {team_labels[j]} (Blue) vs {team_labels[i]} (Red)...")
            total_score = 0
            
            for _ in range(TOTAL_ROUNDS):
                # 随机从两队中各抽一个成员
                red_agent = random.choice(teams[i])
                blue_agent = random.choice(teams[j])
                
                # 加载权重
                actor_red.load_state_dict(torch.load(red_agent['path'], map_location=device))
                actor_blue.load_state_dict(torch.load(blue_agent['path'], map_location=device))
                actor_red.eval(); actor_blue.eval()
                
                # 仿真
                score = run_battle(env, actor_red, actor_blue, device)
                total_score += score
            
            final_ratio = total_score / TOTAL_ROUNDS
            results_matrix[i, j] = final_ratio
            results_matrix[j, i] = 1.0 - final_ratio
            
            # [新增] 打印该队对战耗时
            team_elapsed = time.time() - team_start_time
            print(f"  Result: Blue Win Rate = {final_ratio:.2f} | Time: {team_elapsed:.2f}s")

    # [新增] 打印总耗时
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"Total Simulation Time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
    print(f"{'='*60}\n")


    # 4. 绘图
    # --- 颜色方案调整 ---
    # 使用 "PRGn" (Purple-Green) 预设配色
    # 如果你希望绿色代表高分（胜率高），紫色代表低分，由于 PRGn 默认紫色在左绿色在右，
    # 对应 0.0-1.0 的映射是完美的。

    plt.figure(figsize=(9, 7))

    # 绘图参数优化：
    # square=True: 强制每个格子为正方形
    # linewidths: 格子之间的白线宽度，增加高级感
    # cbar_kws: 侧边栏标签
    sns.heatmap(
        results_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="PRGn",           # 核心修改：紫色配绿色
        center=0.5,            # 设置 0.5 为颜色的中性点（浅色区）
        xticklabels=team_labels, 
        yticklabels=team_labels,
        square=True,           
        linewidths=0.5,        
        cbar_kws={"label": "Score Rate (Win + 0.5*Draw)"}
    )

    plt.title("Cross-Play Score Matrix: Training Progress Evaluation", fontsize=14, pad=20)
    plt.xlabel("Blue Team (Evaluated)", fontsize=12)
    plt.ylabel("Red Team (Opponent)", fontsize=12)

    # 自动调整布局，防止标签被遮挡
    plt.tight_layout()
    plt.show()