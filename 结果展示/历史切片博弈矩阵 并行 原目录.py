import os
import json
import re
import time
import torch
import numpy as np
import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count 
import matplotlib
matplotlib.use('Agg') # 禁止弹出窗口，直接保存
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False
from _context import * # 包含 project_root
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Utilities.LocateDirAndAgents2 import get_latest_log_dir
from read_n_draw_inter_experiment_tests import draw_combat_matrix

# --- 1. 参数配置 ---
action_cycle_multiplier = 30
dt_maneuver = 0.2
TOTAL_ROUNDS = 100 # 每两队之间打100场
TEAM_SIZE = 25     # 每队成员数
using_explore_maneuver = 1  # 是否在实验间测试的时候允许动作有随机性

def get_agent_teams(log_dir, num_teams=3):
    """根据 elo_ratings.json 的 Elo 划分区间，提取各区间最强的组合"""
    json_path = os.path.join(log_dir, "elo_ratings.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 {json_path} 文件，请确保目录下有该文件。")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        elo_data = json.load(f)
        
    agents = []
    max_id = 0
    # 提取所有有效的 actor_rein* 并关联其 Elo
    for k, v in elo_data.items():
        match = re.search(r'^actor_rein(\d+)$', k)
        if match:
            agent_id = int(match.group(1))
            agent_path = os.path.join(log_dir, f"{k}.pt")
            if os.path.exists(agent_path):
                agents.append({'id': agent_id, 'name': k, 'elo': v, 'path': agent_path})
                max_id = max(max_id, agent_id)
            
    if not agents:
        raise ValueError("在 elo_ratings.json 中未找到任何可用的 actor_rein 键。")
        
    # 划分区间
    teams = []
    interval_size = (max_id + 1) / num_teams
    
    for i in range(num_teams):
        start_id = i * interval_size
        end_id = (i + 1) * interval_size if i < num_teams - 1 else float('inf')
        
        # 获取在这个区间内的所有 agents
        pool = [a for a in agents if start_id <= a['id'] < end_id]
        
        # 按照 Elo 降序排列，取前 TEAM_SIZE 个
        pool.sort(key=lambda x: x['elo'], reverse=True)
        team = pool[:TEAM_SIZE]
        
        # 如果取到了就加入球队，否则给个提醒
        if not team:
            print(f"警告: 区间 [{start_id}, {end_id}) 内未找到任何智能体，将从相邻区间凑数")
        teams.append(team)
        
    return teams

# --- 保持原样 ---
def run_battle(env, blue_wrapper, red_wrapper, device):
    """仿真逻辑 (保持与文件 1 一致)"""
    env.reset(red_init_ammo=6, blue_init_ammo=6, ego_side='b')
    env.shielded = 1 # 测试时开启防撞地面
    env.no_out = 1 # 测试时防止出界

    done = False
    r_label, b_label = 0, 0
    
    for count in range(int(20*60/env.dt_maneuver)):
        if done: break
        if count % action_cycle_multiplier == 0:
            r_obs, r_check = env.obs_1v1('r', pomdp=1)
            b_obs, b_check = env.obs_1v1('b', pomdp=1)
            with torch.no_grad():
                explore_dict = {'cat': using_explore_maneuver, 'bern': 1}
                # cat 温度调低以凸显确定性, bern 保持1.0不受干扰
                temp_dict = {'cat': 0.1, 'bern': 1.0}
                # 不再向网络传入 check_obs 执行强力动作屏蔽
                r_act, _, _, _ = red_wrapper.get_action(r_obs, explore=explore_dict, temp=temp_dict)
                b_act, _, _, _ = blue_wrapper.get_action(b_obs, explore=explore_dict, temp=temp_dict)
            # 交给环境物理函数使用 tabu=1 (相对宽松的条件) 拦截无效开火
            if r_act['bern'][0]: launch_missile_immediately(env, 'r', tabu=1)
            if b_act['bern'][0]: launch_missile_immediately(env, 'b', tabu=1)
            r_label, b_label = r_act['cat'][0], b_act['cat'][0]

        r_maneuver = env.maneuver14LR(env.RUAV, r_label)
        b_maneuver = env.maneuver14LR(env.BUAV, b_label)
        env.step(r_maneuver, b_maneuver)
        done, _, _, _ = env.combat_terminate_and_reward('b', b_label, b_act['bern'][0], action_cycle_multiplier)
    
    if env.win: return 1.0   # 蓝胜
    if env.lose: return 0.0  # 红胜
    return 0.5               # 平局

# --- 并行工作函数 ---
def worker_process_battle(args_pack):
    """
    子进程执行函数
    """
    blue_path, red_path = args_pack
    
    # 强制在 Worker 中使用 CPU
    device = torch.device("cpu")
    torch.set_num_threads(1) 
    
    # 1. 初始化环境
    # 注意：这里假设 Namespace 参数是固定的，如果需要动态传参需修改 args_pack
    env = ChooseStrategyEnv(argparse.Namespace(max_episode_len=10 * 60, R_cage=45e3), tacview_show=0)
    state_dim, action_dims = env.obs_dim, {'cont':0, 'cat':env.fly_act_dim, 'bern':env.fire_dim}
    
    # 2. 初始化模型
    blue_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    red_wrapper = HybridActorWrapper(PolicyNetHybrid(state_dim, [128,128,128], action_dims), action_dims, None, device).to(device)
    
    # 3. 加载权重
    try:
        blue_wrapper.load_state_dict(torch.load(blue_path, map_location=device, weights_only=True))
        red_wrapper.load_state_dict(torch.load(red_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"模型加载出错: {e}")
        return 0.5
    
    blue_wrapper.eval()
    red_wrapper.eval()
    
    # 4. 调用原始函数
    return run_battle(env, blue_wrapper, red_wrapper, device)

def plot_elo_sampling(log_dir, teams, team_labels, name):
    """封装 Elo 采样绘图逻辑，采用 Agg 后端防止干扰主进程 GUI"""
    try:
        import matplotlib
        matplotlib.use('Agg', force=True) # 强制局部使用 Agg
        import matplotlib.pyplot as plt
        
        with open(os.path.join(log_dir, "elo_ratings.json"), 'r', encoding='utf-8') as f:
            elo_data = json.load(f)
        
        all_plot_data = []
        for k, v in elo_data.items():
            match = re.search(r'^actor_rein(\d+)$', k)
            if match:
                all_plot_data.append([int(match.group(1)), v])
        
        if not all_plot_data:
            print("警告: 未找到任何 Elo 绘图数据。")
            return

        all_plot_data.sort(key=lambda x: x[0])
        all_plot_data = np.array(all_plot_data)

        plt.figure(figsize=(10, 6))
        plt.plot(all_plot_data[:, 0], all_plot_data[:, 1], color='gray', alpha=0.5, label='Elo Evolution')
        
        # 使用 colormap 生成动态颜色，防止 team 数量超过颜色列表长度
        cmap = plt.get_cmap('tab10')
        for idx, team in enumerate(teams):
            if not team: continue
            team_ids = np.array([a['id'] for a in team])
            team_elos = np.array([a['elo'] for a in team])
            plt.scatter(team_ids, team_elos, color=cmap(idx % 10), label=f'Team {team_labels[idx]} Member', s=20)
        
        plt.title(f"Elo Evolution & Team Sampling - {name}")
        plt.xlabel("Actor ID (Checkpoint)")
        plt.ylabel("Elo Rating")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_save_path = os.path.join(project_root, "结果展示", "outputs", f"elo_sampling_{name}.png")
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path)
        plt.close() # 释放内存
        print(f"Elo 采样分布图已保存至: {plot_save_path}")
    except Exception as e:
        print(f"绘图预览失败（不影响主程序）: {e}")

# --- 主程序 ---
if __name__ == "__main__":
    # 6s
    # name = 'IL_and_MixedPFSP_分阶段_挑战_并行_分层-run-20260326-172341'
    # 2s
    name = 'IL_and_MixedPFSP_分阶段_挑战_并行_分层2s-run-20260408-175230'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(project_root, "logs","combat", name)
    
    # 1. 准备队伍
    team_labels = ['1/4', '2/4', '3/4', '4/4']
    teams = get_agent_teams(log_dir, num_teams=len(team_labels))

    # --- 调用预览绘图 ---
    for idx, team in enumerate(teams):
        print(f"Team {team_labels[idx]} IDs: {[a['id'] for a in team]}")
    plot_elo_sampling(log_dir, teams, team_labels, name)
    
    results_matrix = np.zeros((len(team_labels), len(team_labels)))
    np.fill_diagonal(results_matrix, 0.5)

    # 并行配置
    num_processes = min(cpu_count(), 20)
    print(f"\nStart Parallel Simulation ({num_processes} cores)...")
    total_start_time = time.time()

    # 2. 并行计算
    with Pool(processes=num_processes) as pool:
        for i in range(len(team_labels)):      # Blue (Row)
            for j in range(len(team_labels)):  # Red (Col)
                if i == j: continue
                
                # 只计算下三角 (i > j)，即进度靠后的打进度靠前的
                if i > j:
                    print(f"Testing Blue:{team_labels[i]} vs Red:{team_labels[j]}...")
                    
                    # 准备任务列表
                    tasks = []
                    for _ in range(TOTAL_ROUNDS):
                        # 注意：teams[i] 是一个包含字典的列表，需要取 ['path']
                        blue_agent = random.choice(teams[i])['path']
                        red_agent = random.choice(teams[j])['path']
                        tasks.append((blue_agent, red_agent))
                    
                    # 并行执行
                    # map 会返回一个结果列表 [1.0, 0.0, 0.5 ...]
                    results = pool.map(worker_process_battle, tasks)
                    
                    win_rate = sum(results) / len(results)
                    
                    # 填充矩阵
                    results_matrix[i, j] = win_rate
                    results_matrix[j, i] = 1.0 - win_rate
                    
                    print(f"  -> Result: {win_rate:.2f}")

    total_elapsed = time.time() - total_start_time
    print(f"\nTotal Time: {total_elapsed:.2f}s")

    # 3. 保存结果 CSV
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    csv_path = os.path.join(project_root, "结果展示", "outputs", "history_combat_matrix.csv")
    df = pd.DataFrame(results_matrix, index=team_labels, columns=team_labels)
    df.to_csv(csv_path, float_format="%.4f", encoding="utf-8-sig")
    print(f"博弈矩阵已保存到: {csv_path}")

    # 4. [修改] 尝试修复并显示博弈矩阵
    import matplotlib
    # 试图从原本不可交互的 Agg 切换回交互后端 (这在某些环境下可能需要 reload)
    try:
        import importlib
        importlib.reload(matplotlib)
        importlib.reload(plt)
        # 常见 GUI 后端顺序尝试
        for gui_backend in ['Qt5Agg', 'TkAgg', 'WXAgg']:
            try:
                matplotlib.use(gui_backend, force=True)
                import matplotlib.pyplot as plt
                break
            except: continue
    except: pass

    print("正在绘图 (尝试弹窗显示)...")
    draw_combat_matrix(
        csv_path, 
        team_labels, 
        title="Cross-Play Score Matrix: Training Progress Evaluation",
        xlabel="Opponent / Column",
        ylabel="Evaluated / Row",
        cbar_label="Score Rate",
    )