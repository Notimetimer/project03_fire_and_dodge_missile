import os
import json
import re
import time
import torch
import numpy as np
import argparse
import random
import pandas as pd
from multiprocessing import Pool, cpu_count # 引入多进程库

from _context import * # 包含 project_root
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Envs.battle6dof1v1_missile0309_hierarchical import launch_missile_immediately
from Utilities.LocateDirAndAgents2 import get_latest_log_dir
from read_n_draw_inter_experiment_tests import draw_combat_matrix

# --- 1. 配置参数 ---
action_cycle_multiplier = 10
dt_maneuver = 0.2
TOTAL_ROUNDS = 80    # 每对任务之间对抗 100 场
TEAM_SIZE = 80        # 每队从 Elo 排行中取前 50 名
using_explore_maneuver = 1  # 是否在实验间测试的时候允许动作有随机性

# --- 2. 核心辅助函数 ---

def get_agents_halfprog_desc(log_dir, top_n=80):
    """
    扫描 log_dir 中所有 actor_rein{N}.pt 文件：
    1. 找出最大序号 max_idx
    2. 只保留 N <= max_idx // 2 的文件（训练前半程）
    3. 按序号 N 降序排列，取前 top_n 个
    """
    rein_pattern = re.compile(r'^actor_rein(\d+)\.pt$')
    all_candidates = []
    for fname in os.listdir(log_dir):
        m = rein_pattern.match(fname)
        if m:
            idx = int(m.group(1))
            all_candidates.append((idx, os.path.join(log_dir, fname)))
    if not all_candidates:
        print(f"  [{os.path.basename(log_dir)}] 未找到任何 actor_rein*.pt 文件。")
        return []
    max_idx = max(idx for idx, _ in all_candidates)
    half_threshold = max_idx // 2
    half_candidates = [(idx, p) for idx, p in all_candidates if idx <= half_threshold]
    half_candidates.sort(key=lambda x: x[0], reverse=True)
    paths = [p for _, p in half_candidates[:top_n]]
    print(f"  [{os.path.basename(log_dir)}] max_idx={max_idx}, 半程阈值<={half_threshold}, "
          f"候选{len(half_candidates)}个, 加载{len(paths)}个。")
    return paths

# --- 保持原样，完全不改动 ---
def run_battle(env, blue_wrapper, red_wrapper, device):
    """仿真逻辑 (保持与文件 1 一致)"""
    env.reset(red_init_ammo=6, blue_init_ammo=6, ego_side='b')
    env.shielded = 1 # 测试时开启防撞地
    env.no_out = 0 # 测试时防止出界

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
                temp_dict = {'cat': 0.2, 'bern': 1.0}
                # 不再向网络传入 check_obs 执行强力动作屏蔽
                r_act, _, _, _ = red_wrapper.get_action(r_obs, explore=explore_dict, temperature=temp_dict, check_obs=r_check)
                b_act, _, _, _ = blue_wrapper.get_action(b_obs, explore=explore_dict, temperature=temp_dict, check_obs=b_check)
            r_label = r_act['cat']
            b_label = b_act['cat']
            # 用 about_to_fire 标志位控制发射，与 VsBaseline 保持一致
            if r_act['bern'][0]: env.RUAV.about_to_fire = 1
            if b_act['bern'][0]: env.BUAV.about_to_fire = 1

        # 尝试发射，用导弹 id 是否为 None 来判断是否实际发射
        r_m_id = None
        b_m_id = None
        if getattr(env.RUAV, 'about_to_fire', 0):
            r_m_id = launch_missile_immediately(env, 'r', action_label=r_label, tabu=1)
        if getattr(env.BUAV, 'about_to_fire', 0):
            b_m_id = launch_missile_immediately(env, 'b', action_label=b_label, tabu=1)

        r_maneuver = env.maneuver14LR(env.RUAV, r_label)
        b_maneuver = env.maneuver14LR(env.BUAV, b_label)
        env.step(r_maneuver, b_maneuver)
        # action_shoot 传 bool，避免传 numpy.int32 scalar 引发解包错误
        done, _, _, _ = env.combat_terminate_and_reward('b', b_label, b_m_id is not None, action_cycle_multiplier)
    
    if env.win: return float(1.0)   # 蓝胜
    if env.lose: return float(0.0)  # 红胜
    return float(0.5)               # 平局

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
    env = ChooseStrategyEnv(argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3), tacview_show=0)
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

# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- [在此处修改输入列表] ---
    # 6s
    # mission_names = [
    #     'IL_and_MixedPFSP_分阶段_挑战_并行_分层-run-20260326-172341',
    #     'IL_and_PFSP_挑战_并行_分层-run-20260323-165715',
    #     'MixedPFSP_挑战_并行_分层-run-20260323-165740',
    #     'IL_and_deltaFSP_挑战_并行_分层-run-20260323-152514',
    #     'IL_and_PFSP_分阶段_混规则对手_挑战_并行_分层_A3C-run-20260329-174051',
    #     '纯Rule4训练_分层_挑战-run-20260328-114343',
    # ]

    # 2s
    mission_names = [
        'SLWS-PFSP-run-20260618-221044',
        'HLWS-PFSP-run-20260616-130304',
        'PFSP-run-20260615-234324',
        'SLWS-PFSP(A3C)-run-20260630-220403',
        'FixedOpp-run-20260614-163906',
        'SLWS-FixedOpp-run-20260712-113316',
    ]
    
    # team_labels = range(len(mission_names))
    # 提取任务名称的前半部分作为标签，更具可读性
    team_labels = [name.split('-run-')[0][:25] for name in mission_names]
    # [
    #     '1',
    #     '2',
    #     '3',
    #     '4',
    # ]
    
    # 强制校验长度
    if len(mission_names) != len(team_labels):
        raise ValueError(f"输入错误：任务目录数量({len(mission_names)}) 与 标签数量({len(team_labels)}) 不一致！")

    logs_root_dir = os.path.join(project_root, "logs","combat")
    
    # 1. 准备各算法的 Top 50 精英队
    teams = []
    print("正在准备各任务精英智能体...")
    for name in mission_names:
        log_dir = os.path.join(logs_root_dir, name)
        if not os.path.exists(log_dir):
            # 尝试自动查找
            log_dir = get_latest_log_dir(logs_root_dir, name)
            
        if not log_dir:
            raise FileNotFoundError(f"未找到任务目录: {name}")
        teams.append(get_agents_halfprog_desc(log_dir, TEAM_SIZE))

    num_teams = len(teams)

    # 统一各队有效规模：取所有队伍实际人数与 TEAM_SIZE 的最小值
    effective_size = min(len(t) for t in teams)
    if effective_size < TEAM_SIZE:
        print(f"注意：部分队伍人数不足 {TEAM_SIZE}，统一有效规模调整为 {effective_size}。")
    else:
        effective_size = TEAM_SIZE
    print(f"有效队伍规模: {effective_size}，每对对局数: {TOTAL_ROUNDS}")

    # 为每队生成循环出场序列：按序号降序排列后，循环取模补足 TOTAL_ROUNDS 个上场成员
    lineups = []
    for t in teams:
        roster = t[:effective_size]  # 只使用前 effective_size 名
        lineup = [roster[k % effective_size] for k in range(TOTAL_ROUNDS)]
        lineups.append(lineup)

    results_matrix = np.zeros((num_teams, num_teams))
    np.fill_diagonal(results_matrix, 0.5)

    # 并行配置
    num_processes = min(cpu_count(), 20)  # 限制最大进程数，防止卡死
    print(f"\n开始跨任务博弈矩阵计算 ({num_teams}x{num_teams})...")
    print(f"并行进程数: {num_processes}")
    start_time = time.time()
    
    # 创建进程池
    with Pool(processes=num_processes) as pool:
        for i in range(num_teams):      # 行 i 为蓝方 (Evaluated)
            for j in range(num_teams):  # 列 j 为红方 (Opponent)
                if i == j: continue
                
                # 同样使用对称性，只跑 i > j
                if i > j:
                    print(f"正在对抗: [Row]{team_labels[i]} (Blue) vs [Col]{team_labels[j]} (Red)...")
                    
                    # 一对一配对：第k个蓝方 vs 第k个红方，循环补足 TOTAL_ROUNDS 场
                    battle_tasks = list(zip(lineups[i], lineups[j]))
                    
                    # 并行执行
                    results = pool.map(worker_process_battle, battle_tasks)
                    
                    n_games = len(results)
                    total_score = sum(results)
                    win_rate = total_score / n_games
                    
                    results_matrix[i, j] = win_rate       # i 打赢 j 的胜率
                    results_matrix[j, i] = 1.0 - win_rate # j 打赢 i 的胜率
                    print(f"  -> {team_labels[i]} 对阵 {team_labels[j]} 胜率: {win_rate:.2f} ({n_games} 场)")

    print(f"\n矩阵计算完成！总耗时: {time.time() - start_time:.2f}s")

    # 保存博弈矩阵为 CSV 以便后续分析/绘图
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    csv_path = os.path.join(project_root, "结果展示", "outputs", "combat_matrix_half.csv")
    df = pd.DataFrame(results_matrix, index=team_labels, columns=team_labels)
    df.to_csv(csv_path, float_format="%.4f", encoding="utf-8-sig")
    print(f"博弈矩阵已保存到: {csv_path}")

    # 4. [修改] 调用外部函数进行绘图
    print("正在调用 read_n_draw_inter_experiment_tests 进行绘图...")
    draw_combat_matrix(
        csv_path, 
        team_labels, 
        title=None,
        xlabel="Opponent / Column",
        ylabel="Evaluated / Row",
        cbar_label="Win Rate"
    )