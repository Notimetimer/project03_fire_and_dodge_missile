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
TOTAL_ROUNDS =  80    # 每对任务之间对抗 100 场
TEAM_SIZE =     30        # 每队从 Elo 排行中取前 50 名
using_explore_maneuver = 1  # 是否在实验间测试的时候允许动作有随机性

# 训练进度
progress = 1.0

# --- 2. 核心辅助函数 ---

def get_top_elo_agents(log_dir, top_n=50, prog=1.0):
    """
    只从 elo_ratings.json 里找 actor_rein{N} 格式的 key，
    筛掉编号超过 max_idx * prog 的（按训练进度截断），
    再按编号升序取前 top_n 个存在的 .pt 文件。
    """
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    if not os.path.exists(full_json_path):
        print(f"注意：{log_dir} 不存在 elo_ratings.json，返回空队列。")
        return []

    with open(full_json_path, 'r', encoding='utf-8') as f:
        try:
            elo_dict = json.load(f)
        except Exception:
            print(f"注意：{log_dir} elo_ratings.json 解析失败，返回空队列。")
            return []

    # 只保留 actor_rein{整数} 格式
    rein_pattern = re.compile(r'^actor_rein(\d+)$')
    rein_keys = [(k, int(m.group(1))) for k in elo_dict if (m := rein_pattern.match(k))]

    if not rein_keys:
        print(f"注意：{log_dir} elo_ratings.json 中没有 actor_rein 条目。")
        return []

    max_idx = max(idx for _, idx in rein_keys)
    cutoff  = int(max_idx * prog)

    # 按进度截断后，按编号降序排列
    filtered = [(k, idx) for k, idx in rein_keys if idx <= cutoff]
    filtered.sort(key=lambda x: x[1], reverse=True) # 按编号降序排
    # filtered.sort(key=lambda x: elo_dict[x[0]], reverse=True) # 按Elo分值排

    # 构造完整路径，只保留文件实际存在的
    top_agents_paths = []
    for k, _ in filtered:
        full_path = os.path.join(log_dir, f"{k}.pt")
        if os.path.exists(full_path):
            top_agents_paths.append(full_path)
            if len(top_agents_paths) >= top_n:
                break

    print(f"  [{os.path.basename(log_dir)}] 进度截断 {cutoff}/{max_idx}，找到 {len(top_agents_paths)} 个智能体。")
    return top_agents_paths

def run_battle(env, blue_wrapper, red_wrapper, device):
    """仿真逻辑（参考 VsBaseline_while_... 的正确写法）"""
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
        'SLWSPFSP0.3-run-20260618-221044',
        'SLWSPFSP0.5-run-20260620-211720',
        'HLWSPFSP-run-20260616-130304',
        'CSPFSP-run-20260615-234324',
        'CSVersusRules-run-20260614-163906',
        'SLWSDPC-PFSP-run-20260614-000523',
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
        teams.append(get_top_elo_agents(log_dir, TEAM_SIZE, prog=progress))

    num_teams = len(teams)
    sota_idx = 0  # 第一个 mission 固定作为蓝方 sota

    # --- 预览各队伍成员范围，等待确认 ---
    print(f"\n{'='*65}")
    print(f"队伍预览  (训练进度截断 progress={progress:.0%})")
    print(f"SOTA任务 (Blue): {team_labels[sota_idx]}")
    print(f"{'='*65}")
    for idx, (name, team) in enumerate(zip(mission_names, teams)):
        # 从路径名还原 actor_rein 编号范围
        idxs = []
        for p in team:
            m = re.search(r'actor_rein(\d+)\.pt$', p)
            if m:
                idxs.append(int(m.group(1)))
        if idxs:
            range_str = f"编号 {min(idxs)} ~ {max(idxs)}，共 {len(idxs)} 个"
        else:
            range_str = f"共 {len(team)} 个（无法解析编号）"
        print(f"  [{idx}] {team_labels[idx]}")
        print(f"       {range_str}")
    print(f"{'='*65}")
    input("\n确认以上信息无误，按 Enter 开始对抗...")

    # 并行配置
    num_processes = min(cpu_count(), 20)
    print(f"\n开始SOTA任务 [{team_labels[sota_idx]}] 对抗其他任务 (progress={progress:.0%})...")
    print(f"并行进程数: {num_processes}")
    start_time = time.time()

    stats = {}  # {team_label: {'win': int, 'draw': int, 'loss': int, 'win_rate': float}}

    # 创建进程池
    with Pool(processes=num_processes) as pool:
        for i in range(1, num_teams):  # 其余 mission 作为红方依次挑战 sota_idx
            print(f"正在对抗: {team_labels[sota_idx]} (Blue/SOTA) vs {team_labels[i]} (Red)...")

            if not teams[i] or not teams[sota_idx]:
                print(f"  [SKIP] 某队智能体列表为空，跳过本轮对抗")
                continue

            battle_tasks = []
            for _ in range(TOTAL_ROUNDS):
                blue_path = random.choice(teams[sota_idx])  # SOTA作蓝方
                red_path  = random.choice(teams[i])         # 其余实验作红方
                battle_tasks.append((blue_path, red_path))

            results = pool.map(worker_process_battle, battle_tasks)

            wins   = sum(1 for r in results if r == 1.0)
            draws  = sum(1 for r in results if r == 0.5)
            losses = sum(1 for r in results if r == 0.0)
            score = sum(results)
            win_rate = score / TOTAL_ROUNDS

            stats[team_labels[i]] = {
                'win': wins,
                'draw': draws,
                'loss': losses,
                'win_rate': win_rate,
                'score': score
            }

            print(f"  -> {team_labels[sota_idx]} vs {team_labels[i]}: "
                  f"胜 {wins}, 平 {draws}, 负 {losses} | "
                  f"胜率: {win_rate:.2f} ({score:.1f}/{TOTAL_ROUNDS})")

    print(f"\n对抗计算完成！总耗时: {time.time() - start_time:.2f}s")

    # 打印汇总统计
    print(f"\n{'='*70}")
    print("对抗统计汇总（蓝方 vs 红方）")
    print(f"SOTA任务 (Blue): {team_labels[sota_idx]} (progress={progress:.0%})")
    print(f"{'-'*70}")
    for label, s in stats.items():
        print(f"{label:30s}: 胜 {s['win']:3d}, 平 {s['draw']:3d}, 负 {s['loss']:3d} | 胜率 {s['win_rate']:.2%}")
    print(f"{'='*70}")

    # 可选：保存为 JSON
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    progress_tag = f"progress{int(progress*100):03d}"
    json_path = os.path.join(project_root, "结果展示", "outputs", f"combat_stats_{progress_tag}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'sota': team_labels[sota_idx],
            'stats': stats,
            'total_rounds': TOTAL_ROUNDS,
            'progress': progress
        }, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存到: {json_path}")

    # --- 绘制分组条形图 ---
    labels = list(stats.keys())
    win_rates  = [stats[k]['win']  / TOTAL_ROUNDS * 100 for k in labels]
    draw_rates = [stats[k]['draw'] / TOTAL_ROUNDS * 100 for k in labels]
    loss_rates = [stats[k]['loss'] / TOTAL_ROUNDS * 100 for k in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*2), 6))
    rects1 = ax.bar(x - width, win_rates,  width, label='Win',  color='#2ecc71')
    rects2 = ax.bar(x,         draw_rates, width, label='Draw', color='#f39c12')
    rects3 = ax.bar(x + width, loss_rates, width, label='Loss', color='#e74c3c')

    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Combat Results (SOTA: {team_labels[sota_idx]}) (progress={progress:.0%})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    plt.subplots_adjust(top=0.9, bottom=0.15)

    # 在柱子上标注数值
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.0f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    bar_path = os.path.join(project_root, "结果展示", "outputs", f"combat_stats_bar_{progress_tag}.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"条形图已保存到: {bar_path}")
    plt.show()