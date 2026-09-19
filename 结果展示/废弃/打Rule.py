import os
import sys
import json
import time
import torch
import numpy as np
import argparse
import random
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from multiprocessing import Pool, cpu_count

from _context import * # 包含 project_root
sys.path.insert(0, os.path.join(project_root, "TrainAndTests", "Combats"))
from TrainAndTests.Combats.VsBaseline_while_training_hierarch import test_worker
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper
from Utilities.LocateDirAndAgents2 import get_latest_log_dir

# --- 1. 配置参数 ---
dt_maneuver = 0.2
TOTAL_ROUNDS = 80
RULE_NUM = 3  # 对抗 Rule3

# --- 2. 核心辅助函数 ---

def load_team_from_txt(log_dir):
    """
    从 log_dir/top_batch_names.txt 读取预设队伍名称，
    构造完整 .pt 路径，只保留文件实际存在的。
    """
    txt_path = os.path.join(log_dir, "top_batch_names.txt")
    if not os.path.exists(txt_path):
        print(f"注意：{log_dir} 不存在 top_batch_names.txt，返回空队列。")
        return []

    with open(txt_path, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]

    agents_paths = []
    for name in names:
        full_path = os.path.join(log_dir, f"{name}.pt")
        if os.path.exists(full_path):
            agents_paths.append(full_path)
        else:
            print(f"  警告：{full_path} 不存在，跳过。")

    print(f"  [{os.path.basename(log_dir)}] 从 top_batch_names.txt 加载了 {len(agents_paths)} 个智能体。")
    return agents_paths


def worker_vs_rule(args_pack):
    """
    子进程：加载一个智能体与 Rule 对抗 1 场
    """
    agent_path, rule_num = args_pack

    device = torch.device("cpu")
    torch.set_num_threads(1)

    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3)
    env = ChooseStrategyEnv(env_args, tacview_show=0)
    state_dim = env.obs_dim
    hidden_dim = [128, 128, 128]
    action_dims = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}

    net = PolicyNetHybrid(state_dim, hidden_dim, action_dims).to(device)
    actor = HybridActorWrapper(net, action_dims, None, device).to(device)

    try:
        state_dict = torch.load(agent_path, map_location=device, weights_only=True)
        actor.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"模型加载出错: {e}")
        return 0.5  # 异常时算平局

    actor.eval()

    _, result, _, wins, loses, draws = test_worker(
        model_state_dict=actor.state_dict(),
        rule_num=rule_num,
        env_args=env_args,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dims_dict=action_dims,
        dt_maneuver_val=dt_maneuver,
        device_name='cpu',
        num_runs=1,
        action_cycle_multiplier=10,
        no_out=0,
        deterministic=True,
        restrict_fire=True,
    )
    return result  # 1.0 胜, 0.0 负, 0.5 平

# --- 3. 主程序 ---
if __name__ == "__main__":
    mission_name = 'SLWSPFSP0.3-run-20260618-221044'
    logs_root_dir = os.path.join(project_root, "logs", "combat")

    log_dir = os.path.join(logs_root_dir, mission_name)
    if not os.path.exists(log_dir):
        log_dir = get_latest_log_dir(logs_root_dir, mission_name)
    if not log_dir:
        raise FileNotFoundError(f"未找到任务目录: {mission_name}")

    # 加载预设队伍
    team = load_team_from_txt(log_dir)
    if not team:
        raise RuntimeError("队伍为空，无法进行对抗")

    print(f"\n{'='*65}")
    print(f"任务: {mission_name}")
    print(f"队伍成员数: {len(team)}")
    print(f"对抗对象: Rule{RULE_NUM}")
    print(f"总回合数: {TOTAL_ROUNDS}")
    print(f"{'='*65}")

    # 构建并行任务：每回合随机抽一个智能体与 Rule 对抗
    battle_tasks = []
    for _ in range(TOTAL_ROUNDS):
        agent_path = random.choice(team)
        battle_tasks.append((agent_path, RULE_NUM))

    num_processes = min(cpu_count(), 20)
    print(f"并行进程数: {num_processes}")
    start_time = time.time()

    with Pool(processes=num_processes) as pool:
        results = pool.map(worker_vs_rule, battle_tasks)

    elapsed = time.time() - start_time

    # 统计
    wins   = sum(1 for r in results if r == 1.0)
    draws  = sum(1 for r in results if r == 0.5)
    losses = sum(1 for r in results if r == 0.0)
    win_rate  = wins / TOTAL_ROUNDS
    draw_rate = draws / TOTAL_ROUNDS
    loss_rate = losses / TOTAL_ROUNDS

    print(f"\n对抗完成！耗时: {elapsed:.2f}s")
    print(f"{'='*50}")
    print(f"  vs Rule{RULE_NUM}  ({TOTAL_ROUNDS} 回合)")
    print(f"  胜: {wins:3d} ({win_rate:.2%})")
    print(f"  平: {draws:3d} ({draw_rate:.2%})")
    print(f"  负: {losses:3d} ({loss_rate:.2%})")
    print(f"{'='*50}")

    # 保存结果
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    json_path = os.path.join(project_root, "结果展示", "outputs", "vs_rule_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'mission': mission_name,
            'rule_num': RULE_NUM,
            'total_rounds': TOTAL_ROUNDS,
            'win': wins,
            'draw': draws,
            'loss': losses,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'loss_rate': loss_rate,
        }, f, ensure_ascii=False, indent=2)
    print(f"统计结果已保存到: {json_path}")