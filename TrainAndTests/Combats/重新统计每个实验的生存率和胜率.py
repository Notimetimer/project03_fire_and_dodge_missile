"""
遍历指定日志目录中的 actor_rein*.pt，从最小序号~最大序号等间隔抽取 N 个版本，
对每个版本独立运行一轮 test_worker（No Random 方式：deterministic=True, restrict_fire=True），
将每次运行的重复回合数 num_runs 提升到 5，
导出每个规则的胜率、负率、平率、双杀率以及 score 到 CSV。
"""
import os
import sys
import glob
import re
import csv
import numpy as np
import torch
import argparse
import torch.multiprocessing as mp
from itertools import product
from _context import *

from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path
# 复用训练脚本中的 test_worker
from VsBaseline_while_training_hierarch_plus import test_worker

# ======================= 可配置参数区 =======================
# DIR_NAME = "PPO0.3_flymask_v0h0-run-20260928-111836"
# DIR_NAME = "SAC0.3_flymask_v1h1-run-20260928-093645"
# DIR_NAME = "PPO0.3_flymask_v0h0_fireSL-run-20260924-145554"
# DIR_NAME = "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260930-093137"
DIR_NAME_LIST = [
    # "NoIL_flymask_v0h0-run-20260924-224736",
    # "NoIL_flymask_v0h0-run-20260928-233924",
    # "NoIL_flymask_v0h0-run-20260909-131801",

    "PPO0.3_flymask_v0h0_fireSL-run-20260924-145554",
    "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654",
    "PPO0.3_flymask_v0h0_fireSL-run-20260930-125149",
    
    # "SAC0.3_flymask_v1h1-run-20260923-213238",
    # "SAC0.3_flymask_v1h1-run-20260928-093645",
    # "SAC0.3_flymask_v1h1-run-20260929-194715",
    
    # "PPO0.3_flymask_v0h0-run-20260921-194617",
    # "PPO0.3_flymask_v0h0-run-20260928-111836",
    # "PPO0.3_flymask_v0h0-run-20261001-152601",
    
    # "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260921-122428",
    # "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260928-233900",
    # "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260930-093137",
]
EXPERIMENT_NAME = None

# 抽取的版本数：按最小序号~最大序号等间隔抽取
NUM_PROGRESS_POINTS = 50 # 25

# 测试回合重复次数（由 3 提升到 5）
NUM_RUNS = 5

# 并行测试的 checkpoint 数量上限。每个 worker 负责一个 checkpoint（加载一次模型后
# 串行跑完所有 rule），checkpoint 之间相互独立可并行。
# None 表示自动取 min(CPU核数, checkpoint数)
MAX_PARALLEL_CHECKPOINTS = 5

# 测试对手规则编号列表
TEST_RULE_IDS = [0, 1, 2, 3]

# 测试场景参数（与训练脚本 CombatPPOWithIL3_parallel_hierarch 中测试段保持一致）
DT_MANEUVER = 0.2
ACTION_CYCLE_MULTIPLIER = 30
TEST_RED_INIT_AMMO = 6
TEST_BLUE_INIT_AMMO = 6
VERTICES = None
MAX_EPISODE_LEN = 15 * 60
R_CAGE = 62.00e3

# 模型网络结构（与训练启动脚本 熵实验_混合PFSP有预训练.py 一致）
HIDDEN_DIM = [128, 128, 128]

# 总步数（用于将文件名序号归一化为训练步数）
total_steps = 2e6

# 输出目录（相对 project_root）
OUT_DIR_NAME = os.path.join('结果展示', 'exp_png2')
# ===========================================================


def select_agents_by_interval(log_dir, num_points=20, total_steps=2e6):
    """
    扫描目录中 actor_rein*.pt，按编号排序，等间隔抽取 num_points 个文件。
    文件名数字归一化到 0~100% 后乘以 total_steps 得到实际步数。
    返回 (steps, paths) 两个列表，steps 对应训练步数，paths 对应文件路径。
    """
    files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    step_files = []
    for f in files:
        m = re.fullmatch(r'actor_rein(\d+(?:\.\d+)?)\.pt', os.path.basename(f))
        if m:
            step_files.append((float(m.group(1)), f))
    if not step_files:
        return [], []
    step_files.sort(key=lambda x: x[0])
    raw_numbers = [x[0] for x in step_files]
    max_number = max(raw_numbers) if raw_numbers else 1.0
    # 归一化到 0~100% 后乘以 total_steps
    percentages = [n / max_number * 100.0 for n in raw_numbers]
    steps = [p * total_steps / 100.0 for p in percentages]
    paths = [x[1] for x in step_files]

    # 等间隔抽取 num_points 个索引（最小序号~最大序号）
    if len(steps) <= num_points:
        selected_indices = list(range(len(steps)))
    else:
        selected_indices = np.linspace(0, len(steps) - 1, num_points, dtype=int)
    selected_steps = [steps[i] for i in selected_indices]
    selected_paths = [paths[i] for i in selected_indices]

    print(f"扫描到 {len(steps)} 个 actor_rein 文件，序号范围 [{raw_numbers[0]:.0f}, {raw_numbers[-1]:.0f}]，"
          f"等间隔抽取 {len(selected_steps)} 个")
    for i, (s, p) in enumerate(zip(selected_steps, selected_paths)):
        print(f"  [{i+1}/{len(selected_steps)}] 步数 {s:.0f}: {os.path.basename(p)}")
    return selected_steps, selected_paths


def run_test_for_checkpoint(agent_path, state_dim, hidden_dim, action_dims_dict,
                            env_args, dt_maneuver_val, num_runs, test_rule_ids,
                            action_cycle_multiplier, vertices,
                            red_init_ammo, blue_init_ammo):
    """
    加载一个 actor 版本的 state_dict，对所有规则运行 No Random 测试仿真。
    返回 dict: {rule_num: (score, win, lose, draw, perish_together)}
    """
    # .pt 文件直接就是 actor 的 state_dict
    model_state_dict = torch.load(agent_path, map_location='cpu', weights_only=False)

    # 并行对所有规则运行 test_worker
    pool = mp.Pool(processes=len(test_rule_ids))
    tasks = []
    for rule_num in test_rule_ids:
        kwds = {
            'model_state_dict': model_state_dict,
            'rule_num': rule_num,
            'env_args': env_args,
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'action_dims_dict': action_dims_dict,
            'dt_maneuver_val': dt_maneuver_val,
            'device_name': 'cpu',
            'num_runs': num_runs,
            'action_cycle_multiplier': action_cycle_multiplier,
            'no_out': 0,
            'deterministic': True,     # 机动动作确定化（No Random）
            'restrict_fire': True,      # 动作次序限制打开
            'vertices': vertices,
            'red_init_ammo': red_init_ammo,
            'blue_init_ammo': blue_init_ammo,
        }
        tasks.append(pool.apply_async(test_worker, kwds=kwds))
    results = [t.get() for t in tasks]
    pool.close()
    pool.join()

    # test_worker 返回: (rule_num, result(score), result2(return), wins, loses, draws, BVR_perish_togethers)
    outcomes = {}
    for rule_num, score, result2, wins, loses, draws, perish_together in results:
        outcomes[rule_num] = (score, wins, loses, draws, perish_together)
    return outcomes


def test_checkpoint_worker(task):
    """
    多进程 worker：负责单个 checkpoint 的全部规则测试。

    每个 worker 只加载一次模型 state_dict，然后串行跑完所有 rule，
    避免 checkpoint 内再起子进程池（嵌套 mp 在 Windows spawn 下有问题）。
    checkpoint 之间由外层进程池并行。

    参数 task 是一个元组（方便 pool.map 只传一个参数）：
        (agent_path, rule_ids, state_dim, hidden_dim, action_dims_dict,
         env_args, dt_maneuver_val, num_runs, action_cycle_multiplier,
         vertices, red_init_ammo, blue_init_ammo)
    返回: (agent_path, outcomes_dict) 或 (agent_path, None) 表示失败
    """
    (agent_path, rule_ids, state_dim, hidden_dim, action_dims_dict,
     env_args, dt_maneuver_val, num_runs, action_cycle_multiplier,
     vertices, red_init_ammo, blue_init_ammo) = task

    try:
        model_state_dict = torch.load(agent_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"  [失败] 加载模型出错 {os.path.basename(agent_path)}: {e}")
        return (agent_path, None)

    outcomes = {}
    for rule_num in rule_ids:
        try:
            rule_num_r, score, result2, wins, loses, draws, perish_together = test_worker(
                model_state_dict=model_state_dict,
                rule_num=rule_num,
                env_args=env_args,
                state_dim=state_dim,
                hidden_dim=hidden_dim,
                action_dims_dict=action_dims_dict,
                dt_maneuver_val=dt_maneuver_val,
                device_name='cpu',
                num_runs=num_runs,
                action_cycle_multiplier=action_cycle_multiplier,
                no_out=0,
                deterministic=True,
                restrict_fire=True,
                vertices=vertices,
                red_init_ammo=red_init_ammo,
                blue_init_ammo=blue_init_ammo,
            )
            outcomes[rule_num] = (score, wins, loses, draws, perish_together)
        except Exception as e:
            print(f"  [失败] {os.path.basename(agent_path)} rule={rule_num}: {e}")
            outcomes[rule_num] = (0.0, 0.0, 0.0, 0.0, 0.0)
    return (agent_path, outcomes)


def process_one_experiment(DIR_NAME, state_dim, action_dims_dict, env_args, rule_ids, logs_root_dir, out_dir):
    """
    处理单个实验目录：查找日志目录 -> 等间隔抽取 actor 版本 -> 对所有规则测试 -> 写入 CSV。
    若日志目录不存在则跳过并返回 False。
    """
    # 查找日志目录
    latest_log_dir = os.path.join(logs_root_dir, DIR_NAME) if DIR_NAME else \
        get_latest_log_dir(logs_root_dir, EXPERIMENT_NAME)

    if not latest_log_dir or not os.path.isdir(latest_log_dir):
        print(f"[跳过] 日志目录不存在: {latest_log_dir}")
        return False

    # 等间隔抽取 NUM_PROGRESS_POINTS 个 actor_rein 文件
    steps, agent_paths = select_agents_by_interval(latest_log_dir, NUM_PROGRESS_POINTS, total_steps)
    if not steps:
        print(f"[跳过] '{latest_log_dir}' 中没有 actor_rein 文件")
        return False

    # CSV 列定义
    header = ['step', 'actor_file']
    for r in rule_ids:
        header += [f'rule{r}_score', f'rule{r}_win', f'rule{r}_lose',
                   f'rule{r}_draw', f'rule{r}_perish']
    header += ['avg_score', 'avg_win', 'avg_lose', 'avg_draw', 'avg_perish']

    csv_path = os.path.join(out_dir, f"test_norandom_vs_rules_{DIR_NAME}.csv")
    print(f"\n结果将写入: {csv_path}")
    print(f"测试规则: {rule_ids}，num_runs={NUM_RUNS}，共 {len(steps)} 个版本")

    # 构建所有 checkpoint 的任务列表（每个任务 = 一个 checkpoint 的全部 rule 测试）
    tasks = [
        (agent_path, rule_ids, state_dim, HIDDEN_DIM, action_dims_dict,
         env_args, DT_MANEUVER, NUM_RUNS, ACTION_CYCLE_MULTIPLIER,
         VERTICES, TEST_RED_INIT_AMMO, TEST_BLUE_INIT_AMMO)
        for agent_path in agent_paths
    ]

    # 决定并行 worker 数
    n_checkpoints = len(tasks)
    if MAX_PARALLEL_CHECKPOINTS is not None:
        num_workers = min(MAX_PARALLEL_CHECKPOINTS, n_checkpoints)
    else:
        num_workers = min(os.cpu_count() or 1, n_checkpoints)
    print(f"并行测试: {num_workers} 个 worker × {n_checkpoints} 个 checkpoint\n")

    # checkpoint 之间并行，pool.map 保持输入顺序
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(test_checkpoint_worker, tasks)

    # 按顺序写 CSV
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cp_idx, (step, agent_path, (returned_path, outcomes)) in enumerate(
                zip(steps, agent_paths, results)):
            base = os.path.basename(agent_path)
            if outcomes is None:
                print(f"[{cp_idx+1}/{n_checkpoints}] 步数 {step:.0f} | {base}  -> 跳过（加载失败）")
                continue

            row = [f"{step:.0f}", base]
            scores, wins, loses, draws, perishes = [], [], [], [], []
            for r in rule_ids:
                score, w, l, d, p = outcomes.get(r, (0.0, 0.0, 0.0, 0.0, 0.0))
                row += [f"{score:.4f}", f"{w:.4f}", f"{l:.4f}", f"{d:.4f}", f"{p:.4f}"]
                scores.append(score); wins.append(w); loses.append(l)
                draws.append(d); perishes.append(p)

            row += [f"{np.mean(scores):.4f}", f"{np.mean(wins):.4f}",
                    f"{np.mean(loses):.4f}", f"{np.mean(draws):.4f}",
                    f"{np.mean(perishes):.4f}"]
            writer.writerow(row)
            print(f"[{cp_idx+1}/{n_checkpoints}] 步数 {step:.0f} | {base}  "
                  f"avg_score={np.mean(scores):.3f}  "
                  f"W/L/D={np.mean(wins):.2f}/{np.mean(loses):.2f}/{np.mean(draws):.2f}  "
                  f"perish={np.mean(perishes):.2f}")

    print(f"\n[完成] {DIR_NAME} -> CSV 已保存: {csv_path}")
    return True


def main():
    # 构建 env_args 与维度（与训练脚本测试段一致）—— 只构建一次，所有实验复用
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=MAX_EPISODE_LEN)
    parser.add_argument("--R-cage", type=float, default=R_CAGE)
    args = parser.parse_args([])

    dummy_env = ChooseStrategyEnv(args, tacview_show=False, vertices=VERTICES)
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    del dummy_env

    # 公共路径与规则列表
    logs_root_dir = os.path.join(project_root, "logs/combat")
    out_dir = os.path.join(project_root, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    rule_ids = sorted(TEST_RULE_IDS)

    # 遍历 DIR_NAME_LIST，逐个实验生成 CSV
    print(f"共 {len(DIR_NAME_LIST)} 个实验待处理")
    success, skipped = 0, 0
    for idx, DIR_NAME in enumerate(DIR_NAME_LIST, 1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(DIR_NAME_LIST)}] 处理实验: {DIR_NAME}")
        print("=" * 60)
        ok = process_one_experiment(DIR_NAME, state_dim, action_dims_dict, args,
                                    rule_ids, logs_root_dir, out_dir)
        if ok:
            success += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print(f"全部完成！成功 {success} 个，跳过 {skipped} 个。")


if __name__ == '__main__':
    main()
