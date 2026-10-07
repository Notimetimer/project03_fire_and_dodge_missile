"""
纯模仿学习结果对抗：统一训练 + 统一测试。

流程：
  1. 为每个 epsilon_il (label_smoothing) 训练一个候选策略，各 30 epoch。
  2. 保存 actor_candidate_{epsilon_il}.pt / critic_candidate_{epsilon_il}.pt。
  3. 对每个候选，与 Rule 0,1,2,3 各对战 NUM_RUNS_PER_RULE 场。
  4. 统计平均胜率、策略熵，输出 Pareto 最优解集到 JSON/CSV。

用法：
    conda run -n 38 python -u TrainAndTests/Combats/纯模仿学习结果对抗.py
"""

import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import csv
from datetime import datetime
import torch.multiprocessing as mp
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(cur_dir)
from BasicRules_new_hierarchical import *
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from VsBaseline_while_training_hierarch_plus import test_worker
from calc_Score_pareto import compute_score_pareto

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPSILON_IL_LIST = [0, 0.04, 0.08, 0.16, 0.32, 0.64] # [0.0, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
IL_EPOCHES = 50  # 50  如需严格 50 epoch 可改回 50
IL_BATCH_SIZE = 128
IL_RULE = 2  # 使用的示范数据集规则编号
HIDDEN_DIM = [128, 128, 128]
TEST_RULE_IDS = [0, 1, 2, 3]
NUM_RUNS_PER_RULE = 5  # 每个候选对每个 rule 跑多少组测试；每组 EPISODES_PER_GROUP 场（test_worker 原有设定）
EPISODES_PER_GROUP = 4  # test_worker 一次调用内部跑的场次数（VsBaseline 原有设定）
TEST_WORKERS = 45
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
INIT_AMMO = 4 # 初始4~6枚导弹
ACTOR_LR = 1e-4
CRITIC_LR = 5e-4
GAMMA = 0.97
LMBDA = 0.985
PPO_EPOCHS = 4
PPO_EPS = 0.2
K_ENTROPY = {'cont': 0.01, 'cat': 0.008, 'bern': 0.003}

MAX_EPISODE_DURATION = 15 * 60
R_CAGE = 62.00e3
DT_MANEUVER = 0.2

# 读取 mask_config.json 中的 ver/hor 开关状态，并带上 mask 标记保存结果
MASK_CONFIG_PATH = os.path.join(project_root, 'Algorithms', 'mask_config.json')
try:
    with open(MASK_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _mask_cfg = json.load(f)
    VER = int(_mask_cfg.get('ver', 0))
    HOR = int(_mask_cfg.get('hor', 0))
except Exception as e:
    print(f"[Warning] Failed to load mask_config.json: {e}. Using ver=0, hor=0.", flush=True)
    VER = 0
    HOR = 0
MASK_TAG = f"v{VER}h{HOR}"

CANDIDATE_DIR = os.path.join(cur_dir, f"epsilon_sweep_candidates_{MASK_TAG}")
RESULT_JSON = os.path.join(cur_dir, f"marwil_epsilon_sweep_results_{MASK_TAG}.json")
RESULT_CSV = os.path.join(cur_dir, f"marwil_epsilon_sweep_results_{MASK_TAG}.csv")

# ---------------------------------------------------------------------------
# Helpers copied/adapted from CombatPPOWithIL3_parallel_hierarch.py
# ---------------------------------------------------------------------------
def load_il_and_transitions(folder, il_name, rl_name):
    if folder is None:
        folder = os.getcwd()
    il_path = os.path.join(folder, il_name)
    trans_path = os.path.join(folder, rl_name)
    il = None
    trans = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
        print(f"Loaded IL data from: {il_path}", flush=True)
    else:
        print(f"File NOT found: {il_path}", flush=True)
    if os.path.isfile(trans_path):
        with open(trans_path, "rb") as f:
            trans = pickle.load(f)
    return il, trans


def restructure_actions(actions_data):
    """List[Dict] -> Dict[Array] with keys 'cat' and 'bern'."""
    if isinstance(actions_data, dict):
        return actions_data
    if isinstance(actions_data, list) and len(actions_data) > 0:
        new_actions = {'cat': [], 'bern': []}
        for item in actions_data:
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item()
            if isinstance(act, dict):
                val_cat = act.get('fly', act.get('cat'))
                if val_cat is not None:
                    new_actions['cat'].append(val_cat)
                val_bern = act.get('fire', act.get('bern'))
                if val_bern is not None:
                    new_actions['bern'].append(val_bern)
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                new_actions['cat'].append(act[0])
                new_actions['bern'].append(act[1])
        for k in new_actions:
            arr = np.array(new_actions[k])
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            new_actions[k] = arr
        return new_actions
    return actions_data


def setup_env_args():
    return argparse.Namespace(max_episode_len=MAX_EPISODE_DURATION, R_cage=R_CAGE)


def load_and_prepare_il_data(il_dir=None):
    if il_dir is None:
        il_dir = os.path.join(cur_dir, "IL")
    original_il_transition_dict, _ = load_il_and_transitions(
        il_dir,
        f"il_transitions_combat_LR_rule{IL_RULE}.pkl",
        f"transition_dict_combat_LR_rule{IL_RULE}.pkl"
    )
    if original_il_transition_dict is None:
        raise RuntimeError("Failed to load IL data.")
    original_il_transition_dict['actions'] = restructure_actions(original_il_transition_dict['actions'])
    if 'states' in original_il_transition_dict:
        original_il_transition_dict['states'] = np.array(original_il_transition_dict['states'], dtype=np.float32)
    if 'returns' in original_il_transition_dict:
        original_il_transition_dict['returns'] = np.array(original_il_transition_dict['returns'], dtype=np.float32)
    print(f"IL dataset processed. Samples: {len(original_il_transition_dict['states'])}", flush=True)
    return original_il_transition_dict


def create_fresh_agent(state_dim, hidden_dim, action_dims_dict, device):
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)
    student_agent = PPOHybrid(
        actor=actor_wrapper,
        critic=critic_net,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        lmbda=LMBDA,
        epochs=PPO_EPOCHS,
        eps=PPO_EPS,
        gamma=GAMMA,
        device=device,
        k_entropy=K_ENTROPY,
    )
    student_agent.set_learning_rate(actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    return student_agent


def train_one_epsilon(epsilon_il, original_il_transition_dict, state_dim, action_dims_dict, device, candidate_dir, writer):
    print(f"\n========== Training candidate with epsilon_il = {epsilon_il} ==========", flush=True)
    agent = create_fresh_agent(state_dim, HIDDEN_DIM, action_dims_dict, device)

    losses = []
    for epoch in range(IL_EPOCHES):
        avg_actor_loss, avg_critic_loss, c = agent.MARWIL_update(
            original_il_transition_dict,
            beta=1.0,
            batch_size=IL_BATCH_SIZE,
            label_smoothing=epsilon_il,
            no_bern=0,
        )
        categorical_match_rate = float(agent.marwil_accuracy_cat)
        strategy_entropy = sum(float(value or 0.0) for value in [agent.marwil_entropy_cat, agent.marwil_entropy_bern])
        writer.add_scalars("categorical_match_rate", {f"epsilon_{epsilon_il:g}": categorical_match_rate}, epoch)
        writer.add_scalars("policy_entropy", {f"epsilon_{epsilon_il:g}": strategy_entropy}, epoch)
        losses.append({
            'epoch': epoch,
            'actor_loss': float(avg_actor_loss),
            'critic_loss': float(avg_critic_loss),
            'categorical_match_rate': categorical_match_rate,
            'policy_entropy': strategy_entropy,
        })
        if epoch % 1 == 0 or epoch == IL_EPOCHES - 1:
            print(f"  [epsilon_il={epsilon_il}] Epoch {epoch:3d}/{IL_EPOCHES}: "
                  f"actor_loss={avg_actor_loss:.4f}, critic_loss={avg_critic_loss:.4f}", flush=True)

    actor_path = os.path.join(candidate_dir, f"actor_candidate_{epsilon_il}.pt")
    critic_path = os.path.join(candidate_dir, f"critic_candidate_{epsilon_il}.pt")
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    print(f"  Saved: {actor_path}", flush=True)
    print(f"  Saved: {critic_path}", flush=True)
    return agent, actor_path, critic_path, losses


def compute_policy_entropy(agent, original_il_transition_dict, device, max_samples=5000):
    """Compute policy entropy on a fixed sample of IL states."""
    states = original_il_transition_dict['states']
    actions = original_il_transition_dict['actions']
    n = min(len(states), max_samples)
    if n < len(states):
        idx = np.random.choice(len(states), n, replace=False)
    else:
        idx = np.arange(n)

    states_t = torch.tensor(states[idx], dtype=torch.float).to(device)
    actions_all = {
        'cat': torch.tensor(actions['cat'][idx], dtype=torch.long).to(device),
        'bern': torch.tensor(actions['bern'][idx], dtype=torch.float).to(device),
    }

    metrics = agent.actor.compute_marwil_monitor(states_t, actions_all)
    entropy_cat = metrics.get('entropy_cat', 0.0) or 0.0
    entropy_bern = metrics.get('entropy_bern', 0.0) or 0.0
    return {
        'entropy_cat': float(entropy_cat),
        'entropy_bern': float(entropy_bern),
        'total_entropy': float(entropy_cat + entropy_bern),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_test_job(job):
    """
    Worker function for one (candidate, rule, group) test job.
    Loads the actor and runs test_worker (EPISODES_PER_GROUP 场) for the specified rule.
    """
    actor_path, epsilon_il, rule_num, group_idx, num_runs, env_args, state_dim, action_dims_dict = job
    actor_state_dict = torch.load(actor_path, map_location='cpu')
    result = test_worker(
        model_state_dict=actor_state_dict,
        rule_num=rule_num,
        env_args=env_args,
        state_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        action_dims_dict=action_dims_dict,
        dt_maneuver_val=DT_MANEUVER,
        device_name='cpu',
        num_runs=num_runs,
        action_cycle_multiplier=10,
        no_out=0,
        deterministic=False,
        restrict_fire=False,
        vertices=None,
        red_init_ammo=INIT_AMMO,
        blue_init_ammo=INIT_AMMO,
    )
    return epsilon_il, rule_num, group_idx, result


def evaluate_all_candidates_flat(candidates, env_args, state_dim, action_dims_dict, rule_ids, num_groups, n_workers):
    """
    Flatten all (candidate, rule, group) tests into a single job queue and consume it
    with n_workers processes. Workers pull the next job as soon as they finish,
    eliminating idle time between candidates.

    Each job calls test_worker with EPISODES_PER_GROUP 场. NUM_RUNS_PER_RULE
    is the number of such groups per (candidate, rule) pair.
    """
    jobs = []
    for cand in candidates:
        for rule_num in rule_ids:
            for group_idx in range(num_groups):
                jobs.append((cand['actor_path'], cand['epsilon_il'], rule_num, group_idx, EPISODES_PER_GROUP,
                             env_args, state_dim, action_dims_dict))

    total_episodes = len(jobs) * EPISODES_PER_GROUP
    print(f"  Total test jobs: {len(jobs)} ({len(candidates)} candidates x {len(rule_ids)} rules x {num_groups} groups)", flush=True)
    print(f"  Episodes per group: {EPISODES_PER_GROUP}, total episodes: {total_episodes}", flush=True)
    print(f"  Workers: {n_workers}", flush=True)

    mp.set_start_method('spawn', force=True)
    flat_results = []
    with mp.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(run_test_job, jobs):
            flat_results.append(res)
            print(f"  Finished job {len(flat_results)}/{len(jobs)}: "
                  f"epsilon_il={res[0]}, rule={res[1]}, group={res[2]}", flush=True)

    # Aggregate by epsilon_il -> rule_num, averaging over groups
    results_by_epsilon = {cand['epsilon_il']: {rule: [] for rule in rule_ids} for cand in candidates}
    for epsilon_il, rule_num, group_idx, result in flat_results:
        results_by_epsilon[epsilon_il][int(rule_num)].append(result)

    averaged_results = {cand['epsilon_il']: {} for cand in candidates}
    for epsilon_il in results_by_epsilon:
        for rule_num, result_list in results_by_epsilon[epsilon_il].items():
            if len(result_list) == 0:
                continue
            # result tuple: (rule_num, avg_score, avg_return, win_rate, lose_rate, draw_rate, bvr_perish_together_rate)
            avg_result = list(result_list[0])
            for r in result_list[1:]:
                for i in range(1, len(r)):
                    avg_result[i] += r[i]
            for i in range(1, len(avg_result)):
                avg_result[i] /= len(result_list)
            averaged_results[epsilon_il][int(rule_num)] = tuple(avg_result)
    return averaged_results


# ---------------------------------------------------------------------------
# Main pipeline: train all candidates, then test all candidates
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    env_args = setup_env_args()
    dummy_env = ChooseStrategyEnv(env_args)
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}

    original_il_transition_dict = load_and_prepare_il_data()
    state_dim = original_il_transition_dict['states'].shape[1]
    print(f"state_dim={state_dim}, action_dims_dict={action_dims_dict}", flush=True)

    # ===== Stage 1: train all candidates =====
    print("\n================ Stage 1: Train all candidates ================", flush=True)
    candidates = []
    train_start = datetime.now()
    tensorboard_dir = os.path.join(project_root, "logs", "marwil_epsilon_sweep", f"{MASK_TAG}-{train_start.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"TensorBoard logs: {tensorboard_dir}", flush=True)

    for epsilon_il in EPSILON_IL_LIST:
        agent, actor_path, critic_path, losses = train_one_epsilon(
            epsilon_il, original_il_transition_dict, state_dim, action_dims_dict, DEVICE, CANDIDATE_DIR, writer
        )
        entropy_info = compute_policy_entropy(agent, original_il_transition_dict, DEVICE)

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        candidates.append({
            'epsilon_il': epsilon_il,
            'actor_path': actor_path,
            'critic_path': critic_path,
            'entropy_info': entropy_info,
            'final_losses': losses[-1] if losses else None,
        })
        print(f"  [epsilon_il={epsilon_il}] entropy={entropy_info['total_entropy']:.4f} "
              f"(cat={entropy_info['entropy_cat']:.4f}, bern={entropy_info['entropy_bern']:.4f})", flush=True)

    writer.close()
    print(f"\nStage 1 finished. Training time: {(datetime.now() - train_start).total_seconds()/60:.2f} minutes", flush=True)

    # ===== Stage 2: test all candidates (flat queue) =====
    print("\n================ Stage 2: Test all candidates ================", flush=True)
    all_results = []
    test_start = datetime.now()

    results_by_epsilon = evaluate_all_candidates_flat(
        candidates, env_args, state_dim, action_dims_dict,
        TEST_RULE_IDS, NUM_RUNS_PER_RULE, TEST_WORKERS
    )

    for cand in candidates:
        epsilon_il = cand['epsilon_il']
        entropy_info = cand['entropy_info']
        rule_scores = {}
        for rule_num, result in results_by_epsilon[epsilon_il].items():
            _, result2, wins, loses, draws, bvr_pt = result[1:]
            rule_scores[int(rule_num)] = {
                'avg_score': float(result[1]),
                'avg_return': float(result2),
                'win_rate': float(wins),
                'lose_rate': float(loses),
                'draw_rate': float(draws),
                'bvr_perish_together_rate': float(bvr_pt),
            }
        avg_score = float(np.mean([v['avg_score'] for v in rule_scores.values()]))

        result_entry = {
            'epsilon_il': epsilon_il,
            'actor_path': cand['actor_path'],
            'critic_path': cand['critic_path'],
            'avg_score': avg_score,
            'policy_entropy': entropy_info['total_entropy'],
            'entropy_cat': entropy_info['entropy_cat'],
            'entropy_bern': entropy_info['entropy_bern'],
            'score_vs_rule_0': rule_scores[0]['avg_score'],
            'score_vs_rule_1': rule_scores[1]['avg_score'],
            'score_vs_rule_2': rule_scores[2]['avg_score'],
            'score_vs_rule_3': rule_scores[3]['avg_score'],
            'per_rule_results': rule_scores,
        }
        all_results.append(result_entry)

        print(f"  [epsilon_il={epsilon_il}] avg_score={avg_score:.4f}, "
              f"r0={rule_scores[0]['avg_score']:.2f}, r1={rule_scores[1]['avg_score']:.2f}, "
              f"r2={rule_scores[2]['avg_score']:.2f}, r3={rule_scores[3]['avg_score']:.2f}, "
              f"entropy={entropy_info['total_entropy']:.4f}", flush=True)

    print(f"\nStage 2 finished. Testing time: {(datetime.now() - test_start).total_seconds()/60:.2f} minutes", flush=True)

    # ===== Stage 3: Pareto & save =====
    pareto_objectives = ['avg_score', 'min_rule_score']
    pareto_set = compute_score_pareto(all_results, TEST_RULE_IDS)

    summary = {
        'created_at': datetime.now().isoformat(),
        'config': {
            'epsilon_il_list': EPSILON_IL_LIST,
            'il_epoches': IL_EPOCHES,
            'il_batch_size': IL_BATCH_SIZE,
            'il_rule': IL_RULE,
            'hidden_dim': HIDDEN_DIM,
            'test_rule_ids': TEST_RULE_IDS,
            'num_groups_per_rule': NUM_RUNS_PER_RULE,
            'episodes_per_group': EPISODES_PER_GROUP,
            'total_episodes_per_rule': NUM_RUNS_PER_RULE * EPISODES_PER_GROUP,
            'pareto_objectives': pareto_objectives,
            'pareto_method': 'score_pareto_then_max_entropy',
            'mask_tag': MASK_TAG,
            'ver': VER,
            'hor': HOR,
            'device': str(DEVICE),
        },
        'all_results': all_results,
        'pareto_optimal': pareto_set,
    }

    # Save JSON
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON results to: {RESULT_JSON}", flush=True)

    # Save CSV
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon_il', 'avg_score', 'min_rule_score', 'policy_entropy', 'entropy_cat', 'entropy_bern',
                         'score_vs_rule_0', 'score_vs_rule_1', 'score_vs_rule_2', 'score_vs_rule_3',
                         'is_pareto_optimal', 'is_max_entropy_pareto'])
        for r in all_results:
            writer.writerow([
                r['epsilon_il'],
                f"{r['avg_score']:.6f}",
                f"{r['min_rule_score']:.6f}",
                f"{r['policy_entropy']:.6f}",
                f"{r['entropy_cat']:.6f}",
                f"{r['entropy_bern']:.6f}",
                f"{r['score_vs_rule_0']:.6f}",
                f"{r['score_vs_rule_1']:.6f}",
                f"{r['score_vs_rule_2']:.6f}",
                f"{r['score_vs_rule_3']:.6f}",
                str(r['is_pareto_optimal']),
                str(r['is_max_entropy_pareto']),
            ])
    print(f"Saved CSV results to: {RESULT_CSV}", flush=True)

    # Print summary
    print("\n========== Full Results (score Pareto, then maximum entropy) ==========", flush=True)
    for r in all_results:
        pareto_mark = "[SELECT]" if r['is_max_entropy_pareto'] else "[PARETO]" if r['is_pareto_optimal'] else "        "
        print(f"  {pareto_mark} epsilon_il={r['epsilon_il']:5.2f}: "
              f"r0={r['score_vs_rule_0']:.3f}, r1={r['score_vs_rule_1']:.3f}, "
              f"r2={r['score_vs_rule_2']:.3f}, r3={r['score_vs_rule_3']:.3f}, "
              f"min={r['min_rule_score']:.3f}, avg={r['avg_score']:.3f}, "
              f"entropy={r['policy_entropy']:.4f}", flush=True)

    print("\n========== Pareto Optimal Set (min score, average score, entropy) ==========", flush=True)
    for r in pareto_set:
        print(f"  epsilon_il={r['epsilon_il']:5.2f}: "
              f"r0={r['score_vs_rule_0']:.3f}, r1={r['score_vs_rule_1']:.3f}, "
              f"r2={r['score_vs_rule_2']:.3f}, r3={r['score_vs_rule_3']:.3f}, "
              f"entropy={r['policy_entropy']:.4f}", flush=True)


if __name__ == '__main__':
    main()
