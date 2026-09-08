"""
MARWIL epsilon_il (label_smoothing) sweep for pure imitation learning.

For each epsilon_il in EPSILON_IL_LIST, the script:
  1. Creates a fresh actor/critic from the same architecture used in CombatPPOWithIL3_parallel_hierarch.py.
  2. Runs 50 epochs of MARWIL offline imitation learning with label_smoothing=epsilon_il.
  3. Saves actor_candidate_{epsilon_il}.pt / critic_candidate_{epsilon_il}.pt.
  4. Evaluates the candidate against Rule 0,1,2,3 for 20 rounds each.
  5. Records the average win rate and policy entropy (higher is better for both).
  6. Computes the Pareto-optimal set and writes results to JSON/CSV.
"""

import os
import sys
import numpy as np
import pickle
import torch
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EPSILON_IL_LIST = [0.0, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
IL_EPOCHES = 50
IL_BATCH_SIZE = 128
IL_RULE = 2  # which rule-generated IL dataset to use (matches existing pkl files)
HIDDEN_DIM = [128, 128, 128]
TEST_RULE_IDS = [0, 1, 2, 3]
NUM_RUNS_PER_RULE = 3 # 20 # 测试场次数
TEST_WORKERS = 4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

CANDIDATE_DIR = os.path.join(cur_dir, "epsilon_sweep_candidates")
RESULT_JSON = os.path.join(cur_dir, "marwil_epsilon_sweep_results.json")
RESULT_CSV = os.path.join(cur_dir, "marwil_epsilon_sweep_results.csv")

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
        print(f"Loaded IL data from: {il_path}")
    else:
        print(f"File NOT found: {il_path}")
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
    print(f"IL dataset processed. Samples: {len(original_il_transition_dict['states'])}")
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


def train_one_epsilon(epsilon_il, original_il_transition_dict, state_dim, action_dims_dict, device, candidate_dir):
    print(f"\n========== Training candidate with epsilon_il = {epsilon_il} ==========")
    agent = create_fresh_agent(state_dim, HIDDEN_DIM, action_dims_dict, device)

    for epoch in range(IL_EPOCHES):
        avg_actor_loss, avg_critic_loss, c = agent.MARWIL_update(
            original_il_transition_dict,
            beta=1.0,
            batch_size=IL_BATCH_SIZE,
            label_smoothing=epsilon_il,
            no_bern=0,
        )
        if epoch % 1 == 0 or epoch == IL_EPOCHES - 1:
            print(f"  [epsilon_il={epsilon_il}] Epoch {epoch:3d}/{IL_EPOCHES}: "
                  f"actor_loss={avg_actor_loss:.4f}, critic_loss={avg_critic_loss:.4f}")

    actor_path = os.path.join(candidate_dir, f"actor_candidate_{epsilon_il}.pt")
    critic_path = os.path.join(candidate_dir, f"critic_candidate_{epsilon_il}.pt")
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    print(f"  Saved: {actor_path}")
    print(f"  Saved: {critic_path}")
    return agent, actor_path, critic_path


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
    total_entropy = entropy_cat + entropy_bern
    return {
        'entropy_cat': entropy_cat,
        'entropy_bern': entropy_bern,
        'total_entropy': total_entropy,
    }


def evaluate_candidate(actor_state_dict, env_args, state_dim, action_dims_dict, rule_ids, num_runs, n_workers):
    """Parallel evaluation against each baseline rule."""
    mp.set_start_method('spawn', force=True)
    results = []
    with mp.Pool(processes=n_workers) as pool:
        tasks = []
        for r_idx in rule_ids:
            kwds = {
                'model_state_dict': actor_state_dict,
                'rule_num': r_idx,
                'env_args': env_args,
                'state_dim': state_dim,
                'hidden_dim': HIDDEN_DIM,
                'action_dims_dict': action_dims_dict,
                'dt_maneuver_val': DT_MANEUVER,
                'device_name': 'cpu',
                'num_runs': num_runs,
                'action_cycle_multiplier': 10,
                'no_out': 0,
                'deterministic': True,
                'restrict_fire': True,
                'vertices': None,
                'red_init_ammo': 6,
                'blue_init_ammo': 6,
            }
            tasks.append(pool.apply_async(test_worker, kwds=kwds))
        for t in tasks:
            results.append(t.get())
    return results


def compute_pareto_optimal(results):
    """
    results: list of dicts, each has keys 'avg_win_rate' and 'policy_entropy'.
    Maximize both metrics. A point is Pareto-optimal if no other point has
    >= win rate AND >= entropy with at least one strict inequality.
    """
    pareto = []
    for i, r in enumerate(results):
        dominated = False
        for j, other in enumerate(results):
            if i == j:
                continue
            if (other['avg_win_rate'] >= r['avg_win_rate'] and
                other['policy_entropy'] >= r['policy_entropy']):
                if (other['avg_win_rate'] > r['avg_win_rate'] or
                    other['policy_entropy'] > r['policy_entropy']):
                    dominated = True
                    break
        if not dominated:
            pareto.append(r)
    # Sort by win rate descending for readability
    pareto.sort(key=lambda x: x['avg_win_rate'], reverse=True)
    return pareto


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(CANDIDATE_DIR, exist_ok=True)

    # Environment / dimensions
    env_args = setup_env_args()
    dummy_env = ChooseStrategyEnv(env_args)
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}

    # Load IL data once
    original_il_transition_dict = load_and_prepare_il_data()

    # Use the actual observation dimension from the dataset (POMDP may reduce it).
    state_dim = original_il_transition_dict['states'].shape[1]
    print(f"state_dim={state_dim}, action_dims_dict={action_dims_dict}")

    all_results = []

    for epsilon_il in EPSILON_IL_LIST:
        # 1. Train and save candidate
        agent, actor_path, critic_path = train_one_epsilon(
            epsilon_il, original_il_transition_dict, state_dim, action_dims_dict, DEVICE, CANDIDATE_DIR
        )

        # 2. Compute policy entropy (use same agent still in memory)
        entropy_info = compute_policy_entropy(agent, original_il_transition_dict, DEVICE)
        policy_entropy = entropy_info['total_entropy']

        # Free GPU memory if needed
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # 3. Evaluate against baselines
        print(f"  Evaluating candidate epsilon_il={epsilon_il} against rules {TEST_RULE_IDS} ...")
        actor_state_dict = {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()}
        test_results = evaluate_candidate(
            actor_state_dict, env_args, state_dim, action_dims_dict,
            TEST_RULE_IDS, NUM_RUNS_PER_RULE, TEST_WORKERS
        )

        rule_scores = {}
        for rule_num, result, result2, wins, loses, draws, bvr_pt in test_results:
            rule_scores[int(rule_num)] = {
                'avg_score': float(result),
                'avg_return': float(result2),
                'win_rate': float(wins),
                'lose_rate': float(loses),
                'draw_rate': float(draws),
                'bvr_perish_together_rate': float(bvr_pt),
            }
        avg_win_rate = float(np.mean([v['win_rate'] for v in rule_scores.values()]))

        result_entry = {
            'epsilon_il': epsilon_il,
            'actor_path': actor_path,
            'critic_path': critic_path,
            'avg_win_rate': avg_win_rate,
            'policy_entropy': policy_entropy,
            'entropy_cat': entropy_info['entropy_cat'],
            'entropy_bern': entropy_info['entropy_bern'],
            'per_rule_results': rule_scores,
        }
        all_results.append(result_entry)

        print(f"  [epsilon_il={epsilon_il}] avg_win_rate={avg_win_rate:.4f}, "
              f"policy_entropy={policy_entropy:.4f} (cat={entropy_info['entropy_cat']:.4f}, bern={entropy_info['entropy_bern']:.4f})")

    # 4. Pareto optimal set
    pareto_set = compute_pareto_optimal(all_results)

    summary = {
        'created_at': datetime.now().isoformat(),
        'config': {
            'epsilon_il_list': EPSILON_IL_LIST,
            'il_epoches': IL_EPOCHES,
            'il_batch_size': IL_BATCH_SIZE,
            'hidden_dim': HIDDEN_DIM,
            'test_rule_ids': TEST_RULE_IDS,
            'num_runs_per_rule': NUM_RUNS_PER_RULE,
            'device': str(DEVICE),
        },
        'all_results': all_results,
        'pareto_optimal': pareto_set,
    }

    # 5. Save JSON
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON results to: {RESULT_JSON}")

    # 6. Save CSV
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epsilon_il', 'avg_win_rate', 'policy_entropy', 'entropy_cat', 'entropy_bern',
                         'win_rate_vs_rule_0', 'win_rate_vs_rule_1', 'win_rate_vs_rule_2', 'win_rate_vs_rule_3'])
        for r in all_results:
            writer.writerow([
                r['epsilon_il'],
                f"{r['avg_win_rate']:.6f}",
                f"{r['policy_entropy']:.6f}",
                f"{r['entropy_cat']:.6f}",
                f"{r['entropy_bern']:.6f}",
                f"{r['per_rule_results'][0]['win_rate']:.6f}",
                f"{r['per_rule_results'][1]['win_rate']:.6f}",
                f"{r['per_rule_results'][2]['win_rate']:.6f}",
                f"{r['per_rule_results'][3]['win_rate']:.6f}",
            ])
    print(f"Saved CSV results to: {RESULT_CSV}")

    # 7. Print Pareto summary
    print("\n========== Pareto Optimal Set (maximize win rate & entropy) ==========")
    for r in pareto_set:
        print(f"  epsilon_il={r['epsilon_il']:5.2f}: avg_win_rate={r['avg_win_rate']:.4f}, "
              f"policy_entropy={r['policy_entropy']:.4f}")


if __name__ == '__main__':
    main()
