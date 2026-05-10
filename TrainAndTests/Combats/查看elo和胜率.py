dir = r"D:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\logs\combat\IL_and_Mixed经典PFSP_挑战_并行_分层_训练圆分布-run-20260510-161932"

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_actor_keys(d):
    return [k for k in d.keys() if k.startswith('actor_rein')]


def parse_args():
    p = argparse.ArgumentParser(description='Plot Elo ratings and WinRates for actor_rein keys')
    p.add_argument('--dir', type=str, default=dir, help='Directory containing the JSON files')
    p.add_argument('--elo-file', type=str, default='elite_elo_ratings.json', help='Elo JSON filename')
    p.add_argument('--win-file', type=str, default='Elite_WinRates.json', help='WinRates JSON filename')
    p.add_argument('--out', type=str, default='elo_winrate.png', help='Output figure path')
    p.add_argument('--sort-by', choices=['name','elo','win'], default='name', help='Sort x-axis by this key')
    return p.parse_args()


def main():
    args = parse_args()
    elo_path = os.path.join(args.dir, args.elo_file)
    win_path = os.path.join(args.dir, args.win_file)

    if not os.path.exists(elo_path):
        raise FileNotFoundError(f'Elo file not found: {elo_path}')
    if not os.path.exists(win_path):
        raise FileNotFoundError(f'Win file not found: {win_path}')

    elo = load_json(elo_path)
    win = load_json(win_path)

    # extract actor_rein keys and sort by trailing numeric suffix
    import re
    actor_keys = sorted(set(extract_actor_keys(elo)) | set(extract_actor_keys(win)))
    if not actor_keys:
        raise ValueError('No keys starting with "actor_rein" found in either file')

    parsed = []
    for k in actor_keys:
        m = re.search(r'actor_rein(\d+)', k)
        if m:
            parsed.append((int(m.group(1)), k))
        else:
            # fallback: put at end with large number
            parsed.append((10**9, k))

    parsed.sort(key=lambda x: x[0])
    x_nums = [p[0] for p in parsed]
    keys_ord = [p[1] for p in parsed]

    # collect values in actor order
    elo_ord = []
    win_ord = []
    for k in keys_ord:
        try:
            elo_ord.append(float(elo.get(k, np.nan)))
        except Exception:
            elo_ord.append(np.nan)
        try:
            win_ord.append(float(win.get(k, np.nan)))
        except Exception:
            win_ord.append(np.nan)

    x = np.array(x_nums)

    fig, ax1 = plt.subplots(figsize=(max(6, len(keys_ord)*0.3), 4))
    ax2 = ax1.twinx()

    ax1.plot(x, elo_ord, marker='o', color='tab:blue', label='Actor Elo', markersize=4)
    ax2.plot(x, win_ord, marker='s', color='tab:orange', linestyle='--', label='Actor WinRate', markersize=4)

    # plot Rule entries as scatter across same x-range (evenly spaced)
    rule_keys = sorted([k for k in set(list(elo.keys()) + list(win.keys())) if k.startswith('Rule')])
    if rule_keys:
        n_rules = len(rule_keys)
        if len(x) > 0:
            x_rules = np.linspace(x.min(), x.max(), n_rules)
        else:
            x_rules = np.arange(n_rules)
        rule_elo = []
        rule_win = []
        for k in rule_keys:
            try:
                rule_elo.append(float(elo.get(k, np.nan)))
            except Exception:
                rule_elo.append(np.nan)
            try:
                rule_win.append(float(win.get(k, np.nan)))
            except Exception:
                rule_win.append(np.nan)
        ax1.scatter(x_rules, rule_elo, marker='x', color='red', label='Rule Elo')
        ax2.scatter(x_rules, rule_win, marker='D', color='green', label='Rule WinRate')

    ax1.set_ylabel('Elo')
    ax2.set_ylabel('Win Rate')
    # reduce number of x tick labels to avoid crowding
    max_ticks = 20
    if len(x) <= max_ticks:
        tick_pos = x
        tick_labels = [str(n) for n in x]
    else:
        idxs = np.linspace(0, len(x)-1, max_ticks, dtype=int)
        tick_pos = x[idxs]
        tick_labels = [str(x[i]) for i in idxs]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper left')

    plt.title('Actor Elo (left) and WinRate (right)')
    # add left/right margins and ensure y-axis labels are visible
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.25)
    ax1.tick_params(axis='y', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    plt.show()
    # fig.savefig(args.out, dpi=200)
    # print(f'Saved figure to {args.out}')


if __name__ == '__main__':
    main()
