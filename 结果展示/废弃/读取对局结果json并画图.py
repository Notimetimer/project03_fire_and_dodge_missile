"""
读取 combat_stats.json 并绘制胜/平/负条形图 + 平均score折线
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from _context import *  # 包含 project_root

if __name__ == "__main__":
    json_path = os.path.join(project_root, "结果展示", "outputs", "combat_stats.json")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sota = data['sota']
    stats = data['stats']
    total_rounds = data['total_rounds']

    labels = list(stats.keys())
    win_rates  = [stats[k]['win']  / total_rounds * 100 for k in labels]
    draw_rates = [stats[k]['draw'] / total_rounds * 100 for k in labels]
    loss_rates = [stats[k]['loss'] / total_rounds * 100 for k in labels]
    # score = 胜率 + 平率 * 0.5
    scores = [(stats[k]['win'] + stats[k]['draw'] * 0.5) / total_rounds * 100 for k in labels]

    # 加入 vs Rule 的结果
    rule_json_path = os.path.join(project_root, "结果展示", "outputs", "vs_rule_stats.json")
    if os.path.exists(rule_json_path):
        with open(rule_json_path, 'r', encoding='utf-8') as f:
            rule_data = json.load(f)
        rule_total = rule_data['total_rounds']
        labels.append("Rule")
        win_rates.append(rule_data['win'] / rule_total * 100)
        draw_rates.append(rule_data['draw'] / rule_total * 100)
        loss_rates.append(rule_data['loss'] / rule_total * 100)
        scores.append((rule_data['win'] + rule_data['draw'] * 0.5) / rule_total * 100)

    x = np.arange(len(labels))
    width = 0.22

    fig, ax1 = plt.subplots(figsize=(max(8, len(labels) * 2), 6))

    # 条形图：胜/平/负
    rects1 = ax1.bar(x - width, win_rates,  width, label='Win',  color='#2ecc71')
    rects2 = ax1.bar(x,         draw_rates, width, label='Draw', color='#f39c12')
    rects3 = ax1.bar(x + width, loss_rates, width, label='Loss', color='#e74c3c')

    ax1.set_ylabel('Rate (%)')
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # 在柱子上标注数值
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax1.annotate(f'{height:.0f}',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8)

    # 叠加散点：Score
    ax2 = ax1.twinx()
    ax2.scatter(x, scores, color='#3498db', s=80, zorder=5, label='Score')
    ax2.set_ylabel('Score (%)')
    ax2.set_ylim(0, 105)

    # 在折线点上标注数值
    for i, s in enumerate(scores):
        ax2.annotate(f'{s:.1f}', xy=(x[i], s), xytext=(0, 8),
                     textcoords="offset points", ha='center', fontsize=9,
                     fontweight='bold', color='#2c3e50')

    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')

    ax1.set_title(f'Combat Results (SOTA: {sota})')
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.join(project_root, "结果展示", "outputs"), exist_ok=True)
    bar_path = os.path.join(project_root, "结果展示", "outputs", "combat_stats_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"条形图已保存到: {bar_path}")
    plt.show()
