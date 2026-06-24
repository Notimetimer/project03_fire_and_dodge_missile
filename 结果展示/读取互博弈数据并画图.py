import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _context import project_root

def load_combat_stats(json_path):
    """加载对抗统计数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_combat_results(data, output_dir):
    """绘制对抗结果条形图"""
    stats = data['stats']
    sota = data['baseline']
    total_rounds = data['total_rounds']
    
    labels = list(stats.keys())
    win_rates  = [stats[k]['win']  / total_rounds * 100 for k in labels]
    draw_rates = [stats[k]['draw'] / total_rounds * 100 for k in labels]
    loss_rates = [stats[k]['loss'] / total_rounds * 100 for k in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*2), 6))
    rects1 = ax.bar(x - width, win_rates,  width, label='Win',  color='#2ecc71')
    rects2 = ax.bar(x,         draw_rates, width, label='Draw', color='#f39c12')
    rects3 = ax.bar(x + width, loss_rates, width, label='Loss', color='#e74c3c')
    
    ax.set_ylabel('Rate (%)')
    # ax.set_title(f'Combat Results vs sota: {sota}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
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
    bar_path = os.path.join(output_dir, "combat_stats_bar.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"条形图已保存到: {bar_path}")
    plt.show()

def plot_win_rate_comparison(data, output_dir):
    """绘制胜率对比图"""
    stats = data['stats']
    sota = data['baseline']
    total_rounds = data['total_rounds']
    
    labels = list(stats.keys())
    win_rates = [stats[k]['win_rate'] * 100 for k in labels]
    
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 6))
    colors = ['#2ecc71' if wr >= 50 else '#e74c3c' for wr in win_rates]
    bars = ax.bar(labels, win_rates, color=colors, alpha=0.7)
    
    ax.set_ylabel('Win Rate (%)')
    # ax.set_title(f'Win Rate Comparison vs sota: {sota}')
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    
    # 在柱子上标注数值
    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        ax.annotate(f'{wr:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    win_rate_path = os.path.join(output_dir, "combat_win_rate_comparison.png")
    plt.savefig(win_rate_path, dpi=300, bbox_inches='tight')
    print(f"胜率对比图已保存到: {win_rate_path}")
    plt.show()

def export_to_csv(data, output_dir):
    """导出数据到CSV"""
    stats = data['stats']
    sota = data['baseline']
    total_rounds = data['total_rounds']
    
    # 准备数据
    df_data = []
    for label, s in stats.items():
        df_data.append({
            'Team': label,
            'sota': sota,
            'Win': s['win'],
            'Draw': s['draw'],
            'Loss': s['loss'],
            'Win_Rate': s['win_rate'],
            'Score': s['score'],
            'Total_Rounds': total_rounds
        })
    
    df = pd.DataFrame(df_data)
    csv_path = os.path.join(output_dir, "combat_stats.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV数据已导出到: {csv_path}")
    return df

if __name__ == "__main__":
    # 设置路径
    output_dir = os.path.join(project_root, "结果展示", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, "combat_stats.json")
    
    if not os.path.exists(json_path):
        print(f"错误: 未找到数据文件 {json_path}")
        print("请先运行 '最终智能体博弈，不画矩阵了.py' 生成数据")
    else:
        # 加载数据
        print(f"正在加载数据: {json_path}")
        data = load_combat_stats(json_path)
        
        print(f"\n基准任务 (Red): {data['baseline']}")
        print(f"总轮数: {data['total_rounds']}")
        print(f"\n对抗统计:")
        for label, s in data['stats'].items():
            print(f"  {label:30s}: 胜 {s['win']:3d}, 平 {s['draw']:3d}, 负 {s['loss']:3d} | 胜率 {s['win_rate']:.2%}")
        
        # 导出CSV
        df = export_to_csv(data, output_dir)
        print("\nCSV数据预览:")
        print(df.to_string(index=False))
        
        # 绘图
        print("\n正在生成图表...")
        plot_combat_results(data, output_dir)
        plot_win_rate_comparison(data, output_dir)
        
        print("\n所有图表生成完成！")
