import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from _context import * # 包含 project_root

# --- 1. 环境与绘图配置 ---
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False
TEAM_SIZE = 25     # 每队成员数

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
            if os.path.exists(agent_path) and agent_id != 0: # 排除 ID 为 0 的点
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
        
        # 如果取到了就加入球队，否则从相邻区间凑数（提示）
        if not team:
            print(f"警告: 区间 [{start_id}, {end_id}) 内未找到任何智能体")
        teams.append(team)
        
    return teams

def plot_elo_sampling(log_dir, teams, team_labels, name):
    """绘制 Elo 曲线及采样点分布"""
    print(f"正在读取 Elo 数据并绘图...")
    
    with open(os.path.join(log_dir, "elo_ratings.json"), 'r', encoding='utf-8') as f:
        elo_data = json.load(f)
    
    all_plot_data = []
    for k, v in elo_data.items():
        match = re.search(r'^actor_rein(\d+)$', k)
        if match:
            agent_id = int(match.group(1))
            if agent_id != 0: # 绘图时同样排除 ID 为 0 的点
                all_plot_data.append([agent_id, v])
    
    if not all_plot_data:
        print("错误: 未找到任何 Elo 绘图数据。")
        return

    all_plot_data.sort(key=lambda x: x[0])
    all_plot_data = np.array(all_plot_data)

    # --- [新增] 步数等比例转换逻辑 ---
    TOTAL_STEPS = 13.6e6  # 指定总步数为 13.6M
    max_id = all_plot_data[:, 0].max()
    scale_factor = TOTAL_STEPS / max_id
    all_plot_data[:, 0] *= scale_factor # 转换整个序列的横轴
    # -----------------------------

    plt.figure(figsize=(13, 8))
    plt.plot(all_plot_data[:, 0], all_plot_data[:, 1], color='gray', alpha=0.4, label='Elo Evolution', linewidth=2.0)
    
    # 使用 colormap 生成动态颜色
    cmap = plt.get_cmap('tab10')
    num_teams = len(teams)
    for idx, team in enumerate(teams):
        if not team: continue
        # [修改] 采样点的横轴也需要进行同样的等比例缩放
        team_ids = np.array([a['id'] for a in team]) * scale_factor
        team_elos = np.array([a['elo'] for a in team])
        plt.scatter(team_ids, team_elos, color=cmap(idx % 10), label=f'Team {team_labels[idx]} Member', s=45, edgecolors='white', linewidth=0.5)
    
    # --- [新增] 绘制区间分界虚线 (无图例) ---
    interval_size = (max_id + 1) / num_teams
    for i in range(1, num_teams):
        boundary_step = i * interval_size * scale_factor
        plt.axvline(x=boundary_step, linestyle='--', color='k', alpha=0.7, linewidth=1, label=None)
    # -----------------------------

    # 字体与布局设置
    plt.xlabel("Training Steps", fontsize=18) # 修改横轴标签
    plt.ylabel("Elo Rating", fontsize=18)
    
    # 优化刻度显示：使用科学计数法 (×10^n)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    plt.gca().xaxis.get_offset_text().set_fontsize(14) # 设置偏移量文字大小
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # [修改] 移除标题，手动设置留白 (left, right, top, bottom)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.15)
    
    plot_save_path = os.path.join(project_root, "结果展示", "outputs", f"elo_sampling_{name}.png")
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path, dpi=300)
    print(f"图像已保存至: {plot_save_path}")

# --- 主程序 ---
if __name__ == "__main__":
    # 配置任务名称
    name = 'IL_and_MixedPFSP_分阶段_挑战_并行_分层-run-20260326-172341'
    log_dir = os.path.join(project_root, "logs", "combat", name)
    
    # 1. 准备队伍
    team_labels = ['1/4', '2/4', '3/4', '4/4']
    try:
        teams = get_agent_teams(log_dir, num_teams=len(team_labels))
        
        for idx, team in enumerate(teams):
            ids = [a['id'] for a in team]
            print(f"Team {team_labels[idx]} 包含 {len(ids)} 个成员，ID 范围: {min(ids) if ids else 'N/A'} - {max(ids) if ids else 'N/A'}")
        
        # 2. 绘图
        plot_elo_sampling(log_dir, teams, team_labels, name)
        
        # 3. 开启弹窗可视化
        print("正在弹出窗口显示...")
        plt.show()
        
    except Exception as e:
        print(f"处理失败: {e}")