import numpy as np
import sys
import os
import json
from _context import *

from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

def get_top_elo_agents(log_dir, top_n=50):
    """
    修改后的筛选逻辑 (参考 find_top_agents2.py)：
    1. 优先从 Hall of Fame (hall_of_fame.json) 中选择
    2. 不够则从 Elite Pool (elite_elo_ratings.json) 中补充
    3. 如果还不够，从全局 Elo (elo_ratings.json) 中补充
    4. 排除 Rule 开头的，去重
    """
    # 路径准备
    full_json_path = os.path.join(log_dir, "elo_ratings.json")
    elite_json_path = os.path.join(log_dir, "elite_elo_ratings.json")
    hof_json_path = os.path.join(log_dir, "hall_of_fame.json")

    # 1. 加载所有池子数据
    def load_json_safe(p, default):
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except:
                    return default
        return default

    # 加载数据
    global_elo_dict = load_json_safe(full_json_path, {})
    elite_elo_ratings = load_json_safe(elite_json_path, {})
    hall_of_fame_keys = load_json_safe(hof_json_path, [])

    selected_agents = []
    seen_keys = set()

    # 辅助过滤函数：只保留 'actor_rein' 开头的文件名 (排除 'Rule_' 和 '__CURRENT_MAIN__' 等)
    def is_valid_agent(k):
        return k.startswith('actor_rein')

    # --- 策略 A: 优先检索名人堂 (HoF) ---
    # 兼容 HoF 是 list 或 dict 的情况
    if isinstance(hall_of_fame_keys, dict):
        hof_iterable = hall_of_fame_keys.keys()
    elif isinstance(hall_of_fame_keys, list):
        hof_iterable = hall_of_fame_keys
    else:
        hof_iterable = []

    # 过滤无效的和没有 Elo 记录的
    hof_candidates = [k for k in hof_iterable if k in global_elo_dict and is_valid_agent(k)]
    # 按 Elo 分数降序排列
    hof_sorted = sorted(hof_candidates, key=lambda k: global_elo_dict[k], reverse=True)

    for k in hof_sorted:
        if len(selected_agents) < top_n:
            selected_agents.append(k)
            seen_keys.add(k)

    # --- 策略 B: 补充精英池 (Elite) ---
    # 同样过滤 Rule 开头的
    elite_candidates = [k for k in elite_elo_ratings.keys() if is_valid_agent(k)]
    # 使用 global_elo_dict 的分数来排序 (通常更全)
    elite_sorted = sorted(elite_candidates, key=lambda k: global_elo_dict.get(k, elite_elo_ratings[k]), reverse=True)

    for k in elite_sorted:
        if k not in seen_keys and len(selected_agents) < top_n:
            selected_agents.append(k)
            seen_keys.add(k)

    # --- 策略 C: 如果还不够，从全局 Elo 表中兜底 ---
    if len(selected_agents) < top_n:
        global_candidates = [k for k in global_elo_dict.keys() if is_valid_agent(k)]
        global_sorted = sorted(global_candidates, key=lambda k: global_elo_dict[k], reverse=True)
        for k in global_sorted:
            if k not in seen_keys and len(selected_agents) < top_n:
                selected_agents.append(k)
                seen_keys.add(k)

    # 3. 构造完整文件路径
    top_agents_paths = []
    for k in selected_agents:
        # 拼接 .pt 后缀
        full_path = os.path.join(log_dir, f"{k}.pt")
        # 确保文件实际存在
        if os.path.exists(full_path):
            top_agents_paths.append(full_path)
    
    if len(top_agents_paths) < top_n:
        print(f"注意：{log_dir} 仅找到 {len(top_agents_paths)} 个有效智能体，不足 {top_n} 个。")
        
    return top_agents_paths, hof_sorted, elite_sorted

if __name__=='__main__':
    # 长名称优先（指定日期和时间）
    dir_name = None # "IL_and_MixedPFSP_分阶段_挑战_并行_分层2s-run-20260408-175230"
   
    # 短名称次之（自动找最新实验结果）
    experiment_name = 'IL_and_Mixed经典PFSP_多技术流派_并行_分层_rule3_0.1'
    # --- 查找模型 ---
    logs_root_dir = os.path.join(project_root, "logs/combat")
    
    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, experiment_name)
    
    print(f"Log directory: {latest_log_dir}")
    print("=" * 60)
    
    # 获取顶级智能体
    top_agents_paths, hof_sorted, elite_sorted = get_top_elo_agents(latest_log_dir, top_n=50)
    
    # 加载 hall_of_fame.json 获取完整的 key-value
    hof_json_path = os.path.join(latest_log_dir, "hall_of_fame.json")
    if os.path.exists(hof_json_path):
        with open(hof_json_path, 'r', encoding='utf-8') as f:
            hall_of_fame = json.load(f)
    else:
        hall_of_fame = {}
    
    # 加载 elite_elo_ratings.json 获取完整的 key-value
    elite_json_path = os.path.join(latest_log_dir, "elite_elo_ratings.json")
    if os.path.exists(elite_json_path):
        with open(elite_json_path, 'r', encoding='utf-8') as f:
            elite_elo_ratings = json.load(f)
    else:
        elite_elo_ratings = {}
    
    # 打印 Hall of Fame 的 key 和 value
    print("\n=== Hall of Fame (sorted by Elo) ===")
    for k in hof_sorted:
        if isinstance(hall_of_fame, dict) and k in hall_of_fame:
            print(f"  {k}: Elo={hall_of_fame[k]:.2f}")
        else:
            print(f"  {k}")
    
    # 打印 Elite Pool 的 key 和 value
    print(f"\n=== Elite Pool (sorted by Elo, total={len(elite_sorted)}) ===")
    for k in elite_sorted:
        elo_val = elite_elo_ratings.get(k, "N/A")
        if isinstance(elo_val, float):
            print(f"  {k}: Elo={elo_val:.2f}")
        else:
            print(f"  {k}: Elo={elo_val}")
    
    print(f"\n=== Selected Top Agents (total={len(top_agents_paths)}) ===")
    for i, path in enumerate(top_agents_paths[:10], 1):  # 只打印前10个
        print(f"  {i}. {os.path.basename(path)}")