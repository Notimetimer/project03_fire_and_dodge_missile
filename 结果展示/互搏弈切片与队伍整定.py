import os
import re
import json
import pandas as pd

from _context import *  # 包含 project_root
from Utilities.LocateDirAndAgents2 import get_latest_log_dir


def process_mission(log_dir):
    """
    对单个任务目录执行集合分析，返回按 (-score, batch) 排序的全量 SortedSBS 列表。
    """
    step_batch_path = os.path.join(log_dir, "StepBatch.csv")
    step_wr_path    = os.path.join(log_dir, "StepWR.csv")
    elo_path        = os.path.join(log_dir, "elo_ratings.json")

    for p, fname in [(step_batch_path, "StepBatch.csv"),
                     (step_wr_path,    "StepWR.csv"),
                     (elo_path,        "elo_ratings.json")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"缺少文件: {p}")

    df_batch = pd.read_csv(step_batch_path)
    df_wr    = pd.read_csv(step_wr_path)

    for df, fname in [(df_batch, "StepBatch.csv"), (df_wr, "StepWR.csv")]:
        if "Step" not in df.columns or "Value" not in df.columns:
            raise ValueError(f"{fname} 缺少 'Step' 或 'Value' 列，现有列: {list(df.columns)}")

    set_wr_step    = set(df_wr["Step"].astype(int))
    set_batch_step = set(df_batch["Step"].astype(int))

    with open(elo_path, "r", encoding="utf-8") as f:
        elo_dict = json.load(f)

    rein_pattern = re.compile(r'^actor_rein(\d+)$')
    agent_nums = set(int(m.group(1)) for k in elo_dict.keys() if (m := rein_pattern.match(k)))

    intersection_wr_batch    = set_wr_step & set_batch_step
    scored_batchs            = set(df_batch.loc[df_batch["Step"].isin(intersection_wr_batch), "Value"].astype(int))
    intersection_scored_agent = scored_batchs & agent_nums

    print(f"  StepWR∩StepBatch: {len(intersection_wr_batch)}  "
          f"ScoredBatchs: {len(scored_batchs)}  "
          f"∩agent_nums: {len(intersection_scored_agent)}")

    df_valid_batch = df_batch.loc[df_batch["Value"].astype(int).isin(intersection_scored_agent)].copy()
    df_valid_batch["Value"] = df_valid_batch["Value"].astype(int)
    df_valid_batch["Step"]  = df_valid_batch["Step"].astype(int)

    step_to_score = dict(zip(df_wr["Step"].astype(int), df_wr["Value"].astype(float)))

    sbs_list = []
    for _, row in df_valid_batch.iterrows():
        step  = int(row["Step"])
        batch = int(row["Value"])
        score = step_to_score.get(step, None)
        if score is not None:
            sbs_list.append((step, batch, score))

    # 先按 score 降序，score 相同则按 batch 降序
    SortedSBS = sorted(sbs_list, key=lambda x: (-x[2], -x[1]))
    return SortedSBS


if __name__ == "__main__":
    # 配置任务目录
    mission_names = [
        'SLWSPFSP0.5-run-20260620-211720',
        'SLWSPFSP0.3-run-20260618-221044', # 'SLWSPFSP0.3-run-20260618-221044', SLWSPFSP0.3-run-20260704-175531
        'SLWSPFSP0.2-run-20260622-185856', # 'SLWSPFSP0.5-run-20260620-211720',
        'HLWSPFSP-run-20260616-130304',
        'CSPFSP-run-20260615-234324',
        'SLWSA3C0.3-run-20260630-220403',
    ]

    logs_root_dir = os.path.join(project_root, "logs", "combat")

    for mission_name in mission_names:
        log_dir = os.path.join(logs_root_dir, mission_name)
        if not os.path.exists(log_dir):
            log_dir = get_latest_log_dir(logs_root_dir, mission_name)
        if not log_dir or not os.path.exists(log_dir):
            print(f"[SKIP] 未找到目录: {mission_name}")
            continue

        print(f"\n{'='*60}")
        print(f"处理: {log_dir}")
        SortedSBS = process_mission(log_dir)
        print(f"  全量有效元组总数: {len(SortedSBS)}")

        # 保存全量（不设上限），按 (-score, batch) 顺序写入 txt
        all_batch_names = [f"actor_rein{batch}" for (_, batch, _) in SortedSBS]
        team_file = os.path.join(log_dir, "top_batch_names.txt")
        with open(team_file, "w", encoding="utf-8") as f:
            for bname in all_batch_names:
                f.write(bname + "\n")
        print(f"  已保存 {len(all_batch_names)} 个成员到: {team_file}")