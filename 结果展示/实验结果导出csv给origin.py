"""
本文件用于将 exp_csv 下按实验（算法）分类的原始 run CSV 数据，
经过插值 + 平滑后，重新整理为 Origin 可直接使用的宽表 CSV。

输入目录结构：
exp_csv/
  ├─ 实验1/
  │   ├─ run1_vs_opponent0.csv
  │   ├─ run1_vs_opponent1.csv
  │   ├─ run1_entropy.csv
  │   ├─ run1_mutualkill.csv
  │   ├─ run1_return.csv
  │   ├─ run1_accuracy.csv
  │   ├─ run1_pre_entropy.csv
  │   └─ ...
  ├─ 实验2/
  └─ ...

输出目录结构：
csv4origin/
  ├─ vs_opponent0.csv   第一列为插值平滑后的 Step，其余各列为各实验的均值曲线
  ├─ vs_opponent1.csv
  ├─ entropy.csv
  ├─ mutualkill.csv
  ├─ return.csv
  ├─ accuracy.csv
  ├─ pre_entropy.csv
  └─ ...

横轴范围按曲线分类独立确定：每个分类（如 entropy / vs_opponent0）只从
属于该分类的 CSV 文件中检测最大 Step，并在该范围内插值，不同分类之间不共享横轴。
"""

import os
import re
import numpy as np
import pandas as pd

# ==================== 配置区 ====================
EXP_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv4origin")

X_MIN = 0                # 插值横轴起点
X_MAX = None             # 插值横轴终点，None 则按分类分别自动检测
NUM_POINTS = 500         # 统一插值采样点数

# 按分类指定平滑窗口大小（<=1 不平滑）
# key 为分类名：指标用 metric 名（如 'entropy'），对手用 'vs_opponent'
SMOOTH_WINDOW_BY_CATEGORY = {
    'entropy':       31,
    'mutualkill':    61,
    'return':        61,
    'accuracy':       1,
    'pre_entropy':   1,
    'vs_opponent':  61,   # 所有 vs_opponent{N} 共用此值
}

# 按分类指定横轴名称（写入 CSV 第一列表头）
X_LABEL_BY_CATEGORY = {
    'entropy':       '步数',
    'mutualkill':    '步数',
    'return':        '步数',
    'accuracy':      '迭代',
    'pre_entropy':   '迭代',
    'vs_opponent':  '步数',  # 所有 vs_opponent{N} 共用此值
}

# 默认值（分类未在上述字典中时使用）
DEFAULT_SMOOTH_WINDOW = 61
DEFAULT_X_LABEL = 'Step'
# ===============================================


def smooth_curve(data, window_size):
    """
    使用居中滑动窗口对曲线进行平滑处理，保持前后无相位延迟
    """
    if window_size <= 1 or data is None or len(data) <= 1:
        return np.asarray(data)
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    if window_size >= len(data):
        window_size = len(data) - (1 if len(data) % 2 == 0 else 0)
    if window_size <= 1:
        return np.asarray(data)
    return pd.Series(data).rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()


def find_step_and_value_columns(df):
    """
    自动识别 DataFrame 中的横轴（Step）和纵轴（Value）列名
    """
    cols = list(df.columns)
    step_col = None
    val_col = None

    for c in cols:
        if str(c).strip().lower() in ['step', 'steps', 'epoch', 'epochs', 'iteration', 'iterations', 'x']:
            step_col = c
            break
    if step_col is None:
        step_col = cols[0]

    for c in cols:
        if c == step_col:
            continue
        if str(c).strip().lower() in ['value', 'win_rate', 'winrate', 'score', 'reward', 'y', 'val', 'rate']:
            val_col = c
            break
    if val_col is None:
        candidates = [c for c in cols if c != step_col and str(c).strip().lower() not in ['wall time', 'time', 'timestamp']]
        val_col = candidates[-1] if candidates else (cols[1] if len(cols) > 1 else cols[0])

    return step_col, val_col


def extract_opponent_id(filename):
    """
    从文件名提取对手标识，例如 run1_vs_opponent0.csv -> 0
    """
    name = os.path.splitext(filename)[0]
    if 'vs' in name.lower():
        vs_part = re.split(r'vs', name, flags=re.IGNORECASE)[-1]
    else:
        vs_part = name
    clean = re.sub(r'^[\s\-_]*(?:opp(?:onent|onnet|o)?|rule)?[\s\-_]*', '', vs_part, flags=re.IGNORECASE)
    return clean if clean else '0'


def extract_run_metric(filename):
    """
    从文件名提取 (run_idx, metric_name)
    例如: run1_entropy.csv -> ('1', 'entropy')
    """
    name = os.path.splitext(filename)[0]
    m = re.match(r'^run(\d+)_(.+)$', name, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).lower()
    return None, None


def read_curve(csv_path):
    """
    读取单个 CSV 文件，返回 (steps, vals) 已去 NaN/排序/去重
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, None
    step_col, val_col = find_step_and_value_columns(df)
    steps = df[step_col].to_numpy(dtype=float)
    vals = df[val_col].to_numpy(dtype=float)

    valid_mask = np.isfinite(steps) & np.isfinite(vals)
    steps = steps[valid_mask]
    vals = vals[valid_mask]

    if len(steps) < 2:
        return None, None

    sort_idx = np.argsort(steps)
    steps = steps[sort_idx]
    vals = vals[sort_idx]

    steps, unique_idx = np.unique(steps, return_index=True)
    vals = vals[unique_idx]

    return steps, vals


def load_and_interpolate_category(exp_csv_dir, algo_names, category, category_type, smooth_window):
    """
    对单个分类（如 entropy / vs_opponent0）收集各实验的 run 曲线，
    按该分类独立检测最大 Step 并插值，再求各实验的均值并平滑。

    Args:
        exp_csv_dir (str): exp_csv 根目录
        algo_names (list): 实验名称列表
        category (str): 分类标识，如 'entropy' 或 '0'（对手ID）
        category_type (str): 'metric' 或 'opponent'
        smooth_window (int): 该分类的滑动平均平滑窗口大小

    Returns:
        x_target (np.ndarray): 该分类的横轴插值点
        mean_curves (dict): { algo_name: smoothed_mean_curve(np.ndarray) }
    """
    # 收集各实验属于该分类的文件路径
    files_by_algo = {}
    detected_max_step = 0

    for algo in algo_names:
        algo_dir = os.path.join(exp_csv_dir, algo)
        csv_files = [f for f in os.listdir(algo_dir) if f.endswith('.csv')]
        files_by_algo[algo] = []

        if category_type == 'metric':
            for fname in csv_files:
                run_idx, metric = extract_run_metric(fname)
                if metric is not None and metric == category:
                    files_by_algo[algo].append(os.path.join(algo_dir, fname))
        elif category_type == 'opponent':
            for fname in csv_files:
                if '_vs_' not in fname.lower():
                    continue
                opp_id = extract_opponent_id(fname)
                if opp_id == category:
                    files_by_algo[algo].append(os.path.join(algo_dir, fname))

    # 检测该分类的最大 Step
    if X_MAX is not None:
        cat_xmax = X_MAX
    else:
        detected_max_step = 0
        for algo in algo_names:
            for csv_path in files_by_algo[algo]:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        step_col, _ = find_step_and_value_columns(df)
                        detected_max_step = max(detected_max_step, df[step_col].max())
                except Exception:
                    pass
        cat_xmax = detected_max_step if detected_max_step > 0 else 1e6
        print(f"  [{category}] 最大 Step: {cat_xmax}")

    x_target = np.linspace(X_MIN, cat_xmax, NUM_POINTS)

    # 插值并求均值
    mean_curves = {}
    for algo in algo_names:
        curves = []
        for csv_path in files_by_algo[algo]:
            steps, vals = read_curve(csv_path)
            if steps is None:
                continue
            interp_y = np.interp(x_target, steps, vals, left=vals[0], right=vals[-1])
            curves.append(interp_y)

        if len(curves) > 0:
            stacked = np.vstack(curves)
            mean_y = np.mean(stacked, axis=0)
            mean_y = smooth_curve(mean_y, smooth_window)
            mean_curves[algo] = mean_y

    return x_target, mean_curves


def export_category_csv(output_dir, filename, x_target, mean_curves, algo_names, x_label):
    """
    将单个分类的插值平滑结果导出为 Origin 可用的宽表 CSV。
    第一列为 x_label（横轴名称），其余各列为各实验名称。
    """
    data = {x_label: x_target}
    for algo in algo_names:
        if algo in mean_curves:
            data[algo] = mean_curves[algo]
        else:
            data[algo] = np.full_like(x_target, np.nan)

    df_out = pd.DataFrame(data)
    out_path = os.path.join(output_dir, filename)
    df_out.to_csv(out_path, index=False, float_format='%.6f')
    print(f"  已导出: {out_path}")


def main():
    if not os.path.exists(EXP_CSV_DIR):
        print(f"错误: 目录不存在 -> {EXP_CSV_DIR}")
        return

    algo_names = [d for d in os.listdir(EXP_CSV_DIR)
                  if os.path.isdir(os.path.join(EXP_CSV_DIR, d))]
    algo_names.sort()

    if not algo_names:
        print(f"警告: {EXP_CSV_DIR} 下未找到任何实验子目录。")
        return

    print(f"检测到 {len(algo_names)} 个实验: {algo_names}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有分类
    metric_set = set()
    opp_set = set()

    for algo in algo_names:
        algo_dir = os.path.join(EXP_CSV_DIR, algo)
        for fname in os.listdir(algo_dir):
            if not fname.endswith('.csv'):
                continue
            # 运行级指标
            run_idx, metric = extract_run_metric(fname)
            if metric is not None and '_vs_' not in fname.lower():
                metric_set.add(metric)
                continue
            # 对手胜率
            if '_vs_' in fname.lower():
                opp_id = extract_opponent_id(fname)
                opp_set.add(opp_id)

    # 对对手进行自然排序
    def sort_key(k):
        try:
            return (0, int(k))
        except ValueError:
            return (1, str(k))

    all_opponents = sorted(list(opp_set), key=sort_key)
    all_metrics = sorted(list(metric_set))

    print(f"\n检测到指标分类: {all_metrics}")
    print(f"检测到对手分类: {all_opponents}")

    # 导出指标类 CSV
    print("\n=== 导出运行级指标 CSV ===")
    for metric in all_metrics:
        print(f"处理指标: {metric}")
        sw = SMOOTH_WINDOW_BY_CATEGORY.get(metric, DEFAULT_SMOOTH_WINDOW)
        xl = X_LABEL_BY_CATEGORY.get(metric, DEFAULT_X_LABEL)
        x_target, mean_curves = load_and_interpolate_category(
            EXP_CSV_DIR, algo_names, metric, 'metric', sw)
        export_category_csv(OUTPUT_DIR, f"{metric}.csv", x_target, mean_curves, algo_names, xl)

    # 导出对手胜率类 CSV
    print("\n=== 导出对手胜率 CSV ===")
    for opp_id in all_opponents:
        print(f"处理对手: {opp_id}")
        sw = SMOOTH_WINDOW_BY_CATEGORY.get('vs_opponent', DEFAULT_SMOOTH_WINDOW)
        xl = X_LABEL_BY_CATEGORY.get('vs_opponent', DEFAULT_X_LABEL)
        x_target, mean_curves = load_and_interpolate_category(
            EXP_CSV_DIR, algo_names, opp_id, 'opponent', sw)
        export_category_csv(OUTPUT_DIR, f"vs_opponent{opp_id}.csv", x_target, mean_curves, algo_names, xl)

    print(f"\n全部完成！输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
