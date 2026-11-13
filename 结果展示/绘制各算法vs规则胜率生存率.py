"""
绘制 4 个算法（各 3 次重复实验）在训练过程中对 4 个规则对手的平均指标曲线。

数据来源：
    结果展示/exp_png2/test_norandom_vs_rules_<实验目录名>.csv
每个 CSV 包含 25/50 个训练进度点，列含 avg_score / avg_win / avg_lose /
avg_draw / avg_perish（对 4 规则的平均）。

处理流程：
    1. 同一算法的 3 个重复 CSV，把指标列线性插值到统一横轴（0 ~ 2e6 步）；
    2. 3 条曲线堆叠，求 mean / min / max；
    3. 每个指标一张独立 figure：mean 画实线，min~max 画浅色阴影。

指标：
    - avg_score      平均得分
    - avg_win        平均胜率
    - avg_lose       平均负率
    - avg_draw       平均平率
    - survive        平均生存率 = avg_win + avg_draw - avg_perish

绘图风格参考：对比开火有监督补偿项.py
    - 字体：英文 Times New Roman，中文宋体
    - 图尺寸：7cm × 5cm
    - tab10 配色，第5色亮紫、第8色纯黑
    - 参考线 0 / 0.5 / 1.0
    - 横轴训练步数（科学计数法），纵轴 [-0.05, 1.05]

==================== 合并说明（并入原《对比开火有监督补偿项.py》） ====================
本文件在原有 rules_metrics（规则对手 kill/avg_win/avg_perish）管线之外，
并入原《对比开火有监督补偿项.py》的两条 exp_csv2 管线：
    - win_rates：   exp_csv2/<算法>/run*_vs_opponent*.csv → 与基准对手对抗平均得分
    - run_metrics： exp_csv2/<算法>/run*_<metric>.csv     → return/entropy/accuracy 等
三条管线共用同一套算法顺序与配色（保证所有图曲线颜色和图例顺序一致）：
    顺序 CANONICAL_ALGOS = IL-PPO → 文献[34]方法 → IL-SL-PPO → IL-SLA-PPO
    配色取 tab10 自然序：j=0 蓝、j=1 橙、j=2 绿、j=3 红（与顺序一一对应）。
平滑约定（三条管线统一）：每个 run 插值后先单独平滑，再跨 run 聚合 mean/min/max，
聚合后的 mean/min/max 不再二次平滑。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from _context import *  # 包含 project_root


# ======================== 数据源配置 ========================
CSV_DIR = os.path.join(project_root, "结果展示", "exp_png2")

# exp_csv2：开火监督补偿项对比（run*_vs_opponent*.csv 与 run*_<metric>.csv）
EXP_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_csv2")

# 统一算法顺序：所有图的图例顺序与颜色都以此为准
# tab10 自然序 0=蓝 1=橙 2=绿 3=红，恰好对应：
#   IL-PPO=蓝, 文献[34]方法=橙, IL-SL-PPO=绿, IL-SLA-PPO=红
CANONICAL_ALGOS = ['IL-PPO', '文献[34]方法', 'IL-SL-PPO', 'IL-SLA-PPO']

# (算法显示名, [3 次重复实验的目录名])
# 每组 3 个目录名即 DIR_NAME_LIST 中同算法的三次重复
# 统一在后面给前缀 test_norandom_vs_rules_

ALGORITHM_GROUPS = [
    ("IL-PPO", [
            "PPO0.3_flymask_v0h0-run-20260921-194617",
            "PPO0.3_flymask_v0h0-run-20260928-111836",
            "PPO0.3_flymask_v0h0-run-20261001-152601",
        ]),
    ("文献[34]方法", [
        "SAC0.3_flymask_v1h1-run-20260923-213238",
        "SAC0.3_flymask_v1h1-run-20260928-093645",
        "SAC0.3_flymask_v1h1-run-20260929-194715",
    ]),
    ("IL-SL-PPO", [
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260921-122428",
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260928-233900",
        "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260930-093137",
    ]),
    ("IL-SLA-PPO", [
        "PPO0.3_flymask_v0h0_fireSL-run-20260924-145554",
        "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654",
        "PPO0.3_flymask_v0h0_fireSL-run-20260930-125149",
    ]),    
]

# 指标定义：(CSV 列名, 显示名)；survive 由 win+draw-perish 计算，列名占位
METRICS = [
    ("kill",      "击落基准对手平均概率"),
    ("avg_win",   "对基准对手平均胜率"),
    ("avg_perish", "对基准对手平均双杀率"),
]

# --- 运行级指标的中文标签映射 ---
METRIC_LABELS = {
    'entropy': '在线训练策略熵',
    'pre_entropy': '预训练策略熵',
    'return': '累计奖励',
    'accuracy': '预训练分类准确率',
    'mutualkill': '对基准对手平均双杀率',
    'max_fire_prob': '最大开火概率',
    'min_fire_prob': '最小开火概率',
}

# --- 运行级指标的横轴名称映射（预训练用"迭代"，在线训练用"步数"） ---
METRIC_XLABELS = {
    'entropy': '训练步数',
    'pre_entropy': '迭代',
    'return': '训练步数',
    'accuracy': '迭代',
    'mutualkill': '训练步数',
    'max_fire_prob': '训练步数',
    'min_fire_prob': '训练步数',
}

# 需要按百分数展示的运行级指标（奖励、熵等非概率指标不加百分数）
PERCENT_METRICS = {'accuracy', 'mutualkill', 'max_fire_prob', 'min_fire_prob'}

# 统一横轴
X_MIN = 0.0
X_MAX = 2e6
NUM_POINTS = 100

action_cycle_multiplier = 10 # 动作步和决策步之比


# ======================== 全局视觉参数（与参考文件一致） ========================
label_fontsize = 7.5         # 标题字号，与刻度接近
tick_fontsize = 7
legend_fontsize = 6.5
linewidth = 0.6            # 曲线线宽
refer_linewidth = 0.55     # 参考线宽
legend_linewidth = 1.0     # 图例线条粗细
fill_alpha = 0.10          # min-max 阴影透明度
dpi = 200

COLOR_CYCLE = 5
linestyles = ['-', ':', '--']
smooth_window = 5          # 居中滑动窗口平滑窗口大小；<=1 则不平滑
smooth_window_exp_csv2 = 41  # exp_csv2 两条管线（对抗得分/运行级指标）的单 run 平滑窗口

CM_PER_INCH = 2.54
FIG_WIDTH_CM = 7
FIG_HEIGHT_CM = 5
FIG_WIDTH_IN = FIG_WIDTH_CM / CM_PER_INCH
FIG_HEIGHT_IN = FIG_HEIGHT_CM / CM_PER_INCH

plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
    'axes.formatter.use_mathtext': True,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'axes.axisbelow': True,
    'figure.dpi': dpi,
    'grid.linewidth': 0.3,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '0.35',
})

SAVE_BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "各算法vs规则胜率生存率")
# exp_csv2 两条管线的输出目录（沿用原《对比开火有监督补偿项.py》的出图位置）
SAVE_BASE_DIR_FIRE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "对比开火有监督补偿项")

# ==================== 绘图任务配置（统一在头部管理，新增/删除/禁用都在这里） ====================
# 说明：
#   - enabled = False 可临时禁用某张图
#   - 直接新增一个 dict 即可添加新图
PLOT_TASKS = [
    {
        "name": "rules_metrics",
        "enabled": True,
        "type": "rules_metrics",     # exp_png2：kill / avg_win / avg_perish
        "save_base_dir": SAVE_BASE_DIR,
    },
    {
        "name": "win_rates",
        "enabled": True,
        "type": "win_rates",         # exp_csv2：run*_vs_opponent*.csv 对抗平均得分
        "exp_csv_dir": EXP_CSV_DIR,
        "x_min": 0,
        "x_max": None,
        "num_points": 500,
        "algo_list": None,
        "algo_labels": None,
        "opp_list": None,
        "display_titles": None,
        "show_title": False,
        "smooth_window": smooth_window_exp_csv2,
        "invert_y": False,
        "include_mutualkill": False,
        "mutualkill_label": "对基准对手平均双杀率",
        "save_base_dir": SAVE_BASE_DIR_FIRE,
    },
    {
        "name": "run_metrics",
        "enabled": True,
        "type": "run_metrics",       # exp_csv2：run*_<metric>.csv 运行级指标
        "exp_csv_dir": EXP_CSV_DIR,
        "metric_names": ('accuracy', 'return', 'pre_entropy', 'entropy', 'max_fire_prob', 'min_fire_prob'),
        "x_min": 0,
        "x_max": None,
        "num_points": 500,
        "algo_list": None,
        "algo_labels": None,
        "metric_labels": None,
        "smooth_window": smooth_window_exp_csv2,
        "save_base_dir": SAVE_BASE_DIR_FIRE,
    },
]


def smooth_curve(data, window_size):
    """居中滑动窗口平滑，保持无相位延迟。"""
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


def sort_by_canonical(names):
    """按 CANONICAL_ALGOS 排序；未收录的名字排在最后并按字典序。"""
    order = {name: i for i, name in enumerate(CANONICAL_ALGOS)}
    return sorted(names, key=lambda n: (order.get(n, len(order)), n))


def get_palette(n):
    """
    统一配色：tab10 自然序（0蓝 1橙 2绿 3红），第 j 条曲线取 palette[j % COLOR_CYCLE]。
    配合 sort_by_canonical 的顺序，所有图中同一算法颜色一致。
    """
    cmap = plt.get_cmap('tab10')
    palette = [cmap(i) for i in range(max(10, n))]
    if len(palette) >= 8:
        palette[4] = (0.8, 0.0, 0.8, 1.0)  # 亮紫洋红
        palette[7] = (0.0, 0.0, 0.0, 1.0)  # 纯黑
    return palette


def apply_step_axis(ax, xlabel_text, xt, fontsize):
    """
    横轴统一处理：刻度按 10^x_exp 缩放，并让 (×10^n) 紧贴轴标题。
    x_exp <= 0 时不做换算，直接使用原刻度。
    """
    x_end = float(np.nanmax(xt)) if len(xt) else 0.0
    x_exp = int(np.floor(np.log10(x_end))) if x_end > 0 else 0
    if x_exp <= 0:
        ax.set_xlabel(xlabel_text, fontsize=fontsize)
        return
    _sup = ''.join(chr(c) for c in [0x2070, 0xB9, 0xB2, 0xB3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079])
    ax.set_xlabel(xlabel_text + '(' + chr(0xd7) + '10' + ''.join(_sup[int(d)] for d in str(x_exp)) + ')', fontsize=fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 10 ** x_exp:g}'))


def add_legend(ax, legend_alpha=0.35):
    leg = ax.legend(
        loc="lower right", ncol=2, fontsize=legend_fontsize,
        framealpha=legend_alpha, borderpad=0.25, labelspacing=0.2,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.6,
    )
    if leg is not None:
        leg.set_draggable(True)
        for legobj in leg.get_lines():
            legobj.set_linewidth(legend_linewidth)
            legobj.set_alpha(1.0)


def save_figure(fig, save_name, save_base_dir):
    """在 save_base_dir 下创建 draw_pdf/ 与 draw_svg/ 并分别保存。"""
    if not save_base_dir:
        return
    pdf_dir = os.path.join(save_base_dir, "draw_pdf")
    svg_dir = os.path.join(save_base_dir, "draw_svg")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f"{save_name}.pdf")
    svg_path = os.path.join(svg_dir, f"{save_name}.svg")
    fig.savefig(pdf_path, format='pdf', dpi=dpi)
    fig.savefig(svg_path, format='svg', dpi=dpi)
    print(f"已保存: {pdf_path} / {svg_path}")


# ======================== 管线 1：exp_png2 规则对手指标 ========================

def load_experiment_csv(dir_name):
    """根据实验目录名读取对应 CSV，返回 DataFrame（已按 step 升序）。"""
    csv_path = os.path.join(CSV_DIR, f"test_norandom_vs_rules_{dir_name}.csv")
    df = pd.read_csv(csv_path)
    df = df.sort_values('step').reset_index(drop=True)
    return df


def compute_metric_series(df, metric_key):
    """
    从 DataFrame 提取指标序列。
    survive = avg_win + avg_draw - avg_perish
    """
    if metric_key == 'kill':
        # 击落对手概率 = 胜率 + 双杀率
        return (df['avg_win'].to_numpy() + df['avg_perish'].to_numpy()) * 100.0
    if metric_key == 'survive':
        return (df['avg_win'].to_numpy() + df['avg_draw'].to_numpy() - df['avg_perish'].to_numpy()) * 100.0
    return df[metric_key].to_numpy() * 100.0


def interpolate_to_grid(steps, values, x_target):
    """把 (steps, values) 线性插值到统一 x_target 网格上。"""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(steps) & np.isfinite(values)
    steps, values = steps[valid], values[valid]
    if len(steps) < 2:
        return np.full_like(x_target, np.nan)
    sort_idx = np.argsort(steps)
    steps, values = steps[sort_idx], values[sort_idx]
    steps, unique_idx = np.unique(steps, return_index=True)
    values = values[unique_idx]
    return np.interp(x_target, steps, values, left=values[0], right=values[-1])


def load_group_stats(algo_label, dir_names, x_target):
    """
    对单个算法的多次重复实验，把每个指标插值到统一横轴后，
    返回 { metric_key: {'mean','min','max','runs'} }。
    """
    # 先加载所有重复的 DataFrame
    dfs = []
    for d in dir_names:
        try:
            df = load_experiment_csv(d)
            dfs.append(df)
            print(f"  已加载: {d}  ({len(df)} 个进度点)")
        except Exception as e:
            print(f"  [跳过] {d}: {e}")

    if not dfs:
        return None

    stats = {}
    for metric_key, _ in METRICS:
        curves = []
        for df in dfs:
            vals = compute_metric_series(df, metric_key)
            interp = interpolate_to_grid(df['step'].to_numpy(), vals, x_target)
            # 每个 run 单独做噪声平滑，之后对平滑后的 run 聚合 mean/min/max
            curves.append(smooth_curve(interp, smooth_window))
        stacked = np.vstack(curves)  # (num_runs, num_points)
        stats[metric_key] = {
            'mean': np.mean(stacked, axis=0),
            'min': np.min(stacked, axis=0),
            'max': np.max(stacked, axis=0),
            'runs': len(curves),
        }
    return stats


def style_axis(ax, ylabel_text, metric_key):
    """统一坐标轴样式：参考线、标签、刻度、科学计数法、坐标范围。

    所有指标都画 0 / 0.5 / 1.0 参考线；
    avg_score 不限制 0~1，由数据自动决定 y 轴范围；
    其余指标（win/lose/draw/survive）固定 [-0.05, 1.05]。
    """
    x_exp = int(np.floor(np.log10(X_MAX)))
    _sup = ''.join(chr(c) for c in [0x2070, 0xB9, 0xB2, 0xB3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079])
    for hval in [0, 50, 100]:
        ax.axhline(hval, color='gray', linestyle='-', linewidth=refer_linewidth,
                   alpha=0.8, zorder=1)
    ax.set_ylim(-5, 105)
    ax.set_xlabel('训练步数(' + chr(0xd7) + '10' + ''.join(_sup[int(d)] for d in str(x_exp)) + ')', fontsize=label_fontsize)
    ax.set_ylabel(ylabel_text + ' (%)', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 10 ** x_exp:g}'))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.4)


def plot_rules_metrics(save_base_dir=SAVE_BASE_DIR):
    """管线 1 入口：绘制 exp_png2 的 kill / avg_win / avg_perish 三张图。"""
    x_target = np.linspace(X_MIN, X_MAX, NUM_POINTS)

    # 读取所有算法组的统计数据
    # group_results: [(algo_label, stats_dict)]
    group_results = []
    for algo_label, dir_names in ALGORITHM_GROUPS:
        print(f"\n算法: {algo_label}")
        stats = load_group_stats(algo_label, dir_names, x_target)
        if stats is not None:
            group_results.append((algo_label, stats))

    if not group_results:
        print("没有可绘制的数据，退出。")
        return

    # 统一算法顺序（图例与颜色在所有图中一致）
    order = {name: i for i, name in enumerate(CANONICAL_ALGOS)}
    group_results.sort(key=lambda kv: (order.get(kv[0], len(order)), kv[0]))

    # 配色：matplotlib tab10（与参考文件 sns.color_palette('tab10') 等价）
    palette = get_palette(len(group_results))

    # 每个指标一张独立 figure
    for metric_key, ylabel_text in METRICS:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

        for j, (algo_label, stats) in enumerate(group_results):
            entry = stats.get(metric_key)
            if entry is None:
                continue
            # mean/min/max 已在 load_group_stats 中对单 run 做过平滑，
            # 这里直接取跨 run 聚合后的原始值，不再二次平滑
            mean_y = entry['mean']
            min_y = entry['min']
            max_y = entry['max']

            color = palette[j % COLOR_CYCLE]
            ls = linestyles[(j // COLOR_CYCLE) % len(linestyles)]

            ax.plot(x_target, mean_y, label=algo_label, color=color,
                    linewidth=linewidth, linestyle=ls, alpha=1.0, zorder=3)
            ax.fill_between(x_target, min_y, max_y, color=color,
                            alpha=fill_alpha, edgecolor='none', linewidth=0, zorder=2)

        style_axis(ax, ylabel_text, metric_key)
        add_legend(ax)
        fig.tight_layout(pad=0.2)
        save_figure(fig, metric_key, save_base_dir)
        print(f"绘制: {ylabel_text}")


# ======================== 管线 2/3 共用：exp_csv2 文件解析 ========================

def extract_opponent_id(filename):
    """
    智能提取文件名中的对手标识，兼容 opponent、opponent(手误)、opp、rule 以及带下划线等情况
    例如：
      run1_vs_opponent0.csv   -> 0
      run1_vs_opponent0.csv   -> 0
      run1_vs_opponent_0.csv  -> 0
      run1_vs_rule2.csv       -> 2
    """
    name = os.path.splitext(filename)[0]
    if 'vs' in name.lower():
        vs_part = re.split(r'vs', name, flags=re.IGNORECASE)[-1]
    else:
        vs_part = name
    clean = re.sub(r'^[\s\-_]*(?:opp(?:onent|onnet|o)?|rule)?[\s\-_]*', '', vs_part, flags=re.IGNORECASE)
    return clean if clean else '0'


def find_step_and_value_columns(df):
    """
    自动识别 DataFrame 中的横轴（Step）和纵轴（Value/胜率）列名
    """
    cols = list(df.columns)
    step_col = None
    val_col = None

    # 识别 Step 列
    for c in cols:
        if str(c).strip().lower() in ['step', 'steps', 'epoch', 'epochs', 'iteration', 'iterations', 'x']:
            step_col = c
            break
    if step_col is None:
        step_col = cols[0] # 默认第一列

    # 识别 Value 列
    for c in cols:
        if c == step_col:
            continue
        if str(c).strip().lower() in ['value', 'win_rate', 'winrate', 'score', 'reward', 'y', 'val', 'rate']:
            val_col = c
            break
    if val_col is None:
        # 取除了 Step 和常见时间列之外的最后一列
        candidates = [c for c in cols if c != step_col and str(c).strip().lower() not in ['wall time', 'time', 'timestamp']]
        val_col = candidates[-1] if candidates else (cols[1] if len(cols) > 1 else cols[0])

    return step_col, val_col


def extract_run_metric(filename):
    """
    从文件名提取 (run_idx, metric_name)
    例如: run1_entropy.csv -> ('1', 'entropy')
         run2_mutualkill.csv -> ('2', 'mutualkill')
    """
    name = os.path.splitext(filename)[0]
    m = re.match(r'^run(\d+)_(.+)$', name, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).lower()
    return None, None


# ======================== 管线 2：exp_csv2 对抗平均得分 ========================

def load_and_interpolate_experiments(exp_csv_dir, 
                                     x_min=0, 
                                     x_max=None, 
                                     num_points=500,
                                     algo_list=None,
                                     invert_y=False,
                                     smooth_window=smooth_window):
    """
    读取 exp_csv 下所有算法及对手的 run 数据，并插值对齐到指定的横轴网格上。

    横轴范围按对手分类独立确定：每个对手只从属于该对手的 CSV 文件中检测最大 Step，
    并在该范围内插值，不同对手之间不共享横轴。

    Args:
        exp_csv_dir (str): exp_csv 根目录路径
        x_min (float): 插值横轴起点
        x_max (float): 插值横轴终点，若为 None 则按对手分别自动检测
        num_points (int): 插值点数
        algo_list (list): 算法文件夹名称列表，为 None 时自动扫描子文件夹
        invert_y (bool): 是否使用 (1 - y) 反转胜率，默认为 False
        smooth_window (int): 每条 run 曲线插值后的平滑窗口（先平滑再统计 mean/min/max）
        
    Returns:
        x_targets (dict): { opp_id: x_target(np.ndarray) }，每个对手各自的横轴插值点
        stats_data (dict): 结构为 { algo_name: { opp_id: {'mean': ..., 'min': ..., 'max': ..., 'runs': count} } }
        all_opponents (list): 排序后的所有对手标识列表
        algo_names (list): 算法名称列表
    """
    if not os.path.exists(exp_csv_dir):
        print(f"错误: 目录不存在 -> {exp_csv_dir}")
        return {}, {}, [], []

    if algo_list is None:
        algo_names = [d for d in os.listdir(exp_csv_dir) if os.path.isdir(os.path.join(exp_csv_dir, d))]
        algo_names.sort()
    else:
        algo_names = [a for a in algo_list if os.path.isdir(os.path.join(exp_csv_dir, a))]

    if not algo_names:
        print(f"警告: {exp_csv_dir} 下未找到任何算法子目录。")
        return {}, {}, [], []

    # 第一遍扫描：收集所有文件
    all_opp_set = set()

    parsed_files_by_algo = {} # { algo: [ (opp_id, csv_path) ] }
    for algo in algo_names:
        algo_dir = os.path.join(exp_csv_dir, algo)
        parsed_files_by_algo[algo] = []
        csv_files = [f for f in os.listdir(algo_dir) if f.endswith('.csv')]
        for fname in csv_files:
            # 白名单机制：只有文件名含 "_vs_" 的才算对手胜率文件，
            # 其余（run{idx}_entropy/mutualkill/return/accuracy/pre_entropy 等）一律跳过
            if '_vs_' not in fname.lower():
                continue
            opp_id = extract_opponent_id(fname)
            all_opp_set.add(opp_id)
            csv_path = os.path.join(algo_dir, fname)
            parsed_files_by_algo[algo].append((opp_id, csv_path))

    # 按对手分别检测最大 Step（只在同一对手分类内同步横轴）
    x_targets = {}
    for opp_id in all_opp_set:
        if x_max is not None:
            opp_xmax = x_max
        else:
            detected = 0
            for algo in algo_names:
                for oid, csv_path in parsed_files_by_algo[algo]:
                    if oid != opp_id:
                        continue
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            step_col, _ = find_step_and_value_columns(df)
                            detected = max(detected, df[step_col].max() / action_cycle_multiplier)
                    except Exception:
                        pass
            opp_xmax = detected if detected > 0 else 1e6
        x_targets[opp_id] = np.linspace(x_min, opp_xmax, num_points)
    if x_max is None:
        for opp_id in sorted(x_targets.keys(), key=lambda k: (0, int(k)) if k.isdigit() else (1, k)):
            print(f"[胜率:对手{opp_id}] 最大 Step: {x_targets[opp_id][-1]:.0f}")

    stats_data = {}

    for algo in algo_names:
        stats_data[algo] = {}
        opp_run_curves = {}

        for opp_id, csv_path in parsed_files_by_algo[algo]:
            if opp_id not in opp_run_curves:
                opp_run_curves[opp_id] = []

            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                step_col, val_col = find_step_and_value_columns(df)
                steps = df[step_col].to_numpy(dtype=float) / action_cycle_multiplier
                vals = df[val_col].to_numpy(dtype=float)

                if invert_y:
                    vals = 1.0 - vals

                # 去除 NaN / Inf
                valid_mask = np.isfinite(steps) & np.isfinite(vals)
                steps = steps[valid_mask]
                vals = vals[valid_mask]

                if len(steps) < 2:
                    continue

                # 确保 step 单调递增
                sort_idx = np.argsort(steps)
                steps = steps[sort_idx]
                vals = vals[sort_idx]

                # 去除重复 step
                steps, unique_idx = np.unique(steps, return_index=True)
                vals = vals[unique_idx]

                # 线性插值（使用该对手专属的横轴）
                xt = x_targets[opp_id]
                interp_y = np.interp(xt, steps, vals, left=vals[0], right=vals[-1])
                interp_y = smooth_curve(interp_y, smooth_window)
                opp_run_curves[opp_id].append(interp_y)

            except Exception as e:
                print(f"读取或插值文件 {csv_path} 时出错: {e}")

        # 计算该算法在每个对手下的均值、最大值和最小值
        for opp_id, curves in opp_run_curves.items():
            if len(curves) > 0:
                stacked = np.vstack(curves) # shape: (num_runs, num_points)
                stats_data[algo][opp_id] = {
                    'mean': np.mean(stacked, axis=0),
                    'min': np.min(stacked, axis=0),
                    'max': np.max(stacked, axis=0),
                    'std': np.std(stacked, axis=0),
                    'runs': len(curves)
                }

    # 对所有对手进行自然排序（数字优先）
    def sort_key(k):
        try:
            return (0, int(k))
        except ValueError:
            return (1, str(k))

    all_opponents = sorted(list(all_opp_set), key=sort_key)
    return x_targets, stats_data, all_opponents, algo_names


def plot_interpolated_win_rates(exp_csv_dir,
                                x_min=0,
                                x_max=None,
                                num_points=500,
                                algo_list=None,
                                algo_labels=None,
                                opp_list=None,
                                display_titles=None,
                                show_title=False,
                                smooth_window=smooth_window,
                                invert_y=False,
                                include_mutualkill=False,
                                mutualkill_label="对基准对手平均双杀率",
                                save_base_dir=None):
    """
    绘制不同算法对比不同对手的插值胜率曲线图。
    每个对手（及双杀率）各生成一张独立的 figure，不再使用组图布局。

    Args:
        exp_csv_dir (str): exp_csv 文件夹路径
        x_min (float): 插值横轴起点
        x_max (float): 插值横轴终点（None 则自动检测）
        num_points (int): 插值点数
        algo_list (list): 指定绘制的算法目录名，None 则为所有
        algo_labels (dict or list): 算法在图例中的显示名称
        opp_list (list): 指定绘制的对手标识列表，None 则为所有
        display_titles (list or dict): 每个对手子图的标题
        show_title (bool): 是否在子图上方显示标题（False 则彻底不绘制标题）
        smooth_window (int): 单 run 曲线的平滑窗口大小（值越大越平滑，<=1 表示不平滑）
        invert_y (bool): 是否使用 1 - y 翻转数值
        include_mutualkill (bool): 是否额外生成一张"对基准对手平均双杀率"图
        mutualkill_label (str): mutualkill 图的 y 轴标签（中文显示名）
        save_base_dir (str): 图片保存根目录（可选），会在其下创建 draw_pdf/ 和 draw_svg/，
                             分别保存每张图的 PDF 和 SVG 格式
    """
    x_targets, stats_data, all_opponents, found_algos = load_and_interpolate_experiments(
        exp_csv_dir=exp_csv_dir,
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
        algo_list=algo_list,
        invert_y=invert_y,
        smooth_window=smooth_window
    )

    if not x_targets or not stats_data:
        print("未获取到有效数据，退出绘图。")
        return

    # 对手列表与标题确定
    active_opponents = opp_list if opp_list is not None else all_opponents
    if not active_opponents:
        print("未检测到对手数据，退出绘图。")
        return

    # 统一算法顺序（图例与颜色在所有图中一致）
    active_algos = sort_by_canonical(algo_list if algo_list is not None else found_algos)

    # 配色：tab10 自然序，与 rules/run_metrics 管线一致
    palette = get_palette(len(active_algos))

    # 算法图例显示名映射
    algo_name_map = {}
    for idx, algo in enumerate(active_algos):
        if isinstance(algo_labels, dict) and algo in algo_labels:
            algo_name_map[algo] = algo_labels[algo]
        elif isinstance(algo_labels, list) and idx < len(algo_labels):
            algo_name_map[algo] = algo_labels[idx]
        else:
            algo_name_map[algo] = algo

    # --- 可选：加载 mutualkill 指标数据 ---
    mutualkill_stats = None
    mk_x_target = None
    if include_mutualkill:
        mk_x_targets, mk_stats, _ = load_and_interpolate_run_metrics(
            exp_csv_dir=exp_csv_dir,
            metric_names=('mutualkill',),
            x_min=x_min,
            x_max=x_max,
            num_points=num_points,
            algo_list=active_algos,
            smooth_window=smooth_window,
        )
        if mk_x_targets and mk_stats:
            mk_x_target = mk_x_targets['mutualkill']
            mutualkill_stats = {}
            for algo, md in mk_stats.items():
                if 'mutualkill' in md:
                    entry = md['mutualkill']
                    mutualkill_stats[algo] = {
                        'mean': entry['mean'], 'min': entry['min'],
                        'max': entry['max'], 'runs': entry['runs']
                    }

    def _plot_single_panel(xt, ylabel_text, stats_lookup, save_name=None):
        """在一张独立 figure 上绘制单个面板"""
        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        has_plotted = False
        y_min_total, y_max_total = float('inf'), float('-inf')

        for j, algo in enumerate(active_algos):
            entry = stats_lookup(algo)
            if entry is None:
                continue
            # mean/min/max 在加载侧已对单 run 平滑并聚合，这里直接取值，不再二次平滑；
            # 该面板是 0~1 的对抗得分，按百分数展示
            mean_y = entry['mean'] * 100.0
            min_y = entry['min'] * 100.0
            max_y = entry['max'] * 100.0

            color = palette[j % COLOR_CYCLE]
            ls = linestyles[(j // COLOR_CYCLE) % len(linestyles)]
            label_text = str(algo_name_map[algo])

            ax.plot(xt, mean_y, label=label_text, color=color,
                    linewidth=linewidth, linestyle=ls, alpha=1.0, zorder=3)
            ax.fill_between(xt, min_y, max_y, color=color,
                            alpha=fill_alpha, edgecolor='none', linewidth=0, zorder=2)

            y_min_total = min(y_min_total, np.nanmin(min_y))
            y_max_total = max(y_max_total, np.nanmax(max_y))
            has_plotted = True

        # 参考线 [0, 50, 100]
        for hval in [0, 50, 100]:
            ax.axhline(hval, color='gray', linestyle='-', linewidth=refer_linewidth, alpha=0.8, zorder=1)

        apply_step_axis(ax, '训练步数', xt, label_fontsize)
        ax.set_ylabel(ylabel_text + ' (%)', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        if has_plotted and y_min_total < y_max_total:
            pad = max((y_max_total - y_min_total) * 0.05, 2.0)
            ax.set_ylim(max(-5, y_min_total - pad), min(105, y_max_total + pad))
        else:
            ax.set_ylim(-5, 105)

        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.4)

        if has_plotted:
            add_legend(ax)

        fig.tight_layout(pad=0.2)

        if save_base_dir and save_name:
            # 不使用 bbox_inches='tight'，保持精确尺寸不缩放
            save_figure(fig, save_name, save_base_dir)

    # 逐个对手绘制独立 figure
    for i, opp_id in enumerate(active_opponents):
        if display_titles is not None:
            if isinstance(display_titles, dict) and opp_id in display_titles:
                ylabel_text = display_titles[opp_id]
            elif isinstance(display_titles, list) and i < len(display_titles):
                ylabel_text = display_titles[i]
            else:
                ylabel_text = f"与基准对手对抗平均得分" # f"相对基准对手{i+1}得分"
        else:
            ylabel_text = f"与基准对手对抗平均得分" # f"相对基准对手{i+1}得分"

        xt = x_targets[opp_id]

        def _lookup(algo, oid=opp_id):
            if algo in stats_data and oid in stats_data[algo]:
                return stats_data[algo][oid]
            return None

        print(f"绘制: {ylabel_text}")
        _plot_single_panel(xt, ylabel_text, _lookup,
                           save_name=f"vs_opponent{opp_id}" if save_base_dir else None)

    # 绘制 mutualkill 独立 figure
    if mutualkill_stats is not None:
        def _lookup_mk(algo):
            return mutualkill_stats.get(algo)

        print(f"绘制: {mutualkill_label}")
        _plot_single_panel(mk_x_target, mutualkill_label, _lookup_mk,
                           save_name="mutualkill" if save_base_dir else None)


# ======================== 管线 3：exp_csv2 运行级指标 ========================

def load_and_interpolate_run_metrics(exp_csv_dir,
                                     metric_names=('entropy', 'mutualkill', 'return'),
                                     x_min=0,
                                     x_max=None,
                                     num_points=500,
                                     algo_list=None,
                                     smooth_window=smooth_window):
    """
    读取 exp_csv 下各算法的 run{run_idx}_{metric}.csv 文件并插值对齐到指定横轴网格上。

    横轴范围按指标分类独立确定：每个指标（如 entropy / return / accuracy）只从
    属于该指标的 CSV 文件中检测最大 Step，并在该范围内插值，不同指标之间不共享横轴。

    Args:
        exp_csv_dir (str): exp_csv 根目录路径
        metric_names (tuple): 需要加载的指标名（文件名中 run{run_idx}_ 后的部分）
        x_min (float): 插值横轴起点
        x_max (float): 插值横轴终点，None 则按指标分别自动检测
        num_points (int): 插值点数
        algo_list (list): 算法文件夹名称列表，None 则自动扫描
        smooth_window (int): 每条 run 曲线插值后的平滑窗口（先平滑再统计 mean/min/max）

    Returns:
        x_targets (dict): { metric: x_target(np.ndarray) }，每个指标各自的横轴插值点
        stats_data (dict): { algo: { metric: {'mean','min','max','std','runs'} } }
        algo_names (list): 算法名称列表
    """
    if not os.path.exists(exp_csv_dir):
        print(f"错误: 目录不存在 -> {exp_csv_dir}")
        return {}, {}, []

    if algo_list is None:
        algo_names = [d for d in os.listdir(exp_csv_dir) if os.path.isdir(os.path.join(exp_csv_dir, d))]
        algo_names.sort()
    else:
        algo_names = [a for a in algo_list if os.path.isdir(os.path.join(exp_csv_dir, a))]

    if not algo_names:
        print(f"警告: {exp_csv_dir} 下未找到任何算法子目录。")
        return {}, {}, []

    metric_set = set(m.lower() for m in metric_names)

    # 第一遍扫描：收集文件
    parsed_files_by_algo = {}  # { algo: { metric: [csv_path, ...] } }
    for algo in algo_names:
        algo_dir = os.path.join(exp_csv_dir, algo)
        parsed_files_by_algo[algo] = {m: [] for m in metric_set}
        csv_files = [f for f in os.listdir(algo_dir) if f.endswith('.csv')]
        for fname in csv_files:
            run_idx, metric = extract_run_metric(fname)
            if metric is None or metric not in metric_set:
                continue
            parsed_files_by_algo[algo][metric].append(os.path.join(algo_dir, fname))

    # 按指标分别检测最大 Step（只在同一指标分类内同步横轴）
    x_targets = {}
    for metric in metric_set:
        if x_max is not None:
            mk_xmax = x_max
        else:
            detected = 0
            for algo in algo_names:
                for csv_path in parsed_files_by_algo[algo].get(metric, []):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            step_col, _ = find_step_and_value_columns(df)
                            detected = max(detected, df[step_col].max() / action_cycle_multiplier)
                    except Exception:
                        pass
            mk_xmax = detected if detected > 0 else 1e6
            print(f"[运行级指标:{metric}] 最大 Step: {mk_xmax}")
        x_targets[metric] = np.linspace(x_min, mk_xmax, num_points)

    stats_data = {}

    for algo in algo_names:
        stats_data[algo] = {}
        for metric in metric_set:
            paths = parsed_files_by_algo[algo].get(metric, [])
            curves = []
            xt = x_targets[metric]
            for csv_path in paths:
                try:
                    df = pd.read_csv(csv_path)
                    if df.empty:
                        continue
                    step_col, val_col = find_step_and_value_columns(df)
                    steps = df[step_col].to_numpy(dtype=float) / action_cycle_multiplier
                    vals = df[val_col].to_numpy(dtype=float)

                    valid_mask = np.isfinite(steps) & np.isfinite(vals)
                    steps = steps[valid_mask]
                    vals = vals[valid_mask]

                    if len(steps) < 2:
                        continue

                    sort_idx = np.argsort(steps)
                    steps = steps[sort_idx]
                    vals = vals[sort_idx]

                    steps, unique_idx = np.unique(steps, return_index=True)
                    vals = vals[unique_idx]

                    interp_y = np.interp(xt, steps, vals, left=vals[0], right=vals[-1])
                    interp_y = smooth_curve(interp_y, smooth_window)
                    curves.append(interp_y)
                except Exception as e:
                    print(f"读取或插值文件 {csv_path} 时出错: {e}")

            if len(curves) > 0:
                stacked = np.vstack(curves)
                stats_data[algo][metric] = {
                    'mean': np.mean(stacked, axis=0),
                    'min': np.min(stacked, axis=0),
                    'max': np.max(stacked, axis=0),
                    'std': np.std(stacked, axis=0),
                    'runs': len(curves)
                }

    return x_targets, stats_data, algo_names


def plot_run_metrics(exp_csv_dir,
                     metric_names=('entropy', 'mutualkill', 'return'),
                     x_min=0,
                     x_max=None,
                     num_points=500,
                     algo_list=None,
                     algo_labels=None,
                     metric_labels=None,
                     smooth_window=smooth_window,
                     save_base_dir=None):
    """
    绘制各算法的运行级指标曲线（策略熵 / 累积回报 / 预训练准确率 等）。
    每个指标各生成一张独立的 figure，不再使用组图布局。

    Args:
        exp_csv_dir (str): exp_csv 文件夹路径
        metric_names (tuple): 需要绘制的指标名
        x_min (float): 插值横轴起点
        x_max (float): 插值横轴终点（None 自动检测）
        num_points (int): 插值点数
        algo_list (list): 指定绘制的算法目录名，None 则为所有
        algo_labels (dict or list): 算法图例显示名
        metric_labels (dict): 指标名到中文标签的映射，None 则使用 METRIC_LABELS
        smooth_window (int): 单 run 曲线的平滑窗口大小
        save_base_dir (str): 图片保存根目录（可选），会在其下创建 draw_pdf/ 和 draw_svg/，
                             分别保存每张图的 PDF 和 SVG 格式
    """
    x_targets, stats_data, found_algos = load_and_interpolate_run_metrics(
        exp_csv_dir=exp_csv_dir,
        metric_names=metric_names,
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
        algo_list=algo_list,
        smooth_window=smooth_window,
    )

    if not x_targets or not stats_data:
        print("未获取到运行级指标数据，退出绘图。")
        return

    active_metrics = [m for m in metric_names if any(m in stats_data.get(a, {}) for a in stats_data)]
    if not active_metrics:
        print("未检测到任何运行级指标数据，退出绘图。")
        return

    # 统一算法顺序（图例与颜色在所有图中一致）
    active_algos = sort_by_canonical(algo_list if algo_list is not None else found_algos)

    # 配色：tab10 自然序，与 rules/win_rates 管线一致
    palette = get_palette(len(active_algos))

    # 算法图例显示名映射
    algo_name_map = {}
    for idx, algo in enumerate(active_algos):
        if isinstance(algo_labels, dict) and algo in algo_labels:
            algo_name_map[algo] = algo_labels[algo]
        elif isinstance(algo_labels, list) and idx < len(algo_labels):
            algo_name_map[algo] = algo_labels[idx]
        else:
            algo_name_map[algo] = algo

    # 指标中文标签映射
    if metric_labels is None:
        metric_labels = METRIC_LABELS

    for metric in active_metrics:
        ylabel_text = metric_labels.get(metric, metric)
        if metric in PERCENT_METRICS:
            ylabel_text = ylabel_text + ' (%)'
        xlabel_text = METRIC_XLABELS.get(metric, 'Step')
        xt = x_targets[metric]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        has_plotted = False
        y_min_total, y_max_total = float('inf'), float('-inf')

        for j, algo in enumerate(active_algos):
            if algo in stats_data and metric in stats_data[algo]:
                entry = stats_data[algo][metric]
                # mean/min/max 在加载侧已对单 run 平滑并聚合，这里直接取值，不再二次平滑
                mean_y = entry['mean']
                min_y = entry['min']
                max_y = entry['max']
                if metric in PERCENT_METRICS:
                    mean_y = mean_y * 100.0
                    min_y = min_y * 100.0
                    max_y = max_y * 100.0

                color = palette[j % COLOR_CYCLE]
                ls = linestyles[(j // COLOR_CYCLE) % len(linestyles)]
                label_text = str(algo_name_map[algo])

                ax.plot(xt, mean_y, label=label_text, color=color,
                        linewidth=linewidth, linestyle=ls, alpha=1.0, zorder=3)
                ax.fill_between(xt, min_y, max_y, color=color,
                                alpha=fill_alpha, edgecolor='none', linewidth=0, zorder=2)

                y_min_total = min(y_min_total, np.nanmin(min_y))
                y_max_total = max(y_max_total, np.nanmax(max_y))
                has_plotted = True

        # 坐标轴设置
        apply_step_axis(ax, xlabel_text, xt, label_fontsize)
        ax.set_ylabel(ylabel_text, fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

        if has_plotted and y_min_total < y_max_total:
            pad = max((y_max_total - y_min_total) * 0.05, 0.02)
            ax.set_ylim(y_min_total - pad, y_max_total + pad)

        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.4)

        if has_plotted:
            add_legend(ax)

        fig.tight_layout(pad=0.2)

        if save_base_dir:
            # 不使用 bbox_inches='tight'，保持精确尺寸不缩放
            save_figure(fig, metric, save_base_dir)

        print(f"绘制: {ylabel_text}")


# ======================== 任务调度与入口 ========================

def run_plot_tasks(plot_tasks=None):
    """统一执行头部配置的绘图任务，支持 enable 开关、增删任务。"""
    if plot_tasks is None:
        plot_tasks = PLOT_TASKS

    for task in plot_tasks:
        if not task.get("enabled", True):
            print(f"跳过任务: {task.get('name', 'unnamed')} (enabled=False)")
            continue

        task_type = task.get("type")
        if task_type == "rules_metrics":
            plot_rules_metrics(
                save_base_dir=task.get("save_base_dir", SAVE_BASE_DIR),
            )
        elif task_type == "win_rates":
            plot_interpolated_win_rates(
                exp_csv_dir=task.get("exp_csv_dir", EXP_CSV_DIR),
                x_min=task.get("x_min", 0),
                x_max=task.get("x_max", None),
                num_points=task.get("num_points", 500),
                algo_list=task.get("algo_list", None),
                algo_labels=task.get("algo_labels", None),
                opp_list=task.get("opp_list", None),
                display_titles=task.get("display_titles", None),
                show_title=task.get("show_title", False),
                smooth_window=task.get("smooth_window", smooth_window_exp_csv2),
                invert_y=task.get("invert_y", False),
                include_mutualkill=task.get("include_mutualkill", False),
                mutualkill_label=task.get("mutualkill_label", "对基准对手平均双杀率"),
                save_base_dir=task.get("save_base_dir", SAVE_BASE_DIR_FIRE),
            )
        elif task_type == "run_metrics":
            plot_run_metrics(
                exp_csv_dir=task.get("exp_csv_dir", EXP_CSV_DIR),
                metric_names=task.get("metric_names", ('entropy', 'mutualkill', 'return')),
                x_min=task.get("x_min", 0),
                x_max=task.get("x_max", None),
                num_points=task.get("num_points", 500),
                algo_list=task.get("algo_list", None),
                algo_labels=task.get("algo_labels", None),
                metric_labels=task.get("metric_labels", None),
                smooth_window=task.get("smooth_window", smooth_window_exp_csv2),
                save_base_dir=task.get("save_base_dir", SAVE_BASE_DIR_FIRE),
            )
        else:
            print(f"未知绘图任务类型: {task_type!r}，任务名={task.get('name', 'unnamed')}")


def main():
    # 统一在头部的 PLOT_TASKS 中维护要画的图，想加就新增 dict，想删就删除，想停用就 enabled=False
    run_plot_tasks(PLOT_TASKS)

    # 一次性显示所有 figure（不再逐张阻塞）
    plt.show()


if __name__ == "__main__":
    main()
