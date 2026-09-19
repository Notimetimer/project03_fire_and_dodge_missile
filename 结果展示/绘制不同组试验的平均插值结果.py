"""
本文件用于对比不同组实验对比不同对手的插值胜率曲线

1、exp_csv里面，先有不同组对比实验的文件夹，
每个文件夹里面包含有不同训练次数下对不同对手的胜率曲线，文件目录结构大致如下：
exp_csv
    算法1
        run{run_idx}_vs_opponent{opp_idx}.csv
        run_idx = 1,2,3....
        opp_idx = 1,2,3....
    算法2
        run{run_idx}_vs_opponent{opp_idx}.csv
        run_idx = 1,2,3....
        opp_idx = 1,2,3....
    算法3
        run{run_idx}_vs_opponent{opp_idx}.csv
        run_idx = 1,2,3....
        opp_idx = 1,2,3....


2、需要先指定曲线横轴数值范围（最小、最大与插值点数）
在每个算法中对相同对手的胜率曲线进行插值，插到指定横轴上，得到横轴对齐的胜率值
随后求每个算法与每个对手的平均胜率值、最大与最小值。

3、绘制曲线规则是：
subplot1(绘制每个算法对对手1的胜率曲线)
    把每个算法下的平均数值绘制为实线，最大和最小值用浅色阴影块表示。
    每个算法用不同的颜色表示。
subplot2(绘制每个算法对对手2的胜率曲线)
subplot3(绘制每个算法对对手3的胜率曲线)
subplot4(绘制每个算法对对手4的胜率曲线)
subplot5(绘制每个算法对对手5的胜率曲线)
...

"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from _context import * # 包含 project_root

# 全局视觉参数（先定义，供 rcParams 和后续绘图使用）
label_fontsize = 7.5      # 六号 = 7.5 pt
tick_fontsize = 6.5
legend_fontsize = 5.5
linewidth = 0.6           # 曲线线宽（pt），在 5 cm 小图中保持纤细
refer_linewidth = 0.55    # 参考线宽
legend_linewidth = 1.0    # 图例中展示的线条粗细（pt）
dpi = 200                 # 屏幕显示与保存的统一 DPI

# 线型与颜色策略：前 5 种颜色优先，超过 5 条曲线时重复这 5 色并切换线型
# 5 条用 '-'，5 条用 '--'，5 条用 ':'，最多 15 条组合
COLOR_CYCLE = 5
linestyles = ['-', ':', '--']

# ==================== 单位换算（严格写出过程） ====================
# matplotlib 的 figsize 单位为英寸（inch），fontsize 单位为磅（pt）
# 换算关系：
#   1 inch = 2.54 cm          → 1 cm = 1/2.54 inch ≈ 0.3937 inch
#   1 pt   = 1/72 inch        → 1 inch = 72 pt
# 中文字号对照：六号 = 7.5 pt
CM_PER_INCH = 2.54
PT_PER_INCH = 72.0

FIG_WIDTH_CM = 5.0          # 图宽：5 cm
FIG_HEIGHT_CM = 4.5         # 图高：4.5 cm
FIG_WIDTH_IN = FIG_WIDTH_CM / CM_PER_INCH   # 5 / 2.54 ≈ 1.9685 inch
FIG_HEIGHT_IN = FIG_HEIGHT_CM / CM_PER_INCH  # 4.5 / 2.54 ≈ 1.7717 inch

# --- 1. 环境与绘图配置 ---
# 字体：英文用 Times New Roman，中文用宋体（SimSun）
# matplotlib 3.6+ 支持按字符回退：将 font.family 直接设为字体列表，
# 遇到 Times New Roman 无中文字形时自动回退到 SimSun，实现中英文混排各用其字体。
# 注意：不使用 sns.set_theme，因为它会重置 font.family 导致中文字体回退失效。
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],   # 英文 Times New Roman，中文回退宋体
    'axes.unicode_minus': False,                     # 负号正常显示
    'mathtext.fontset': 'stix',                       # 数学公式用类 Times 字体
    'axes.formatter.use_mathtext': True,              # 1e7 → ×10⁷ 样式
    'axes.grid': True,                                # darkgrid 风格
    'grid.alpha': 0.4,
    'axes.axisbelow': True,
    'figure.dpi': dpi,                                # 屏幕显示与保存文件统一 DPI
    'grid.linewidth': 0.3,
})
# ================================================================

# --- 运行级指标的中文标签映射 ---
METRIC_LABELS = {
    'entropy': '在线训练策略熵',
    'pre_entropy': '预训练策略熵',
    'return': '奖励',
    'accuracy': '预训练分类准确率',
    'mutualkill': '对基准对手平均双杀率',
}

# --- 运行级指标的横轴名称映射（预训练用"迭代"，在线训练用"步数"） ---
METRIC_XLABELS = {
    'entropy': '步数',
    'pre_entropy': '迭代',
    'return': '步数',
    'accuracy': '迭代',
    'mutualkill': '步数',
}

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


def load_and_interpolate_experiments(exp_csv_dir, 
                                     x_min=0, 
                                     x_max=None, 
                                     num_points=500,
                                     algo_list=None,
                                     invert_y=False):
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
                            detected = max(detected, df[step_col].max())
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
                steps = df[step_col].to_numpy(dtype=float)
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
                                smooth_window=61,
                                fill_alpha=0.10,
                                linewidth=1.08,
                                legend_alpha=0.35,
                                legend_linewidth=2.2,
                                invert_y=False,
                                show_grid=True,
                                include_mutualkill=False,
                                mutualkill_label="对基准对手平均双杀率",
                                label_fontsize=label_fontsize,
                                tick_fontsize=tick_fontsize,
                                legend_fontsize=legend_fontsize,
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
        smooth_window (int): 平均曲线平滑窗口大小（值越大越平滑，<=1 表示不平滑）
        fill_alpha (float): 极值阴影区透明度（较小值使色块更浅）
        linewidth (float): 绘制的曲线线宽（默认 1.08）
        legend_alpha (float): 图例背景透明度（较小值更透明）
        legend_linewidth (float): 图例中线条粗细
        invert_y (bool): 是否使用 1 - y 翻转数值
        show_grid (bool): 是否显示网格
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
        invert_y=invert_y
    )

    if not x_targets or not stats_data:
        print("未获取到有效数据，退出绘图。")
        return

    # 对手列表与标题确定
    active_opponents = opp_list if opp_list is not None else all_opponents
    if not active_opponents:
        print("未检测到对手数据，退出绘图。")
        return

    active_algos = algo_list if algo_list is not None else found_algos

    # 配色配置（参考展示决策训练曲线.py）
    palette = list(sns.color_palette("tab10", max(10, len(active_algos))))
    if len(palette) >= 8:
        palette[4] = (0.8, 0.0, 0.8) # 亮紫洋红，增强对比
        palette[7] = (0.0, 0.0, 0.0) # 纯黑

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
            raw_mean = entry['mean']
            raw_min = entry['min']
            raw_max = entry['max']

            mean_y = smooth_curve(raw_mean, smooth_window)
            min_y = smooth_curve(raw_min, smooth_window)
            max_y = smooth_curve(raw_max, smooth_window)

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

        # 参考线 [0, 0.5, 1.0]
        for hval in [0, 0.5, 1.0]:
            ax.axhline(hval, color='gray', linestyle='-', linewidth=refer_linewidth, alpha=0.8, zorder=1)

        ax.set_xlabel('步数', fontweight='bold', fontsize=label_fontsize)
        ax.set_ylabel(ylabel_text, fontweight='bold', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(tick_fontsize)  # 1e7 等科学计数法偏移文字与刻度同号

        if has_plotted and y_min_total < y_max_total:
            pad = max((y_max_total - y_min_total) * 0.05, 0.02)
            ax.set_ylim(max(-0.05, y_min_total - pad), min(1.05, y_max_total + pad))
        else:
            ax.set_ylim(-0.05, 1.05)

        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4)

        if has_plotted:
            leg = ax.legend(
                loc="lower right",
                ncol=2,
                fontsize=legend_fontsize,
                framealpha=legend_alpha,
                borderpad=0.25,
                labelspacing=0.2,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=0.6
            )
            if leg is not None:
                leg.set_draggable(True)
                for legobj in leg.get_lines():
                    legobj.set_linewidth(legend_linewidth)
                    legobj.set_alpha(1.0)

        fig.tight_layout(pad=0.2)

        if save_base_dir and save_name:
            pdf_dir = os.path.join(save_base_dir, "draw_pdf")
            svg_dir = os.path.join(save_base_dir, "draw_svg")
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)
            # 不使用 bbox_inches='tight'，保持精确尺寸不缩放
            pdf_path = os.path.join(pdf_dir, f"{save_name}.pdf")
            svg_path = os.path.join(svg_dir, f"{save_name}.svg")
            fig.savefig(pdf_path, format='pdf', dpi=dpi)
            fig.savefig(svg_path, format='svg', dpi=dpi)
            print(f"已保存: {pdf_path} / {svg_path}")

    # 逐个对手绘制独立 figure
    for i, opp_id in enumerate(active_opponents):
        if display_titles is not None:
            if isinstance(display_titles, dict) and opp_id in display_titles:
                ylabel_text = display_titles[opp_id]
            elif isinstance(display_titles, list) and i < len(display_titles):
                ylabel_text = display_titles[i]
            else:
                ylabel_text = f"相对基准对手{i+1}比分"
        else:
            ylabel_text = f"相对基准对手{i+1}比分"

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


def load_and_interpolate_run_metrics(exp_csv_dir,
                                     metric_names=('entropy', 'mutualkill', 'return'),
                                     x_min=0,
                                     x_max=None,
                                     num_points=500,
                                     algo_list=None):
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
                            detected = max(detected, df[step_col].max())
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
                    steps = df[step_col].to_numpy(dtype=float)
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
                     smooth_window=61,
                     fill_alpha=0.10,
                     linewidth=1.08,
                     legend_alpha=0.35,
                     legend_linewidth=2.2,
                     show_grid=True,
                     label_fontsize=label_fontsize,
                     tick_fontsize=tick_fontsize,
                     legend_fontsize=legend_fontsize,
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
        smooth_window (int): 平均曲线平滑窗口大小
        fill_alpha (float): 极值阴影区透明度
        linewidth (float): 曲线线宽
        legend_alpha (float): 图例背景透明度
        legend_linewidth (float): 图例中线条粗细
        show_grid (bool): 是否显示网格
        label_fontsize (int): 轴标签字体大小
        tick_fontsize (int): 刻度字体大小
        legend_fontsize (int): 图例字体大小
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
    )

    if not x_targets or not stats_data:
        print("未获取到运行级指标数据，退出绘图。")
        return

    active_metrics = [m for m in metric_names if any(m in stats_data.get(a, {}) for a in stats_data)]
    if not active_metrics:
        print("未检测到任何运行级指标数据，退出绘图。")
        return

    active_algos = algo_list if algo_list is not None else found_algos

    # 配色配置（与 figure1 保持一致）
    palette = list(sns.color_palette("tab10", max(10, len(active_algos))))
    if len(palette) >= 8:
        palette[4] = (0.8, 0.0, 0.8)
        palette[7] = (0.0, 0.0, 0.0)

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
        xlabel_text = METRIC_XLABELS.get(metric, 'Step')
        xt = x_targets[metric]

        fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
        has_plotted = False
        y_min_total, y_max_total = float('inf'), float('-inf')

        for j, algo in enumerate(active_algos):
            if algo in stats_data and metric in stats_data[algo]:
                entry = stats_data[algo][metric]
                raw_mean = entry['mean']
                raw_min = entry['min']
                raw_max = entry['max']

                mean_y = smooth_curve(raw_mean, smooth_window)
                min_y = smooth_curve(raw_min, smooth_window)
                max_y = smooth_curve(raw_max, smooth_window)

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
        ax.set_xlabel(xlabel_text, fontweight='bold', fontsize=label_fontsize)
        ax.set_ylabel(ylabel_text, fontweight='bold', fontsize=label_fontsize)
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.offsetText.set_fontsize(tick_fontsize)  # 1e7 等科学计数法偏移文字与刻度同号

        if has_plotted and y_min_total < y_max_total:
            pad = max((y_max_total - y_min_total) * 0.05, 0.02)
            ax.set_ylim(y_min_total - pad, y_max_total + pad)

        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4)

        if has_plotted:
            leg = ax.legend(
                loc="lower right",
                ncol=2,
                fontsize=legend_fontsize,
                framealpha=legend_alpha,
                borderpad=0.25,
                labelspacing=0.2,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=0.6
            )
            if leg is not None:
                leg.set_draggable(True)
                for legobj in leg.get_lines():
                    legobj.set_linewidth(legend_linewidth)
                    legobj.set_alpha(1.0)

        fig.tight_layout(pad=0.2)

        if save_base_dir:
            pdf_dir = os.path.join(save_base_dir, "draw_pdf")
            svg_dir = os.path.join(save_base_dir, "draw_svg")
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(svg_dir, exist_ok=True)
            # 不使用 bbox_inches='tight'，保持精确尺寸不缩放
            pdf_path = os.path.join(pdf_dir, f"{metric}.pdf")
            svg_path = os.path.join(svg_dir, f"{metric}.svg")
            fig.savefig(pdf_path, format='pdf', dpi=dpi)
            fig.savefig(svg_path, format='svg', dpi=dpi)
            print(f"已保存: {pdf_path} / {svg_path}")

        print(f"绘制: {ylabel_text}")


if __name__ == "__main__":
    # 配置 exp_csv 路径
    EXP_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_csv")

    # ==================== 1. 横轴范围与插值点数 ====================
    X_MIN = 0
    X_MAX = None          # None 表示自动检测所有 CSV 中的最大步数（如 20000000 即 20M）
    NUM_POINTS = 500      # 统一插值采样点数

    # ==================== 2. 算法与对手配置 ====================
    ALGO_LIST = None      # 算法列表，None 表示自动扫描 exp_csv 目录下所有子文件夹
    ALGO_LABELS = None    # 算法图例别名映射，支持 dict 如 {"PFSP_0": "PFSP (delta=0)"} 或 list
    OPP_LIST = None       # 对手列表，None 表示自动按 0, 1, 2, 3 排序读取
    
    # 子图标题配置（已注释留作调试，SHOW_TITLE=False 默认彻底不绘制子图标题）
    # DISPLAY_TITLES = ['Agents WinRate Vs Opp. 1', 'Agents WinRate Vs Opp. 2', 'Agents WinRate Vs Opp. 3', 'Agents WinRate Vs Opp. 4']
    DISPLAY_TITLES = None
    SHOW_TITLE = False    # 是否显示各子图上方的标题（设为 False 则彻底不显示任何标题）

    # ==================== 3. 视觉样式设置（可在此调节） ====================
    SMOOTH_WINDOW = 61    # 平均曲线的滑动平均平滑窗口大小（推荐 51 ~ 101，数值越大越平滑，<=1 不平滑）
    FILL_ALPHA = 0.10     # 极值阴影区透明度（0.05 ~ 0.15，使阴影块更浅）
    LINEWIDTH = linewidth # 曲线线宽（pt），由顶部 linewidth 统一控制
    LEGEND_ALPHA = 0.35   # 图例背景透明度（更通透）
    LEGEND_LINEWIDTH = legend_linewidth  # 图例中展示的线条粗细（pt），由顶部统一控制
    INVERT_Y = False      # 是否 1 - y 反转数值（默认 False）

    # 字号统一为六号（7.5pt），由文件顶部 FONT_SIZE_HAO 定义，无需在此重复设置
    # 图片尺寸统一为宽5cm×高4cm，由文件顶部 FIG_WIDTH_IN / FIG_HEIGHT_IN 定义

    # 保存根目录：在其下自动创建 draw_pdf/ 和 draw_svg/ 两个子目录
    SAVE_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ==================== 4. 执行绘图 ====================
    # 比分 + 双杀率：每个对手各一张独立 figure，双杀率一张独立 figure
    plot_interpolated_win_rates(
        exp_csv_dir=EXP_CSV_DIR,
        x_min=X_MIN,
        x_max=X_MAX,
        num_points=NUM_POINTS,
        algo_list=ALGO_LIST,
        algo_labels=ALGO_LABELS,
        opp_list=OPP_LIST,
        display_titles=DISPLAY_TITLES,
        show_title=SHOW_TITLE,
        smooth_window=SMOOTH_WINDOW,
        fill_alpha=FILL_ALPHA,
        linewidth=LINEWIDTH,
        legend_alpha=LEGEND_ALPHA,
        legend_linewidth=LEGEND_LINEWIDTH,
        invert_y=INVERT_Y,
        include_mutualkill=True,    # 额外生成双杀率图
        mutualkill_label="对基准对手平均双杀率",
        save_base_dir=SAVE_BASE_DIR
    )

    # ==================== 5. 运行级指标曲线 ====================
    # 每个指标各一张独立 figure
    # 依次为：预训练准确率 / 累积回报 / 预训练策略熵 / 在线训练策略熵
    METRIC_NAMES = ('accuracy', 'return', 'pre_entropy', 'entropy')

    plot_run_metrics(
        exp_csv_dir=EXP_CSV_DIR,
        metric_names=METRIC_NAMES,
        x_min=X_MIN,
        x_max=X_MAX,
        num_points=NUM_POINTS,
        algo_list=ALGO_LIST,
        algo_labels=ALGO_LABELS,
        metric_labels=None,        # None 则使用内置 METRIC_LABELS 中文映射
        smooth_window=SMOOTH_WINDOW,
        fill_alpha=FILL_ALPHA,
        linewidth=LINEWIDTH,
        legend_alpha=LEGEND_ALPHA,
        legend_linewidth=LEGEND_LINEWIDTH,
        show_grid=True,
        save_base_dir=SAVE_BASE_DIR
    )

    # 一次性显示所有 figure（不再逐张阻塞）
    plt.show()