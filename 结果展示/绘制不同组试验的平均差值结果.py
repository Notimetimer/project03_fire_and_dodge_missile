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

# --- 1. 环境与绘图配置 ---
sns.set_theme(style="darkgrid", font="SimHei", rc={"axes.unicode_minus": False})

EXP_CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_csv")


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
    智能提取文件名中的对手标识，兼容 opponent、opponnet(手误)、opp、rule 以及带下划线等情况
    例如：
      run1_vs_opponent0.csv   -> 0
      run1_vs_opponnet0.csv   -> 0
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
    
    Args:
        exp_csv_dir (str): exp_csv 根目录路径
        x_min (float): 插值横轴起点
        x_max (float): 插值横轴终点，若为 None 则自动根据所有 CSV 数据的最大 Step 确定
        num_points (int): 插值点数
        algo_list (list): 算法文件夹名称列表，为 None 时自动扫描子文件夹
        invert_y (bool): 是否使用 (1 - y) 反转胜率，默认为 False
        
    Returns:
        x_target (np.ndarray): 统一的横轴插值点
        stats_data (dict): 结构为 { algo_name: { opp_id: {'mean': ..., 'min': ..., 'max': ..., 'runs': count} } }
        all_opponents (list): 排序后的所有对手标识列表
        algo_names (list): 算法名称列表
    """
    if not os.path.exists(exp_csv_dir):
        print(f"错误: 目录不存在 -> {exp_csv_dir}")
        return None, {}, [], []

    if algo_list is None:
        algo_names = [d for d in os.listdir(exp_csv_dir) if os.path.isdir(os.path.join(exp_csv_dir, d))]
        algo_names.sort()
    else:
        algo_names = [a for a in algo_list if os.path.isdir(os.path.join(exp_csv_dir, a))]

    if not algo_names:
        print(f"警告: {exp_csv_dir} 下未找到任何算法子目录。")
        return None, {}, [], []

    # 第一遍扫描：收集所有文件并确定最大 Step
    all_opp_set = set()
    detected_max_step = 0

    parsed_files_by_algo = {} # { algo: [ (opp_id, csv_path) ] }
    for algo in algo_names:
        algo_dir = os.path.join(exp_csv_dir, algo)
        parsed_files_by_algo[algo] = []
        csv_files = [f for f in os.listdir(algo_dir) if f.endswith('.csv')]
        for fname in csv_files:
            opp_id = extract_opponent_id(fname)
            all_opp_set.add(opp_id)
            csv_path = os.path.join(algo_dir, fname)
            parsed_files_by_algo[algo].append((opp_id, csv_path))

    if x_max is None:
        for algo, file_tuples in parsed_files_by_algo.items():
            for opp_id, csv_path in file_tuples:
                try:
                    df = pd.read_csv(csv_path)
                    if not df.empty:
                        step_col, _ = find_step_and_value_columns(df)
                        detected_max_step = max(detected_max_step, df[step_col].max())
                except Exception:
                    pass
        x_max = detected_max_step if detected_max_step > 0 else 1e6
        print(f"自动检测到全局最大 Step: {x_max}")

    x_target = np.linspace(x_min, x_max, num_points)
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

                # 线性插值（超出范围的使用边界值填充）
                interp_y = np.interp(x_target, steps, vals, left=vals[0], right=vals[-1])
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
    return x_target, stats_data, all_opponents, algo_names


def plot_interpolated_win_rates(exp_csv_dir,
                                x_min=0,
                                x_max=None,
                                num_points=500,
                                algo_list=None,
                                algo_labels=None,
                                opp_list=None,
                                display_titles=None,
                                show_title=False,
                                layout="2x2",
                                smooth_window=61,
                                fill_alpha=0.10,
                                linewidth=1.08,
                                legend_alpha=0.35,
                                legend_linewidth=2.2,
                                invert_y=False,
                                show_grid=True,
                                save_path=None):
    """
    绘制不同算法对比不同对手的插值胜率曲线图。
    
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
        layout (str): 子图排布方式，可选 "2x2"(2行2列网格), "vertical"(上下排列), "horizontal"(左右单行)
        smooth_window (int): 平均曲线平滑窗口大小（值越大越平滑，<=1 表示不平滑）
        fill_alpha (float): 极值阴影区透明度（较小值使色块更浅）
        linewidth (float): 绘制的曲线线宽（默认 1.08）
        legend_alpha (float): 图例背景透明度（较小值更透明）
        legend_linewidth (float): 图例中线条粗细
        invert_y (bool): 是否使用 1 - y 翻转数值
        show_grid (bool): 是否显示网格
        save_path (str): 图片保存路径（可选）
    """
    x_target, stats_data, all_opponents, found_algos = load_and_interpolate_experiments(
        exp_csv_dir=exp_csv_dir,
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
        algo_list=algo_list,
        invert_y=invert_y
    )

    if x_target is None or not stats_data:
        print("未获取到有效数据，退出绘图。")
        return

    # 对手列表与标题确定
    active_opponents = opp_list if opp_list is not None else all_opponents
    if not active_opponents:
        print("未检测到对手数据，退出绘图。")
        return

    n_opps = len(active_opponents)
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

    # 子图布局（支持 2x2 网格、垂直 vertical、横排 horizontal）
    if layout in ["2x2", "grid"]:
        n_rows, n_cols = 2, 2
        fig_w, fig_h = 10.0, 7.2
    elif layout == "vertical":
        n_rows, n_cols = n_opps, 1
        fig_w, fig_h = 8.0, 2.8 * n_opps + 0.6
    elif layout == "horizontal":
        n_rows, n_cols = 1, n_opps
        fig_w, fig_h = 4.2 * n_cols, 4.5
    else:
        n_cols = min(4, n_opps)
        n_rows = int(np.ceil(n_opps / n_cols))
        fig_w, fig_h = 4.2 * n_cols, 3.5 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.flatten()

    for i, opp_id in enumerate(active_opponents):
        ax = axes_flat[i]
        
        # 确定子图标题
        if display_titles is not None:
            if isinstance(display_titles, dict) and opp_id in display_titles:
                title = display_titles[opp_id]
            elif isinstance(display_titles, list) and i < len(display_titles):
                title = display_titles[i]
            else:
                title = f"Agents WinRate Vs Opp. {i+1}"
        else:
            title = f"Agents WinRate Vs Opp. {i+1}"

        has_plotted = False
        y_min_total, y_max_total = float('inf'), float('-inf')

        for j, algo in enumerate(active_algos):
            if algo in stats_data and opp_id in stats_data[algo]:
                entry = stats_data[algo][opp_id]
                raw_mean = entry['mean']
                raw_min = entry['min']
                raw_max = entry['max']
                num_runs = entry['runs']

                # 对均值及极值上下界进行滑动平均平滑
                mean_y = smooth_curve(raw_mean, smooth_window)
                min_y = smooth_curve(raw_min, smooth_window)
                max_y = smooth_curve(raw_max, smooth_window)

                color = palette[j % len(palette)]
                label_text = str(algo_name_map[algo])

                # 1. 绘制平均值曲线：纯实线 (linestyle='-')，完全不透明 (alpha=1.0)
                ax.plot(x_target, mean_y, label=label_text, color=color, 
                        linewidth=linewidth, linestyle='-', alpha=1.0, zorder=3)
                
                # 2. 绘制极值半透明浅色阴影（无边缘描边，纯阴影）
                ax.fill_between(x_target, min_y, max_y, color=color, 
                                alpha=fill_alpha, edgecolor='none', linewidth=0, zorder=2)

                y_min_total = min(y_min_total, np.nanmin(min_y))
                y_max_total = max(y_max_total, np.nanmax(max_y))
                has_plotted = True

        # 添加参考线 [0, 0.5, 1.0]
        for hval in [0, 0.5, 1.0]:
            ax.axhline(hval, color='gray', linestyle='-', linewidth=0.8, alpha=0.8, zorder=1)

        # 确定 y 轴标签（直接显示 WinRate Vs Opp. 1 ~ 4）
        if display_titles is not None:
            if isinstance(display_titles, dict) and opp_id in display_titles:
                ylabel_text = display_titles[opp_id]
            elif isinstance(display_titles, list) and i < len(display_titles):
                ylabel_text = display_titles[i]
            else:
                ylabel_text = f"WinRate Vs Opp. {i+1}"
        else:
            ylabel_text = f"WinRate Vs Opp. {i+1}"

        # 坐标轴设置
        ax.set_xlabel('Step', fontweight='bold')
        ax.set_ylabel(ylabel_text, fontweight='bold')

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # 调整 y 轴范围
        if has_plotted and y_min_total < y_max_total:
            pad = max((y_max_total - y_min_total) * 0.05, 0.02)
            ax.set_ylim(max(-0.05, y_min_total - pad), min(1.05, y_max_total + pad))
        else:
            ax.set_ylim(-0.05, 1.05)

        # 设置网格
        if show_grid:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4)

        # 图例：超紧凑面积（1/3大小）、多列排布、半透明背景、线条加粗、支持鼠标拖动
        if has_plotted:
            leg = ax.legend(
                loc="lower right", 
                ncol=2, 
                fontsize=8, 
                framealpha=legend_alpha,
                borderpad=0.25,
                labelspacing=0.2,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=0.6
            )
            if leg is not None:
                # 允许鼠标拖动图例
                leg.set_draggable(True)
                # 设置图例中的线条加粗突出
                for legobj in leg.get_lines():
                    legobj.set_linewidth(legend_linewidth)
                    legobj.set_alpha(1.0)

    # 隐藏多余子图
    for k in range(n_opps, len(axes_flat)):
        fig.delaxes(axes_flat[k])

    # 上下左右充足间距 (pad=2.0, w_pad=2.2, h_pad=2.2)
    fig.tight_layout(pad=2.0, w_pad=2.2, h_pad=2.2)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")

    plt.show()


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

    # ==================== 3. 子图布局与视觉样式设置（可在此调节） ====================
    LAYOUT = "2x2"        # 子图排列方式："2x2"(2行2列网格), "vertical"(上下垂直), "horizontal"(左右单行)
    SMOOTH_WINDOW = 61    # 平均曲线的滑动平均平滑窗口大小（推荐 51 ~ 101，数值越大越平滑，<=1 不平滑）
    FILL_ALPHA = 0.10     # 极值阴影区透明度（0.05 ~ 0.15，使阴影块更浅）
    LINEWIDTH = 1.08      # 平均值实线粗细（原1.2的0.9倍：1.08）
    LEGEND_ALPHA = 0.35   # 图例背景透明度（更通透）
    LEGEND_LINEWIDTH = 2.2 # 图例中展示的线条粗细（加粗便于辨认）
    INVERT_Y = False      # 是否 1 - y 反转数值（默认 False）

    # ==================== 4. 执行绘图 ====================
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
        layout=LAYOUT,
        smooth_window=SMOOTH_WINDOW,
        fill_alpha=FILL_ALPHA,
        linewidth=LINEWIDTH,
        legend_alpha=LEGEND_ALPHA,
        legend_linewidth=LEGEND_LINEWIDTH,
        invert_y=INVERT_Y,
        save_path=None
    )