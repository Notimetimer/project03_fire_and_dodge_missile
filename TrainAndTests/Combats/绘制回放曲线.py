"""
读取 logs/CombatRecord/ 下的 JSON 回放文件，用 matplotlib 绘制 3D/2D 轨迹图。
用法:
    python plot_combat_replay.py                       # 自动列出并选择文件
    python plot_combat_replay.py path/to/replay.json  # 直接指定文件
"""

import os
import sys
import json
import glob
import pickle
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings

# 抑制所有中文字体缺失警告
warnings.filterwarnings('ignore')
# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# axis equal
def set_axes_equal(ax, z_min=0, z_max=20, pad=0.05):
    """x/y 各自紧凑显示（只加 pad），z 轴固定 [z_min, z_max]。
    用 set_box_aspect 按实际跨度比保证单位长度相等。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    # x_half = x_range / 2 * (1 + pad)
    # y_half = y_range / 2 * (1 + pad)
    # ax.set_xlim3d([x_middle - x_half, x_middle + x_half])
    # ax.set_ylim3d([y_middle - y_half, y_middle + y_half])
    # 取 x/y 最大跨度，强制 N/E 等宽
    xy_half = max(x_range, y_range) / 2 * (1 + pad)
    ax.set_xlim3d([x_middle - xy_half, x_middle + xy_half])
    ax.set_ylim3d([y_middle - xy_half, y_middle + xy_half])
    ax.set_zlim3d([z_min, z_max])
    ax.set_zticks([0, 10, 20])

    # box_aspect 按实际跨度比，使三轴单位长度视觉相等
    z_range = z_max - z_min
    xy_span = xy_half * 2
    ax.set_box_aspect([xy_span, xy_span, z_range])


def _perp_basis(d):
    """返回两个单位向量，均垂直于 d 且互相正交。"""
    d = np.asarray(d, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-12:
        return np.array([1, 0, 0]), np.array([0, 1, 0])
    d = d / norm
    ref = np.array([0, 0, 1], dtype=float) if abs(d[2]) < 0.9 else np.array([1, 0, 0], dtype=float)
    u = np.cross(d, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    v = v / np.linalg.norm(v)
    return u, v


def _draw_cone_arrow(ax, origin, direction, color, radius_ratio=0.3, n_segments=12):
    """在 3D 轴上绘制一个实心圆锥箭头。origin 为尾部，direction 指向尖端。"""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    length = np.linalg.norm(direction)
    if length < 1e-9:
        return
    d = direction / length
    radius = length * radius_ratio

    u, v = _perp_basis(d)
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    base_points = np.array([origin + radius * (np.cos(th) * u + np.sin(th) * v) for th in theta])
    apex = origin + direction

    # 圆锥侧面：apex 与相邻两个底面点组成三角形
    side_verts = [[apex, base_points[i], base_points[(i + 1) % n_segments]]
                  for i in range(n_segments)]
    # 底面封口
    base_verts = [[origin, base_points[i], base_points[(i + 1) % n_segments]]
                  for i in range(n_segments)]

    verts = [np.array(tri, dtype=float) for tri in side_verts + base_verts]
    poly3d = Poly3DCollection(verts, facecolors=color, edgecolors=None,
                              alpha=0.85, shade=True)
    ax.add_collection3d(poly3d)


def load_replay(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _detach_legend_from_3d_rotation(ax3, leg):
    """
    让可拖动图例与 3D 轴旋转互不干扰。

    Axes3D 在 button_press_event 上绑定了 self._button_press，
    只要 event.inaxes == self 就会启动旋转。而 fig.legend 绘制在 3D 轴上方、
    与之重叠，鼠标点击图例时 event.inaxes 仍是 3D 轴，导致拖图例时曲线跟着转。

    解决：在 canvas 的回调注册表中找到 3D 轴的 _button_press 回调，
    替换为「先判断点击是否命中图例，命中则 return，否则执行原逻辑」的包装版。
    """
    registry = ax3.figure.canvas.callbacks
    cid_dict = registry.callbacks.get('button_press_event', {})
    for cid, ref in list(cid_dict.items()):
        # matplotlib 把回调包成 _StrongRef / WeakMethod，需 ref() 取出真实函数
        try:
            func = ref()
        except Exception:
            func = None
        if func is None:
            continue
        # 匹配绑定到 ax3 的 _button_press 方法
        if getattr(func, '__name__', '') == '_button_press' and getattr(func, '__self__', None) is ax3:
            original = func

            def wrapped(event, _orig=original, _legend=leg):
                # 图例包含该点击点 → 只拖图例，不触发 3D 旋转
                if _legend.contains(event)[0]:
                    return
                _orig(event)

            registry.disconnect(cid)
            registry.connect('button_press_event', wrapped)
            return
    # 没找到（理论上不会发生），静默跳过


def plot_replay(data: dict, save_path: str = None, show: bool = True):
    meta = data.get('meta', {})
    red_name = meta.get('red_name', '红方')
    blue_name = meta.get('blue_name', '蓝方')
    title = (f"{blue_name}  vs  {red_name}  |  结果: {meta.get('result', '?')}")

    ruav_pos = np.array(data['RUAV']['pos_'])   # (T, 3)  [N, H, E]
    buav_pos = np.array(data['BUAV']['pos_'])   # (T, 3)
    t_arr = np.array(data.get('t', []))         # 时间戳数组

    rmis = {mid: np.array(traj) for mid, traj in data.get('RMIS', {}).items()}
    bmis = {mid: np.array(traj) for mid, traj in data.get('BMIS', {}).items()}

    # ---- 坐标轴映射: pos_ = [N, H, E] -> 画图用 (E, N, H) ----
    def split(arr):
        return arr[:, 2] / 1e3, arr[:, 0] / 1e3, arr[:, 1] / 1e3  # E(km), N(km), H(km)

    # 绘图窗口 10cm x 10cm（1英寸=2.54cm）
    fig = plt.figure(figsize=(10 / 2.54, 10 / 2.54))
    fig.suptitle(title, fontsize=11)

    rx, ry, rz = split(ruav_pos)
    bx, by, bz = split(buav_pos)

    # ── 3D 轨迹图 ────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 1, 1, projection='3d')

    # 红蓝实线轨迹（label 用 show_name + "轨迹"）
    ax3.plot(rx, ry, rz, color='crimson', lw=1.5, label=f'{red_name}轨迹')
    ax3.plot(bx, by, bz, color='royalblue', lw=1.5, label=f'{blue_name}轨迹')

    # # 任务区域限制：以 (0,0,0) 为圆心，62km 为半径的圆（E-N 平面，H=0）
    # radius_km = 62.0
    # theta = np.linspace(0, 2 * np.pi, 256)
    # circle_e = radius_km * np.cos(theta)
    # circle_n = radius_km * np.sin(theta)
    # circle_h = np.zeros_like(theta)
    # ax3.plot(circle_e, circle_n, circle_h, color='black', linestyle='-.', lw=1.2, label='作战区域')

    # 导弹虚线（不显示在图例中）
    for mid, traj in rmis.items():
        mx, my, mz = split(traj)
        ax3.plot(mx, my, mz, color='crimson', lw=1.0, linestyle='--', alpha=0.8)
    for mid, traj in bmis.items():
        mx, my, mz = split(traj)
        ax3.plot(mx, my, mz, color='royalblue', lw=1.0, linestyle='--', alpha=0.8)

    def add_direction_arrows(ax, x, y, z, t_arr, color, interval=30, arrow_len=1.2):
        """从进场开始，每隔 interval 秒在轨迹上绘制一个指向前进方向的圆锥箭头。"""
        if len(t_arr) < 2 or len(x) < 2:
            return
        start_t = t_arr[0]
        end_t = t_arr[-1]
        sample_ts = np.arange(start_t, end_t + 1e-9, interval)
        idxs = [int(np.argmin(np.abs(t_arr - st))) for st in sample_ts]
        idxs = sorted(set(idxs))

        for i in idxs:
            if i == 0:
                dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
            elif i == len(x) - 1:
                dx, dy, dz = x[-1] - x[-2], y[-1] - y[-2], z[-1] - z[-2]
            else:
                dx, dy, dz = x[i+1] - x[i-1], y[i+1] - y[i-1], z[i+1] - z[i-1]
            norm = np.sqrt(dx*dx + dy*dy + dz*dz)
            if norm < 1e-9:
                continue
            scale = arrow_len / norm
            dx, dy, dz = dx * scale, dy * scale, dz * scale
            # 箭头胖瘦比 radius_ratio
            _draw_cone_arrow(ax, origin=(x[i], y[i], z[i]), direction=(dx, dy, dz),
                             color=color, radius_ratio=0.3, n_segments=12)

    # 方向箭头：从进场开始每隔 30s 绘制一次
    # 箭头长度 arrow_len
    add_direction_arrows(ax3, rx, ry, rz, t_arr, color='crimson', interval=30, arrow_len=4)
    add_direction_arrows(ax3, bx, by, bz, t_arr, color='royalblue', interval=30, arrow_len=4)

    # 起点（大点）
    ax3.scatter(rx[0],  ry[0],  rz[0],  color='crimson',   marker='o', s=40,  zorder=5)
    ax3.scatter(bx[0],  by[0],  bz[0],  color='royalblue', marker='o', s=40,  zorder=5)

    # 终点：死亡方画黄色大点（图例统一叫"命中"，只注册一次）；存活方画从倒数第二点指向最后一点的箭头
    red_dead = bool(meta.get('red_dead', False))
    blue_dead = bool(meta.get('blue_dead', False))
    hit_label_used = False

    if red_dead:
        ax3.scatter(rx[-1], ry[-1], rz[-1], color='gold', marker='o', s=160,
                    edgecolors='black', linewidths=1.0, zorder=6,
                    label='' if hit_label_used else '命中')
        hit_label_used = True
    elif len(rx) >= 2:
        d = (rx[-1] - rx[-2], ry[-1] - ry[-2], rz[-1] - rz[-2])
        _draw_cone_arrow(ax3, origin=(rx[-2], ry[-2], rz[-2]), direction=d,
                         color='crimson', radius_ratio=0.3, n_segments=12)

    if blue_dead:
        ax3.scatter(bx[-1], by[-1], bz[-1], color='gold', marker='o', s=160,
                    edgecolors='black', linewidths=1.0, zorder=6,
                    label='' if hit_label_used else '命中')
        hit_label_used = True
    elif len(bx) >= 2:
        d = (bx[-1] - bx[-2], by[-1] - by[-2], bz[-1] - bz[-2])
        _draw_cone_arrow(ax3, origin=(bx[-2], by[-2], bz[-2]), direction=d,
                         color='royalblue', radius_ratio=0.3, n_segments=12)

    ax3.set_xlabel('东 (km)')
    ax3.set_ylabel('北 (km)')
    ax3.set_zlabel('高度 (km)')
    ax3.set_title('三维轨迹')
    leg = fig.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.01, 0.95))
    leg.set_draggable(True)

    # 修复：拖动图例时 3D 曲线跟着旋转。
    # Axes3D._button_press 会在鼠标落在本轴内时启动旋转，
    # 而图例（fig.legend）与 3D 轴重叠，导致拖图例也触发旋转。
    # 这里在 canvas 回调注册表中替换 3D 轴的 _button_press：
    # 若点击命中图例则直接返回，不进入旋转逻辑。
    _detach_legend_from_3d_rotation(ax3, leg)

    set_axes_equal(ax3, z_min=0, z_max=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

    if show:
        plt.show()


def find_record_dir():
    """从本文件位置向上定位 logs/CombatRecord/"""
    cur = os.path.dirname(os.path.abspath(__file__))
    # 本文件在 TrainAndTests/Combats/，向上三层到项目根目录，再进 logs/CombatRecord
    project_root = os.path.dirname(os.path.dirname(cur))
    record_dir = os.path.join(project_root, 'logs', 'CombatRecord')
    if not os.path.isdir(record_dir):
        # 也尝试 _context 风格：向上两层
        project_root2 = os.path.dirname(cur)
        record_dir2 = os.path.join(project_root2, 'logs', 'CombatRecord')
        if os.path.isdir(record_dir2):
            return record_dir2
    return record_dir


def pick_file(record_dir: str, experiment: str = None) -> str:
    files = sorted(glob.glob(os.path.join(record_dir, '*.json')))
    if experiment:
        files = [f for f in files if experiment.lower() in os.path.basename(f).lower()]
    if not files:
        print(f"未在 {record_dir} 中找到匹配实验名称的 JSON 文件。")
        sys.exit(1)
    print(f"\n找到 {len(files)} 个回放文件:")
    for i, f in enumerate(files):
        print(f"  [{i}] {os.path.basename(f)}")
    idx_str = input("请输入编号 (默认 0): ").strip()
    idx = int(idx_str) if idx_str.isdigit() else 0
    return files[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot combat replay JSON')
    parser.add_argument('json_file', nargs='?', default=None, help='Path to replay JSON file')
    parser.add_argument('--save', action='store_true', help='Save figure alongside JSON file')
    parser.add_argument('--experiment', type=str, default=None, help='Filter replay files by experiment name substring')
    args = parser.parse_args()

    if args.json_file and os.path.isfile(args.json_file):
        json_path = args.json_file
    else:
        record_dir = find_record_dir()
        json_path = pick_file(record_dir, experiment=args.experiment)

    print(f"\n正在读取: {json_path}")
    data = load_replay(json_path)

    save_path = None
    if args.save:
        save_path = os.path.splitext(json_path)[0] + '_trajectory.png'

    plot_replay(data, save_path=save_path)


# =========================== 新增：离线图绘功能 ===========================

# --- Figure 6/7 坐标轴显示比例（y轴数据单位 / x轴数据单位；1=等比，2=y轴拉伸2倍） ---
FIG6_YX_ASPECT = 1.5      # Figure 6 俯视图 北(y)/东(x) 显示比例
FIG7_YX_ASPECT = 1.5      # Figure 7 侧视图 高度(y)/东(x) 显示比例


def _setup_fonts():
    """统一设置中文及字号"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    _fs = 8
    plt.rcParams['font.size'] = _fs
    plt.rcParams['axes.labelsize'] = _fs
    plt.rcParams['xtick.labelsize'] = _fs
    plt.rcParams['ytick.labelsize'] = _fs
    plt.rcParams['legend.fontsize'] = _fs


def _fmt_min_sec(t_seconds):
    """秒 -> 分:秒 字符串，如 65 -> '01:05'"""
    m = int(t_seconds) // 60
    s = int(t_seconds) % 60
    return f"{m:02d}:{s:02d}"


def plot_figures_1_to_4(history):
    """根据 history 绘制 Figure 1~4（高度、马赫数、双方距离、ATA）"""
    _setup_fonts()
    t = np.array(history['time'])

    # --- Figure 1: 红蓝双方高度 ---
    plt.figure(1, figsize=(10/2, 4/2))
    plt.clf()
    plt.plot(t, [a / 1000 for a in history['r_alt']], label='红方高度', color='crimson', linestyle='-', linewidth=1.2)
    plt.plot(t, [a / 1000 for a in history['b_alt']], label='蓝方高度', color='royalblue', linestyle='-', linewidth=1.2)
    for i, ft in enumerate(history['r_fire_time']):
        plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='红方开火' if i == 0 else '')
    for i, ft in enumerate(history['b_fire_time']):
        plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='蓝方开火' if i == 0 else '')
    plt.xlabel('时间/(s)')
    plt.ylabel('高度/(km)')
    leg = plt.legend()
    leg.set_draggable(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Figure 2: 马赫数 ---
    plt.figure(2, figsize=(10/2, 4/2))
    plt.clf()
    plt.plot(t, history['r_mach'], label='红方马赫数', color='crimson', linestyle='-', linewidth=1.2)
    plt.plot(t, history['b_mach'], label='蓝方马赫数', color='royalblue', linestyle='-', linewidth=1.2)
    for i, ft in enumerate(history['r_fire_time']):
        plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='红方开火' if i == 0 else '')
    for i, ft in enumerate(history['b_fire_time']):
        plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='蓝方开火' if i == 0 else '')
    plt.xlabel('时间/(s)')
    plt.ylabel('马赫数')
    leg = plt.legend()
    leg.set_draggable(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Figure 3: 双方距离 ---
    plt.figure(3, figsize=(10/2, 4/2))
    plt.clf()
    plt.plot(t, [d / 10000 for d in history['distance']], label='双方距离', color='black', linestyle='-', linewidth=1.2)
    for i, ft in enumerate(history['r_fire_time']):
        plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='红方开火' if i == 0 else '')
    for i, ft in enumerate(history['b_fire_time']):
        plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='蓝方开火' if i == 0 else '')
    plt.xlabel('时间/(s)')
    plt.ylabel('双方距离/(10km)')
    leg = plt.legend()
    leg.set_draggable(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # --- Figure 4: ATA ---
    plt.figure(4, figsize=(10/2, 4/2))
    plt.clf()
    plt.plot(t, history['r_ata'], label='红方夹角', color='crimson', linestyle='-', linewidth=1.2)
    plt.plot(t, history['b_ata'], label='蓝方夹角', color='royalblue', linestyle='-', linewidth=1.2)
    for i, ft in enumerate(history['r_fire_time']):
        plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='红方开火' if i == 0 else '')
    for i, ft in enumerate(history['b_fire_time']):
        plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='蓝方开火' if i == 0 else '')
    plt.xlabel('时间/(s)')
    plt.ylabel('夹角/(°)')
    leg = plt.legend()
    leg.set_draggable(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_figures_6_to_8(history, replay_data, hit_markers, fig6_aspect=1, fig7_aspect=1):
    """根据 replay_data 绘制 Figure 6~8（俯视图、侧视图、threat_distance）"""
    _setup_fonts()

    ruav6 = np.array(replay_data['RUAV']['pos_'])  # [N, H, E, t]
    buav6 = np.array(replay_data['BUAV']['pos_'])

    # --- Figure 6: 俯视图（北-东） ---
    plt.figure(6, figsize=(10/2, 4/2))
    plt.clf()
    ax6 = plt.gca()
    ax6.plot(ruav6[:, 2] / 1e3, ruav6[:, 0] / 1e3, color='crimson', lw=1.2, label='红方飞行轨迹')
    ax6.plot(buav6[:, 2] / 1e3, buav6[:, 0] / 1e3, color='royalblue', lw=1.2, label='蓝方飞行轨迹')
    _r_mis_used = False
    for _mid, _traj in replay_data['RMIS'].items():
        _m = np.array(_traj)
        ax6.plot(_m[:, 2] / 1e3, _m[:, 0] / 1e3, color='crimson', lw=0.8, linestyle='--', alpha=0.8,
                 label='' if _r_mis_used else '红方导弹轨迹')
        _r_mis_used = True
    _b_mis_used = False
    for _mid, _traj in replay_data['BMIS'].items():
        _m = np.array(_traj)
        ax6.plot(_m[:, 2] / 1e3, _m[:, 0] / 1e3, color='royalblue', lw=0.8, linestyle='--', alpha=0.8,
                 label='' if _b_mis_used else '蓝方导弹轨迹')
        _b_mis_used = True
    _hit_used = False
    for _side, _color in (('r', 'crimson'), ('b', 'royalblue')):
        for _p in hit_markers[_side]:
            ax6.scatter(_p[2] / 1e3, _p[0] / 1e3, marker='o', s=60, facecolors='gold',
                        edgecolors='black', linewidths=0.8, zorder=6,
                        label='' if _hit_used else '命中')
            _hit_used = True
    ax6.set_xlabel('东/(km)')
    ax6.set_ylabel('北/(km)')
    ax6.set_aspect(fig6_aspect, adjustable='box')
    ax6.grid(True, alpha=0.3)
    leg6 = ax6.legend()
    leg6.set_draggable(True)
    plt.tight_layout()

    # --- Figure 7: 侧视图（东-高） ---
    plt.figure(7, figsize=(10/2, 4/2))
    plt.clf()
    ax7 = plt.gca()
    ax7.plot(ruav6[:, 2] / 1e3, ruav6[:, 1] / 1e3, color='crimson', lw=1.2, label='红方轨迹')
    ax7.plot(buav6[:, 2] / 1e3, buav6[:, 1] / 1e3, color='royalblue', lw=1.2, label='蓝方轨迹')
    _r_mis_used = False
    for _mid, _traj in replay_data['RMIS'].items():
        _m = np.array(_traj)
        ax7.plot(_m[:, 2] / 1e3, _m[:, 1] / 1e3, color='crimson', lw=0.8, linestyle='--', alpha=0.8,
                 label='' if _r_mis_used else '红方导弹轨迹')
        _r_mis_used = True
    _b_mis_used = False
    for _mid, _traj in replay_data['BMIS'].items():
        _m = np.array(_traj)
        ax7.plot(_m[:, 2] / 1e3, _m[:, 1] / 1e3, color='royalblue', lw=0.8, linestyle='--', alpha=0.8,
                 label='' if _b_mis_used else '蓝方导弹轨迹')
        _b_mis_used = True
    _hit_used = False
    for _side in ('r', 'b'):
        for _p in hit_markers[_side]:
            ax7.scatter(_p[2] / 1e3, _p[1] / 1e3, marker='o', s=60, facecolors='gold',
                        edgecolors='black', linewidths=0.8, zorder=6,
                        label='' if _hit_used else '命中')
            _hit_used = True
    ax7.set_xlabel('东/(km)')
    ax7.set_ylabel('高度/(km)')
    ax7.set_aspect(fig7_aspect, adjustable='box')
    ax7.grid(True, alpha=0.3)
    leg7 = ax7.legend()
    leg7.set_draggable(True)
    plt.tight_layout()

    # --- Figure 8: threat_distance ---
    plt.figure(8, figsize=(10/2, 4/2))
    plt.clf()
    t = np.array(history['time'])
    plt.plot(t, [d / 1e3 for d in history['r_threat_dist']], label='红方 threat_distance',
             color='crimson', linestyle='-', linewidth=1.2)
    plt.plot(t, [d / 1e3 for d in history['b_threat_dist']], label='蓝方 threat_distance',
             color='royalblue', linestyle='-', linewidth=1.2)
    for i, ft in enumerate(history['r_fire_time']):
        plt.axvline(x=ft, color='red', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='红方开火' if i == 0 else '')
    for i, ft in enumerate(history['b_fire_time']):
        plt.axvline(x=ft, color='blue', linestyle='--', linewidth=0.8, alpha=0.7,
                    label='蓝方开火' if i == 0 else '')
    plt.xlabel('时间/(s)')
    plt.ylabel('threat_distance/(km)')
    leg8 = plt.legend()
    leg8.set_draggable(True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_all_figures(history, replay_data, hit_markers,
                     fig6_aspect=1, fig7_aspect=1, show=False):
    """一站式绘制 Figure 1~8（含 3D 轨迹 Figure 5）。
    show=False 时不弹窗，只创建 figure，由外部统一 plt.show()。
    """
    plot_figures_1_to_4(history)
    plot_replay(replay_data, save_path=None, show=False)
    plot_figures_6_to_8(history, replay_data, hit_markers, fig6_aspect, fig7_aspect)
    if show:
        plt.show()


def find_acmi_dir():
    """从本文件位置向上定位到项目根目录，返回 acmi_records 路径"""
    cur = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(cur))
    return os.path.join(project_root, 'acmi_records')


def save_replay_data(save_dir, history, replay_data, hit_markers):
    """用 pickle 保存仿真数据（history / replay_data / hit_markers）"""
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S") + "_replay.pkl"
    filepath = os.path.join(save_dir, filename)
    data = {
        'history': history,
        'replay_data': replay_data,
        'hit_markers': hit_markers,
        'saved_at': now.isoformat(),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"仿真数据已保存: {filepath}")
    return filepath


def load_replay_data(load_path):
    """从 pickle 文件加载仿真数据"""
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    print(f"仿真数据已加载: {load_path}")
    return data['history'], data['replay_data'], data['hit_markers']


def find_latest_replay_pkl(directory):
    """返回目录下最新的 *_replay.pkl 文件路径"""
    files = sorted(glob.glob(os.path.join(directory, '*_replay.pkl')), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    return files[0]


# =========================== 新的 __main__ ===========================


if __name__ == '__main__':
    # 默认从 acmi_records 目录读取最新保存的仿真数据并画图
    acmi_dir = find_acmi_dir()
    latest = find_latest_replay_pkl(acmi_dir)
    if latest is None:
        print(f"未在 {acmi_dir} 中找到 *_replay.pkl 文件，尝试旧版 JSON 逻辑...")
        parser = argparse.ArgumentParser(description='Plot combat replay JSON')
        parser.add_argument('json_file', nargs='?', default=None, help='Path to replay JSON file')
        parser.add_argument('--save', action='store_true', help='Save figure alongside JSON file')
        parser.add_argument('--experiment', type=str, default=None,
                            help='Filter replay files by experiment name substring')
        args = parser.parse_args()
        if args.json_file and os.path.isfile(args.json_file):
            json_path = args.json_file
        else:
            record_dir = find_record_dir()
            json_path = pick_file(record_dir, experiment=args.experiment)
        print(f"\n正在读取: {json_path}")
        data = load_replay(json_path)
        save_path = None
        if args.save:
            save_path = os.path.splitext(json_path)[0] + '_trajectory.png'
        plot_replay(data, save_path=save_path)
        sys.exit(0)

    print(f"\n加载最新仿真数据: {latest}")
    history, replay_data, hit_markers = load_replay_data(latest)
    plot_all_figures(history, replay_data, hit_markers,
                     fig6_aspect=FIG6_YX_ASPECT, fig7_aspect=FIG7_YX_ASPECT, show=True)
