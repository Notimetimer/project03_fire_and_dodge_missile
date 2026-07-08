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
import argparse
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
    x_half = x_range / 2 * (1 + pad)
    y_half = y_range / 2 * (1 + pad)
    ax.set_xlim3d([x_middle - x_half, x_middle + x_half])
    ax.set_ylim3d([y_middle - y_half, y_middle + y_half])
    ax.set_zlim3d([z_min, z_max])
    ax.set_zticks(np.arange(z_min, z_max + 1, 5))

    # box_aspect 按实际跨度比，使三轴单位长度视觉相等
    z_range = z_max - z_min
    ax.set_box_aspect([x_range, y_range, z_range])


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


def _draw_cone_arrow(ax, origin, direction, color, radius_ratio=0.35, n_segments=12):
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


def plot_replay(data: dict, save_path: str = None):
    meta = data.get('meta', {})
    title = (f"Agent: {meta.get('agent', '?')}  |  "
             f"vs Rule_{meta.get('rule_num', '?')}  |  "
             f"Result: {meta.get('result', '?')}")

    ruav_pos = np.array(data['RUAV']['pos_'])   # (T, 3)  [N, H, E]
    buav_pos = np.array(data['BUAV']['pos_'])   # (T, 3)
    t_arr = np.array(data.get('t', []))         # 时间戳数组

    rmis = {mid: np.array(traj) for mid, traj in data.get('RMIS', {}).items()}
    bmis = {mid: np.array(traj) for mid, traj in data.get('BMIS', {}).items()}

    # ---- 坐标轴映射: pos_ = [N, H, E] -> 画图用 (E, N, H) ----
    def split(arr):
        return arr[:, 2] / 1e3, arr[:, 0] / 1e3, arr[:, 1] / 1e3  # E(km), N(km), H(km)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, fontsize=11)

    rx, ry, rz = split(ruav_pos)
    bx, by, bz = split(buav_pos)

    # ── 3D 轨迹图 ────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 1, 1, projection='3d')

    # 红蓝实线轨迹
    ax3.plot(rx, ry, rz, color='crimson', lw=1.5, label='R_UAV')
    ax3.plot(bx, by, bz, color='royalblue', lw=1.5, label='B_UAV')

    # 导弹虚线（只有第一条带 label）
    r_mis_labeled = False
    for mid, traj in rmis.items():
        mx, my, mz = split(traj)
        lbl = 'R_Missile' if not r_mis_labeled else ''
        ax3.plot(mx, my, mz, color='crimson', lw=1.0, linestyle='--', alpha=0.8, label=lbl)
        r_mis_labeled = True
    b_mis_labeled = False
    for mid, traj in bmis.items():
        mx, my, mz = split(traj)
        lbl = 'B_Missile' if not b_mis_labeled else ''
        ax3.plot(mx, my, mz, color='royalblue', lw=1.0, linestyle='--', alpha=0.8, label=lbl)
        b_mis_labeled = True

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
                             color=color, radius_ratio=0.4, n_segments=12)

    # 方向箭头：从进场开始每隔 30s 绘制一次
    # 箭头长度 arrow_len
    add_direction_arrows(ax3, rx, ry, rz, t_arr, color='crimson', interval=30, arrow_len=2.0)
    add_direction_arrows(ax3, bx, by, bz, t_arr, color='royalblue', interval=30, arrow_len=2.0)

    # 起点（大点）和终点（小点）
    ax3.scatter(rx[0],  ry[0],  rz[0],  color='crimson',   marker='o', s=80,  zorder=5)
    ax3.scatter(bx[0],  by[0],  bz[0],  color='royalblue',  marker='o', s=80,  zorder=5)
    ax3.scatter(rx[-1], ry[-1], rz[-1], color='crimson',   marker='o', s=20,  zorder=5)
    ax3.scatter(bx[-1], by[-1], bz[-1], color='royalblue',  marker='o', s=20,  zorder=5)

    ax3.set_xlabel('E (km)')
    ax3.set_ylabel('N (km)')
    ax3.set_zlabel('H (km)')
    ax3.set_title('3D Trajectory')
    leg = fig.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.01, 0.95))
    leg.set_draggable(True)

    set_axes_equal(ax3, z_min=0, z_max=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")

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
