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
