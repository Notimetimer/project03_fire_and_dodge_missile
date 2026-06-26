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

# axis equal
def set_axes_equal(ax):
    """确保3D图的坐标轴单位长度相等。"""
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    # 设置等显示缩放比例
    ax.set_box_aspect([1, 1, 1])

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

    # ── 收集关键事件 ────────────────────────────────────────────
    # 事件结构: (time, position, side, event_type)
    # 导弹发射点使用对应 UAV 位置，确保发射时刻导弹与飞机位置一致
    events = []
    
    rx, ry, rz = split(ruav_pos)
    bx, by, bz = split(buav_pos)
    
    # 无人机进入战场
    if len(t_arr) > 0:
        events.append((t_arr[0], (rx[0], ry[0], rz[0]), 'red', 'uav'))
        events.append((t_arr[0], (bx[0], by[0], bz[0]), 'blue', 'uav'))
    
    # 导弹发射和爆炸
    # 发射时间：导弹第一次出现在主记录的前一帧（step 之前）
    for mid, traj in rmis.items():
        launch_time_idx = len(t_arr) - len(traj) - 1
        if launch_time_idx >= 0 and launch_time_idx < len(t_arr):
            events.append((t_arr[launch_time_idx], (rx[launch_time_idx], ry[launch_time_idx], rz[launch_time_idx]), 'red', 'launch'))
        end_time_idx = len(t_arr) - 1
        if end_time_idx >= 0:
            mx, my, mz = split(traj)
            events.append((t_arr[end_time_idx], (mx[-1], my[-1], mz[-1]), 'red', 'explode'))
    
    for mid, traj in bmis.items():
        launch_time_idx = len(t_arr) - len(traj) - 1
        if launch_time_idx >= 0 and launch_time_idx < len(t_arr):
            events.append((t_arr[launch_time_idx], (bx[launch_time_idx], by[launch_time_idx], bz[launch_time_idx]), 'blue', 'launch'))
        end_time_idx = len(t_arr) - 1
        if end_time_idx >= 0:
            mx, my, mz = split(traj)
            events.append((t_arr[end_time_idx], (mx[-1], my[-1], mz[-1]), 'blue', 'explode'))
    
    # 无人机轨迹结束
    if len(t_arr) > 0:
        events.append((t_arr[-1], (rx[-1], ry[-1], rz[-1]), 'red', 'uav'))
        events.append((t_arr[-1], (bx[-1], by[-1], bz[-1]), 'blue', 'uav'))
    
    # 按时间排序并合并 2s 内的事件
    events.sort(key=lambda x: x[0])
    grouped_events = []
    if len(events) > 0:
        current_group = [events[0]]
        for event in events[1:]:
            if event[0] - current_group[0][0] >= 2.0:
                grouped_events.append(current_group)
                current_group = [event]
            else:
                current_group.append(event)
        grouped_events.append(current_group)
    
    # 每组的代表时间和索引
    group_times = [group[0][0] for group in grouped_events]
    group_indices = [np.argmin(np.abs(t_arr - t)) for t in group_times]
    
    # ── 3D 轨迹图 ────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 1, 1, projection='3d')

    # 绘制红方无人机轨迹（打断显示）
    r_segments = []
    start = 0
    for idx in group_indices:
        if idx > start:
            r_segments.append((start, idx))
        start = idx + 1
    if start < len(rx):
        r_segments.append((start, len(rx)))
    
    for seg_start, seg_end in r_segments:
        ax3.plot(rx[seg_start:seg_end], ry[seg_start:seg_end], rz[seg_start:seg_end], 
                 color='crimson', lw=1.5, label='Red UAV' if seg_start == 0 else "")
    
    # 绘制蓝方无人机轨迹（打断显示）
    b_segments = []
    start = 0
    for idx in group_indices:
        if idx > start:
            b_segments.append((start, idx))
        start = idx + 1
    if start < len(bx):
        b_segments.append((start, len(bx)))
    
    for seg_start, seg_end in b_segments:
        ax3.plot(bx[seg_start:seg_end], by[seg_start:seg_end], bz[seg_start:seg_end], 
                 color='royalblue', lw=1.5, label='Blue UAV' if seg_start == 0 else "")

    # 红方导弹：红色虚线
    for mid, traj in rmis.items():
        mx, my, mz = split(traj)
        ax3.plot(mx, my, mz, color='crimson', lw=1.0, linestyle='--', alpha=0.8)

    # 蓝方导弹：蓝色虚线
    for mid, traj in bmis.items():
        mx, my, mz = split(traj)
        ax3.plot(mx, my, mz, color='royalblue', lw=1.0, linestyle='--', alpha=0.8)
    
    # ── 标记时间点编号 ────────────────────────────────────────────
    print(f"\n--- 关键时间点信息 ---")
    print(f"原始事件数: {len(events)}")
    print(f"合并后时间组数: {len(grouped_events)}")
    print(f"t_arr 长度: {len(t_arr)}")
    print(f"group_times: {group_times}")
    print(f"group_indices: {group_indices}")
    
    for i, (time, time_idx) in enumerate(zip(group_times, group_indices)):
        if time_idx >= len(t_arr):
            print(f"  编号 {i}: time_idx={time_idx} 超出范围 len(t_arr)={len(t_arr)}")
            continue
        
        r_pos = (rx[time_idx], ry[time_idx], rz[time_idx])
        b_pos = (bx[time_idx], by[time_idx], bz[time_idx])
        print(f"  编号 {i}: t={time:.3f}, idx={time_idx}")
        print(f"    红方 UAV: ({r_pos[0]:.2f}, {r_pos[1]:.2f}, {r_pos[2]:.2f})")
        print(f"    蓝方 UAV: ({b_pos[0]:.2f}, {b_pos[1]:.2f}, {b_pos[2]:.2f})")
        
        # 判断是否为进场时间（第一个时间点），进场点最大
        is_start = (time == t_arr[0])
        size = 60 if is_start else 30
        
        # 标记 UAV 位置
        ax3.scatter(rx[time_idx], ry[time_idx], rz[time_idx], color='crimson', marker='o', s=size)
        ax3.text(rx[time_idx], ry[time_idx], rz[time_idx] + 0.5, str(i), fontsize=12, color='black', fontweight='bold')
        ax3.scatter(bx[time_idx], by[time_idx], bz[time_idx], color='royalblue', marker='o', s=size)
        ax3.text(bx[time_idx], by[time_idx], bz[time_idx] + 0.5, str(i), fontsize=12, color='black', fontweight='bold')
        
        # 标记该组内所有导弹事件
        for event_time, pos, side, event_type in grouped_events[i]:
            if event_type != 'uav':
                x, y, z = pos
                color = 'crimson' if side == 'red' else 'royalblue'
                print(f"    {side}方 {event_type}: ({x:.2f}, {y:.2f}, {z:.2f})")
                ax3.scatter(x, y, z, color=color, marker='o', s=size)
                ax3.text(x, y, z + 0.5, str(i), fontsize=12, color='black', fontweight='bold')

    ax3.set_xlabel('E (km)')
    ax3.set_ylabel('N (km)')
    ax3.set_zlabel('H (km)')
    ax3.legend(fontsize=8)
    ax3.set_title('3D Trajectory')

    set_axes_equal(ax3)

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


def pick_file(record_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(record_dir, '*.json')))
    if not files:
        print(f"未在 {record_dir} 中找到任何 JSON 文件。")
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
    args = parser.parse_args()

    if args.json_file and os.path.isfile(args.json_file):
        json_path = args.json_file
    else:
        record_dir = find_record_dir()
        json_path = pick_file(record_dir)

    print(f"\n正在读取: {json_path}")
    data = load_replay(json_path)

    save_path = None
    if args.save:
        save_path = os.path.splitext(json_path)[0] + '_trajectory.png'

    plot_replay(data, save_path=save_path)
