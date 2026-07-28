"""
可视化 parallel_FlightControl_Train_dual_a_out2偏好.py 中使用的随机控制指令信号。
展示高度、航向、速度三个目标信号在一个 episode 内的随机游走轨迹。
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 设置字体以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 参数（与训练脚本一致） ============
dt_decide = 0.16          # 决策步长 (s)
max_episode_len = 5 * 60  # 最大 episode 时长 (s)
max_decisions = int(np.ceil(max_episode_len / dt_decide)) + 10

num_traces = 5  # 绘制多少条随机轨迹

# ============ 辅助函数 ============
def sub_of_radian(angle):
    """将角度限制在 [-pi, pi]"""
    return (angle + pi) % (2 * pi) - pi


# ============ 正弦叠加参数 ============
sin_period = 3 * 60        # 正弦周期 3 分钟 (s)
sin_amp_height = 2000      # 高度正弦幅度 (m)
sin_amp_psi = pi / 2       # 航向正弦幅度 (rad)
sin_amp_v = 80            # 速度正弦幅度 (m/s)


def generate_signal_worker_mode(dt_decide, max_decisions):
    """
    Worker 就地随机模式 (worker_random=1):
    每步独立 randn 叠加，并在每步 clip。
    在随机游走基础上叠加正弦信号。
    """
    init_height = np.random.uniform(4000, 10000)
    height_req = np.clip(init_height + np.random.uniform(-1, 1) * 5000, 1000, 15000)
    psi_req = np.random.uniform(-pi, pi)
    v_req = np.random.uniform(0.5, 1.3) * 340

    # 正弦初始相位随机
    phase_h = np.random.uniform(0, 2 * pi)
    phase_psi = np.random.uniform(0, 2 * pi)
    phase_v = np.random.uniform(0, 2 * pi)

    heights = []
    psis = []
    vs = []

    for step in range(max_decisions):
        if step > 0:
            height_req += np.random.randn() * 80 * dt_decide
            psi_req += np.random.randn() * 10 * pi / 180 * dt_decide
            v_req += np.random.randn() * 3 * dt_decide

        t = step * dt_decide
        sin_h = sin_amp_height * np.sin(2 * pi / sin_period * t + phase_h)
        sin_psi = sin_amp_psi * np.sin(2 * pi / sin_period * t + phase_psi)
        sin_v = sin_amp_v * np.sin(2 * pi / sin_period * t + phase_v)

        h_out = np.clip(height_req + sin_h, 1000, 13000)
        psi_out = sub_of_radian(psi_req + sin_psi)
        v_out = np.clip(v_req + sin_v, 0.5 * 340, 1.3 * 340)

        heights.append(h_out)
        psis.append(psi_out)
        vs.append(v_out)

    return np.array(heights), np.array(psis), np.array(vs)


def generate_signal_master_mode(dt_decide, max_decisions):
    """
    Master 统一生成模式 (worker_random=0):
    先生成全部噪声序列，再 cumsum + clip。
    在随机游走基础上叠加正弦信号。
    """
    init_height = np.random.uniform(4000, 10000)
    height_req0 = np.clip(init_height + np.random.uniform(-1, 1) * 5000, 1000, 15000)
    psi_req0 = np.random.uniform(-pi, pi)
    v_req0 = np.random.uniform(0.5, 1.3) * 340

    height_noise = np.random.randn(max_decisions) * 80 * dt_decide
    psi_noise = np.random.randn(max_decisions) * 10 * pi / 180 * dt_decide
    v_noise = np.random.randn(max_decisions) * 3 * dt_decide

    # 随机游走基线
    height_base = height_req0 + np.cumsum(height_noise)
    psi_base = psi_req0 + np.cumsum(psi_noise)
    v_base = v_req0 + np.cumsum(v_noise)

    # 叠加正弦信号
    t_arr = np.arange(max_decisions) * dt_decide
    phase_h = np.random.uniform(0, 2 * pi)
    phase_psi = np.random.uniform(0, 2 * pi)
    phase_v = np.random.uniform(0, 2 * pi)

    sin_h = sin_amp_height * np.sin(2 * pi / sin_period * t_arr + phase_h)
    sin_psi = sin_amp_psi * np.sin(2 * pi / sin_period * t_arr + phase_psi)
    sin_v = sin_amp_v * np.sin(2 * pi / sin_period * t_arr + phase_v)

    height_req_seq = np.clip(height_base + sin_h, 1000, 13000)
    psi_req_seq = np.array([sub_of_radian(a) for a in (psi_base + sin_psi)])
    v_req_seq = np.clip(v_base + sin_v, 0.5 * 340, 1.3 * 340)

    return height_req_seq, psi_req_seq, v_req_seq


# ============ 绘图 ============
time_axis = np.arange(max_decisions) * dt_decide  # 时间轴 (秒)

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')
fig.suptitle('随机控制指令信号可视化', fontsize=14)

# 左列: Worker 就地随机模式
axes[0, 0].set_title('Worker 模式 — 高度目标 (m)')
axes[1, 0].set_title('Worker 模式 — 航向目标 (°)')
axes[2, 0].set_title('Worker 模式 — 速度目标 (m/s)')

# 右列: Master 统一生成模式
axes[0, 1].set_title('Master 模式 — 高度目标 (m)')
axes[1, 1].set_title('Master 模式 — 航向目标 (°)')
axes[2, 1].set_title('Master 模式 — 速度目标 (m/s)')

np.random.seed(42)

for i in range(num_traces):
    # Worker mode
    h_w, psi_w, v_w = generate_signal_worker_mode(dt_decide, max_decisions)
    axes[0, 0].plot(time_axis, h_w, alpha=0.7, label=f'trace {i}')
    axes[1, 0].plot(time_axis, np.degrees(psi_w), alpha=0.7)
    axes[2, 0].plot(time_axis, v_w, alpha=0.7)

    # Master mode
    h_m, psi_m, v_m = generate_signal_master_mode(dt_decide, max_decisions)
    axes[0, 1].plot(time_axis, h_m, alpha=0.7, label=f'trace {i}')
    axes[1, 1].plot(time_axis, np.degrees(psi_m), alpha=0.7)
    axes[2, 1].plot(time_axis, v_m, alpha=0.7)

# 添加参考线和标签
for col in range(2):
    axes[0, col].axhline(1000, color='r', ls='--', lw=0.8, label='下界 1000m')
    axes[0, col].axhline(13000, color='r', ls='--', lw=0.8, label='上界 13000m')
    axes[0, col].set_ylabel('高度 (m)')
    axes[0, col].legend(loc='upper right', fontsize=7)

    axes[1, col].axhline(-180, color='r', ls='--', lw=0.8)
    axes[1, col].axhline(180, color='r', ls='--', lw=0.8)
    axes[1, col].set_ylabel('航向 (°)')

    axes[2, col].axhline(0.5 * 340, color='r', ls='--', lw=0.8, label='下界 170 m/s')
    axes[2, col].axhline(1.3 * 340, color='r', ls='--', lw=0.8, label='上界 442 m/s')
    axes[2, col].set_ylabel('速度 (m/s)')
    axes[2, col].set_xlabel('时间 (s)')
    axes[2, col].legend(loc='upper right', fontsize=7)

plt.tight_layout()
plt.show()
