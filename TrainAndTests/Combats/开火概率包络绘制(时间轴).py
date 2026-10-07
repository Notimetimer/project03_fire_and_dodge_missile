import os
import sys
import glob
import re
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from math import pi, sin, cos
from itertools import product
from _context import *

from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import ChooseStrategyEnv
from Math_calculates.sub_of_angles import sub_of_radian
from Utilities.LocateDirAndAgents2 import get_latest_log_dir, find_latest_agent_path

# ======================= 可配置参数区 =======================
# 模型来源：True 加载 Algorithms.SACHybrid；False 加载 Algorithms.PPOHybrid23_0
from Algorithms.SACHybrid import PolicyNetHybrid, HybridActorWrapper
# from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper

# 优先使用 dir_name 指定日志目录；为 None 时用 experiment_name 自动找最新
# DIR_NAME = "PPO0.3_flymask_v0h0_fireSL-run-20260921-194654"
# DIR_NAME = "PPO0.3_flymask_v0h0-run-20260921-194617"
# DIR_NAME = "切断PPObern梯度0.3_flymask_v0h0_fireSL-run-20260921-122428"
DIR_NAME = "SAC0.3_flymask_v1h1-run-20260923-213238" # "SAC0.3_flymask_v1h1-run-20260928-093645"

# DIR_NAME = "PPO0.3_flymask_v0h0-run-20260928-111836"

EXPERIMENT_NAME = None

# 抽取进度为几%的actor参数 / 弹药量（循环遍历所有组合）
NUM_PROGRESS_POINTS = 20   # 等间隔抽取的进度点数
AMMO_LIST = [6, 1]            # ammo 初始弹药量

# 观测覆盖：中制导标志 / 自发射以来的等待时间
MID_TERM = 0      # r_obs[3]  missile_in_mid_term
T_SHOOT = 31.0 if MID_TERM else 120     # r_obs[21] = T_SHOOT / 120

# 场景参数
RED_HEIGHT = 8e3
BLUE_HEIGHT = 8e3
AA_HOR_DEG = 180
DEVICE = 'cpu'

# 扫描网格
DIST_MIN_KM, DIST_MAX_KM, DIST_STEP_KM = 8, 100, 15
# 锁定 delta_psi=0（正前方），不再扫描角度

# 绘图：开火概率不归一化，固定 0=最深蓝、1=白色
COLOR_LEVELS = 21     # 颜色采样等级（越大过渡越细）

# 总步数
total_steps = 2e6

# 出图规格
FIG_DPI = 600 # 300            # 保存分辨率 (dpi)
FIG_WIDTH_CM = 10 # 4.0       # 图宽 (cm)
FIG_HEIGHT_CM = None     # 图高 (cm)；None 时极坐标按正方形、直角坐标按 0.7 倍宽自动

delta_psi_refer = np.radians(0)  # 只考虑正前方

# 输出目录（相对 project_root）与是否弹窗显示
OUT_DIR_NAME = os.path.join('结果展示', 'exp_png2')
SHOW_PLOT = True
# ===========================================================


def create_initial_states(red_height, blue_height, distance, delta_psi, AA_hor=0):
    """
    创建初始状态配置，借用BasicRules_new_hierarchical找攻击区.py的初始化方法

    Args:
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m) 
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad) - 这是我机-目标视线方位角和初始我机航向角之间的差值
        AA_hor: 水平进入角偏移 (rad)

    说明：
    - delta_psi = 目标视线方位角 - 我机航向角
    - 红方(我机)初始位置：[0, h, 0]，航向角0 (向北)
    - 蓝方(目标)初始位置：[distance*cos(delta_psi), h, distance*sin(delta_psi)]
      航向角 = 0 + AA_hor (基础向东 + 进入角偏移)
    - 通过调整蓝方的横向位置来实现不同的delta_psi，而不是调整航向角
    """
    # 红方在西边[0, h, 0]面向东
    DEFAULT_RED_BIRTH_STATE = {
        'position': np.array([0.0, red_height, 0.0]),
        'psi': 0,  # 面向北
        'e2e': False
    }
    
    # 蓝方在东边，航向角 = 基础向东(0) + AA_hor(进入角偏移)
    # delta_psi通过调整蓝方位置来实现，而不是航向角
    blue_psi = sub_of_radian(delta_psi + AA_hor, 0)
    
    # 根据delta_psi调整蓝方的横向位置，实现不同的视线角度
    blue_N = distance * cos(delta_psi)  # 横向偏移
    blue_E = distance * sin(delta_psi)  # 纵向距离
    
    DEFAULT_BLUE_BIRTH_STATE = {
        'position': np.array([blue_N, blue_height, blue_E]),
        'psi': blue_psi,
        'e2e': False
    }
    
    return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

def select_agents_by_interval(log_dir, num_points=20, total_steps=2e6):
    """
    扫描目录中 actor_rein*.pt，按编号排序，等间隔抽取 num_points 个文件。
    文件名数字归一化到 0~100% 后乘以 total_steps 得到实际步数。
    返回 (steps, paths) 两个列表，steps 对应训练步数，paths 对应文件路径。
    """
    files = glob.glob(os.path.join(log_dir, "actor_rein*.pt"))
    step_files = []
    for f in files:
        m = re.fullmatch(r'actor_rein(\d+(?:\.\d+)?)\.pt', os.path.basename(f))
        if m:
            step_files.append((float(m.group(1)), f))
    if not step_files:
        return [], []
    step_files.sort(key=lambda x: x[0])
    raw_numbers = [x[0] for x in step_files]
    max_number = max(raw_numbers) if raw_numbers else 1.0
    # 归一化到 0~100% 后乘以 total_steps
    percentages = [n / max_number * 100.0 for n in raw_numbers]
    steps = [p * total_steps / 100.0 for p in percentages]
    paths = [x[1] for x in step_files]

    # 等间隔抽取 num_points 个索引
    if len(steps) <= num_points:
        # 文件不足 20 个，全部使用
        selected_indices = list(range(len(steps)))
    else:
        selected_indices = np.linspace(0, len(steps) - 1, num_points, dtype=int)
    selected_steps = [steps[i] for i in selected_indices]
    selected_paths = [paths[i] for i in selected_indices]

    print(f"扫描到 {len(steps)} 个 actor_rein 文件，最大文件名数字 {max_number:.0f}，等间隔抽取 {len(selected_steps)} 个")
    for i, (s, p) in enumerate(zip(selected_steps, selected_paths)):
        print(f"  [{i+1}/{len(selected_steps)}] 步数 {s:.0f}: {os.path.basename(p)}")
    return selected_steps, selected_paths

def load_trained_actor(model_path, device='cpu'):
    """加载训练好的actor模型"""
    # 加载模型参数
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建网络结构（需要与训练时一致）
    state_dim = 22  # 根据环境观测维度
    hidden_dims = [256, 128]
    action_dims_dict = {'cont': 0, 'cat': [5, 6], 'bern': 1}  # 根据你的动作空间
    
    # 创建网络
    policy_net = PolicyNetHybrid(state_dim, hidden_dims, action_dims_dict)
    actor = HybridActorWrapper(policy_net, action_dims_dict, device=device)
    
    # 加载权重
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    return actor

def run_single_step_firing_probability(actor, red_height, blue_height, distance, delta_psi, AA_hor=0, device='cpu', ammo=6):
    """
    运行单步并获取开火概率

    Args:
        actor: 策略网络
        red_height: 红方高度 (m)
        blue_height: 蓝方高度 (m)
        distance: 水平距离 (m)
        delta_psi: 目标方位角偏移 (rad)
        AA_hor: 水平进入角偏移 (rad)
        device: 计算设备
    """
    # 创建环境
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=120.0)
    args = parser.parse_args([])
    args.R_cage = 100e3
    
    env = ChooseStrategyEnv(args, tacview_show=False, vertices=None)
    env.dt_move = 0.04
    env.shielded = 1
    
    # 创建初始状态
    red_state, blue_state = create_initial_states(
        red_height=red_height, 
        blue_height=blue_height, 
        distance=distance, 
        delta_psi=delta_psi, 
        AA_hor=AA_hor
    )
    
    # 重置环境
    env.reset(red_birth_state=red_state, blue_birth_state=blue_state,
              red_init_ammo=ammo, blue_init_ammo=ammo)
    
    # 获取观测
    r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)

    # 验证开火等待时间和中制导标志位是否起到效果
    t_since_launch = r_obs[21]
    missile_in_mid_term = r_obs[3]
    
    # 修改数值
    r_obs[21] = T_SHOOT / 120  # 设置等待时间
    r_obs[3] = MID_TERM        # 设置中制导标志位
    
    # 转换为tensor
    if isinstance(r_obs, np.ndarray):
        r_obs_tensor = torch.tensor(r_obs, dtype=torch.float).unsqueeze(0).to(device)
    else:
        r_obs_tensor = r_obs.unsqueeze(0).to(device)
    
    # 获取动作和开火概率
    with torch.no_grad():
        actions_exec, actions_raw, _, actions_dist_check = actor.get_action(
            r_obs_tensor, explore=False, check_obs=r_check_obs, temperature=1.0
        ) # check_obs=r_check_obs, None
    
    # 提取开火概率
    firing_probability = actions_dist_check['bern'][0] if 'bern' in actions_dist_check else 0.0
    
    
    return firing_probability

def plot_firing_probability_heatmap_timeline(steps, distances, probabilities, ammo):
    """
    在直角坐标系中绘制开火概率热图（时间轴版本）
    横轴：训练步数，纵轴：距离

    Args:
        steps: 训练步数数组
        distances: 距离数组 (m)
        probabilities: 概率矩阵，形状为 (len(steps), len(distances))
        ammo: 导弹数量
    """
    import matplotlib.ticker as ticker
    import matplotlib as mpl

    # 字体配置：中文用默认字体，英文/数学用 Times New Roman
    mpl.rcParams['font.family'] = ['sans-serif']
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 中文支持
    mpl.rcParams['mathtext.fontset'] = 'stix'  # 数学公式用类似 Times New Roman 的字体

    # 图尺寸：cm -> inch；高度默认 0.7 倍宽
    fig_w = FIG_WIDTH_CM / 2.54
    fig_h = (FIG_HEIGHT_CM if FIG_HEIGHT_CM is not None else FIG_WIDTH_CM * 0.7) / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 创建网格 - 横轴步数，纵轴距离
    Steps, Distances = np.meshgrid(steps, distances / 1000)

    # probabilities形状: (len(steps), len(distances))
    # 网格形状: (len(distances), len(steps))
    probabilities_plot = probabilities.T

    # 绘制热图：固定 0~1 映射，0=最深蓝，1=白色，不做数据归一化
    c = ax.contourf(Steps, Distances, probabilities_plot,
                    levels=np.linspace(0, 1, COLOR_LEVELS + 1),
                    cmap='Blues_r', norm=plt.Normalize(vmin=0, vmax=1))

    # 添加颜色条：不显示标签，刻度固定为百分比
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # 设置标签
    ax.set_xlabel('训练步数')
    ax.set_ylabel('距离/(km)')
    ax.set_title(f'$n_{{msl}} = {ammo}$', fontsize=12)

    # 横轴科学记数法格式
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # # 纵轴单位标注（放在图外左侧顶部）
    # ax.text(-0.15, 1.02, '$\\times 1$ km', transform=ax.transAxes,
    #         ha='right', va='bottom', fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax

def main():
    # 固定参数（全部来自文件顶部的可配置区）
    red_height = RED_HEIGHT
    blue_height = BLUE_HEIGHT
    AA_hor = np.radians(AA_HOR_DEG)
    device = DEVICE

    # 网格搜索参数：锁定 delta_psi=0（正前方）
    distances = np.arange(DIST_MIN_KM * 1e3, DIST_MAX_KM * 1e3 + DIST_STEP_KM * 1e3, DIST_STEP_KM * 1e3)
    delta_psi = delta_psi_refer

    print(f"开始计算开火概率...")
    print(f"距离范围: {DIST_MIN_KM:.0f}km - {DIST_MAX_KM:.0f}km, 间隔: {DIST_STEP_KM:.0f}km")
    print(f"目标方位角: delta_psi=0（正前方）")
    print(f"总计算点数: {len(distances)}")

    # 查找日志目录
    dir_name = DIR_NAME
    experiment_name = EXPERIMENT_NAME

    logs_root_dir = os.path.join(project_root, "logs/combat")
    latest_log_dir = os.path.join(logs_root_dir, dir_name) if dir_name else \
        get_latest_log_dir(logs_root_dir, experiment_name)

    if not latest_log_dir:
        raise FileNotFoundError(f"No log directory found for mission '{experiment_name}'")

    # 输出目录（相对 project_root），不存在则创建
    out_dir = os.path.join(project_root, OUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # 网络结构参数（只需建一次环境读取维度）
    env_args = argparse.Namespace(max_episode_len=15*60, R_cage=62.00e3)
    env = ChooseStrategyEnv(env_args, tacview_show=False, vertices=None)
    state_dim = env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
    hidden_dim = [128, 128, 128]

    # 等间隔抽取 NUM_PROGRESS_POINTS 个 actor_rein 文件
    steps, agent_paths = select_agents_by_interval(latest_log_dir, NUM_PROGRESS_POINTS, total_steps)
    if not steps:
        print(f"错误: '{latest_log_dir}' 中没有 actor_rein 文件")
        return

    # 循环遍历 ammo（改为外层）
    figures = []  # 存储所有 figure 用于统一展示
    for ammo in AMMO_LIST:
        print(f"\n===== ammo={ammo} =====")

        # 初始化概率矩阵：(len(steps), len(distances))
        probabilities = np.zeros((len(steps), len(distances)))

        # 遍历每个进度点
        for step_idx, (step, agent_path) in enumerate(zip(steps, agent_paths)):
            print(f"\n加载模型: {os.path.basename(agent_path)} (步数 {step:.0f})")

            try:
                actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
                actor = HybridActorWrapper(actor_net, action_dims_dict, device=device).to(device)
                actor.load_state_dict(torch.load(agent_path, map_location=device, weights_only=True), strict=False)
                actor.eval()
                print("模型加载成功")
            except Exception as e:
                print(f"模型加载失败: {e}")
                continue

            # 对所有距离点计算开火概率
            for dist_idx, distance in enumerate(distances):
                print(f"  距离 {distance/1000:.0f}km ({dist_idx+1}/{len(distances)})", end='\r')

                try:
                    prob = run_single_step_firing_probability(
                        actor=actor,
                        red_height=red_height,
                        blue_height=blue_height,
                        distance=distance,
                        delta_psi=delta_psi,
                        AA_hor=AA_hor,
                        device=device,
                        ammo=ammo
                    )
                    probabilities[step_idx, dist_idx] = prob
                except Exception as e:
                    print(f"\n计算错误 (距离={distance/1000:.0f}km): {e}")
                    probabilities[step_idx, dist_idx] = 0.0

        print(f"\n计算完成！")

        base_name = f"fire_prob_timeline_ammo{ammo}"

        # 保存结果
        results = {
            'steps': steps,
            'distances_km': distances / 1000,
            'probabilities': probabilities
        }
        np.savez(os.path.join(out_dir, base_name + '.npz'), **results)

        # 绘制图形并保存 png + svg
        fig1, ax1 = plot_firing_probability_heatmap_timeline(
            steps=steps,
            distances=distances,
            probabilities=1 - np.power(1 - probabilities, 5),
            ammo=ammo
        )
        fig1.savefig(os.path.join(out_dir, base_name + '.png'), dpi=FIG_DPI, bbox_inches='tight')
        fig1.savefig(os.path.join(out_dir, base_name + '.svg'), bbox_inches='tight')
        figures.append(fig1)
        print(f"已保存: {os.path.join(out_dir, base_name)}.png/.svg")

    print("\n全部组合完成！")

    # 统一展示所有 figure
    if SHOW_PLOT:
        plt.show()

    # 展示完后关闭所有 figure
    for fig in figures:
        plt.close(fig)

if __name__ == '__main__':
    main()
