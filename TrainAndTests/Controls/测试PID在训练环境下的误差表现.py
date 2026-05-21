import sys
import os
import numpy as np
import time
import torch.multiprocessing as mp
from math import *
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 引入训练环境与PID控制器
from TrainAndTests.Controls.FlightControl_Train_dual_a_out2 import track_env
from TrainAndTests.Controls.UPolicyWrapper import UnifiedPolicyWrapper
from Math_calculates.sub_of_angles import sub_of_radian

# =============================================================================
# 超参数与环境配置 (严格对齐训练时的并行参数)
# =============================================================================
num_episodes = 100
max_episode_len = 5 * 60
dt_decide = 0.16 # 0.16
dt_move = 0.02 # 0.02
beta_ao_95_time = 10.0
beta_ao = 0.05 ** (dt_decide / beta_ao_95_time)

def run_single_episode(episode_id):
    # 为每个进程设置独立随机数种子，防止并行导致的伪随机同构
    seed = int(time.time() * 1000) % 1000000 + episode_id * 1000
    np.random.seed(seed)
    random.seed(seed)
    
    # 初始化环境和 PID 控制器
    env = track_env(dt_move=dt_move, tacview_show=0, time_limit=max_episode_len)
    env.realistic = 0
    
    pid_controller = UnifiedPolicyWrapper(env, dt_decide=dt_decide)
    
    # 完全复制训练初态生成逻辑
    init_height = np.random.uniform(4000, 10000)
    birth_state = {'position': np.array([0.0, init_height, 0.0]),
                   'psi': np.random.uniform(-pi/6, pi/6)}
    
    height_req = np.clip(init_height + 1 * np.random.uniform(-1, 1) * 5000, 3000, 15000)
    psi_req = np.random.uniform(-pi, pi)
    v_req = np.random.uniform(0.5, 2.5) * 340
    
    env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
    
    obs, obs_check = env.get_obs()
    done = False
    
    # 误差记录容器
    v_error_ema = 0.0
    psi_error_ema = 0.0
    theta_error_ema = 0.0
    ao_ema = 0.0
    
    v_error_sum = 0.0
    psi_error_sum = 0.0
    theta_error_sum = 0.0
    ao_sum = 0.0
    steps = 0
    episode_return = 0.0
    
    while not done:
        # 完全复制训练时“目标会跑”的动态机动生成逻辑
        height_req += np.random.randn() * 80 * dt_decide
        env.height_req = np.clip(height_req, 3000, 13000)
        
        psi_req += np.random.randn() * 10 * (pi/180) * dt_decide
        env.psi_req = sub_of_radian(psi_req)
        
        v_req += np.random.randn() * 3 * dt_decide
        env.v_req = np.clip(v_req, 0.5 * 340, 2.5 * 340)  # 这个虚高的速度输入边界是导致速度基准误差偏大的原因
        
        # PID 计算决策
        obs, obs_check = env.get_obs()
        action = pid_controller.get_action(obs, explore=False)
        
        # 推进环境
        next_obs, reward, done = env.step(action)
        steps += 1
        episode_return += reward
        
        # 获取误差 (包含 EMA 和算术平均累加)
        v_error_ema = beta_ao * v_error_ema + (1 - beta_ao) * abs(env.v_error)
        psi_error_ema = beta_ao * psi_error_ema + (1 - beta_ao) * abs(env.psi_error)
        theta_error_ema = beta_ao * theta_error_ema + (1 - beta_ao) * abs(env.theta_error)
        ao_ema = beta_ao * ao_ema + (1 - beta_ao) * env.AO
        
        v_error_sum += abs(env.v_error)
        psi_error_sum += abs(env.psi_error)
        theta_error_sum += abs(env.theta_error)
        ao_sum += env.AO
        
    # 回合结束，进行 EMA 无偏矫正计算
    unbias_factor = (1 - beta_ao ** max(1, steps))
    final_v_ema = v_error_ema / unbias_factor
    final_psi_ema = (psi_error_ema / unbias_factor)
    final_theta_ema = (theta_error_ema / unbias_factor)
    final_ao_ema = (ao_ema / unbias_factor)
    
    # 算术平均对照组
    avg_v = v_error_sum / max(1, steps)
    avg_psi = (psi_error_sum / max(1, steps))
    avg_theta = (theta_error_sum / max(1, steps))
    avg_ao = (ao_sum / max(1, steps))
    
    # 打包输出分析数据
    return {
        'ep': episode_id,
        'steps': steps,
        'fail': env.fail,
        'stall': env.stall,
        'crash': env.crash,
        'break_up': getattr(env, 'break_up', False),
        'fail_neg_alpha': (env.RUAV.alpha_air * 180 / pi < -8) if env.fail else False,
        'fail_pos_alpha': (env.RUAV.alpha_air * 180 / pi > 26) if env.fail else False,
        'fail_pos_ny': (env.RUAV.Ny > 9.5) if env.fail else False,
        'fail_neg_ny': (env.RUAV.Ny < -3) if env.fail else False,
        'return': episode_return,
        'v_ema': final_v_ema,
        'psi_ema': final_psi_ema,
        'theta_ema': final_theta_ema,
        'ao_ema': final_ao_ema,
        'v_avg': avg_v,
        'psi_avg': avg_psi,
        'theta_avg': avg_theta,
        'ao_avg': avg_ao,
    }

if __name__ == '__main__':
    print("="*60)
    print(f" 开始并行运行 {num_episodes} 个回合的 PID 控制器基准测试")
    print(" 注意：目标机动范围生成受训练环境限制，部分指标可能不符合真实飞机机动极限。")
    print("="*60)
    
    # 配置多线程启动方式，防止 CUDA 调用被污染
    mp.set_start_method('spawn', force=True)
    
    start_time = time.time()
    results = []
    
    # 采用映射多进程加速测试 (开启和 episode_num 相等进程进行并行计算)
    workers_cnt = min(num_episodes, os.cpu_count() or 4)
    with mp.Pool(workers_cnt) as pool:
        res_iter = pool.imap_unordered(run_single_episode, range(num_episodes))
        # 异步收取回报以提供实时打印
        for idx, res in enumerate(res_iter):
            results.append(res)
            print(f"[{idx+1:02d}/{num_episodes}] 回合{'成功' if not res['fail'] else '坠机/失速/解体'} | "
                  f"存活步数: {res['steps']:04d} | 奖励: {res['return']:7.1f} | "
                  f"EMA -> V: {res['v_ema']:6.2f}m/s, AO: {res['ao_ema']:5.2f}°, Psi: {res['psi_ema']:5.2f}°, Theta: {res['theta_ema']:5.2f}°")
                  
    # 对收集到的所有回合作汇总统计
    fail_cnt = sum([1 for r in results if r['fail']])
    stall_cnt = sum([1 for r in results if r['stall']])
    crash_cnt = sum([1 for r in results if r['crash']])
    breakup_cnt = sum([1 for r in results if r.get('break_up', False)])
    
    # 细化失败统计
    neg_alpha_cnt = sum([1 for r in results if r.get('fail_neg_alpha', False)])
    pos_alpha_cnt = sum([1 for r in results if r.get('fail_pos_alpha', False)])
    pos_ny_cnt = sum([1 for r in results if r.get('fail_pos_ny', False)])
    neg_ny_cnt = sum([1 for r in results if r.get('fail_neg_ny', False)])
    
    v_emas = [r['v_ema'] for r in results]
    psi_emas = [r['psi_ema'] for r in results]
    theta_emas = [r['theta_ema'] for r in results]
    ao_emas = [r['ao_ema'] for r in results]

    mean_v_ema, max_v_ema, min_v_ema = np.mean(v_emas), np.max(v_emas), np.min(v_emas)
    mean_psi_ema, max_psi_ema, min_psi_ema = np.mean(psi_emas), np.max(psi_emas), np.min(psi_emas)
    mean_theta_ema, max_theta_ema, min_theta_ema = np.mean(theta_emas), np.max(theta_emas), np.min(theta_emas)
    mean_ao_ema, max_ao_ema, min_ao_ema = np.mean(ao_emas), np.max(ao_emas), np.min(ao_emas)
    
    mean_v_avg = np.mean([r['v_avg'] for r in results])
    mean_psi_avg = np.mean([r['psi_avg'] for r in results])
    mean_theta_avg = np.mean([r['theta_avg'] for r in results])
    mean_ao_avg = np.mean([r['ao_avg'] for r in results])
    
    mean_return = np.mean([r['return'] for r in results])
    survive_rate = (num_episodes - fail_cnt) / num_episodes * 100.0
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print("                PID 测试汇总计算报告")
    print("="*50)
    print(f" 总耗时:     {elapsed:.2f} 秒")
    print(f" 平均奖励 (Return): {mean_return:.2f}")
    print(f" 成功存活率: {survive_rate:.1f}% ({num_episodes - fail_cnt}/{num_episodes})")
    print(f" 失败分解:   失速 {stall_cnt}次 | 坠毁 {crash_cnt}次 | 过载解体 {breakup_cnt}次")
    print(f" 详细诱因:   负迎角 {neg_alpha_cnt}次 | 正迎角 {pos_alpha_cnt}次 | 正过载 {pos_ny_cnt}次 | 负过载 {neg_ny_cnt}次")
    print("-" * 50)
    print("[由于训练场景生成的机动过激，PID 暴露的均值 EMA 基准底板]")
    print(f" 基准速度误差 (v_ema):       {mean_v_ema:7.3f} m/s | 范围: [{min_v_ema:6.2f}, {max_v_ema:6.2f}]")
    print(f" 基准航向误差 (psi_ema):     {mean_psi_ema:7.3f} °   | 范围: [{min_psi_ema:6.2f}, {max_psi_ema:6.2f}]")
    print(f" 基准俯仰角误差 (theta_ema): {mean_theta_ema:7.3f} °   | 范围: [{min_theta_ema:6.2f}, {max_theta_ema:6.2f}]")
    print(f" 基准指向误差 (ao_ema): {mean_ao_ema:7.3f} °   | 范围: [{min_ao_ema:6.2f}, {max_ao_ema:6.2f}]")
    print("-" * 50)
    print("[算术平均误差值对照]")
    print(f" 基准速度算术误差:           {mean_v_avg:.3f} m/s")
    print(f" 基准航向算术误差:           {mean_psi_avg:.3f} °")
    print(f" 基准俯仰算术误差:           {mean_theta_avg:.3f} °")
    print("="*50)
