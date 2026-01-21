import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
import re
import time  # 确保引入 time 模块
from datetime import datetime
import torch.multiprocessing as mp  # 使用 torch 的多进程模块
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger

# ==========================================
# [修改] 并行测试 Worker 函数 (增加了 dt_maneuver_val 参数)
# ==========================================
def test_worker(model_state_dict, rule_num, env_args, state_dim, hidden_dim, action_dims_dict, dt_maneuver_val, device_name='cpu'):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    """
    在独立进程中运行一场对战。
    """
    device = torch.device(device_name)
    
    # 1. 局部初始化环境 (必须在子进程内创建)
    # 关闭渲染以节省资源
    test_env = ChooseStrategyEnv(env_args, tacview_show=0)
    test_env.shielded = 1
    test_env.dt_move = 0.05
    test_env.dt_maneuver = dt_maneuver_val # 使用传入的值，不依赖全局变量
    
    # 2. 局部初始化网络并加载权重
    net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(net, action_dims_dict, None, device).to(device)
    actor.load_state_dict(model_state_dict)
    actor.eval() # 设置为评估模式

    # 3. 运行对战逻辑
    test_env.reset(red_init_ammo=6, blue_init_ammo=6)
    action_cycle = 30
    steps = 0
    done = False
    
    b_action_label = 0
    
    while not done and test_env.running:
        # 获取观测
        r_obs, r_check = test_env.obs_1v1('r', pomdp=1)
        b_obs, _ = test_env.obs_1v1('b', pomdp=1)
        
        # 决策点
        if steps % action_cycle == 0:
            # 红方 (规则)
            r_state_check = test_env.unscale_state(r_check)
            r_action_label, r_fire = basic_rules(r_state_check, rule_num)
            if r_fire: launch_missile_immediately(test_env, 'r')
            
            # 蓝方 (神经网络 - 确定性决策)
            with torch.no_grad():
                # [修复] 调用 actor.get_action 而不是 take_action
                # get_action 返回 4 个值: actions_exec, actions_raw, h_state, actions_dist_check
                b_act_exec, _, _, _ = actor.get_action(b_obs, explore={'cont':0, 'cat':0, 'bern':1})
                b_action_label = b_act_exec['cat'][0]
                if b_act_exec['bern'][0]: launch_missile_immediately(test_env, 'b')

        # 物理步
        r_maneuver = test_env.maneuver14LR(test_env.RUAV, r_action_label)
        b_maneuver = test_env.maneuver14LR(test_env.BUAV, b_action_label)
        test_env.step(r_maneuver, b_maneuver)
        
        # 判定
        done, _, _, _ = test_env.combat_terminate_and_reward('b', b_action_label, False, action_cycle)
        steps += 1
        
        if steps * dt_maneuver_val > env_args.max_episode_len: break

    # 返回结果：1 赢, 0 输, 0.5 平
    result = 1 if test_env.win else (0 if test_env.lose else 0.5)
    return rule_num, result


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) # 重要：CUDA 环境下推荐使用 spawn
    
    # [修改] 使用 maxtasksperchild 防止内存泄漏 (如 JSBSim 未完全释放)
    test_pool = mp.Pool(processes=3, maxtasksperchild=10) 
    
    pending_tests = [] # 用于存放正在运行的测试任务
    
    while total_steps < int(max_steps):
        i_episode += 1
        
        # # -- 新增：熵系数变化 --
        # direction = 1 if total_steps<=2e6 else -1
        # # k_entropy['bern'] = max(0.0001, 0.1 - total_steps/2e6 * (0.1-0.0001)) * direction
        # k_entropy['bern'] = 0
        # # ----
        
        # --- 异步测试触发逻辑 ---
        if total_steps >= trigger:
            print(f"\n>>> Triggering Parallel Test at steps {total_steps}...")
            # 1. 深度拷贝当前 Actor 权重到 CPU 内存
            current_weights = {k: v.cpu().clone() for k, v in student_agent.actor.state_dict().items()}
            
            # 2. 异步启动 3 个对战
            for r_idx in [0, 1, 2]:
                res_obj = test_pool.apply_async(
                    test_worker, 
                    args=(current_weights, r_idx, args, state_dim, hidden_dim, action_dims_dict, dt_maneuver, 'cpu') # [修改] 显式传入 dt_maneuver
                )
                pending_tests.append((res_obj, total_steps)) # 记录任务对象和触发时的步数
            
            trigger += trigger_delta # 更新下一次触发阈值

        # --- 结果轮询记录 (非阻塞) ---
        if len(pending_tests) > 0:
            finished_tasks = []
            for task in pending_tests:
                res_obj, recorded_step = task
                if res_obj.ready(): # 检查进程是否跑完
                    # [修改] 直接获取结果，不再包裹 try-except
                    rule_num, outcome = res_obj.get()
                    # 在主进程的 Logger 中记录结果
                    logger.add(f"test/agent_vs_rule{rule_num}", outcome, recorded_step)
                    print(f"  [Async Test Result] Rule_{rule_num}: {outcome} (Triggered at {recorded_step})")
                    finished_tasks.append(task)
            
            # 清理已完成的任务
            for task in finished_tasks:
                pending_tests.remove(task)
            
            
    # 训练代码
    # ......
    
    # [修改] 关闭进程池
    test_pool.close()
    test_pool.join()