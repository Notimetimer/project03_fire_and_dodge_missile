'''
同步并行化改进（每个仿真进程同步开始，结束后等待其他仿真进程结束）
放弃非阻塞的并行测试，改为严格的并行测试完成后再并行采样，都完成了再并行测试
'''

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
import traceback # [新增]
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new import *
from Envs.Tasks.ChooseStrategyEnv2_2 import *
from Algorithms.PPOHybrid23_0 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Algorithms.Utils import compute_monte_carlo_returns
from prepare_il_datas import run_rules
from VsBaseline_while_training import test_worker

dt_move = 0.04

def get_current_file_dir():
    return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()


def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)



def run_MLP_simulation(
    mission_name='无名',
    actor_lr=1e-4,
    critic_lr=5e-4,
    hidden_dim=None,
    gamma=0.995,
    lmbda=0.995,
    epochs=4,
    eps=0.2,
    k_entropy=None,
    label_smoothing=0.3,
    max_episode_duration=15*60,
    R_cage = 62.00e3, # 45e3 # 55e3,
    device = torch.device("cpu"),
):

    # 1. 设置随机数种子 (Master)
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 2. 参数与环境配置 (Master 用于获取维度)
    parser = argparse.ArgumentParser("UAV swarm confrontation")
    parser.add_argument("--max-episode-len", type=float, default=max_episode_duration, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=R_cage, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = ChooseStrategyEnv(args)
    state_dim = dummy_env.obs_dim
    action_dims_dict = {'cont': 0, 'cat': dummy_env.fly_act_dim, 'bern': dummy_env.fire_dim}
    del dummy_env

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Master training device: {device}")

    # 3. 创建神经网络
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, None, device).to(device)

    student_agent = PPOHybrid(
        actor=actor_wrapper, 
        critic=critic_net, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        lmbda=lmbda, 
        epochs=epochs, 
        eps=eps, 
        gamma=gamma, 
        device=device, 
        k_entropy=k_entropy, 
        max_std=label_smoothing
    )
    
    
    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    # 保存供人读取的网络结构参数
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())

    # 保存神经网络权重参数
    init_opponent_name = "actor_rein0"
    torch.save(student_agent.actor.state_dict(), os.path.join(log_dir, f"{init_opponent_name}.pt"))
    torch.save(student_agent.critic.state_dict(), os.path.join(log_dir, f"critic.pt"))

    # 保存onnx模型
    # 前提：假设此时 student_agent 已经创建好，且 state_dim 已经定义
    # 构建一个与 state 维度相同的 dummy input (batch_size=1)
    dummy_state = torch.randn(1, state_dim).to(device)

    # ==========================================
    # 1. 导出 Actor 的底层网络（PolicyNetHybrid）
    # ==========================================
    actor_onnx_path = os.path.join(log_dir, "student_actor.onnx")

    # 对于你的 PolicyNetHybrid，它返回的是一个 dict {'cont': ..., 'cat': ..., 'bern': ...}
    # 在高版本的 PyTorch 中，ONNX 对返回 dict 有支持（自动解包），或者你可以写一个简单的 wrapper 解包
    try:
        torch.onnx.export(
            student_agent.actor.net,           # 只导出纯网络结构，避开 Wrapper里的复杂采样操作
            dummy_state,                       # 伪造的输入状态
            actor_onnx_path,                   # 输出的文件名 / 路径
            export_params=True,                # 是否连同参数一起导出（选 True 可以看权重信息）
            opset_version=11,                  # 建议使用 11 或以上的算子集
            do_constant_folding=True,          # 是否执行常量折叠优化
            input_names=['state'],             # 命名的输入节点名称
            output_names=['cat_output', 'bern_output'] # 按照返回顺序手动指定名字
        )
        print(f"Actor ONNX successfully exported to {actor_onnx_path}")
    except Exception as e:
        print(f"Error exporting Actor ONNX: {e}")

    # ==========================================
    # 2. 导出 Critic 的底层网络（ValueNet）
    # ==========================================
    critic_onnx_path = os.path.join(log_dir, "student_critic.onnx")

    try:
        torch.onnx.export(
            student_agent.critic,              
            dummy_state,                       
            critic_onnx_path,                  
            export_params=True,                
            opset_version=11,                 
            do_constant_folding=True,          
            input_names=['state'],             
            output_names=['value_estimate']    # Critic 返回的一般是标量价值
        )
        print(f"Critic ONNX successfully exported to {critic_onnx_path}")
    except Exception as e:
        print(f"Error exporting Critic ONNX: {e}")

    # ==========================================
    # 3. 保存 JIT 模型 (TorchScript)
    # ==========================================
    # todo


if __name__ == "__main__":
    hidden_dim = [128, 128, 128]
    run_MLP_simulation(hidden_dim=hidden_dim)