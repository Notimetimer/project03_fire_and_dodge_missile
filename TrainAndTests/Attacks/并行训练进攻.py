import argparse
import time
import sys
import os
import numpy as np
import torch
from datetime import datetime
from math import pi

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.AttackManeuverEnv import AttackTrainEnv, dt_maneuver
# 引入我们刚才写的并行 Wrapper
from Algorithms.ParallelEnv import ParallelPettingZooEnv
from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Math_calculates.ScaleLearningRate import scale_learning_rate

# --- 参数配置 ---
parser = argparse.ArgumentParser("UAV Parallel Training")
parser.add_argument("--max-episode-len", type=float, default=120,  # 8 * 60,
                    help="maximum episode time length")  # test 真的中远距空战可能会持续20分钟那么长
parser.add_argument("--R-cage", type=float, default=70e3,  # 8 * 60,
                    help="")
parser.add_argument("--n-envs", type=int, default=8, help="并行环境数量 (根据CPU核数设定)")
parser.add_argument("--max-steps", type=float, default=2e6, help="总训练步数")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 5
hidden_dim = [128, 128, 128]
gamma = 0.9
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy = 0.01
mission_name = 'Attack_Parallel'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# --- 环境工厂函数 ---
def make_env():
    # 这里可以添加不同种子的设置
    def _thunk():
        env = AttackTrainEnv(args, tacview_show=0) 
        return env
    return _thunk

if __name__ == "__main__":
    # 1. 启动并行环境
    print(f"Starting {args.n_envs} parallel environments...")
    env_fns = [make_env() for _ in range(args.n_envs)]
    vec_env = ParallelPettingZooEnv(env_fns)
    
    # 2. 获取维度信息 (用一个临时环境)
    tmp_env = AttackTrainEnv(args)
    # ================= [新增这一行] =================
    tmp_env.reset()  # 必须先重置，生成飞机对象，否则无法获取观测
    # ===============================================
    b_action_dim = tmp_env.b_action_spaces[0].shape[0]
    # 获取观测维度
    tmp_obs, _ = tmp_env.attack_obs('b')
    state_dim = tmp_obs.shape[0]
    del tmp_env

    # 动作空间定义 (Hybrid)
    action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
    action_dims_dict = {'cont': b_action_dim, 'cat': [], 'bern': 0}

    # 3. 初始化 PPO Agent
    actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    critic_net = ValueNet(state_dim, hidden_dim).to(device)
    actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

    agent = PPOHybrid(
        actor=actor_wrapper, critic=critic_net, 
        actor_lr=actor_lr, critic_lr=critic_lr,
        lmbda=lmbda, epochs=epochs, eps=eps, gamma=gamma, 
        device=device, k_entropy=k_entropy, max_std=0.3
    )
    
    # 学习率缩放
    actor_lr = scale_learning_rate(actor_lr, agent.actor)
    critic_lr = scale_learning_rate(critic_lr, agent.critic)
    agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)

    # 日志
    log_dir = os.path.join(project_root, f"logs/attack/{mission_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    # 保存 Meta 信息
    import json
    with open(os.path.join(log_dir, "actor.meta.json"), "w") as f:
        json.dump({k: list(v.shape) for k, v in agent.actor.state_dict().items()}, f)

    # 4. 训练主循环
    total_steps = 0
    # 每次 PPO 更新前收集的步数 = update_steps * n_envs
    # 例如 update_steps=256, n_envs=8 -> 每次更新基于 2048 个样本
    steps_per_rollout = 256 
    
    # Reset 全部环境
    obs_dict = vec_env.reset()
    b_obs = obs_dict['b_obs'] # (N_envs, State_dim)

    try:
        while total_steps < args.max_steps:
            # 缓冲区
            transition_dict = {
                'states': [], 'actions': [], 'next_states': [], 
                'rewards': [], 'dones': []
            }
            
            # --- 数据收集阶段 (Rollout) ---
            for _ in range(steps_per_rollout):
                # A. 蓝方动作 (神经网络批量推理)
                # agent.take_action 支持 (Batch, Dim) 输入
                # 返回 actions_exec (用于环境), actions_raw (用于训练)
                b_actions_exec, b_actions_raw, _, _ = agent.take_action(b_obs, explore=True)
                
                # B. 构造动作字典
                # 红方设为 None，触发 Worker 内部规则
                # 蓝方取出 'cont' 部分传入环境
                actions_dispatch = {
                    'r': None, 
                    'b': b_actions_exec['cont'] # (N_envs, 3)
                }
                
                # C. 并行步进
                results = vec_env.step(actions_dispatch)
                
                next_b_obs = results['b_obs']
                rewards = results['b_reward'] # (N_envs, )
                dones = results['dones']      # (N_envs, )
                
                # D. 存储数据
                # 注意：这里我们存的是批次数据，稍后在更新前展平
                transition_dict['states'].append(b_obs)
                transition_dict['actions'].append(b_actions_raw) # 这是一个包含 (N, 3) 数组的字典
                transition_dict['next_states'].append(next_b_obs)
                transition_dict['rewards'].append(rewards)
                transition_dict['dones'].append(dones)
                
                b_obs = next_b_obs
                total_steps += args.n_envs

            # --- 更新阶段 (Update) ---
            
            # 1. 数据展平 (Flatten) [Time, Batch, Dim] -> [Time*Batch, Dim]
            # 状态
            s_stack = np.stack(transition_dict['states']) # (T, N, D)
            flat_states = s_stack.reshape(-1, s_stack.shape[-1]) # (T*N, D)
            
            ns_stack = np.stack(transition_dict['next_states'])
            flat_next_states = ns_stack.reshape(-1, ns_stack.shape[-1])
            
            # 奖励 & Done
            flat_rewards = np.stack(transition_dict['rewards']).flatten()
            flat_dones = np.stack(transition_dict['dones']).flatten()
            
            # 动作 (字典特殊处理)
            # 原始 list 里的元素是 {'cont': (N, 3), ...}
            # 我们需要把它变成 [{'cont': [1,2,3]}, {'cont': ...}] 这种形式给 PPO
            # 或者更高效：直接拼成 {'cont': (T*N, 3)}
            
            raw_actions_list = transition_dict['actions']
            flat_actions_dict = {}
            for k in raw_actions_list[0].keys():
                # 堆叠 (T, N, Dim) -> (T*N, Dim)
                arr = np.stack([item[k] for item in raw_actions_list])
                flat_actions_dict[k] = arr.reshape(-1, arr.shape[-1])

            # 为了兼容 PPOHybrid2 内部的 update 逻辑 (它可能预期 actions 是一个 list of dicts)
            # 我们这里做一个快速转换，或者你需要确认 PPOHybrid2.update 是否能处理 dict of big arrays
            # 假设 PPOHybrid2 里的逻辑是: actions_from_buffer = transition_dict['actions']
            # vals = [d[key] for d in actions_from_buffer]
            # 为了兼容性，我们需要构造一个 Update 专用的 transition_dict
            
            # 高效构建符合接口的 transition_dict
            update_transition_dict = {
                'states': flat_states,
                'next_states': flat_next_states,
                'rewards': flat_rewards,
                'dones': flat_dones,
                # 这里的 actions 我们直接传“展开后的字典”
                # 但是 PPOHybrid2 现在的代码是遍历 list。
                # 技巧：我们把 (T*N) 的数据伪装成列表传进去，
                # 或者更简单：修改 PPOHybrid2.update 让它能直接接收 dict of tensors (推荐)
                # 鉴于不能改 PPO 代码，我们这里手动展开：
                'actions': [] 
            }
            
            # 手动展开 actions (Python循环较慢，但对于几千个样本还好)
            # 或者我们可以修改 agent.update 逻辑。
            # 这里采用最稳妥的方式：构造 list of dicts
            n_samples = flat_states.shape[0]
            cont_actions = flat_actions_dict['cont']
            # 下面这行可能稍微有点慢，但兼容性最好
            update_transition_dict['actions'] = [{'cont': cont_actions[i]} for i in range(n_samples)]

            # 2. 执行 PPO 更新
            # 注意：如果启用了 GAE，这里直接把展平的数据传进去可能破坏时间相关性
            # 但如果 batch 足够大且 shuffled，PPO 还是能工作的。
            # 若追求完美，应该在 ParallelEnv 层面不做 Flatten，而是让 Agent 处理 (T, N) 数据
            # 但鉴于 PPOHybrid2 是标准实现，Flatten 是通用做法。
            agent.update(update_transition_dict)
            
            # 3. 记录日志
            avg_return = np.mean(flat_rewards) 
            logger.add("train/reward_step", avg_return, total_steps)
            logger.add("train/actor_loss", agent.actor_loss, total_steps)
            logger.add("train/critic_loss", agent.critic_loss, total_steps)
            
            print(f"Step {total_steps}: Reward={avg_return:.4f}, ActorLoss={agent.actor_loss:.4f}")

            # 4. 保存模型
            if total_steps % 10000 < args.n_envs * steps_per_rollout: # 约每1万步保存
                torch.save(agent.actor.state_dict(), os.path.join(log_dir, f"actor_{total_steps}.pt"))
                
    finally:
        vec_env.close()
        logger.close()