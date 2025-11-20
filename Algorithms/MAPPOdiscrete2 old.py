from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# ================================================================= #
#                         辅助函数 (未修改)                           #
# ================================================================= #

# 计算并记录 actor / critic 的梯度范数（L2）
def model_grad_norm(model):
    total_sq = 0.0
    found = False
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach().cpu()
            total_sq += float(g.norm(2).item()) ** 2
            found = True
    return float(total_sq ** 0.5) if found else float('nan')

def check_weights_bias_nan(model, model_name="model", place=None):
    """检查模型中名为 weight/bias 的参数是否包含 NaN，发现则抛出异常。
    参数:
      model: torch.nn.Module
      model_name: 用于错误消息中标识模型（如 "actor"/"critic"）
      place: 字符串，调用位置/上下文（如 "update_loop","pretrain_step"），用于更明确的错误报告
    """
    for name, param in model.named_parameters():
        if ("weight" in name) or ("bias" in name):
            if param is None:
                continue
            if torch.isnan(param).any():
                loc = f" at {place}" if place else ""
                raise ValueError(f"NaN detected in {model_name} parameter '{name}'{loc}")

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)

# ================================================================= #
#                         网络结构定义 (未修改)                        #
# ================================================================= #

class PolicyNetDiscrete(torch.nn.Module):
    """
    个体 Actor 网络 (策略网络)
    输入：单个智能体的观测 obs
    输出：该智能体的动作概率分布
    """
    def __init__(self, obs_dim, hidden_dims, action_dim):
        super(PolicyNetDiscrete, self).__init__()
        layers = []
        prev_size = obs_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, action_dim)

    def forward(self, x):
        x = self.net(x)
        return F.softmax(self.fc_out(x), dim=1)


class CentralizedValueNet(torch.nn.Module):
    """
    中心化 Critic 网络 (价值网络)
    输入:
      - global_state: 所有智能体观测拼接成的全局状态
      - agent_id_one_hot: 标记当前智能体的 one-hot 向量
    输出:
      - 对应智能体的状态价值
    """
    def __init__(self, global_state_dim, back_hidden_dims, front_hidden_dims, num_agents):
        super(CentralizedValueNet, self).__init__()
        
        # 躯干网络 (Backbone)
        back_layers = []
        prev_size = global_state_dim
        for layer_size in back_hidden_dims:
            back_layers.append(torch.nn.Linear(prev_size, layer_size))
            back_layers.append(nn.ReLU())
            prev_size = layer_size
        self.back_net = nn.Sequential(*back_layers)

        # 头部网络 (Heads)
        self.front_heads = nn.ModuleList()
        for _ in range(num_agents):
            front_layers = []
            head_prev_size = prev_size # 从backbone的输出开始
            for layer_size in front_hidden_dims:
                front_layers.append(torch.nn.Linear(head_prev_size, layer_size))
                front_layers.append(nn.ReLU())
                head_prev_size = layer_size
            front_layers.append(torch.nn.Linear(head_prev_size, 1))
            self.front_heads.append(nn.Sequential(*front_layers))

    def forward(self, global_state, agent_id_one_hot):
        # 1. 全局状态通过共享的躯干网络
        back_output = self.back_net(global_state)
        
        # 2. 计算所有头部的输出
        head_outputs = [head(back_output) for head in self.front_heads]
        all_values = torch.cat(head_outputs, dim=1) # shape: (batch_size, num_agents)
        
        # 3. 根据 one-hot id 选择对应的头部输出
        # agent_id_one_hot shape: (batch_size, num_agents)
        value = (all_values * agent_id_one_hot).sum(dim=1, keepdim=True) # shape: (batch_size, 1)
        
        return value

# ================================================================= #
#                      MAPPO 主算法                                 #
# ================================================================= #

class MAPPO:
    ''' MAPPO 算法, 采用 PPO 截断方式 '''
    def __init__(self, obs_dim, action_dim, num_agents, 
                 actor_hidden_dims, critic_hidden_dims_back, critic_hidden_dims_front,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.01, actor_max_grad=2, critic_max_grad=2):
        
        self.num_agents = num_agents
        
        # Actor 网络 (去中心化)
        self.actor = PolicyNetDiscrete(obs_dim, actor_hidden_dims, action_dim).to(device)
        
        # Critic 网络 (中心化)
        global_state_dim = obs_dim * num_agents
        self.critic = CentralizedValueNet(
            global_state_dim, critic_hidden_dims_back, critic_hidden_dims_front, num_agents
        ).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.actor_max_grad=actor_max_grad
        self.critic_max_grad=critic_max_grad

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr    

    def take_action(self, obs, explore=True):
        """根据单个智能体的观测 obs 采取行动"""
        state = torch.tensor(np.array([obs]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) # 离散的输出为类别分布
        if explore:
            action = action_dist.sample()
        else:
            action = torch.argmax(probs)
        # 返回动作索引与对应的概率分布（numpy array）
        probs_np = probs.detach().cpu().numpy()[0].copy() # [0]是batch维度
        return probs_np, action.item()

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=True, mini_batches=4):
        # 从 transition_dict 中提取数据
        obs = torch.tensor(np.array(transition_dict['obs']), dtype=torch.float).to(self.device)
        global_states = torch.tensor(np.array(transition_dict['global_states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_global_states = torch.tensor(np.array(transition_dict['next_global_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        agent_ids = torch.tensor(np.array(transition_dict['agent_ids']), dtype=torch.long).view(-1, 1).to(self.device)
        agent_ids_one_hot = F.one_hot(agent_ids.squeeze(), num_classes=self.num_agents).float().to(self.device)
        
        # --- 计算优势函数 (在整个 batch 上计算) ---
        v_current = self.critic(global_states, agent_ids_one_hot)
        v_next = self.critic(next_global_states, agent_ids_one_hot)
        td_target = rewards + self.gamma * v_next * (1 - dones)
        td_delta = td_target - v_current
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        if adv_normed:
            adv_mean, adv_std = advantage.detach().mean(), advantage.detach().std(unbiased=False) 
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)

        # --- 提前计算旧策略的对数概率和旧的价值预测 (在整个 batch 上计算) ---
        old_log_probs = torch.log(self.actor(obs).gather(1, actions)).detach()
        v_pred_old = self.critic(global_states, agent_ids_one_hot).detach()

        # --- 初始化日志列表 ---
        actor_grad_list, critic_grad_list = [], []
        actor_loss_list, critic_loss_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []
        entropy_list, ratio_list = [], []

        batch_size = obs.shape[0]
        if mini_batches <= 0:
            mini_batch_size = batch_size
        else:
            mini_batch_size = batch_size // mini_batches
        
        # --- 开始多轮 Epoch 训练 ---
        for _ in range(self.epochs):
            # 1. 数据打乱 (如果启用)
            if shuffled:
                indices = torch.randperm(batch_size)
            else:
                indices = torch.arange(batch_size)
            
            # 2. 在 Mini-batch 上进行更新
            for i in range(0, batch_size, mini_batch_size):
                # 获取 mini-batch 的索引
                mb_indices = indices[i:i + mini_batch_size]
                
                # --- 从整个 batch 中切分出 mini-batch 数据 ---
                mb_obs = obs[mb_indices]
                mb_global_states = global_states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_agent_ids_one_hot = agent_ids_one_hot[mb_indices]
                mb_advantage = advantage[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_td_target = td_target[mb_indices]
                mb_v_pred_old = v_pred_old[mb_indices]

                # --- Actor 更新 ---
                log_probs = torch.log(self.actor(mb_obs).gather(1, mb_actions))
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                
                probs = self.actor(mb_obs)
                action_dist = torch.distributions.Categorical(probs)
                entropy_factor = action_dist.entropy().mean()

                actor_loss = torch.mean(-torch.min(surr1, surr2)) - self.k_entropy * entropy_factor

                # --- Critic 更新 ---
                current_values = self.critic(mb_global_states, mb_agent_ids_one_hot)

                if clip_vf:
                    v_pred_clipped = torch.clamp(current_values, mb_v_pred_old - clip_range, mb_v_pred_old + clip_range)
                    vf_loss1 = (current_values - mb_td_target.detach()).pow(2)
                    vf_loss2 = (v_pred_clipped - mb_td_target.detach()).pow(2)
                    critic_loss = torch.max(vf_loss1, vf_loss2).mean()
                else:
                    critic_loss = torch.mean(F.mse_loss(current_values, mb_td_target.detach()))
                
                # --- 优化器步骤 ---
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                
                # 裁剪前梯度
                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic))  

                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # --- 记录日志 ---
                actor_grad_list.append(model_grad_norm(self.actor))
                actor_loss_list.append(actor_loss.detach().cpu().item())
                critic_grad_list.append(model_grad_norm(self.critic))            
                critic_loss_list.append(critic_loss.detach().cpu().item())
                entropy_list.append(entropy_factor.detach().cpu().item())
                ratio_list.append(ratio.mean().detach().cpu().item())
        
        # --- 保存平均指标 ---
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")


def take_action_from_policy_discrete(policy_net, obs, device, explore=False):
    """
    独立的离散策略推理函数。
    输入:
      policy_net: PolicyNetDiscrete 实例
      obs: 单个智能体的观测 (可被 np.array 接受的状态)
      device: torch.device
      explore: True 则从分布采样，False 则选择 argmax（确定性）
    返回:
      action: int 动作索引
    """
    policy_net.eval()
    state_t = torch.tensor(np.array([obs]), dtype=torch.float).to(device)
    with torch.no_grad():
        probs = policy_net(state_t)  # (1, action_dim)
        if explore:
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
        else:
            action = int(torch.argmax(probs, dim=1).item())
    return action

# ### 主要改动说明：

# 1.  **`update` 方法签名**：
#     *   `def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=True, mini_batches=4):`
#     *   新增了 `shuffled` 和 `mini_batches` 参数，并设置了常用的默认值。

# 2.  **数据计算范围**:
#     *   **优势函数 `advantage`**、**旧的对数概率 `old_log_probs`** 和 **旧的价值预测 `v_pred_old`** 依然在整个 batch 上计算一次。这是 PPO 的标准做法，它们在整个 epoch 期间保持不变。

# 3.  **Epoch 和 Mini-batch 循环**:
#     *   在 `for _ in range(self.epochs):` 循环的**内部**，我们添加了 mini-batch 的逻辑。
#     *   `indices = torch.randperm(batch_size)`：在每个 epoch 开始时，都会生成一组新的随机索引，以确保每个 epoch 的 mini-batch 组成都不同。如果 `shuffled=False`，则使用顺序索引。
#     *   `for i in range(0, batch_size, mini_batch_size):`: 这是一个新的内层循环，它遍历整个 batch，每次切分出一个大小为 `mini_batch_size` 的数据块。
#     *   所有的损失计算、反向传播和优化器更新 (`.step()`) 现在都在这个最内层的 mini-batch 循环中完成。

# 4.  **日志记录**:
#     *   日志列表（如 `actor_loss_list`）现在会记录**每个 mini-batch** 的损失值。因此，在训练结束后计算的 `np.mean()` 将是所有 mini-batch 更新的平均值，这能更精细地反映训练过程。

# ### 如何在您的 `train_spread.py` 中使用？

# 您之前的 `train_spread.py` 文件无需做任何修改即可兼容。`agent.update(transition_dict, adv_normed=True)` 这行代码会自动使用 `update` 方法中 `shuffled=True` 和 `mini_batches=4` 的默认值。

# 如果您想调整这些参数，可以像这样调用：
# ```python
# # 例如，使用 8 个 mini-batch 并且不打乱数据（不推荐）
# agent.update(transition_dict, adv_normed=True, shuffled=False, mini_batches=8)