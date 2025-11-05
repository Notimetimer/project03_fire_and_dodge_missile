import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli

# --- 复用已有的辅助函数和类 ---

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

class SquashedNormal:
    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, device=mu.device, dtype=mu.dtype)
        self.std = torch.clamp(std, min=float(eps))
        self.normal = Normal(mu, self.std)
        self.eps = eps
        self.mean = mu

    def sample(self):
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        log_prob_u = self.normal.log_prob(u)
        jacobian = 0 # 保存u则不需要修正项
        return log_prob_u - jacobian

    def entropy(self):
        return self.normal.entropy().sum(-1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dim:
            layers.append(torch.nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = torch.nn.Linear(prev_size, 1)

    def forward(self, x):
        y = self.net(x)
        return self.fc_out(y)

# --- 新的混合动作空间 Actor ---

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络。
    - 统一的MLP主干。
    - 根据 action_dims_dict 动态创建输出头。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict

        # 1. 共享主干网络
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 2. 动态创建输出头
        # 连续动作头
        if self.action_dims.get('Cont', 0) > 0:
            cont_dim = self.action_dims['Cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_param = nn.Parameter(torch.log(torch.ones(cont_dim, dtype=torch.float) * init_std))

        # 分类/离散动作头
        if self.action_dims.get('Cat', 0) > 0:
            cat_dim = self.action_dims['Cat']
            self.fc_cat = nn.Linear(prev_size, cat_dim)

        # 伯努利动作头
        if self.action_dims.get('Bern', 0) > 0:
            bern_dim = self.action_dims['Bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)

    def forward(self, x, min_std=1e-6, max_std=0.4):
        # 通过共享网络
        shared_features = self.net(x)
        
        outputs = {'Cont': None, 'Cat': None, 'Bern': None}

        # 计算每个头的输出
        if self.action_dims.get('Cont', 0) > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_param)
            std = torch.clamp(std, min=min_std, max=max_std)
            # 广播到 batch
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['Cont'] = (mu, std)

        if self.action_dims.get('Cat', 0) > 0:
            cat_logits = self.fc_cat(shared_features)
            outputs['Cat'] = F.softmax(cat_logits, dim=-1) # 输出现在是概率

        if self.action_dims.get('Bern', 0) > 0:
            bern_logits = self.fc_bern(shared_features)
            outputs['Bern'] = bern_logits

        return outputs

# --- 新的 PPOHybrid 算法 ---

class PPOHybrid:
    def __init__(self, state_dim, hidden_dim, action_dims_dict, action_bounds, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3):
        
        self.action_dims = action_dims_dict
        self.actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad=critic_max_grad
        self.actor_max_grad=actor_max_grad
        self.max_std = max_std

        # 将 action_bounds 作为常量存储在 device 上
        if self.action_dims.get('Cont', 0) > 0:
            assert action_bounds is not None, "action_bounds must be provided for continuous actions"
            self.action_bounds = torch.tensor(action_bounds, dtype=torch.float, device=self.device)
            if self.action_bounds.dim() != 2 or self.action_bounds.shape[0] != self.action_dims['Cont']:
                 raise ValueError(f"action_bounds shape must be ({self.action_dims['Cont']}, 2)")
            self.amin = self.action_bounds[:, 0]
            self.amax = self.action_bounds[:, 1]
            self.action_span = self.amax - self.amin
            
    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态设置 actor 和 critic 的学习率"""
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  

    def _scale_action_to_exec(self, a_norm):
        """将标准化的动作 a_norm (在 [-1,1] 范围内) 缩放到环境执行区间。"""
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    def take_action(self, state, explore=True, max_std=None):
        '''
        输出为两项，前一项为可被裁剪的动作部分，输入环境（cat分布需要重新采样一次）
        后一项为用于被update输入的部分，作为神经网络原始输出
        '''
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        max_action_std = max_std if max_std is not None else self.max_std

        # 从actor获取所有头的输出
        actor_outputs = self.actor(state, max_std=max_action_std)
        
        actions_exec = {} # 用于环境执行 (numpy)
        actions_raw = {}  # 用于存储和训练 (torch tensor on device)

        # --- 处理连续动作 ---
        if actor_outputs['Cont'] is not None:
            mu, std = actor_outputs['Cont']
            dist = SquashedNormal(mu, std)
            if explore:
                a_norm, u = dist.sample()
            else:
                u = mu
                a_norm = torch.tanh(u)
            
            a_exec = self._scale_action_to_exec(a_norm)
            actions_exec['Cont'] = a_exec[0].cpu().detach().numpy().flatten()
            actions_raw['Cont'] = u[0].cpu().detach().numpy().flatten() # 存储 pre-tanh 的 u

        # --- 处理分类动作 ---
        if actor_outputs['Cat'] is not None:
            cat_probs = actor_outputs['Cat']
            dist = Categorical(probs=cat_probs)
            if explore:
                cat_action_idx = dist.sample()
            else:
                cat_action_idx = torch.argmax(dist.probs, dim=-1)
            # probs_np = cat_probs.cpu().detach().numpy()[0].copy() # [0]是batch维度
            actions_exec['Cat'] = cat_probs.cpu().detach().numpy()[0].copy() # 返回概率向量
            actions_raw['Cat'] = cat_action_idx.cpu().detach().numpy().flatten() # 存储采样索引

        # --- 处理伯努利动作 ---
        if actor_outputs['Bern'] is not None:
            bern_logits = actor_outputs['Bern']
            dist = Bernoulli(logits=bern_logits)
            if explore:
                bern_action = dist.sample()
            else:
                bern_action = (dist.probs > 0.5).float() # 确定性动作

            actions_exec['Bern'] = bern_action[0].cpu().detach().numpy().flatten()
            actions_raw['Bern'] = actions_exec['Bern'] # 存储采样结果 (0. or 1.)
        
        return actions_exec, actions_raw

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)

        # 从字典中提取动作
        actions_from_buffer = transition_dict['actions']

        # actions_from_buffer 是 list[dict]，需要按 key 聚合
        actions_on_device = {}
        all_keys = actions_from_buffer[0].keys()
        for key in all_keys:
            vals = [d[key] for d in actions_from_buffer]
            if key == 'Cat':
                # 分类动作的索引需要是 LongTensor
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
            else:
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)
        
        # 计算 TD-target 和 advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        
        if adv_normed:
            adv_mean = advantage.detach().mean()
            adv_std = advantage.detach().std(unbiased=False)
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
        
        # 提前计算一次旧的 value 预测（用于 value clipping）
        v_pred_old = self.critic(states).detach()  # (N,1)

        # 计算旧策略下的 log_probs
        with torch.no_grad():
            old_actor_outputs = self.actor(states, max_std=self.max_std)
            old_log_probs = torch.zeros(states.size(0), 1).to(self.device)

            # Cont
            if self.action_dims['Cont'] > 0:
                mu_old, std_old = old_actor_outputs['Cont']
                dist_old = SquashedNormal(mu_old, std_old)
                u_old = actions_on_device['Cont']
                old_log_probs += dist_old.log_prob(0, u_old).sum(-1, keepdim=True)
            # Cat
            if self.action_dims.get('Cat', 0) > 0:
                cat_probs_old = old_actor_outputs['Cat'] # 现在直接是概率
                cat_action_old = actions_on_device['Cat']
                # 与 PPOdiscrete 保持一致，使用 gather 方法
                old_log_probs += torch.log(cat_probs_old.gather(1, cat_action_old))  #.detach()
            # Bern
            if self.action_dims['Bern'] > 0:
                bern_logits_old = old_actor_outputs['Bern']
                dist_old = Bernoulli(logits=bern_logits_old)
                bern_action_old = actions_on_device['Bern']
                old_log_probs += dist_old.log_prob(bern_action_old).sum(-1, keepdim=True)
        
        # 日志记录列表
        actor_loss_list, critic_loss_list, entropy_list = [], [], []
        actor_grad_list = []
        critic_grad_list = []
        pre_clip_actor_grad = []
        pre_clip_critic_grad = []
        ratio_list = []


        for _ in range(self.epochs):
            # --- Actor Loss ---
            actor_outputs = self.actor(states, max_std=self.max_std)
            log_probs = torch.zeros(states.size(0), 1).to(self.device)
            entropy = torch.zeros(states.size(0), 1).to(self.device)

            # Cont
            if self.action_dims['Cont'] > 0:
                mu, std = actor_outputs['Cont']
                dist = SquashedNormal(mu, std)
                u = actions_on_device['Cont']
                log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
                entropy += dist.entropy().unsqueeze(-1)
            # Cat
            if self.action_dims.get('Cat', 0) > 0:
                cat_probs = actor_outputs['Cat']
                cat_action = actions_on_device['Cat'].long() # 确保是 long 类型
                log_probs += torch.log(cat_probs.gather(1, cat_action))
                dist = Categorical(probs=cat_probs)
                entropy += dist.entropy().unsqueeze(-1)
            # Bern
            if self.action_dims['Bern'] > 0:
                bern_logits = actor_outputs['Bern']
                dist = Bernoulli(logits=bern_logits)
                bern_action = actions_on_device['Bern']
                log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
                entropy += dist.entropy().sum(-1, keepdim=True)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy.mean()

            # --- Critic Loss ---
            # 计算 critic_loss：支持可选的 value clipping（PPO 风格）
            if clip_vf:
                v_pred = self.critic(states)                                  # 当前预测 (N,1)
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target.detach()).pow(2)               # (N,1)
                vf_loss2 = (v_pred_clipped - td_target.detach()).pow(2)       # (N,1)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(self.critic(states), td_target.detach())
            
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

            # # 保存用于日志/展示的数值（断开计算图并搬到 CPU）
            actor_grad_list.append(model_grad_norm(self.actor))
            actor_loss_list.append(actor_loss.detach().cpu().item())
            critic_grad_list.append(model_grad_norm(self.critic))            
            critic_loss_list.append(critic_loss.detach().cpu().item())
            entropy_list.append(dist.entropy().mean().detach().cpu().item())
            ratio_list.append(ratio.mean().detach().cpu().item())

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().detach().cpu().item()
        # 权重/偏置 NaN 检查（在每次前向后、反向前检查参数）
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")
