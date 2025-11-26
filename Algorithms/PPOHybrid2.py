'''
修改进行中：
原先的"cat"部分动作头是扁平结构，以n选1为动作特征，现在允许分维度多选1，维度之间平行决策，
cat动作头的形状构建现在以list而非int值为输入，就算是一个n选1的问题，也必须写成[n]的形式
'''

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

def compute_advantage(gamma, lmbda, td_delta, dones):
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # [新增] 转为 numpy
    advantage_list = []
    advantage = 0.0
    
    # [修改] 同时遍历 delta 和 done
    for delta, done in zip(td_delta[::-1], dones[::-1]):
        # 如果当前是 done，说明这是序列的最后一步（或者该步之后没有未来），
        # 此时不应该加上一步（时间上的未来）的 advantage。
        # 注意：这里的 advantage 变量存的是“下一步的优势”，所以要乘 (1-done)
        advantage = delta + gamma * lmbda * advantage * (1 - done)
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
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            if isinstance(self.action_dims['cont'], list):
                raise ValueError("action_dims['cont'] must be an integer, not a list.")
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_param = nn.Parameter(torch.log(torch.ones(cont_dim, dtype=torch.float) * init_std))

        # 分类/离散动作头 [修改]
        if 'cat' in self.action_dims:
            if not isinstance(self.action_dims['cat'], list):
                raise ValueError("action_dims['cat'] must be a list of integers (e.g. [3, 2]).")
            self.cat_dims = self.action_dims['cat']
            total_cat_dim = sum(self.cat_dims)
            self.fc_cat = nn.Linear(prev_size, total_cat_dim)

        # 伯努利动作头
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            if isinstance(self.action_dims['bern'], list):
                raise ValueError("action_dims['bern'] must be an integer, not a list.")
            bern_dim = self.action_dims['bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)

    def forward(self, x, min_std=1e-6, max_std=0.4):
        # 通过共享网络
        shared_features = self.net(x)
        
        outputs = {'cont': None, 'cat': None, 'bern': None}

        # 计算每个头的输出
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_param)
            std = torch.clamp(std, min=min_std, max=max_std)
            # 广播到 batch
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        # [修改] 处理多头分类
        if 'cat' in self.action_dims:
            cat_logits_all = self.fc_cat(shared_features)
            # 按维度切分
            cat_logits_list = torch.split(cat_logits_all, self.cat_dims, dim=-1)
            # 对每个头做 softmax
            outputs['cat'] = [F.softmax(logits, dim=-1) for logits in cat_logits_list]

        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)
            outputs['bern'] = bern_logits

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
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            assert action_bounds is not None, "action_bounds must be provided for continuous actions"
            self.action_bounds = torch.tensor(action_bounds, dtype=torch.float, device=self.device)
            if self.action_bounds.dim() != 2 or self.action_bounds.shape[0] != self.action_dims['cont']:
                 raise ValueError(f"action_bounds shape must be ({self.action_dims['cont']}, 2)")
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
        输出为两项，先a后u
        连续动作 a 为限幅和缩放后结果，u 为原始采样
        分类动作 a 为动作分布(list of arrays)，u 为采样索引(array)
        伯努利动作 a 和 u 都是采样结果
        '''
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        max_action_std = max_std if max_std is not None else self.max_std

        # 从actor获取所有头的输出
        actor_outputs = self.actor(state, max_std=max_action_std)
        
        actions_exec = {} # 用于环境执行 (numpy)
        actions_raw = {}  # 用于存储和训练 (torch tensor on device)

        # --- 处理连续动作 --- a考虑action_bound，u不考虑
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            if explore:
                a_norm, u = dist.sample()
            else:
                u = mu
                a_norm = torch.tanh(u)
            
            a_exec = self._scale_action_to_exec(a_norm)
            actions_exec['cont'] = a_exec[0].cpu().detach().numpy().flatten()
            actions_raw['cont'] = u[0].cpu().detach().numpy().flatten() # 存储 pre-tanh 的 u

        # --- 处理分类动作 --- [修改]
        if actor_outputs['cat'] is not None:
            cat_probs_list = actor_outputs['cat'] # list of tensors
            
            cat_probs_exec_list = []
            cat_indices_raw_list = []

            for probs in cat_probs_list:
                dist = Categorical(probs=probs)
                if explore:
                    idx = dist.sample()
                else:
                    idx = torch.argmax(dist.probs, dim=-1)
                
                # probs shape (1, dim), idx shape (1,)
                cat_probs_exec_list.append(probs.cpu().detach().numpy()[0].copy())
                cat_indices_raw_list.append(idx.item())

            actions_exec['cat'] = cat_probs_exec_list # list of probability arrays
            actions_raw['cat'] = np.array(cat_indices_raw_list) # array of indices [idx1, idx2, ...]

        # --- 处理伯努利动作 --- 不区分a和u
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            if explore:
                bern_action = dist.sample()
            else:
                bern_action = (dist.probs > 0.5).float() # 确定性动作

            actions_exec['bern'] = bern_action[0].cpu().detach().numpy().flatten()
            actions_raw['bern'] = actions_exec['bern'] # 存储采样结果 (0. or 1.)
        
        return actions_exec, actions_raw

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=0):
        N = len(transition_dict['states'])
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        # 从字典中提取动作
        actions_from_buffer = transition_dict['actions']

        # 计算 TD-target 和 advantage
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
        
        # 首先按原始顺序把数据载入（不要一开始就打乱）

        indices = np.arange(N)
        if shuffled:
            np.random.shuffle(indices)
            states = states[indices]
            actions_from_buffer = actions_from_buffer[indices]
            rewards = rewards[indices]
            dones = dones[indices]
            next_states = next_states[indices]
            advantage = advantage[indices]
            # max_stds = max_stds[indices] if isinstance(max_stds, torch.Tensor) and max_stds.shape[0] == N else max_stds
            # # 重新整理 h0s
            # h0s_stack = h0s_stack[indices]
            # h0s = h0s_stack.squeeze(2).permute(1, 0, 2).contiguous()
        else:
            # # 未打乱时直接使用之前为 critic 准备好的 h0s_for_critic
            # h0s = h0s_for_critic
            pass


        # actions_from_buffer 是 list[dict]，需要按 key 聚合
        actions_on_device = {}
        all_keys = actions_from_buffer[0].keys()
        for key in all_keys:
            vals = [d[key] for d in actions_from_buffer]
            if key == 'cat':
                # 分类动作的索引需要是 LongTensor
                # vals is list of arrays, np.array(vals) -> (N, num_heads)
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
            else:
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)
        
        # 优势归一化
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
            if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
                mu_old, std_old = old_actor_outputs['cont']
                dist_old = SquashedNormal(mu_old, std_old)
                u_old = actions_on_device['cont']
                old_log_probs += dist_old.log_prob(0, u_old).sum(-1, keepdim=True)
            # Cat [修改]
            if 'cat' in self.action_dims:
                cat_probs_list_old = old_actor_outputs['cat'] # list of tensors
                cat_action_old = actions_on_device['cat'] # (N, num_heads)
                
                for i, probs_old in enumerate(cat_probs_list_old):
                    # 取出第 i 个头的动作索引
                    act_i = cat_action_old[:, i].unsqueeze(-1) # (N, 1)
                    # 累加 log_prob
                    old_log_probs += torch.log(probs_old.gather(1, act_i))

            # Bern
            if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
                bern_logits_old = old_actor_outputs['bern']
                dist_old = Bernoulli(logits=bern_logits_old)
                bern_action_old = actions_on_device['bern']
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
            if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
                mu, std = actor_outputs['cont']
                dist = SquashedNormal(mu, std)
                u = actions_on_device['cont']
                log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
                entropy += dist.entropy().unsqueeze(-1)
            # Cat [修改]
            if 'cat' in self.action_dims:
                cat_probs_list = actor_outputs['cat']
                cat_action = actions_on_device['cat'].long()
                
                for i, probs in enumerate(cat_probs_list):
                    act_i = cat_action[:, i].unsqueeze(-1)
                    log_probs += torch.log(probs.gather(1, act_i))
                    
                    dist = Categorical(probs=probs)
                    entropy += dist.entropy().unsqueeze(-1)

            # Bern
            if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
                bern_logits = actor_outputs['bern']
                dist = Bernoulli(logits=bern_logits)
                bern_action = actions_on_device['bern']
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
