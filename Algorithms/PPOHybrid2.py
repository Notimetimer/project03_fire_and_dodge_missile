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

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络 (纯 MLP)。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict

        # 共享主干网络
        layers = []
        prev_size = state_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 动态创建输出头
        # 连续动作头
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            self.log_std_param = nn.Parameter(torch.log(torch.ones(cont_dim, dtype=torch.float) * init_std))

        # 分类/离散动作头，self.action_dims['cat']必须是list或者tuple，作为多维离散动作空间处理
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']  # list
            total_cat_dim = sum(self.cat_dims)
            self.fc_cat = nn.Linear(prev_size, total_cat_dim)

        # 伯努利动作头
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            self.fc_bern = nn.Linear(prev_size, bern_dim)

    def forward(self, x, min_std=1e-6, max_std=0.4):
        shared_features = self.net(x)
        outputs = {'cont': None, 'cat': None, 'bern': None}

        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_param)
            std = torch.clamp(std, min=min_std, max=max_std)
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_all = self.fc_cat(shared_features)
            cat_logits_list = torch.split(cat_logits_all, self.cat_dims, dim=-1)
            outputs['cat'] = [F.softmax(logits, dim=-1) for logits in cat_logits_list]

        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)
            outputs['bern'] = bern_logits

        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(nn.Module):
    """
    统一接口适配器。
    将具体的 PolicyNetHybrid 封装起来，对外提供标准的 get_action 和 evaluate_actions 接口。
    未来如果引入 GRU，只需修改这个 Wrapper 或替换为 RecurrentActorWrapper，PPO 算法本身无需修改。
    """
    def __init__(self, policy_net, action_dims_dict, action_bounds=None, device='cpu'):
        super(HybridActorWrapper, self).__init__()
        self.net = policy_net
        self.action_dims = action_dims_dict
        self.device = device
        
        # 处理 Action Bounds
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            # 如果有连续动作，必须提供 action_bounds
            if action_bounds is None:
                raise ValueError("Continuous action space requires action_bounds")
            self.register_buffer('action_bounds', torch.tensor(action_bounds, dtype=torch.float, device=device))
            self.register_buffer('amin', self.action_bounds[:, 0])
            self.register_buffer('amax', self.action_bounds[:, 1])
            self.register_buffer('action_span', self.amax - self.amin)

    def _scale_action_to_exec(self, a_norm):
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    def get_action(self, state, h=None, explore=True, max_std=None):
        """
        推理接口。
        Args:
            state: numpy array or tensor
            h: hidden state (预留接口，目前未使用)
        Returns:
            actions_exec: dict (numpy), 用于环境执行
            actions_raw: dict (numpy/tensor), 用于存入 buffer
            next_h: hidden state (预留接口)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        
        # 调用网络
        actor_outputs = self.net(state, max_std=max_std)  # 如果需要gru，改动这一行
        
        actions_exec = {}
        actions_raw = {}

        # --- Cont ---
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
            actions_raw['cont'] = u[0].cpu().detach().numpy().flatten()

        # --- Cat ---
        if actor_outputs['cat'] is not None:
            cat_probs_list = actor_outputs['cat']
            cat_probs_exec_list = []
            cat_indices_raw_list = []
            for probs in cat_probs_list:
                dist = Categorical(probs=probs)
                idx = dist.sample() if explore else torch.argmax(dist.probs, dim=-1)
                cat_probs_exec_list.append(probs.cpu().detach().numpy()[0].copy())
                cat_indices_raw_list.append(idx.item())
            actions_exec['cat'] = cat_probs_exec_list
            actions_raw['cat'] = np.array(cat_indices_raw_list)

        # --- Bern ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            bern_action = dist.sample() if explore else (dist.probs > 0.5).float()
            actions_exec['bern'] = bern_action[0].cpu().detach().numpy().flatten()
            actions_raw['bern'] = actions_exec['bern']

        return actions_exec, actions_raw, None # None for hidden state

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None):
        """
        训练接口。计算 log_probs 和 entropy。
        Args:
            states: tensor (B, D)
            actions_raw: dict of tensors
        Returns:
            log_probs: tensor (B, 1)
            entropy: tensor (B, 1)
            next_h: None
        """
        actor_outputs = self.net(states, max_std=max_std)
        log_probs = torch.zeros(states.size(0), 1).to(self.device)
        entropy = torch.zeros(states.size(0), 1).to(self.device)

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u = actions_raw['cont']
            log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
            entropy += dist.entropy().unsqueeze(-1) # 近似熵

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_probs_list = actor_outputs['cat']
            cat_action = actions_raw['cat'].long()
            for i, probs in enumerate(cat_probs_list):
                act_i = cat_action[:, i].unsqueeze(-1)
                log_probs += torch.log(probs.gather(1, act_i) + 1e-8)
                dist = Categorical(probs=probs)
                entropy += dist.entropy().unsqueeze(-1)

        # --- Bern ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            bern_action = actions_raw['bern']
            log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
            entropy += dist.entropy().sum(-1, keepdim=True)

        return log_probs, entropy, None
    
    def compute_il_loss(self, states, expert_actions, label_smoothing=0.1):
        """
        计算模仿学习 Loss (MARWIL / BC)。
        
        Args:
            states: (Batch, State_Dim)
            expert_actions: 字典, 包含 {'cont': u, 'cat': index, 'bern': float}
                            注意：对于连续动作，这里通常假设传入的是 pre-tanh 的 u，
                            或者你需要在外部处理好。
            label_smoothing: 标签平滑系数
            
        Returns:
            total_loss_per_sample: (Batch, ) 每个样本的 Loss 总和 (未平均)
        """
        actor_outputs = self.net(states) # 获取 raw output (mu/std, logits)
        
        # 初始化一个全 0 的 loss tensor，形状 (Batch, )
        total_loss_per_sample = torch.zeros(states.size(0), device=self.device)

        # --- 1. 连续动作 (Continuous) ---
        # 依据提供的 PPOContinuous 代码，MARWIL 使用 log_prob(u)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u_expert = expert_actions['cont'] # 假设传入的是 pre-tanh value
            
            # 计算 log_prob，维度求和保持 (Batch, 1) -> squeeze 为 (Batch, )
            # Loss = - log_prob
            cont_loss = -dist.log_prob(0, u_expert).sum(dim=-1)
            total_loss_per_sample += cont_loss

        # --- 2. 离散/多离散动作 (Categorical) ---
        # 依据提供的 Multi-Discrete 代码，使用 CrossEntropy
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_list = actor_outputs['cat'] # 注意：这里 net forward 返回的是 softmax 后的 probs 还是 logits? 
            # 修正：你的 PolicyNetHybrid forward 返回的是 [F.softmax(logits)...]
            # 为了数值稳定性，建议 PolicyNetHybrid 改为返回 logits，或者在这里取 log
            
            # 假设 expert_actions['cat'] 是 (Batch, Num_Heads)
            expert_cat = expert_actions['cat'].long()
            
            for i, probs in enumerate(cat_logits_list):
                # probs: (Batch, N_Class)
                expert_idx = expert_cat[:, i] # (Batch, )
                
                log_probs = torch.log(probs + 1e-10)
                
                if label_smoothing > 0:
                    # Label Smoothing 逻辑
                    n_classes = probs.size(1)
                    one_hot = torch.zeros_like(probs).scatter_(1, expert_idx.unsqueeze(1), 1.0)
                    smooth_target = one_hot * (1.0 - label_smoothing) + (label_smoothing / n_classes)
                    # CrossEntropy: - sum(target * log_p)
                    ce_loss = -torch.sum(smooth_target * log_probs, dim=1)
                else:
                    # 标准 CE: - log_p[target]
                    # gather 需要 index 维度为 (Batch, 1)
                    ce_loss = -log_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
                
                total_loss_per_sample += ce_loss

        # --- 3. 伯努利动作 (Bernoulli) ---
        # 依据提供的 Bernoulli 代码，使用 BCE
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            probs = torch.sigmoid(bern_logits) # 或者是 net 直接输出 logits，这里转 prob
            
            target = expert_actions['bern'] # (Batch, Dim)
            
            # Label Smoothing
            if label_smoothing > 0:
                target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            
            # BCE: - [y*log(p) + (1-y)*log(1-p)]
            # 加上 clamp 防止 log(0)
            probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
            bce_loss = -(target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
            
            # 对动作维度求和 (Batch, Dim) -> (Batch, )
            total_loss_per_sample += bce_loss.sum(dim=-1)

        return total_loss_per_sample

# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy=0.01, critic_max_grad=2, actor_max_grad=2, max_std=0.3):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.k_entropy = k_entropy
        self.critic_max_grad = critic_max_grad
        self.actor_max_grad = actor_max_grad
        self.max_std = max_std
        
        # 记录指标
        self.actor_loss = 0
        self.critic_loss = 0
        self.actor_grad = 0
        self.critic_grad = 0
        self.entropy_mean = 0
        self.ratio_mean = 0
        self.pre_clip_actor_grad = 0
        self.pre_clip_critic_grad = 0

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  

    def take_action(self, state, explore=True, max_std=None):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        # 注意：这里返回了 hidden_state (虽然是 None)，保持接口一致性
        actions_exec, actions_raw, _ = self.actor.get_action(state, h=None, explore=explore, max_std=max_s)
        return actions_exec, actions_raw

    def update(self, transition_dict, adv_normed=False, clip_vf=False, clip_range=0.2, shuffled=1):

        # RL 更新阶段：解冻 std 参数，允许策略调整探索方差
        if hasattr(self.actor.net, 'log_std_param'):
            self.actor.net.log_std_param.requires_grad = True

        
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        actions_from_buffer = transition_dict['actions']
        
        # todo action_mask 防止“死后动作”干扰决策
        # todo truncs
        # todo global states，适配集中式Critic

        # 1. 准备动作数据 (转 Tensor)
        actions_on_device = {}
        all_keys = actions_from_buffer[0].keys()
        for key in all_keys:
            vals = [d[key] for d in actions_from_buffer]
            if key == 'cat':
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
            else:
                actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)

        # 2. 计算 Advantage
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu()).to(self.device)
            
            # 3. 计算旧策略的 log_probs (使用 Wrapper)
            old_log_probs, _, _ = self.actor.evaluate_actions(states, actions_on_device, h=None, max_std=self.max_std)
            v_pred_old = self.critic(states)
            
            # --- [2. 优势归一化] ---
            if adv_normed:
                adv_mean = advantage.mean()
                adv_std = advantage.std(unbiased=False)
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)

            # --- [3. Shuffle 逻辑: 在此处打乱所有相关 Tensor] ---
            if shuffled:
                # 生成随机索引
                idx = torch.randperm(states.size(0), device=self.device)
                
                # 打乱基础数据
                states = states[idx]
                td_target = td_target[idx]
                advantage = advantage[idx]
                old_log_probs = old_log_probs[idx]
                v_pred_old = v_pred_old[idx]
                
                # 打乱字典形式的动作
                for key in actions_on_device:
                    actions_on_device[key] = actions_on_device[key][idx]
                
                # 如果有其他 tensor 需要 shuffle (如 truncs) 也要在这里处理
                # todo truncs的处理逻辑 当前env里面没有做，所以算法先不做
                

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        for _ in range(self.epochs):
            # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
            log_probs, entropy, _ = self.actor.evaluate_actions(states, actions_on_device, h=None, max_std=self.max_std)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean() - self.k_entropy * entropy.mean()

            # Critic Loss
            v_pred = self.critic(states)
            if clip_vf:
                v_pred_clipped = torch.clamp(v_pred, v_pred_old - clip_range, v_pred_old + clip_range)
                vf_loss1 = (v_pred - td_target).pow(2)
                vf_loss2 = (v_pred_clipped - td_target).pow(2)
                critic_loss = torch.max(vf_loss1, vf_loss2).mean()
            else:
                critic_loss = F.mse_loss(v_pred, td_target)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()

            pre_clip_actor_grad.append(model_grad_norm(self.actor))
            pre_clip_critic_grad.append(model_grad_norm(self.critic)) 
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Logging
            actor_grad_list.append(model_grad_norm(self.actor))
            critic_grad_list.append(model_grad_norm(self.critic))            
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            entropy_list.append(entropy.mean().item())
            ratio_list.append(ratio.mean().item())

        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        self.advantage = advantage.abs().mean().item()
        
        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    def preprocess_parallel_buffer(self, transition_dict):
        """
        统一将 (T, N, D) 或 (T, D) 的数据转为扁平的 (Batch, D)，并计算 GAE。
        """
        # 1. 提取并转 Tensor (逻辑与你上传的 PPOContinuous 一致)
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).to(self.device)
        
        # 2. 维度统一化 (升维到 [T, N, 1])
        if rewards.dim() == 1: # 单环境
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)
            rewards = rewards.unsqueeze(1).unsqueeze(2)
            dones = dones.unsqueeze(1).unsqueeze(2)
        elif rewards.dim() == 2: # 多环境 [T, N]
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            # states 已经是 [T, N, D]

        # 3. 处理 Truncs (PPOContinuous 逻辑)
        use_truncs = 'truncs' in transition_dict and len(transition_dict['truncs']) > 0
        if use_truncs:
            truncs = torch.tensor(np.array(transition_dict['truncs']), dtype=torch.float).to(self.device)
            if truncs.dim() == 1: truncs = truncs.unsqueeze(1).unsqueeze(2)
            elif truncs.dim() == 2: truncs = truncs.unsqueeze(-1)
        else:
            truncs = None

        # 4. GAE 计算
        with torch.no_grad():
            T, N = states.shape[0], states.shape[1]
            flat_s = states.reshape(T * N, -1)
            flat_ns = next_states.reshape(T * N, -1)
            curr_vals = self.critic(flat_s).view(T, N, 1)
            next_vals = self.critic(flat_ns).view(T, N, 1)
            
            # 使用 Utils 中的 compute_advantage (支持 truncs)
            td_delta = rewards + self.gamma * next_vals * (1.0 - dones) - curr_vals
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta, dones, truncs)
            returns = advantage + curr_vals

        # 5. 扁平化并回填 (Flatten)
        def flat(t): return t.reshape(-1, *t.shape[2:]).cpu().numpy().tolist()
        
        transition_dict['states'] = flat(states)
        transition_dict['next_states'] = flat(next_states)
        transition_dict['rewards'] = flat(rewards)
        transition_dict['dones'] = flat(dones)
        transition_dict['advantages'] = flat(advantage)
        transition_dict['td_targets'] = flat(returns)
        if use_truncs: transition_dict['truncs'] = flat(truncs)
        
        # Actions 比较特殊，是字典，需要单独处理扁平化
        # 假设 actions 是 list of dicts -> dict of lists (flat)
        # 这步通常在 Buffer 采样时做比较好，或者在这里手动 loop key
        pass # 具体实现视你的 Buffer 结构而定，核心是上面的 Advantage 计算

        return transition_dict

    # --- 新增功能 2: MARWIL Update ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.1, max_weight=100.0):
        """
        MARWIL 离线更新函数 (混合动作空间支持版 + Mini-batch)
        """
        # --- [1. MARWIL模式：冻结连续动作 std 参数] ---
        if hasattr(self.actor.net, 'log_std_param'):
            self.actor.net.log_std_param.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 处理动作字典 (转 Tensor)
        actions_all = {}
        for k, v in il_transition_dict['actions'].items():
            if k == 'cat':
                actions_all[k] = torch.tensor(np.array(v), dtype=torch.long).to(self.device)
            else:
                actions_all[k] = torch.tensor(np.array(v), dtype=torch.float).to(self.device)

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_c = 0
        batch_count = 0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            # 切片取 Batch 数据
            s_batch = states_all[batch_indices]
            r_batch = returns_all[batch_indices]
            
            # 动作字典切片
            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # ----------------------------------------------------
            # A. 计算优势 (Advantage) 和 权重 (Weights)
            # ----------------------------------------------------
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                # 动态更新 c^2 (Moving Average)
                if not hasattr(self, 'c_sq'): 
                    self.c_sq = torch.tensor(1.0, device=self.device)
                
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                # 归一化优势
                advantage = residual / (c + 1e-8)
                
                # 计算权重并截断
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

            # ----------------------------------------------------
            # B. 计算 Actor Loss (委托给 Wrapper)
            # ----------------------------------------------------
            # compute_il_loss 返回的是 (Batch, ) 的 loss，没有 reduce
            raw_il_loss = self.actor.compute_il_loss(s_batch, actions_batch, label_smoothing)
            
            # 加权平均: mean( alpha * w * loss )
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # ----------------------------------------------------
            # C. 计算 Critic Loss
            # ----------------------------------------------------
            v_pred = self.critic(s_batch)
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v

            # ----------------------------------------------------
            # D. 反向传播
            # ----------------------------------------------------
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_c += c.item()
            batch_count += 1

        # 返回平均 Loss
        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_c