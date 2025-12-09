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
class CascadePolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dims_dict):
        super().__init__()
        self.action_dims = action_dims_dict
        
        # --- Level 0: 基础特征提取 & Level 0 动作头 ---
        self.l0_net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        # 假设 Level 0 是一个 5选1 的离散动作
        self.l0_dim = action_dims_dict['level0_cat'][0]  # 这是one-hot吗？
        self.fc_l0_logits = nn.Linear(64, self.l0_dim)

        # --- Level 1: 接收 (状态特征 + Level 0 动作的 One-Hot) ---
        # 输入维度 = 状态特征(64) + Level 0 动作维度(5)
        self.l1_input_dim = 64 + self.l0_dim
        self.l1_net = nn.Sequential(nn.Linear(self.l1_input_dim, 64), nn.ReLU())
        
        # Level 1 的动作头 (Cat + Bern)
        self.fc_l1_cat = nn.Linear(64, sum(action_dims_dict['level1_cat']))
        self.fc_l1_bern = nn.Linear(64, action_dims_dict['level1_bern'])

    def forward(self, state, l0_action=None):
        """
        Args:
            state: (Batch, Dim)
            l0_action: (Batch, ) 或 None。
                       如果是 None (推理模式)，返回 Level 0 的 logits，Level 1 返回 None。
                       如果有值 (训练模式)，返回 Level 0 和 Level 1 的所有 logits。
        """
        # 1. Level 0 计算
        feat0 = self.l0_net(state)
        l0_logits = self.fc_l0_logits(feat0)
        
        outputs = {'l0_logits': l0_logits, 'l1_logits': None, 'l1_bern': None}

        # 2. Level 1 计算
        if l0_action is not None:
            # --- 训练模式 (Teacher Forcing) ---
            # l0_action 是整数索引，需要转 one-hot
            # l0_action: (Batch, ) -> (Batch, l0_dim)
            l0_one_hot = F.one_hot(l0_action.long(), num_classes=self.l0_dim).float()
            
            # 拼接特征：原始特征 feat0 + 动作特征
            l1_input = torch.cat([feat0, l0_one_hot], dim=-1)
            
            feat1 = self.l1_net(l1_input)
            
            outputs['l1_logits'] = self.fc_l1_cat(feat1)
            outputs['l1_bern'] = self.fc_l1_bern(feat1)
            
        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class CascadeActorWrapper(nn.Module):
    def __init__(self, policy_net, action_dims_dict, device='cpu'):
        super().__init__()
        self.net = policy_net
        self.device = device
        

    def _scale_action_to_exec(self, a_norm):
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    def get_action(self, state, explore=True, **kwargs):
        """
        推理模式：自回归采样 (Autoregressive Sampling)
        Step 1: 算 L0 -> 采样 L0
        Step 2: 把采样到的 L0 喂回去 -> 算 L1 -> 采样 L1
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)

        # --- Step 1: 决策 Level 0 ---
        # 此时只给 state，不给 action，网络只跑一半
        out0 = self.net(state, l0_action=None) 
        l0_logits = out0['l0_logits']
        
        # 采样 Level 0
        dist0 = Categorical(logits=l0_logits)
        l0_idx = dist0.sample() if explore else torch.argmax(l0_logits, dim=-1)
        
        # --- Step 2: 决策 Level 1 (基于 Level 0 的结果) ---
        # 将刚才决策出的 l0_idx 喂给网络
        out1 = self.net(state, l0_action=l0_idx)
        
        # 采样 Level 1 (Cat)
        l1_probs = F.softmax(out1['l1_logits'], dim=-1) # 简化写了，假设只有一个cat头
        dist1_cat = Categorical(probs=l1_probs)
        l1_idx = dist1_cat.sample() if explore else torch.argmax(l1_probs, dim=-1)
        
        # 采样 Level 1 (Bern)
        l1_bern_logits = out1['l1_bern']
        dist1_bern = Bernoulli(logits=l1_bern_logits)
        l1_bern_val = dist1_bern.sample() if explore else (l1_bern_logits > 0).float()

        # 组装返回给 Buffer 和 环境
        actions_exec = {
            'l0_cat': l0_idx.cpu().numpy(),
            'l1_cat': l1_idx.cpu().numpy(), 
            'l1_bern': l1_bern_val.cpu().numpy()
        }
        # actions_raw 也是同样的结构
        return actions_exec, actions_exec, None

    def evaluate_actions(self, states, actions_raw, **kwargs):
        """
        训练模式：Teacher Forcing
        从 Buffer 里拿到真实的 l0_action，直接喂给网络，一次性算出所有概率。
        """
        l0_truth = actions_raw['l0_cat'] # (Batch, )
        
        # 一次 Forward 搞定，不需要 Loop
        outputs = self.net(states, l0_action=l0_truth)
        
        log_probs = 0
        entropy = 0
        
        # --- 计算 Level 0 的 LogProb ---
        dist0 = Categorical(logits=outputs['l0_logits'])
        log_probs += dist0.log_prob(l0_truth).unsqueeze(-1)
        entropy += dist0.entropy().unsqueeze(-1)
        
        # --- 计算 Level 1 的 LogProb ---
        # 注意：这里不需要再采样了，直接用 Buffer 里的 l1 动作去算概率
        
        # L1 Cat
        l1_cat_truth = actions_raw['l1_cat']
        l1_probs = F.softmax(outputs['l1_logits'], dim=-1)
        dist1_cat = Categorical(probs=l1_probs)
        log_probs += dist1_cat.log_prob(l1_cat_truth).unsqueeze(-1)
        entropy += dist1_cat.entropy().unsqueeze(-1)
        
        # L1 Bern
        l1_bern_truth = actions_raw['l1_bern']
        dist1_bern = Bernoulli(logits=outputs['l1_bern'])
        log_probs += dist1_bern.log_prob(l1_bern_truth).unsqueeze(-1) # Bern可能是多维的，注意求和
        entropy += dist1_bern.entropy().unsqueeze(-1)

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