
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal, CircularDiscretizedDistribution
from Algorithms.MLP_heads import ValueNet

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
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

        # 计算共享 std 的总维度 (Continuous + Circular)
        self.cont_dim = self.action_dims.get('cont', 0)
        self.circ_dim = self.action_dims.get('circ', 0)
        total_shared_std_dim = self.cont_dim + self.circ_dim

        if total_shared_std_dim > 0:
            # 这里的 log_std 是状态无关的，由 cont 和 circ 共享
            self.log_std_shared = nn.Parameter(torch.log(torch.ones(total_shared_std_dim) * init_std))

        # 1. 连续动作头
        if self.cont_dim > 0:
            self.fc_mu = nn.Linear(prev_size, self.cont_dim)

        # 2. 圆周动作头 (Circular) - 每个维度 2 个输出 (x, y)
        if self.circ_dim > 0:
            self.fc_circ = nn.Linear(prev_size, self.circ_dim * 2)

        # 3. 离散动作头 (Categorical)
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']
            self.fc_cat = nn.Linear(prev_size, sum(self.cat_dims))

        # 4. 伯努利动作头 (Bernoulli)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            self.fc_bern = nn.Linear(prev_size, self.action_dims['bern'])
            nn.init.constant_(self.fc_bern.bias, -2.0)
    
    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temp=1.0):
        if isinstance(temp, dict):
            temp_cat = temp.get('cat', 1.0)
            temp_bern = temp.get('bern', 1.0)
        else:
            temp_cat = temp_bern = temp

        shared_features = self.net(x)
        outputs = {'cont': None, 'circ': None, 'cat': None, 'bern': None}

        # --- 处理共享 Std ---
        std_all = None
        if (self.cont_dim + self.circ_dim) > 0:
            std_all = torch.exp(self.log_std_shared).clamp(min=min_std, max=max_std)
            if shared_features.dim() > 1:
                std_all = std_all.unsqueeze(0).expand(shared_features.size(0), -1)

        # --- Continuous ---
        if self.cont_dim > 0:
            mu = self.fc_mu(shared_features)
            std_cont = std_all[:, :self.cont_dim]
            outputs['cont'] = (mu, std_cont)

        # --- Circular ---
        if self.circ_dim > 0:
            vec_all = self.fc_circ(shared_features)
            # 重塑为 (Batch, circ_dim, 2)
            vec_all = vec_all.view(-1, self.circ_dim, 2)
            std_circ = std_all[:, self.cont_dim:]
            outputs['circ'] = (vec_all, std_circ)

        # --- Categorical ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_all = self.fc_cat(shared_features)
            cat_logits_list = torch.split(cat_logits_all, self.cat_dims, dim=-1)
            final_probs_list = [F.softmax(logits / (temp_cat + 1e-8), dim=-1) for logits in cat_logits_list]
            outputs['cat'] = final_probs_list

        # --- Bernoulli ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)
            if action_masks is not None and 'bern' in action_masks:
                bern_logits = bern_logits.masked_fill(action_masks['bern'] == 0, -1e9)
            outputs['bern'] = bern_logits / (temp_bern + 1e-8)
            
        return outputs
# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================
class HybridActorWrapper(nn.Module):
    def __init__(self, policy_net, action_dims_dict, action_bounds=None, device='cpu'):
        super(HybridActorWrapper, self).__init__()
        self.net = policy_net
        self.action_dims = action_dims_dict
        self.device = device
        
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            if action_bounds is None:
                raise ValueError("Continuous action space requires action_bounds")
            self.register_buffer('action_bounds', torch.tensor(action_bounds, dtype=torch.float, device=device))
            self.register_buffer('amin', self.action_bounds[:, 0])
            self.register_buffer('amax', self.action_bounds[:, 1])
            self.register_buffer('action_span', self.amax - self.amin)

    def _scale_action_to_exec(self, a_norm):
        return self.amin + (a_norm + 1.0) * 0.5 * self.action_span

    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5, temp=1.0):
        # 简化的 Batch 检测
        is_batch = True if (isinstance(state, torch.Tensor) and state.dim() > 1) or (isinstance(state, np.ndarray) and state.ndim > 1) else False
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state if is_batch else [state], dtype=torch.float).to(self.device)

        explore_opts = explore if isinstance(explore, dict) else {k: explore for k in ['cont', 'cat', 'bern', 'circ']}
        
        # --- 原始 Action Mask 逻辑保持不变 ---
        action_masks = None
        if (check_obs is not None) and isinstance(check_obs, dict):
            # ... (此处省略原有的 check_obs -> mask 生成代码，保持一致) ...
            # 假设生成的 mask 为 mask_tensor
            mask_tensor = torch.ones((state.size(0), 1), device=self.device) # 占位
            action_masks = {'bern': mask_tensor}

        actor_outputs = self.net(state, max_std=max_std, action_masks=action_masks, temp=temp)
        
        actions_exec, actions_raw, actions_dist_check = {}, {}, {}

        # --- Cont ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u = dist.sample()[1] if explore_opts.get('cont', True) else mu
            a_norm = torch.tanh(u)
            a_exec = self._scale_action_to_exec(a_norm)
            actions_exec['cont'] = a_exec.cpu().detach().numpy() if is_batch else a_exec[0].cpu().detach().numpy().flatten()
            actions_raw['cont'] = u.cpu().detach().numpy() if is_batch else u[0].cpu().detach().numpy().flatten()

        # --- Circ [新增] ---
        if actor_outputs['circ'] is not None:
            vec_all, std_all = actor_outputs['circ']
            circ_indices = []
            circ_probs = []
            for i in range(self.action_dims['circ']):
                vec_h = vec_all[:, i, :] # (Batch, 2)
                std_h = std_all[:, i:i+1] # (Batch, 1)
                dist = CircularDiscretizedDistribution(vec_h, std_h)
                
                idx = dist.sample()[0] if explore_opts.get('circ', True) else dist.mean_idx
                circ_indices.append(idx)
                circ_probs.append(dist.probs.cpu().detach().numpy())
            
            idx_stack = torch.stack(circ_indices, dim=-1).cpu().detach().numpy()
            actions_exec['circ'] = idx_stack if is_batch else idx_stack[0]
            actions_raw['circ'] = actions_exec['circ']
            actions_dist_check['circ'] = circ_probs

        # --- Cat & Bern 保持不变 ---
        # ... (此处省略原有的 cat 和 bern 处理逻辑) ...

        return actions_exec, actions_raw, None, actions_dist_check

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None):
        actor_outputs = self.net(states, max_std=max_std)
        log_probs = torch.zeros(states.size(0), 1).to(self.device)
        entropy = torch.zeros(states.size(0), 1).to(self.device)
        entropy_details = {'cont': None, 'cat': None, 'bern': None, 'circ': None}

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            log_probs += dist.log_prob(0, actions_raw['cont']).sum(-1, keepdim=True)
            e_cont = dist.entropy().unsqueeze(-1)
            entropy += e_cont
            entropy_details['cont'] = e_cont

        # --- Circ [新增] ---
        if 'circ' in self.action_dims and self.action_dims['circ'] > 0:
            vec_all, std_all = actor_outputs['circ']
            circ_actions = actions_raw['circ']
            if not isinstance(circ_actions, torch.Tensor):
                circ_actions = torch.tensor(circ_actions, dtype=torch.long, device=self.device)
            
            e_circ_sum = torch.zeros_like(entropy)
            for i in range(self.action_dims['circ']):
                vec_h = vec_all[:, i, :]
                std_h = std_all[:, i:i+1]
                dist = CircularDiscretizedDistribution(vec_h, std_h)
                
                act_i = circ_actions[:, i]
                log_probs += dist.log_prob(act_i).unsqueeze(-1)
                
                e_head = dist.entropy().unsqueeze(-1)
                entropy += e_head
                e_circ_sum += e_head
            entropy_details['circ'] = e_circ_sum

        # --- Cat & Bern 保持不变 ---
        # ... (此处省略原有的 cat 和 bern 评估逻辑) ...

        return log_probs, entropy, entropy_details, actor_outputs, None