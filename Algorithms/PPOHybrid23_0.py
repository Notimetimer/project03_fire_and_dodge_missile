'''
actor内置开火mask
'''

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import copy
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
    引入了可学习的温度参数来控制离散和伯努利动作的熵。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5, head_hidden_layer_num=1):
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

        # 1. 连续动作头 (Continuous)
        # 参数: log_std (控制高斯分布宽度)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

            # # # 原·单层动作头
            # self.fc_mu = nn.Linear(prev_size, cont_dim)
            
            # 现·2层动作头
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), cont_dim))
            self.fc_mu = nn.Sequential(*layers)
            

        # 2. 离散动作头 (Categorical)
        # 参数: log_temp_cat (控制 Softmax 温度)
        if 'cat' not in self.action_dims:
            self.action_dims['cat'] = []
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']  # list, e.g., [4, 10]
            total_cat_dim = sum(self.cat_dims)
            # # 原·单层输出
            # self.fc_cat = nn.Linear(prev_size, total_cat_dim)
            # 现·2层动作头
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), total_cat_dim))
            self.fc_cat = nn.Sequential(*layers)
            
            # 为每一个独立的离散头 (Head) 创建一个温度参数
            # 比如有 [4, 10] 两个头，我们就需要 2 个温度参数
            # 初始化为 0 (即 temperature=1.0)，保持原网络特性，让网络自己学去增大熵
            # self.log_temp_cat = nn.Parameter(torch.zeros(len(self.cat_dims))) 

        # 3. 伯努利动作头 (Bernoulli)
        # 参数: log_temp_bern (控制 Sigmoid 陡峭度)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            # # 原·单层输出
            # self.fc_bern = nn.Linear(prev_size, bern_dim)
            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            # nn.init.constant_(self.fc_bern.bias, -2.0)

            # 现·2层输出
            layers = []
            # for _ in range(head_hidden_layer_num):
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), bern_dim))
            self.fc_bern = nn.Sequential(*layers)
            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            nn.init.constant_(self.fc_bern[-1].bias, -2.0)
            
            # 为每一个伯努利动作维度创建一个温度参数
            # 初始化为 0 (即 temperature=1.0)
            # self.log_temp_bern = nn.Parameter(torch.zeros(bern_dim))
    
    # [修改] 增加 action_masks 参数, [新增] 增加 temp 参数
    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temperature=1.0, mask_on=1):
        if isinstance(temp, dict):
            temp_cat = temp.get('cat', 1.0)
            temp_bern = temp.get('bern', 1.0)
        else:
            temp_cat = temp
            temp_bern = temp

        shared_features = self.net(x)
        outputs = {'cont': None, 'cat': None, 'bern': None}

        # --- Continuous ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(shared_features)
            # 计算 std
            std = torch.exp(self.log_std_cont)
            std = torch.clamp(std, min=min_std, max=max_std)
            # 扩展维度以匹配 batch
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        # --- Categorical ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_all = self.fc_cat(shared_features)
            
            # 1. 切分 Logits
            cat_logits_list = torch.split(cat_logits_all, self.cat_dims, dim=-1)
            
            # 2. 获取温度 (Temp = exp(log_temp))
            # temp_cat 形状: (num_heads, )
            # temps = 1.0  # [修改] 使用传入的 temp
            # >2 强随机，<0.1 强确定性
            
            # 3. 应用温度缩放 (Logits / Temp) 并 Softmax
            # 较高的 Temp -> Logits 数值变小 -> Softmax 后分布趋向均匀 (熵增大)
            # 较低的 Temp -> Logits 数值差距拉大 -> Softmax 后分布趋向 One-hot (熵减小)
            final_probs_list = []
            for i, logits in enumerate(cat_logits_list):
                # 对应的温度: temps[i]
                # 使用 temp_cat 进行缩放, 防止除0
                scaled_logits = logits / (temp_cat + 1e-8)
                final_probs_list.append(F.softmax(scaled_logits, dim=-1))
            
            outputs['cat'] = final_probs_list

        # --- Bernoulli (核心修改区域) ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)

            # Compute can_fire mask from flattened observation x (always applied)
            xb = x
            if xb.dim() == 1:
                xb = xb.unsqueeze(0)

            # Indices (0-based): cos_ata_hor -> x[:,6], ata -> x[:,10], locked -> x[:,2], ammo -> x[:,20], distance_scaled -> x[:,9]
            cos_ata_hor = torch.clamp(xb[:, 6], -0.999999, 0.999999)
            delta_theta = xb[:, 8]
            ata = xb[:, 10]
            sin_theta = xb[:, 17]
            locked = xb[:, 2]
            ammo = xb[:, 20]
            dist = xb[:, 9] * 10e3
            t_since_launch = xb[:, 21] * 120

            ata_hor = torch.acos(cos_ata_hor)
            # 新代码1 Use Python/math pi (float) to avoid creating a constant tensor via numpy
            pi = math.pi
            ata_cond = (ata <= (60.0 * pi / 180.0)) & (ata_hor <= (30.0 * pi / 180.0))
            # 旧代码1
            # pi_val = torch.tensor(np.pi, device=shared_features.device)
            # ata_cond = (ata <= (60.0 * pi_val / 180.0)) & (ata_hor <= (30.0 * pi_val / 180.0))
            locked_cond = (locked >= 0.5) # == 1
            ammo_cond = (ammo > 0.0)
            # Use elementwise logical ops so this works on tensors
            timd_cond = (t_since_launch >= 40) | ((dist < 30e3) & (t_since_launch >= 10))
            dist_cond = (dist < 95e3)
            delta_theta_cond = (delta_theta < pi * (30) / 180.0) # 之前的 > -30度可能写反了
            cont_plus_1 = ~((delta_theta > 15.0 * pi / 180.0) & (torch.asin(sin_theta) <= -15.0 * pi / 180.0)) # & (dist >= 50e3))

            # 新代码2 Avoid Python branching on tensor-like `mask_on` (which breaks torch.jit.trace).
            # Create a scalar boolean tensor and select between the two candidate masks.
            if torch.is_tensor(mask_on):
                mask_on_tensor = mask_on.to(device=shared_features.device)
            else:
                mask_on_tensor = torch.tensor(bool(mask_on), device=shared_features.device)
            mask_on_bool = mask_on_tensor.to(dtype=torch.bool)

            can_fire_full = ata_cond & locked_cond & ammo_cond & timd_cond & dist_cond & delta_theta_cond & cont_plus_1
            can_fire = torch.where(mask_on_bool, can_fire_full, ammo_cond)

            # # 旧代码2
            # if mask_on:
            #     can_fire = ata_cond & locked_cond & ammo_cond & timd_cond & dist_cond & delta_theta_cond
            # else:
            #     can_fire = ammo_cond

            # build mask for bern dims and apply to first bern dimension only
            bern_dim = self.action_dims.get('bern', 0)
            batch_size = shared_features.size(0)
            mask = torch.ones((batch_size, bern_dim), dtype=torch.bool, device=shared_features.device)
            mask[:, 0] = can_fire.to(dtype=torch.bool)

            # If external action_masks provided (e.g., death masks), combine them (AND)
            if action_masks is not None and 'bern' in action_masks:
                ext_mask = action_masks['bern']
                if isinstance(ext_mask, torch.Tensor):
                    if ext_mask.dim() == 1:
                        ext_mask = ext_mask.unsqueeze(1)
                    ext_bool = (ext_mask != 0).to(dtype=torch.bool, device=shared_features.device)
                else:
                    ext_mask = torch.tensor(np.array(ext_mask), device=shared_features.device)
                    if ext_mask.dim() == 1:
                        ext_mask = ext_mask.unsqueeze(1)
                    ext_bool = (ext_mask != 0).to(dtype=torch.bool, device=shared_features.device)

                if ext_bool.size(1) == 1 and bern_dim > 1:
                    ext_bool = ext_bool.expand(-1, bern_dim)

                mask = mask & ext_bool

            # Apply mask: False -> set logits very small
            bern_logits = bern_logits.masked_fill(mask == 0, -1e8)

            # [修改] 使用我们提取的 temp_bern
            # temps = 1.0 
            scaled_bern_logits = bern_logits / (temp_bern + 1e-8)
            outputs['bern'] = scaled_bern_logits
            
        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(nn.Module):
    """
    统一接口适配器。
    将具体的 PolicyNetHybrid 封装起来，对外提供标准的 get action 和 evaluate actions 接口。
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

    # [修改] 增加 check_obs 参数，默认为 None， [新增] 增加 temp 参数
    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5, temperature=1.0, mask_on=1):
        """
        推理接口。
        Args:
            state: numpy array or tensor
            h: hidden state (预留接口，目前未使用)
            explore: bool or dict. If bool, applies to all action types.
                     If dict, e.g., {'cont': True, 'cat': False, 'bern': True}, controls exploration for each type.
        Returns:
            actions_exec: dict (numpy), 用于环境执行
            actions_raw: dict (numpy/tensor), 用于存入 buffer
            next_h: hidden state (预留接口)
            
        注意： 仅在推理时传入check_obs, 训练时禁止传入!!!
        1、目前 get action 中的 mask 生成只处理单个 check_obs（推理时），
            并把同一 mask 广播到整个 batch；如果要对 batch 内每个样本分别判断需扩展生成逻辑。
        2、evaluate_action（训练/计算 log_prob）默认未把 action_masks 传给 net ,
            若希望训练时也应用 mask，需要在 evaluate_action 调用 net 时传入 action_masks。
        """
        #  增强的 Batch 检测逻辑
        is_batch = False
        if not isinstance(state, torch.Tensor):
            if isinstance(state, np.ndarray) and state.ndim > 1:
                is_batch = True
                state = torch.tensor(state, dtype=torch.float).to(self.device)
            else:
                state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        else:
            if state.dim() > 1:
                is_batch = True
        
        # [修改] 处理 explore 参数，使其支持字典
        if isinstance(explore, bool):
            explore_opts = {'cont': explore, 'cat': explore, 'bern': explore}
        elif isinstance(explore, dict):
            # 如果传入字典，使用字典的值，对缺失的键默认为 True
            explore_opts = {
                'cont': explore.get('cont', True),
                'cat': explore.get('cat', True),
                'bern': explore.get('bern', True),
            }
        else:
            # 对于其他意外的输入类型，默认全部探索
            explore_opts = {'cont': True, 'cat': True, 'bern': True}

        # # =====================================================================
        # # [迁移] 解析 check_obs 并构建 Action Mask
        # # =====================================================================
        # action_masks = None
        # can_fire = True
        # # 当且仅当传入了单个 dict 类型的 check_obs 时启用 mask, 不受explore影响
        # if (check_obs is not None) and isinstance(check_obs, dict):  # and (not explore_opts['bern']):
        #     # 默认允许开火，下面按规则逐项收敛（保留注释）
        #     can_fire = True
        #     # 如果是Batch训练模式，通常check_obs会增加维度，这里只在推理的时候启用

        #     # 1. ATA <= 60度 (0.5236 rad)
        #     ata_hor = np.arccos(check_obs["target_information"][0])
        #     ata = check_obs["target_information"][4]
        #     ata_condition = (ata <= 60 * np.pi / 180 and ata_hor <= 20 * np.pi / 180)
        #     # [新增] ata_hor 是第一个漂亮结果后新增的mask项
        #     can_fire = can_fire and ata_condition

        #     # 2. Target Locked == 1
        #     locked = check_obs["target_locked"]
        #     locked_condition = (locked == 1)
        #     can_fire = can_fire and locked_condition

        #     # 3. Ammo > 0 (ego_main 最后一个元素是 ammo)
        #     ammo = check_obs["ego_main"][6]
        #     ammo_condition = (ammo > 0)
        #     can_fire = can_fire and ammo_condition

        #     # 4. 超远距离尾追不打（使用 AA_hor 判断尾追）
        #     distance = check_obs["target_information"][3]
        #     AA_hor = check_obs["target_information"][6]
        #     if (distance > 30e3) and (abs(AA_hor) < np.pi/6):
        #         can_fire = False

        #     # 5. 30km 外12s内禁止重复发射第二枚 或 mid-term 有在飞导弹
        #     # weapon 计时单位兼容原逻辑
        #     if (distance > 30e3 and check_obs["weapon"] * 120 < 12) or check_obs.get("missile_in_mid_term", False):
        #         can_fire = False

        #     # 构建 Tensor Mask: (Batch_Size, Bern_Dim) -> (1, 1)
        #     # 1.0 表示允许 (保留 Logits)，0.0 表示禁止 (Logits -> -inf)
        #     mask_val = 1.0 if can_fire else 0.0
            
        #     # 适配 state 的 batch size
        #     batch_size = state.size(0)
        #     mask_tensor = torch.full((batch_size, 1), mask_val, device=self.device, dtype=torch.float)
            
        #     action_masks = {'bern': mask_tensor}
        # # =====================================================================

        # [修改] 调用网络时传入 action_masks 和 temp
        actor_outputs = self.net(state, max_std=max_std, temperature=temp, mask_on=mask_on)
        
        # # [原有] 调用网络
        # actor_outputs = self.net(state, max_std=max_std)  # 如果需要gru，改动这一行
        
        actions_exec = {}
        actions_raw = {}
        actions_dist_check = {} #  诊断输出

        # --- Cont ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            if explore_opts['cont']:
                a_norm, u = dist.sample()
            else:
                u = mu
                a_norm = torch.tanh(u)
            
            a_exec = self._scale_action_to_exec(a_norm)
            
            #  根据是否 Batch 返回不同形状
            if is_batch:
                actions_exec['cont'] = a_exec.cpu().detach().numpy() # (Batch, Dim)
                actions_raw['cont'] = u.cpu().detach().numpy()
                actions_dist_check['cont'] = u.cpu().detach().numpy()
            else:
                actions_exec['cont'] = a_exec[0].cpu().detach().numpy().flatten()
                actions_raw['cont'] = u[0].cpu().detach().numpy().flatten()
                actions_dist_check['cont'] = u[0].cpu().detach().numpy().flatten()

        # --- Cat ---
        if actor_outputs['cat'] is not None:
            cat_probs_list = actor_outputs['cat']
            cat_exec_list = []      # 用于 actions_exec
            cat_indices_raw_list = [] # 用于 actions_raw
            cat_probs_check_list = [] #  记录 Cat 概率
            
            for probs in cat_probs_list:
                dist = Categorical(probs=probs)
                idx = dist.sample() if explore_opts['cat'] else torch.argmax(dist.probs, dim=-1)
                
                if is_batch:
                    cat_exec_list.append(idx.cpu().detach().numpy()) # (Batch, )
                    cat_indices_raw_list.append(idx.cpu().detach().numpy())
                    cat_probs_check_list.append(probs.cpu().detach().numpy())
                else:
                    cat_exec_list.append(idx.item())
                    cat_indices_raw_list.append(idx.item())
                    cat_probs_check_list.append(probs[0].cpu().detach().numpy().copy())
            
            # 这里的 actions_exec['cat'] 现在变成了一个包含索引的 numpy 数组
            if is_batch:
                actions_exec['cat'] = np.stack(cat_exec_list, axis=-1) # (Batch, N_Heads)
                actions_raw['cat'] = np.stack(cat_indices_raw_list, axis=-1)
            else:
                actions_exec['cat'] = np.array(cat_exec_list) 
                actions_raw['cat'] = np.array(cat_indices_raw_list)
            
            #  将所有 Cat 概率分布以列表形式存入诊断输出
            actions_dist_check['cat'] = cat_probs_check_list

        # --- Bern ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            dist = Bernoulli(logits=bern_logits)
            bern_action = dist.sample() if explore_opts['bern'] else (dist.probs > bern_threshold).float()
            
            if is_batch:
                actions_exec['bern'] = bern_action.cpu().detach().numpy() # (Batch, Dim)
                actions_raw['bern'] = actions_exec['bern']
                actions_dist_check['bern'] = dist.probs.cpu().detach().numpy()
            else:
                actions_exec['bern'] = bern_action[0].cpu().detach().numpy().flatten()
                actions_raw['bern'] = actions_exec['bern']
                actions_dist_check['bern'] = dist.probs[0].cpu().detach().numpy().flatten()

        return actions_exec, actions_raw, None, actions_dist_check # None for hidden state

    def evaluate_actions(self, states, actions_raw, h=None, max_std=None, mask_on=1):
        """
        训练接口。计算 log_probs 和 entropy。
        Args:
            states: tensor (B, D)
            actions_raw: dict of tensors
        Returns:
            log_probs: tensor (B, 1)
            entropy: tensor (B, 1)
            next_h: None
            actor_outputs: dict (raw outputs from net) [新增]
        """
        actor_outputs = self.net(states, max_std=max_std, mask_on=mask_on)
        log_probs = torch.zeros(states.size(0), 1).to(self.device)
        entropy = torch.zeros(states.size(0), 1).to(self.device)
        
        #  用于记录分项 Entropy 的字典
        entropy_details = {'cont': None, 'cat': None, 'bern': None}

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u = actions_raw['cont']
            log_probs += dist.log_prob(0, u).sum(-1, keepdim=True)
            # entropy += dist.entropy().unsqueeze(-1) # 近似熵
            
            #  单独记录 cont entropy
            e_cont = dist.entropy().unsqueeze(-1)
            entropy += e_cont
            entropy_details['cont'] = e_cont # [修改] 保持 Tensor 用于 Loss 计算

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_probs_list = actor_outputs['cat']
            cat_action = actions_raw['cat'].long()
            
            #  临时列表用于计算 cat 总熵
            e_cat_sum = torch.zeros_like(entropy)
            
            for i, probs in enumerate(cat_probs_list):
                act_i = cat_action[:, i].unsqueeze(-1)
                log_probs += torch.log(probs.gather(1, act_i) + 1e-8)
                dist = Categorical(probs=probs)
                
                # 累加每个离散头的熵
                e_head = dist.entropy().unsqueeze(-1)
                entropy += e_head
                e_cat_sum += e_head
            
            #  记录 cat entropy
            entropy_details['cat'] = e_cat_sum # [修改] 保持 Tensor 用于 Loss 计算

        # --- Bern ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = actor_outputs['bern']
            # Replace -inf logits (from masking) with a large negative finite value for numerical stability during training
            bern_logits = bern_logits.clamp(min=-1e8)
            dist = Bernoulli(logits=bern_logits)
            bern_action = actions_raw['bern']
            log_probs += dist.log_prob(bern_action).sum(-1, keepdim=True)
            
            #  单独记录 bern entropy
            e_bern = dist.entropy().sum(-1, keepdim=True)
            entropy += e_bern
            entropy_details['bern'] = e_bern # [修改] 保持 Tensor 用于 Loss 计算

        # [修改] 返回 actor_outputs 以便外部访问 logits
        return log_probs, entropy, entropy_details, actor_outputs, None
    
    def compute_il_loss(self, states, expert_actions, label_smoothing=0.1, no_bern=False, mask_on=0):
        """
        计算模仿学习 Loss (MARWIL / BC)。
        
        Args:
            states: (Batch, State_Dim)
            expert_actions: 字典, 包含 {'cont': u, 'cat': index, 'bern': float}
                            注意：对于连续动作，这里通常假设传入的是 pre-tanh 的 u，
                            或者你需要在外部处理好。
            label_smoothing: 标签平滑系数
            
        Returns:
            total_loss_per_sample: (Batch, ) 每个样本的 Loss 总和，未加权
        """
        '''
        会增加复杂度的可选改进：模仿学习的时候alpha 传入向量，从而区分密集和稀疏动作的学习强度（密集应该高一些）
        '''
        actor_outputs = self.net(states, mask_on=mask_on) # 获取 raw output (mu/std, logits)
        
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
                    '''
                    log_probs.gather()
                    从所有动作的概率分布 log_probs 中，精准地抽取出“实际执行了的那个动作” expert_idx 对应的概率值。
                    - 1 (第一个参数)：表示在第 1 维（列维度）进行选取。
                    - expert_idx.unsqueeze(1)：将原来形状为(Batch,)的索引变成(Batch, 1)。
                     这是因为 gather 要求索引的维度必须和原张量一致。
                    - .squeeze(1)：取完值后，形状还是(Batch, 1)用 squeeze 把那个多余的维度删掉，
                    变成平铺的 (Batch,)，方便后续算 Loss。
                    '''
                
                total_loss_per_sample += ce_loss

        # --- 3. 伯努利动作 (Bernoulli) ---
        # -- Focal Loss --
        if not no_bern:
            if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
                bern_logits = actor_outputs['bern']
                # Clamp masked -inf logits to a large negative finite value for stable sigmoid/log calculations
                bern_logits = bern_logits.clamp(min=-1e8)
                probs = torch.sigmoid(bern_logits)
                probs = torch.clamp(probs, 1e-10, 1.0 - 1e-10)
                target = expert_actions['bern'] # (Batch, 1)
                # Label Smoothing
                if label_smoothing > 0:
                    target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
                # === 方案2：Focal Loss (针对敏感度问题) ===
                # alpha: 平衡因子，类似于 pos_weight 的作用，但范围是 0-1
                # gamma: 聚焦因子，通常设为 2.0。值越大，越忽视简单背景，越关注难分类的发射瞬间
                
                # 建议参数组合：
                # alpha = 0.75 (意味着正样本本身权重是 0.75，负样本是 0.25，自带 3:1 的加权)
                # gamma = 2.0 (标准设置)
                
                alpha = 0.75
                gamma = 2.0
                
                # Focal Loss 公式
                # 对于正样本 (target=1): -alpha * (1-p)^gamma * log(p)
                # 对于负样本 (target=0): -(1-alpha) * p^gamma * log(1-p)
                
                loss_pos = -alpha * torch.pow(1.0 - probs, gamma) * torch.log(probs) * target
                loss_neg = -(1 - alpha) * torch.pow(probs, gamma) * torch.log(1.0 - probs) * (1.0 - target)
                
                bce_loss = loss_pos + loss_neg
                
                total_loss_per_sample += bce_loss.sum(dim=-1)
        
        return total_loss_per_sample

    @torch.no_grad()
    def compute_marwil_monitor(self, states, expert_actions, mask_on=0, advantages=None):
        """
        [监控用] 在固定 batch 上计算每个动作头独立的 NLL 与策略熵。
        - 全程 no_grad，不影响主网络更新；
        - NLL 不带 label smoothing / focal / weights，是纯粹的负对数似然；
        - 熵是策略分布本身的熵（不依赖 expert action）。
        - advantages: 可选，(N,) 或 (N,1) 的 advantage tensor，用于计算 adv_positive_frac。

        Returns:
            dict: 包含 nll_cont/nll_cat/nll_bern/entropy_cont/entropy_cat/entropy_bern/adv_positive_frac,
                  缺失的动作头或未传入 advantages 时对应键返回 None。
        """
        actor_outputs = self.net(states, mask_on=mask_on)
        metrics = {
            'nll_cont': None, 'nll_cat': None, 'nll_bern': None,
            'entropy_cont': None, 'entropy_cat': None, 'entropy_bern': None,
            'accuracy_cont': None, 'accuracy_cat': None, 'accuracy_bern': None,
            'adv_positive_frac': None,
        }

        # --- Cont ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0 and actor_outputs.get('cont') is not None:
            mu, std = actor_outputs['cont']
            dist = SquashedNormal(mu, std)
            u_expert = expert_actions['cont']
            # 纯 NLL：sum over dim, mean over batch
            nll_cont = (-dist.log_prob(0, u_expert).sum(dim=-1)).mean()
            ent_cont_raw = dist.entropy()
            ent_cont = ent_cont_raw.sum(dim=-1).mean() if ent_cont_raw.dim() > 1 else ent_cont_raw.mean()
            # Accuracy: mu的误差模均值
            accuracy_cont = torch.norm(mu - u_expert, dim=-1).mean()
            metrics['nll_cont'] = nll_cont.item()
            metrics['entropy_cont'] = ent_cont.item()
            metrics['accuracy_cont'] = accuracy_cont.item()

        # --- Cat ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0 and actor_outputs.get('cat') is not None:
            cat_probs_list = actor_outputs['cat']
            expert_cat = expert_actions['cat'].long()
            nll_cat_sum = torch.zeros(states.size(0), device=self.device)
            ent_cat_sum = torch.zeros(states.size(0), device=self.device)
            correct_cat_sum = torch.zeros(states.size(0), device=self.device)
            for i, probs in enumerate(cat_probs_list):
                expert_idx = expert_cat[:, i]
                log_probs = torch.log(probs + 1e-10)
                nll_cat_sum += -log_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)
                dist = Categorical(probs=probs)
                ent_cat_sum += dist.entropy()
                # Accuracy: argmax和示范的action完全一样
                pred_idx = torch.argmax(probs, dim=1)
                correct_cat_sum += (pred_idx == expert_idx).float()
            metrics['nll_cat'] = nll_cat_sum.mean().item()
            metrics['entropy_cat'] = ent_cat_sum.mean().item()
            # 计算所有cat动作的平均准确度
            total_cat_dims = len(cat_probs_list)
            metrics['accuracy_cat'] = (correct_cat_sum / total_cat_dims).mean().item()

        # --- Bern ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0 and actor_outputs.get('bern') is not None:
            bern_logits = actor_outputs['bern'].clamp(min=-1e8)
            dist = Bernoulli(logits=bern_logits)
            target = expert_actions['bern']
            nll_bern = (-dist.log_prob(target).sum(dim=-1)).mean()
            ent_bern = dist.entropy().sum(dim=-1).mean()
            # Accuracy: argmax和示范的action完全一样
            pred_bern = (bern_logits > 0).float()
            accuracy_bern = (pred_bern == target).float().mean()
            metrics['nll_bern'] = nll_bern.item()
            metrics['entropy_bern'] = ent_bern.item()
            metrics['accuracy_bern'] = accuracy_bern.item()

        # --- adv_positive_frac ---
        if advantages is not None:
            adv = advantages.detach().view(-1)
            metrics['adv_positive_frac'] = (adv > 0).float().mean().item()

        return metrics
# =============================================================================
# 3. PPO 算法类 (精简版)
# =============================================================================

class PPOHybrid:
    def __init__(self, actor, critic, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device, 
                 k_entropy={'cont':0.01, 'cat':0.005, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.7):
        
        self.actor = actor # 这是一个 HybridActorWrapper 实例
        self.critic = critic
        
        # critic优化器不动了，保持adam
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        "Adam优化器（有动量）"
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        "RMSprop优化器（有二阶矩但是没有一阶动量），初步测试效果更差"
        # self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=actor_lr, alpha=0.99, eps=1e-5)
        "AdamW优化器（weight_decay）"
        # self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=1e-4, eps=1e-5)
        "RAdam优化器(为冷启动优化，但非平稳环境不建议用)"
        # self.actor_optimizer = torch.optim.RAdam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        
        # [修改] 解析 k_entropy，支持字典输入
        if isinstance(k_entropy, dict):
            self.k_entropy = k_entropy
        else:
            self.k_entropy = {'cont': k_entropy, 'cat': k_entropy, 'bern': k_entropy}
        
        # [新增] SAC 风格的可学习 Cat 熵系数 (初始值对应原配置)
        init_k_cat = k_entropy.get('cat', k_entropy)
        self.log_k_cat = torch.nn.Parameter(torch.log(torch.tensor(init_k_cat, device=device)))
        # [新增] 为自适应熵系数配备独立的优化器 (学习率通常与 Actor 保持一致或略大)
        self.k_cat_optim = torch.optim.Adam([self.log_k_cat], lr=actor_lr)
        
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
        
        #  额外的监控指标
        self.approx_kl = 0        # 近似 KL 散度 (判断策略变化幅度)
        self.clip_frac = 0        # 裁剪触发比例 (判断 eps 或 lr 是否合适)
        self.explained_var = 0    # 解释方差 (判断 Critic 拟合程度)
        #  分项 Entropy 监控
        self.entropy_cat = 0
        self.entropy_bern = 0
        self.entropy_cont = 0
        
        # [新增] 监控指标
        self.td_error_var = 0     # TD error 的分布方差
        self.grad_norm_ratio = 0  # actor 梯度与 critic 梯度的范数比

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        if actor_lr is not None:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = actor_lr
        if critic_lr is not None:
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = critic_lr  


    # 重置优化器，这里仍然是adam，目前只有PBT在使用这个方法
    def reset_optimizer(self):
        """
        清除 optimizer 的动量等 state，但保留当前学习率/其他 param_group 设置。
        通过重建 optimizer（保留 param_group 的 lr/betas/eps/weight_decay）来达到清空 state 的目的。
        """
        def _recreate_from(old_optim, params):
            pg = old_optim.param_groups[0]
            lr = pg.get('lr', 1e-3)
            betas = pg.get('betas', (0.9, 0.999))
            eps = pg.get('eps', 1e-8)
            weight_decay = pg.get('weight_decay', 0.0)
            return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # 重新创建 optim（以清除内部 state），但使用当前的 lr/betas/eps/weight_decay
        self.actor_optimizer = _recreate_from(self.actor_optimizer, self.actor.parameters())
        self.critic_optimizer = _recreate_from(self.critic_optimizer, self.critic.parameters())

        # 清除可能残留的梯度
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
    
    
    def take_action(self, state, h0=None, explore=True, max_std=None, check_obs=None, temperature=1.0, mask_on=1):
        # 委托给 Actor Wrapper
        max_s = max_std if max_std is not None else self.max_std
        
        # [修改] 透传 check_obs
        actions_exec, actions_raw, h_state, actions_dist_check = self.actor.get_action(
            state, h=h0, explore=explore, max_std=max_s, check_obs=check_obs, temperature=temperature, mask_on=mask_on
        )
        #  保持原有的返回两个字典的接口，或者根据需要返回 diagnostic output
        return actions_exec, actions_raw, h_state, actions_dist_check

    def update(self, transition_dict, adv_normed=False, 
                clip_vf=False, clip_range=0.2, shuffled=1, 
                mini_batch_size=None, alpha_logit_reg=0.05,
                v_trace=None, target_p1=0.65, target_p1_b=0.8, 
                k_nonlinear=0.89, mask_on=1, actor_frozen=0): 
                # [新增] target_p1 默认“一超”概率，剩下来的留给“多强”)
                # [修改] 增加 target_p1_b 参数，对应开火控制的“笃定程度”

        # RL 更新阶段：确保所有分布参数都参与梯度更新
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = True
        # if hasattr(self.actor.net, 'log_temp_cat'):
        #     self.actor.net.log_temp_cat.requires_grad = False
        # if hasattr(self.actor.net, 'log_temp_bern'):
        #     self.actor.net.log_temp_bern.requires_grad = False

        #  6. 智能数据转换：如果已经是 np.ndarray (来自 HybridReplayBuffer)，直接转 Tensor
        # 否则 (来自 list append)，先转 np 再转 Tensor
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        states = to_tensor(transition_dict['states'], torch.float)
        
        dones = to_tensor(transition_dict['dones'], torch.float).view(-1, 1)
        rewards = to_tensor(transition_dict['rewards'], torch.float).view(-1, 1)

        #  处理 active_masks (可选输入)
        # 如果 transition_dict 中没有 active_masks，则默认所有样本均有效
        if 'active_masks' in transition_dict:
            active_masks = to_tensor(transition_dict['active_masks'], torch.float).view(-1, 1)
        else:
            # 创建与 dones 形状一致的全 1 张量
            active_masks = torch.ones_like(dones)

        # 处理 obs (如果存在)
        if 'obs' in transition_dict:
            actor_inputs = to_tensor(transition_dict['obs'], torch.float)
            critic_inputs = states
        else:
            actor_inputs = states
            critic_inputs = states
        
        # 1. 准备动作数据
        actions_from_buffer = transition_dict['actions']
        
        # todo action_mask 防止“死后动作”干扰决策
        # todo truncs
        # todo global states，适配集中式Critic

        # 1. 准备动作数据 (转 Tensor)
        actions_on_device = {}
        
        # Buffer 传来的 actions 已经是 dict of arrays
        if isinstance(actions_from_buffer, dict):
            for key, val in actions_from_buffer.items():
                if key == 'cat':
                    actions_on_device[key] = to_tensor(val, torch.long)
                else:
                    actions_on_device[key] = to_tensor(val, torch.float)
        else:
            # 兼容旧代码 (list of dicts)
            # 旧逻辑：List of Dicts (较慢)
            all_keys = actions_from_buffer[0].keys()
            for key in all_keys:
                vals = [d[key] for d in actions_from_buffer]
                if key == 'cat':
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.long).to(self.device)
                else:
                    actions_on_device[key] = torch.tensor(np.array(vals), dtype=torch.float).to(self.device)


        # 2. 获取 Advantage (优先使用 Buffer 算好的)
        if 'advantages' in transition_dict and 'td_targets' in transition_dict:
            advantage = to_tensor(transition_dict['advantages'], torch.float).view(-1, 1)
            td_target = to_tensor(transition_dict['td_targets'], torch.float).view(-1, 1)
        else:
            # 现场计算 GAE (不推荐用于并行展平后的数据)
            
            #  处理 truncs
            if 'truncs' in transition_dict:
                truncs = to_tensor(transition_dict['truncs'], torch.float).view(-1, 1)
            else:
                truncs = None # 或者 torch.zeros_like(dones)

            # 以下为公共部分
            # 如果没有预计算，则现场计算 (注意：如果是并行数据直接展平进来的，这里计算会有偏差)
            with torch.no_grad():
                # [修改] 优先使用预计算的 next_values
                if 'next_values' in transition_dict:
                    next_vals = to_tensor(transition_dict['next_values'], torch.float).view(-1, 1)
                else:
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_states = to_tensor(transition_dict['next_states'], torch.float)
                    # Critic 使用全局 next_states 计算 Target
                    # 注意：对于截断的步，next_value 应该是 V(s_t+1) 而不是 0
                    next_vals = self.critic(next_states)
                
                # td_target的计算不应考虑truncs。仅当dones=1时，next_value才为0。
                # truncs的影响由compute_advantage函数内部处理。
                td_target = rewards + self.gamma * next_vals * (1 - dones)

                # Critic 使用全局 states 计算当前 Value
                td_delta = td_target - self.critic(critic_inputs)
                advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu(), dones.cpu(), truncs.cpu() if truncs is not None else None).to(self.device)
                
        # 3. 计算旧策略的 log_probs (使用 Wrapper)
        with torch.no_grad():
            # Actor 使用 actor_inputs (可能是 obs)
            # [修改] 接收 5 个返回值
            old_log_probs, _, _, _ ,_ = self.actor.evaluate_actions(actor_inputs, actions_on_device, h=None, max_std=self.max_std, mask_on=mask_on)
            # Critic 使用 critic_inputs (全局 states)
            v_pred_old = self.critic(critic_inputs)
            
        # --- [2. 优势归一化 - 适配 active_masks] ---
        if adv_normed:
            #  仅使用 active 的数据计算统计量
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            
            if active_adv.numel() > 1: # 防止 active 数据过少导致 NaN
                adv_mean = active_adv.mean()
                adv_std = active_adv.std(unbiased=False)
                advantage = (advantage - adv_mean) / (adv_std + 1e-8)
            else:
                # 降级策略：如果有效数据太少，不进行归一化或者只做去中心化
                pass
        else:
            # 推荐: 即使不归一化，也建议减去均值 (Centering)
            # 这有助于降低方差，且不改变梯度的方向
            #  使用 mask 计算均值
            active_adv = advantage[active_masks.squeeze(-1).bool()]
            if active_adv.numel() > 0:
                adv_mean = active_adv.mean()
                advantage = advantage - adv_mean

        # =====================================================================
        # [新增] 计算 Cat 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_cat_total = 0.0
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            for n_actions in self.actor.action_dims['cat']:
                if n_actions > 1:
                    # “一超”概率为 target_p1，其余 (n_actions - 1) 个动作平分剩余概率
                    p_rest = (1.0 - target_p1) / (n_actions - 1)
                    # 香农熵: -p1*ln(p1) - (N-1) * p_rest*ln(p_rest)
                    h_dim = - target_p1 * np.log(target_p1 + 1e-8) - (n_actions - 1) * p_rest * np.log(p_rest + 1e-8)
                    target_entropy_cat_total += h_dim
        target_entropy_cat_tensor = torch.tensor(target_entropy_cat_total, device=self.device)
        
        # =====================================================================
        # [新增] 计算 Bernoulli 分布的动态目标熵 (Target Entropy)
        # =====================================================================
        target_entropy_bern_total = 0.0
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            # 伯努利熵公式: -p*ln(p) - (1-p)*ln(1-p)
            p = target_p1_b
            h_bern_dim = - p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
            # 伯努利熵按动作维度累加
            target_entropy_bern_total = h_bern_dim * self.actor.action_dims['bern']
        target_entropy_bern_tensor = torch.tensor(target_entropy_bern_total, device=self.device)

        # 4. PPO Update Loop
        actor_loss_list, critic_loss_list, entropy_list, ratio_list = [], [], [], []
        actor_grad_list, critic_grad_list = [], []
        pre_clip_actor_grad, pre_clip_critic_grad = [], []

        #  监控列表
        kl_list = []
        clip_frac_list = []
        #  分项 Entropy 列表
        entropy_cat_list = []
        entropy_bern_list = []
        entropy_cont_list = []
        grad_norm_ratio_list = [] # [新增] 范数比列表
        
        # [新增] 初始化样本统计计数器 (包含重复更新累加)
        ppo_samples_total = 0
        ppo_valid_samples_total = 0

        mask_eps = 1e-5
        
        num_samples = actor_inputs.size(0)
        if mini_batch_size is None:
            mini_batch_size = num_samples

        for _ in range(self.epochs):
            
            # --- [Shuffle 逻辑移入 Epoch 循环] ---
            # 每个 epoch 生成新的随机索引，确保数据打乱
            if shuffled:
                idx = torch.randperm(num_samples, device=self.device)
            else:
                idx = torch.arange(num_samples, device=self.device)
            
            # --- [引入 Mini-Batch 循环] ---
            for start in range(0, num_samples, mini_batch_size):
                end = min(start + mini_batch_size, num_samples)
                batch_idx = idx[start:end]
                
                # 切片 mini-batch 数据
                mb_actor_inputs = actor_inputs[batch_idx]
                mb_critic_inputs = critic_inputs[batch_idx]
                mb_advantage = advantage[batch_idx]
                mb_td_target = td_target[batch_idx]
                mb_old_log_probs = old_log_probs[batch_idx]
                mb_active_masks = active_masks[batch_idx]
                # v_pred_old_batch = v_pred_old[batch_idx] # 如果需要用到旧 Value
                
                # 切片 Actions (Dict 结构)
                mb_actions = {}
                for k, v in actions_on_device.items():
                    mb_actions[k] = v[batch_idx]

                # 计算当前策略的 log_probs 和 entropy (使用 Wrapper)
                #  接收 entropy_details 和 actor_outputs
                log_probs, entropy, entropy_details, actor_outputs, _ = self.actor.evaluate_actions(mb_actor_inputs, mb_actions, h=None, max_std=self.max_std, mask_on=mask_on)
                
                #  计算 log_ratio 用于更精准的 KL 计算
                log_ratio = log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                
                #  计算 Approximate KL Divergence
                with torch.no_grad():
                    #  KL 计算也最好应用 mask，但为了监控方便，这里先保持全局均值或应用mask均值
                    active_sum = mb_active_masks.sum()
                    approx_kl = (((ratio - 1) - log_ratio) * mb_active_masks).sum() / (active_sum + mask_eps)
                    kl_list.append(approx_kl.item())
                    
                    #  计算 Clip Fraction (有多少样本触发了裁剪)
                    clip_fracs = (((ratio - 1.0).abs() > self.eps).float() * mb_active_masks).sum() / (active_sum + mask_eps)
                    clip_frac_list.append(clip_fracs.item())
                
                # 借用一点IMPALA的经验，防止ratio过大导致梯度被炸飞
                # 建议设置数值：2.0~5.0
                if v_trace is not None:
                    ratio = torch.clamp(ratio, max=v_trace)

                surr1 = ratio * mb_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * mb_advantage
                
                #  Actor Loss 使用 mask 加权
                surrogate_loss = -torch.min(surr1, surr2)
                # active_sum 已经在上面计算过
                
                # [新增] 统计 PPO 样本
                ppo_samples_total += mb_active_masks.numel()
                ppo_valid_samples_total += mb_active_masks.sum().item()

                actor_loss = (surrogate_loss * mb_active_masks).sum() / (active_sum + mask_eps)
                
                # [修改] 分项 Entropy Loss 计算
                e_cont = entropy_details['cont'] if entropy_details['cont'] is not None else torch.tensor(0., device=self.device)
                e_cat = entropy_details['cat'] if entropy_details['cat'] is not None else torch.tensor(0., device=self.device)
                e_bern = entropy_details['bern'] if entropy_details['bern'] is not None else torch.tensor(0., device=self.device)

                loss_ent_cont = (e_cont * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_cat = (e_cat * mb_active_masks).sum() / (active_sum + mask_eps)
                loss_ent_bern = (e_bern * mb_active_masks).sum() / (active_sum + mask_eps)

                # =====================================================================
                # [修改] 使用动态 k_cat 计算 Actor Loss
                # =====================================================================
                k_cont = self.k_entropy.get('cont', 0.0)
                k_cat = self.k_entropy.get('cat', 0.0)
                k_bern = self.k_entropy.get('bern', 0.0)
                
                # # 当前的 k_cat 是 e^(log_k_cat)
                # curr_k_cat = torch.exp(self.log_k_cat)
                # --- 关键修改：跟踪目标熵, 熵过小，加熵。熵过大，减熵 ---
                # 比平方误差更加平缓的损失函数候选：
                # 1. pseudo-huber loss:     torch.sqrt(1 + torch.square(target - current)) - 1
                # 2. log-cosh loss:         torch.log(torch.cosh(target_entropy_cat_tensor - loss_ent_cat))

                # 裁剪的范围，绝不能太小
                theoretical_ent_cat_max = 0.0
                for dim in self.actor.action_dims['cat']:
                    theoretical_ent_cat_max += np.log(float(dim))
                diff_cat0 = theoretical_ent_cat_max - target_entropy_cat_tensor
                k_nonlinear_max_cat = torch.sqrt(1+diff_cat0**2)/(diff_cat0+1e-8)

                theoretical_ent_bern_max = -np.log(0.5)*self.actor.action_dims['bern']
                diff_bern0 = theoretical_ent_bern_max - target_entropy_bern_tensor
                k_nonlinear_max_bern = torch.sqrt(1+diff_bern0**2)/(diff_bern0+1e-8)

                k_nonlinear_cat = min(max(k_nonlinear, 0.0), k_nonlinear_max_cat)
                k_nonlinear_bern = min(max(k_nonlinear, 0.0), k_nonlinear_max_bern)

                # 1. Categorical 约束项
                cat_constraint_term = k_cat * (
                    - loss_ent_cat # + 
                    # torch.where(loss_ent_cat > target_entropy_cat_tensor, 
                    # min(max(k_nonlinear_cat, 0.0), 1.0)/(2 * diff_cat0) * \
                    #     torch.square(loss_ent_cat - target_entropy_cat_tensor)
                    # , 0.0)
                )
                # 2. Bernoulli 约束项
                bern_constraint_term = k_bern * (
                    - loss_ent_bern # + 
                    # torch.where(loss_ent_bern > target_entropy_bern_tensor,
                    # min(max(k_nonlinear_bern, 0.0), 1.0)/(2 * diff_bern0) * \
                    #     torch.square(loss_ent_bern - target_entropy_bern_tensor)
                    # , 0.0)
                )
                # 3. 组合最终 Actor Loss
                actor_loss = actor_loss + cat_constraint_term + bern_constraint_term - (k_cont * loss_ent_cont)
                
                # 原有非目标熵正则项
                # actor_loss = actor_loss - (k_cont * loss_ent_cont + k_cat * loss_ent_cat + k_bern * loss_ent_bern)

                # =====================================================================
                # [修改/重构] Bernoulli 开火头的专属正则化、稀疏惩罚与冷却约束
                # =====================================================================
                if actor_outputs['bern'] is not None:
                    bern_logits = actor_outputs['bern']
                    bern_probs = torch.sigmoid(bern_logits)
                    
                    # # 1. 基础 Logit 越界惩罚 (保持 Logit 在可激活区间)
                    # over = F.relu(torch.abs(bern_logits) - 4.0)
                    # 只惩罚允许开火的位置
                    # fire_mask = (bern_logits > -1e4).float()
                    # logit_loss = ((over ** 2) * fire_mask).mean()
                    # actor_loss = actor_loss + alpha_logit_reg * logit_loss

                    # 2. 基础稀疏惩罚 (也采用方案 A，防止高概率时失效)
                    # 惩罚项 = -log(1 - p)。其梯度为 alpha * p
                    alpha_sparsity = 0.001
                    eps = 1e-7
                    # 这里的 -log(1-p) 在数学上等于 softplus(logits)，更数值稳定
                    # 但为了直观对应方案 A，我们写成对数形式
                    sparsity_loss_term = -torch.log(1.0 - bern_probs + eps)
                    sparsity_loss = (sparsity_loss_term * mb_active_masks).sum() / (active_sum + mask_eps)
                    actor_loss = actor_loss + alpha_sparsity * sparsity_loss

                #     if time_since_shoot_location is not None:
                #         # 3. [核心新增] 冷却时间强制压制 (方案 A 版)
                #         t_since_launch = mb_actor_inputs[:, time_since_shoot_location:time_since_shoot_location+1] 
                        
                #         # 构建惩罚权重: t < 0.333 时生效
                #         cooldown_weight = torch.clamp(1.0 - t_since_launch * 3.0, min=0.0)
                        
                #         # 严厉惩罚系数：由于梯度不再消失，0.4 的力度已经非常有震慑力
                #         alpha_cooldown = 0.3
                        
                #         # 核心修改：将原来的 bern_probs 替换为 -log(1 - p)
                #         # 这样当网络“极度想开火”(p->1) 时，惩罚项的梯度达到峰值 alpha_cooldown
                #         cooldown_log_penalty = -torch.log(1.0 - bern_probs + eps)
                        
                #         cooldown_loss = (cooldown_weight * cooldown_log_penalty * mb_active_masks).sum() / (active_sum + mask_eps)
                #         actor_loss = actor_loss + alpha_cooldown * cooldown_loss
                # # =====================================================================

                # Critic Loss
                # Critic 使用 critic_inputs
                v_pred = self.critic(mb_critic_inputs)
                if clip_vf:
                    v_pred_old_batch = v_pred_old[batch_idx]
                    v_pred_clipped = torch.clamp(v_pred, v_pred_old_batch - clip_range, v_pred_old_batch + clip_range)
                    vf_loss1 = (v_pred - mb_td_target).pow(2)
                    vf_loss2 = (v_pred_clipped - mb_td_target).pow(2)
                    critic_loss_per_sample = torch.max(vf_loss1, vf_loss2)
                else:
                    #  reduction='none' 使得我们可以应用 mask
                    critic_loss_per_sample = F.mse_loss(v_pred, mb_td_target, reduction='none')
                
                #  Critic Loss 使用 mask 加权
                critic_loss = (critic_loss_per_sample * mb_active_masks).sum() / (active_sum + mask_eps)
                
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                # if alpha_loss.requires_grad:    # [新增]
                #     self.k_cat_optim.zero_grad()
                if not actor_frozen:
                    actor_loss.backward()
                critic_loss.backward()
                # if alpha_loss.requires_grad:    # [新增] 反向传播温度损失
                #     alpha_loss.backward()

                pre_clip_actor_grad.append(model_grad_norm(self.actor))
                pre_clip_critic_grad.append(model_grad_norm(self.critic)) 
                
                # [新增] 计算范数比 (Pre-clip)
                grad_norm_ratio_list.append(pre_clip_actor_grad[-1] / (pre_clip_critic_grad[-1] + 1e-8))

                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
                if not actor_frozen:
                    self.actor_optimizer.step()
                self.critic_optimizer.step()
                # if alpha_loss.requires_grad:    # [新增] 步进自适应熵系数
                #     self.k_cat_optim.step()

                # Logging
                actor_grad_list.append(model_grad_norm(self.actor))
                critic_grad_list.append(model_grad_norm(self.critic))            
                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                
                # 记录总 Entropy (未加权)
                entropy_total = (entropy * mb_active_masks).sum() / (active_sum + mask_eps)
                entropy_list.append(entropy_total.item()) 
                ratio_list.append(ratio.mean().item()) 
                
                #  [修改] 记录分项 Entropy (现在是 Tensor，需要 .mean().item())
                if entropy_details['cont'] is not None:
                    entropy_cont_list.append(entropy_details['cont'].mean().item())
                if entropy_details['cat'] is not None:
                    entropy_cat_list.append(entropy_details['cat'].mean().item())
                if entropy_details['bern'] is not None:
                    entropy_bern_list.append(entropy_details['bern'].mean().item())
        # # =====================================================================
        # # 第二阶段：规则强制修正 (Post-PPO Rule Enforcement)
        # # 移出 Epoch 循环，只执行 1 次或独立的少数次数，避免干扰 PPO 的 clip 指标
        # # =====================================================================
        # if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
        #     # 重新打乱索引，进行一次专门针对开火规则的微调
        #     rule_idx = torch.randperm(num_samples, device=self.device)
        #     for start in range(0, num_samples, mini_batch_size):
        #         end = min(start + mini_batch_size, num_samples)
        #         batch_idx = rule_idx[start:end]
                
        #         mb_actor_inputs = actor_inputs[batch_idx]
        #         mb_active_masks = active_masks[batch_idx]
                
        #         # 重新前向传播获取当前 logits
        #         actor_outputs = self.actor.net(mb_actor_inputs, min_std=self.min_std, max_std=self.max_std)
        #         bern_logits = actor_outputs['bern']
        #         bern_probs = torch.sigmoid(bern_logits)
                
        #         # --- 方案 A 对数惩罚项 ---
        #         eps = 1e-7
        #         rule_loss = 0

        #         # 基础稀疏性惩罚
        #         sparsity_log_penalty = -torch.log(1.0 - bern_probs + eps)
        #         rule_loss += (0.001 * sparsity_log_penalty * mb_active_masks).sum() / (active_sum + mask_eps)

        #         # 冷却时间强制压制 (方案 A 版)
        #         if time_since_shoot_location is not None:
        #             t_since_launch = mb_actor_inputs[:, time_since_shoot_location:time_since_shoot_location+1]
        #             cooldown_weight = torch.clamp(1.0 - t_since_launch * 3.0, min=0.0)
                    
        #             # 方案 A：梯度 w.r.t logits = alpha * p (单调不消失)
        #             cooldown_log_penalty = -torch.log(1.0 - bern_probs + eps)
        #             rule_loss += (0.2 * cooldown_log_penalty * cooldown_weight * mb_active_masks).sum() / (active_sum + mask_eps)

        #         # 独立执行规则更新
        #         self.actor_optimizer.zero_grad()
        #         rule_loss.backward()
        #         nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad)
        #         self.actor_optimizer.step()
                
        self.actor_loss = np.mean(actor_loss_list)
        self.actor_grad = np.mean(actor_grad_list)
        self.critic_loss = np.mean(critic_loss_list)
        self.critic_grad = np.mean(critic_grad_list)
        self.entropy_mean = np.mean(entropy_list)
        self.ratio_mean = np.mean(ratio_list)
        self.pre_clip_critic_grad = np.mean(pre_clip_critic_grad)
        self.pre_clip_actor_grad = np.mean(pre_clip_actor_grad)
        
        #  记录 active 的 advantage 均值
        active_sum_total = active_masks.sum().item()
        if active_sum_total > 0:
            self.advantage = (advantage.abs() * active_masks).sum().item() / active_sum_total
        else:
            self.advantage = 0
        
        #  汇总新指标
        self.approx_kl = np.mean(kl_list)
        self.clip_frac = np.mean(clip_frac_list)
        #  计算分项 Entropy 均值
        self.entropy_cont = np.mean(entropy_cont_list) if len(entropy_cont_list) > 0 else 0
        self.entropy_cat = np.mean(entropy_cat_list) if len(entropy_cat_list) > 0 else 0
        self.entropy_bern = np.mean(entropy_bern_list) if len(entropy_bern_list) > 0 else 0
        
        # [新增] 汇总监控项
        self.grad_norm_ratio = np.mean(grad_norm_ratio_list) if len(grad_norm_ratio_list) > 0 else 0
        
        # [新增] 赋值有效样本监控项
        self.PPO_samples = ppo_samples_total
        self.PPO_valid_samples = ppo_valid_samples_total
        
        #  计算 Explained Variance
        # y_true: td_target, y_pred: v_pred_old (更新前的值) 或 v_pred (更新后的值，通常用更新前比较多，或者直接对比)
        # 这里使用 numpy 计算以防 tensor 维度广播问题
        #  explained_var 最好也只看 active 的，但为了简单起见，这里先保持原样或简单过滤
        mask_bool = active_masks.squeeze(-1).bool().cpu().numpy()
        y_true = td_target.flatten().cpu().numpy()[mask_bool]
        y_pred = v_pred_old.flatten().cpu().numpy()[mask_bool] # 比较更新前的 Value 网络预测能力
        
        # if len(y_true) > 1:
        #     var_y = np.var(y_true)
        #     if var_y == 0:
        #         self.explained_var = np.nan
        #     else:
        #         self.explained_var = 1 - np.var(y_true - y_pred) / var_y
        # else:
        #     self.explained_var = 0

        if len(y_true) > 1:
            var_y = np.var(y_true)
            self.td_error_var = np.var(y_true - y_pred) # [新增] TD error 方差
            if var_y < 1e-8:
                self.explained_var = 0.0
            else:
                self.explained_var = 1 - self.td_error_var / var_y
        else:
            self.td_error_var = 0.0
            self.explained_var = 0.0

        check_weights_bias_nan(self.actor, "actor", "update后")
        check_weights_bias_nan(self.critic, "critic", "update后")

    # =========================================================================
    #  [New Helper] 提取出的 MSE 计算逻辑 (供 mixed_update 和 BC_update 复用)
    # =========================================================================
    def _compute_mse_loss_with_f(self, actor_input_batch, actions_batch, returns_batch, 
                                 critic_s_batch, # 用于计算 V
                                 max_weight=100.0, use_F=True):
        """
        计算基于 F 函数 (Advantage > 0) 门控的 MSE Loss。
        """
        # 1. 计算 F 函数 (优势加权门控)
        with torch.no_grad():
            # Critic 计算 V 值
            # 注意：如果 actor_input 和 critic_input 不一样 (如 obs vs state)，这里需要传入正确的 tensor
            # 在本类中，通常传入的是用于 Actor 的输入，如果 Actor 用 Obs，Critic 用 State，
            # 外部调用时需确保 critic_s_batch 是正确的全局 State 或与 Actor 输入一致(视具体配置)
            # 这里为了通用性，假设 critic_s_batch 是正确的 Critic 输入
            v_pred = self.critic(critic_s_batch)
            
            # 计算优势
            # 确保 returns_batch 是 (Batch, 1)
            adv = (returns_batch - v_pred) / (torch.sqrt(self.c_sq) + 1e-8)

            if use_F:
                # F 函数：只有 Adv > 0 的样本权重为 1，否则为极小值
                F_mask = torch.where(adv > 0, torch.ones_like(adv), torch.full_like(adv, 1e-6))
                F_mask = torch.clamp(F_mask, max=max_weight)
            else:
                F_mask = torch.ones_like(adv)

        # 2. 获取网络原始输出
        outputs = self.actor.net(actor_input_batch)
        
        # 初始化 Loss Sum (Batch, 1)
        mse_loss_sum = torch.zeros_like(adv)

        # --- Continuous MSE ---
        if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
            mu_current, _ = outputs['cont']
            u_expert = actions_batch['cont']
            mse_loss_sum += F.mse_loss(mu_current, u_expert, reduction='none').sum(dim=-1, keepdim=True)

        # --- Categorical MSE ---
        if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
            cat_probs_current = outputs['cat']
            expert_cat = actions_batch['cat'].long()
            for i, probs in enumerate(cat_probs_current):
                target_one_hot = F.one_hot(expert_cat[:, i], num_classes=probs.size(-1)).float()
                mse_loss_sum += F.mse_loss(probs, target_one_hot, reduction='none').sum(dim=-1, keepdim=True)

        # --- Bernoulli MSE ---
        if 'bern' in self.actor.action_dims and self.actor.action_dims['bern'] > 0:
            bern_logits = outputs['bern']
            bern_probs = torch.sigmoid(bern_logits)
            target_bern = actions_batch['bern']
            mse_loss_sum += F.mse_loss(bern_probs, target_bern, reduction='none').sum(dim=-1, keepdim=True)

        # 3. 应用 FMask 并求平均
        # F_mask: [Batch, 1], mse_loss_sum: [Batch, 1]
        if use_F:
            final_loss = torch.mean(F_mask * mse_loss_sum)
        else:
            final_loss = torch.mean(mse_loss_sum)
        
        return final_loss, F_mask

    
    # =========================================================================
    #  [New Method] BC_update (Critic 同 MARWIL, Actor 用 MSE+F)
    # =========================================================================
    def BC_update(self, il_transition_dict, batch_size=64, c_v=1.0, shuffled=1, max_weight=100.0):
        """
        行为克隆更新 (Behavior Cloning with F-function Constraint)。
        Critic: 使用 MARWIL 风格的回归更新 (拟合 R)。
        Actor: 使用 MSE 回归，但仅在 Advantage > 0 时通过 F 函数生效。
        """
        # 1. 数据准备
        # 可能的局部观测
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 冻结分布参数，只训练均值/Logits
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 处理 Actions
        raw_actions = il_transition_dict['actions']
        actions_all = {}
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)

        # 2. 索引准备
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_valid_samples = 0
        batch_count = 0

        # 初始化 c_sq 用于 Advantage 归一化 (如果不存在)
        if not hasattr(self, 'c_sq'): 
            self.c_sq = torch.tensor(1.0, device=self.device)

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices] 
            r_batch = returns_all[batch_indices]
            
            if use_obs:
                actor_input_batch = obs_all[batch_indices]
            else:
                actor_input_batch = s_batch
            
            # Critic input (通常 state)
            critic_input_batch = s_batch

            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # --- A. Actor Loss (MSE + F) ---
            # 使用 helper 函数计算
            actor_loss, F_mask = self._compute_mse_loss_with_f(
                actor_input_batch, actions_batch, r_batch, critic_input_batch, max_weight, use_F=0
            )

            # --- B. Critic Loss (同 MARWIL) ---
            v_pred = self.critic(critic_input_batch)
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            
            # --- C. Optimize ---
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss.backward()
            critic_loss.backward()
            
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_valid_samples += F_mask.sum().item()
            batch_count += 1

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, total_valid_samples
    
    
    
    # --- 修改后的 MARWIL_update， 注意原先是0 ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.1, max_weight=100.0,
                      tau=0.8, no_bern=1):
        """
        MARWIL 离线更新函数
        输入 actions 结构支持: [{'cat': array([v]), 'bern': array([v])}, ...]
        tau: 非对称损失权重 (Expectile Regression). tau=0.5 为 MSE; tau>0.5 (如0.9) 倾向于高估 Value (拟合好样本)
        """
        # 1. 数据准备
        if 'obs' in il_transition_dict and len(il_transition_dict['obs']) > 0:
            obs_all = torch.tensor(np.array(il_transition_dict['obs']), dtype=torch.float).to(self.device)
            use_obs = True
        else:
            use_obs = False
            
        # 预训练阶段通常不训练探索 std
        if hasattr(self.actor.net, 'log_std_cont'):
            self.actor.net.log_std_cont.requires_grad = False

        # 1. 提取全量数据并转为 Tensor
        states_all = torch.tensor(np.array(il_transition_dict['states']), dtype=torch.float).to(self.device)
        returns_all = torch.tensor(np.array(il_transition_dict['returns']), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 统一处理 Actions：List of Dicts -> Dict of Tensors
        raw_actions = il_transition_dict['actions']
        actions_all = {}
        
        # 1. 如果是列表 (List of Dicts)，先堆叠成 Dict of Numpy Arrays
        if isinstance(raw_actions, list):
            keys = raw_actions[0].keys()
            temp_dict = {}
            for k in keys:
                # np.stack 会把 [array([1]), array([2])] 变成 array([[1], [2]]) -> (N, 1)
                temp_dict[k] = np.stack([d[k] for d in raw_actions], axis=0)
            raw_actions = temp_dict # 现在变成了 Dict of Arrays

        # 2. Dict of Arrays -> Dict of Tensors
        if isinstance(raw_actions, dict):
            for k, v in raw_actions.items():
                if k == 'cat':
                    actions_all[k] = torch.tensor(v, dtype=torch.long).to(self.device)
                else:
                    actions_all[k] = torch.tensor(v, dtype=torch.float).to(self.device)
        # ============================================================

        # 2. 准备 Batch 索引
        total_size = states_all.size(0)
        indices = np.arange(total_size)
        if shuffled:
            np.random.shuffle(indices)

        total_actor_loss = 0
        total_critic_loss = 0
        total_c = 0
        batch_count = 0

        # [新增] 权重与 advantage 监控累加器
        total_weight_mean = 0.0
        total_weight_max = 0.0
        total_weight_min = 0.0
        total_clip_frac = 0.0
        total_adv_std = 0.0
        total_adv_p95 = 0.0
        total_adv_max = 0.0
        total_adv_mean = 0.0

        # 3. Mini-batch 循环
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            s_batch = states_all[batch_indices] 
            r_batch = returns_all[batch_indices]
            
            if use_obs:
                actor_input_batch = obs_all[batch_indices]
            else:
                actor_input_batch = s_batch 
            
            # 动作字典切片
            actions_batch = {}
            for k, v in actions_all.items():
                actions_batch[k] = v[batch_indices]

            # A. Advantage & Weights
            with torch.no_grad():
                values = self.critic(s_batch)
                residual = r_batch - values
                
                if not hasattr(self, 'c_sq'): 
                    self.c_sq = torch.tensor(1.0, device=self.device)
                
                batch_mse = (residual ** 2).mean().item()
                self.c_sq = self.c_sq + 1e-8 * (batch_mse - self.c_sq)
                c = torch.sqrt(self.c_sq)
                
                advantage = residual / (c + 1e-8)
                raw_weights = torch.exp(beta * advantage)
                weights = torch.clamp(raw_weights, max=max_weight)

                # [新增] 记录本 batch 的权重统计
                total_weight_mean += weights.mean().item()
                total_weight_max += weights.max().item()
                total_weight_min += weights.min().item()
                total_clip_frac += (weights >= max_weight - 1e-6).float().mean().item()

                # [新增] 记录本 batch advantage 分布统计
                adv = advantage.detach()
                total_adv_std += adv.std().item()
                total_adv_p95 += torch.quantile(adv, 0.95).item()
                total_adv_max += adv.max().item()
                total_adv_mean += adv.mean().item()

            # B. Actor Loss
            raw_il_loss = self.actor.compute_il_loss(actor_input_batch, actions_batch, label_smoothing, no_bern=no_bern)
            actor_loss = torch.mean(alpha * weights * raw_il_loss)

            # C. Critic Loss
            v_pred = self.critic(s_batch)
            
            # 原有
            critic_loss = F.mse_loss(v_pred, r_batch) * c_v
            


            # D. Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.critic_max_grad)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_c += c.item()
            batch_count += 1

        avg_weight_mean = total_weight_mean / batch_count if batch_count > 0 else 0
        avg_weight_max = total_weight_max / batch_count if batch_count > 0 else 0
        avg_weight_min = total_weight_min / batch_count if batch_count > 0 else 0
        avg_clip_frac = total_clip_frac / batch_count if batch_count > 0 else 0
        avg_adv_std = total_adv_std / batch_count if batch_count > 0 else 0
        avg_adv_p95 = total_adv_p95 / batch_count if batch_count > 0 else 0
        avg_adv_max = total_adv_max / batch_count if batch_count > 0 else 0
        avg_adv_mean = total_adv_mean / batch_count if batch_count > 0 else 0

        avg_actor_loss = total_actor_loss / batch_count if batch_count > 0 else 0
        avg_critic_loss = total_critic_loss / batch_count if batch_count > 0 else 0
        avg_c = total_c / batch_count if batch_count > 0 else 0

        # ============================================================
        # [新增] 监控：在固定的全量 batch 上独立统计每个动作头的 NLL 与熵
        # 全程 no_grad，不参与反传，因此不会干扰现有的网络更新。
        # ============================================================
        if use_obs:
            monitor_input_all = obs_all
        else:
            monitor_input_all = states_all
        with torch.no_grad():
            values_all = self.critic(states_all)
            residual_all = returns_all - values_all
            advantage_all = residual_all / (torch.sqrt(self.c_sq) + 1e-8)
        monitor_metrics = self.actor.compute_marwil_monitor(monitor_input_all, actions_all, advantages=advantage_all)
        # 缓存到 agent 属性，便于训练脚本拉取写入 logger
        self.marwil_nll_cont = monitor_metrics['nll_cont']
        self.marwil_nll_cat = monitor_metrics['nll_cat']
        self.marwil_nll_bern = monitor_metrics['nll_bern']
        self.marwil_entropy_cont = monitor_metrics['entropy_cont']
        self.marwil_entropy_cat = monitor_metrics['entropy_cat']
        self.marwil_entropy_bern = monitor_metrics['entropy_bern']
        self.marwil_accuracy_cont = monitor_metrics['accuracy_cont']
        self.marwil_accuracy_cat = monitor_metrics['accuracy_cat']
        self.marwil_accuracy_bern = monitor_metrics['accuracy_bern']
        self.marwil_weight_mean = avg_weight_mean
        self.marwil_weight_max = avg_weight_max
        self.marwil_weight_min = avg_weight_min
        self.marwil_weight_clip_frac = avg_clip_frac
        self.marwil_adv_std = avg_adv_std
        self.marwil_adv_p95 = avg_adv_p95
        self.marwil_adv_max = avg_adv_max
        self.marwil_adv_mean = avg_adv_mean
        self.marwil_adv_positive_frac = monitor_metrics['adv_positive_frac']

        return avg_actor_loss, avg_critic_loss, avg_c
    
    def Stir(self, transition_dict, target_entropies, max_steps=50, lr=0.05):
        """
        策略搅拌：深拷贝网络并使用纯熵正则项更新，将策略分布推平到目标熵，
        同时尽可能保持原有动作的相对优先级排序不变。
        
        Args:
            transition_dict: 同 update 方法的输入，用于提供计算图所需的 states (和 mask)
            target_entropies: dict, 例如 {'cont': 2.0, 'cat': 1.5, 'bern': 0.6}
            max_steps: 最大允许的搅拌迭代次数
            lr: SGD 学习率 (不宜过大，否则容易导致离散动作排序翻转)
            
        Returns:
            stirred_state_dict: 搅拌后的独立网络参数字典，与原计算图解耦
        """
        # 1. 深度拷贝网络，彻底解耦计算图
        stirred_net = copy.deepcopy(self.actor)
        stirred_net.net.train() # 确保开启梯度计算

        # 2. 冻结特征提取层，仅放开动作头和 std 参数
        # 这是为了最大程度保护原始动作提取出的特征表达，防止由于大跨度更新彻底破坏策略逻辑
        for name, param in stirred_net.net.named_parameters():
            if 'fc_' in name or 'log_std' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # 3. 必须使用无动量的 SGD！
        # 如果使用 Adam，动量积攒会导致 Logits 大幅超调，直接破坏原有的动作优先级排序
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, stirred_net.net.parameters()), lr=lr)

        # 4. 解析输入状态 (复用 update 的接口逻辑)
        if isinstance(transition_dict['states'], np.ndarray):
            states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        else:
            states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        

        # 5. 执行搅拌循环
        for step in range(max_steps):
            outputs = stirred_net.net(states)
            
            loss = 0.0
            current_entropies = {}

            # --- Cont ---
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                # 平均到每个样本上
                ent_cont = dist.entropy().mean()
                current_entropies['cont'] = ent_cont.item()
                # 如果当前熵低于目标值，则加上负熵作为 Loss (最小化负熵 = 最大化熵)
                if ent_cont < target_entropies.get('cont', -float('inf')):
                    loss -= ent_cont 

            # --- Cat ---
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                ent_cat = 0.0
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    ent_cat += dist.entropy().mean()
                current_entropies['cat'] = ent_cat.item()
                if ent_cat < target_entropies.get('cat', -float('inf')):
                    loss -= ent_cat

            # --- Bern ---
            # 跳过 Bernoulli 部分的搅拌，避免 mask_on 相关的复杂性

            # 6. 如果没有需要优化的项，直接退出
            if isinstance(loss, float):
                break

            # 7. 反向传播与搅拌更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 8. 更新后检查退出条件：所有指定的动作头都达到了目标熵则停止
            # 注意：必须在 step() 之后检查，避免更新后过冲
            all_met = True
            for key, target in target_entropies.items():
                if current_entropies.get(key, float('inf')) < target:
                    all_met = False
                    break
            if all_met:
                break

        # 8. 计算搅拌后的熵值
        with torch.no_grad():
            # 使用输入的状态计算熵值
            if isinstance(transition_dict['states'], np.ndarray):
                states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
            else:
                states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
            
            outputs = stirred_net.net(states)
            
            # 计算cat熵
            cat_entropy = 0.0
            if 'cat' in self.actor.action_dims and sum(self.actor.action_dims['cat']) > 0:
                cat_probs_list = outputs['cat']
                for probs in cat_probs_list:
                    dist = Categorical(probs=probs)
                    cat_entropy += dist.entropy().mean()
                cat_entropy = cat_entropy.item()
            
            # Bern 熵不计算（搅拌时已跳过）
            bern_entropy = 0.0
            
            # 计算cont熵
            cont_entropy = 0.0
            if 'cont' in self.actor.action_dims and self.actor.action_dims['cont'] > 0:
                mu, std = outputs['cont']
                dist = SquashedNormal(mu, std)
                cont_entropy = dist.entropy().mean().item()
        
        # 9. 转换为 CPU 张量并 clone，彻底切断与本次优化过程及 GPU 的羁绊
        # stirred_net 本身是 HybridActorWrapper 的深拷贝，直接取其 state_dict 即可与外部保存/加载保持一致
        stirred_state_dict = {k: v.cpu().clone() for k, v in stirred_net.state_dict().items()}
        
        # 返回搅拌后的状态字典和熵值信息
        entropy_info = {
            'cat_entropy': cat_entropy,
            'bern_entropy': bern_entropy,
            'cont_entropy': cont_entropy
        }
        
        return stirred_state_dict, entropy_info