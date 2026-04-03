'''
混合动作空间空间的PPO改为SAC
'''
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import collections
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet


class ReplayBufferHybrid:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action_dict, reward, next_state, done):
        self.buffer.append((state, action_dict, reward, next_state, done))

    def sample(self, batch_size):
        # 1. 随机抽样
        transitions = random.sample(self.buffer, batch_size)
        
        # 2. 解包
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 3. 规整动作字典 (List[Dict] -> Dict[Array])
        actions_dict_np = {}
        if len(actions) > 0:
            for key in actions[0].keys():
                actions_dict_np[key] = np.array([act[key] for act in actions])

        # 4. 【核心】：打包成一个干净的字典返回
        # 此时所有值都是已经对齐好的 NumPy Array
        batch_dict = {
            'states': np.array(states, dtype=np.float32),
            'actions': actions_dict_np,
            'rewards': np.array(rewards, dtype=np.float32).reshape(-1, 1), # 预处理形状
            'next_states': np.array(next_states, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32).reshape(-1, 1)      # 预处理形状
        }
        
        return batch_dict

    def size(self):
        return len(self.buffer)

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络 (纯 MLP)。
    引入了可学习的温度参数来控制离散和伯努利动作的熵。
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

        # 1. 连续动作头 (Continuous)
        # 参数: log_std (控制高斯分布宽度)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.fc_mu = nn.Linear(prev_size, cont_dim)
            # 这里的 log_std 依然是状态无关的，对应 PPO 的标准做法
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

        # 2. 离散动作头 (Categorical)
        # 参数: log_temp_cat (控制 Softmax 温度)
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = self.action_dims['cat']  # list, e.g., [4, 10]
            total_cat_dim = sum(self.cat_dims)
            self.fc_cat = nn.Linear(prev_size, total_cat_dim)
            
            # 为每一个独立的离散头 (Head) 创建一个温度参数
            # 比如有 [4, 10] 两个头，我们就需要 2 个温度参数
            # 初始化为 0 (即 temp=1.0)，保持原网络特性，让网络自己学去增大熵
            # self.log_temp_cat = nn.Parameter(torch.zeros(len(self.cat_dims))) 

        # 3. 伯努利动作头 (Bernoulli)
        # 参数: log_temp_bern (控制 Sigmoid 陡峭度)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            # 原·单层输出
            self.fc_bern = nn.Linear(prev_size, bern_dim)
            # # 现·多层输出
            # layers = []
            # for _ in range(1):
            #     layers.append(nn.Linear(prev_size, 64))
            #     layers.append(nn.ReLU())
            #     prev_size = 64
            # layers.append(nn.Linear(prev_size, bern_dim))
            # self.fc_bern = nn.Sequential(*layers)

            # 初始化 bias 为 -2，使初始开火概率较低（sigmoid(-2) ≈ 0.12）
            nn.init.constant_(self.fc_bern.bias, -2.0) # 原·单层输出
            # nn.init.constant_(self.fc_bern[-1].bias, -2.0) # 现·多层输出
            
            # 为每一个伯努利动作维度创建一个温度参数
            # 初始化为 0 (即 temp=1.0)
            # self.log_temp_bern = nn.Parameter(torch.zeros(bern_dim))
    
    # [修改] 增加 action_masks 参数, [新增] 增加 temp 参数
    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temp=1.0):
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
            # [新增] Action Masking 逻辑
            # 如果提供了 mask，将 mask 为 0 (False) 的位置的 logit 设为 -1e9
            if action_masks is not None and 'bern' in action_masks:
                mask = action_masks['bern']
                # 确保 mask 和 logits 维度匹配 (Batch, Dim)
                # mask == 0 代表禁止开火，设为极小值
                bern_logits = bern_logits.masked_fill(mask == 0, -1e9)

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
    def get_action(self, state, h=None, explore=True, max_std=None, check_obs=None, bern_threshold=0.5, temp=1.0):
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
        2、evaluate_actions（训练/计算 log_prob）默认未把 action_masks 传给 net ,
            若希望训练时也应用 mask，需要在 evaluate_actions 调用 net 时传入 action_masks。
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
                'bern': explore.get('bern', True)
            }
        else:
            # 对于其他意外的输入类型，默认全部探索
            explore_opts = {'cont': True, 'cat': True, 'bern': True}

        # =====================================================================
        # [新增] 解析 check_obs 并构建 Action Mask
        # =====================================================================
        action_masks = None
        can_fire = True
        # 当且仅当传入了单个 dict 类型的 check_obs 时启用 mask, 不受explore影响
        if (check_obs is not None) and isinstance(check_obs, dict):  # and (not explore_opts['bern']):
            # 默认允许开火，下面按规则逐项收敛（保留注释）
            can_fire = True
            # 如果是Batch训练模式，通常check_obs会增加维度，这里只在推理的时候启用

            # 1. ATA <= 60度 (0.5236 rad)
            ata_hor = np.arccos(check_obs["target_information"][0])
            ata = check_obs["target_information"][4]
            ata_condition = (ata <= 60 * np.pi / 180 and ata_hor <= 20 * np.pi / 180)
            # [新增] ata_hor 是第一个漂亮结果后新增的mask项
            can_fire = can_fire and ata_condition

            # 2. Target Locked == 1
            locked = check_obs["target_locked"]
            locked_condition = (locked == 1)
            can_fire = can_fire and locked_condition

            # 3. Ammo > 0 (ego_main 最后一个元素是 ammo)
            ammo = check_obs["ego_main"][6]
            ammo_condition = (ammo > 0)
            can_fire = can_fire and ammo_condition

            # 4. 超远距离尾追不打（使用 AA_hor 判断尾追）
            distance = check_obs["target_information"][3]
            AA_hor = check_obs["target_information"][6]
            if (distance > 30e3) and (abs(AA_hor) < np.pi/6):
                can_fire = False

            # 5. 30km 外12s内禁止重复发射第二枚 或 mid-term 有在飞导弹
            # weapon 计时单位兼容原逻辑
            if (distance > 30e3 and check_obs["weapon"] * 120 < 12) or check_obs.get("missile_in_mid_term", False):
                can_fire = False

            # 构建 Tensor Mask: (Batch_Size, Bern_Dim) -> (1, 1)
            # 1.0 表示允许 (保留 Logits)，0.0 表示禁止 (Logits -> -inf)
            mask_val = 1.0 if can_fire else 0.0
            
            # 适配 state 的 batch size
            batch_size = state.size(0)
            mask_tensor = torch.full((batch_size, 1), mask_val, device=self.device, dtype=torch.float)
            
            action_masks = {'bern': mask_tensor}
        # =====================================================================

        # [修改] 调用网络时传入 action_masks 和 temp
        actor_outputs = self.net(state, max_std=max_std, action_masks=action_masks, temp=temp)
        
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

    def sample_for_sac(self, states, action_masks=None):
        """
        专门为 SAC 提供的采样方法。
        返回可导的 actions，以及总的 log_prob。
        """
        actor_outputs = self.net(states, action_masks=action_masks)
        
        actions_differentiable = {}
        log_probs = torch.zeros(states.size(0), 1).to(self.device)

        # --- Cont (连续动作，使用 rsample) ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            # 注意：此处需确保 SquashedNormal 支持 rsample 并且正确计算了 tanh 的 log_prob
            dist = Normal(mu, std)
            u = dist.rsample() # 重参数化采样
            a_norm = torch.tanh(u)
            # 计算 Squash 的 log_prob
            log_prob_cont = dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-7)
            log_probs += log_prob_cont.sum(-1, keepdim=True)
            
            actions_differentiable['cont'] = a_norm # 直接输出 -1~1 的范围给 Q 网络

        # --- Cat (离散动作，使用 Gumbel-Softmax) ---
        if actor_outputs['cat'] is not None:
            cat_logits_list = actor_outputs['cat'] # 假设 net 稍作修改，返回了未经过 softmax 的 logits
            cat_actions = []
            for logits in cat_logits_list:
                # hard=True 表示前向传播输出 One-hot(例如[0,1,0])，反向传播用 softmax 的梯度
                gumbel_out = F.gumbel_softmax(logits, tau=1.0, hard=True)
                cat_actions.append(gumbel_out)
                
                # 计算 log_prob (近似)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                # 由于 hard=True 返回的是 one-hot，可以通过与 log_probs 相乘来提取选中项的 log_prob
                log_p = torch.sum(torch.log(probs + 1e-8) * gumbel_out, dim=-1, keepdim=True)
                log_probs += log_p
                
            actions_differentiable['cat'] = torch.cat(cat_actions, dim=-1)

        # --- Bern (伯努利动作，使用 Binary Gumbel-Softmax / 缓和的 Sigmoid) ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            # 将 logits 转换为 [prob_0, prob_1] 的形式以便使用 gumbel_softmax
            logits_2d = torch.stack([-bern_logits, bern_logits], dim=-1) 
            gumbel_out = F.gumbel_softmax(logits_2d, tau=1.0, hard=True)
            bern_action = gumbel_out[..., 1] # 取出代表 1(True) 的那一列
            
            actions_differentiable['bern'] = bern_action
            
            probs = torch.sigmoid(bern_logits)
            # 同样提取对应的 log_prob
            log_p = torch.log(probs + 1e-8) * bern_action + torch.log(1 - probs + 1e-8) * (1 - bern_action)
            log_probs += log_p.view(states.size(0), -1).sum(-1, keepdim=True)

        return actions_differentiable, log_probs

class QNetHybrid(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dims_dict):
        super(QNetHybrid, self).__init__()
        
        # 计算所有动作展平后的总维度
        act_dim = 0
        if 'cont' in action_dims_dict: act_dim += action_dims_dict['cont']
        if 'cat' in action_dims_dict: act_dim += sum(action_dims_dict['cat']) # Cat 需要转为 One-hot 输入
        if 'bern' in action_dims_dict: act_dim += action_dims_dict['bern']
        
        layers = []
        prev_size = state_dim + act_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, 1)

    def forward(self, state, action_dict):
        # 拼接动作
        actions_list = []
        if 'cont' in action_dict and action_dict['cont'] is not None:
            actions_list.append(action_dict['cont'])
        if 'cat' in action_dict and action_dict['cat'] is not None:
            actions_list.append(action_dict['cat']) # 这里必须已经是 one-hot 或 gumbel-softmax 的输出
        if 'bern' in action_dict and action_dict['bern'] is not None:
            actions_list.append(action_dict['bern'])
            
        action_cat = torch.cat(actions_list, dim=-1)
        x = torch.cat([state, action_cat], dim=-1)
        return self.fc_out(self.net(x))

# =============================================================================
# 3. SAC 算法类 (精简版)
# =============================================================================
class SACHybrid:
    def __init__(self, actor, critic_1, critic_2, target_critic_1, target_critic_2, 
                 actor_lr, critic_lr, alpha_lr, action_dims_dict,
                 gamma, tau, device):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        
        # 初始化目标网络
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # 自动调节温度参数 Alpha
        # 针对 Hybrid，可以设一个全局 Alpha，也可以为 cont, cat, bern 各设一个。这里用一个全局的演示。
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # Target Entropy: 启发式设为动作维度的负数
        total_action_dim = sum(action_dims_dict.values()) if not isinstance(action_dims_dict['cat'], list) else action_dims_dict['cont'] + len(action_dims_dict['cat']) + action_dims_dict['bern']
        self.target_entropy = -total_action_dim 
        
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, explore=True, check_obs=None):
        # 推理时仍然使用你原来的 get_action，因为原来的 get_action 用于环境交互，包含了动作还原
        return self.actor.get_action(state, explore=explore, check_obs=check_obs)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, batch):
        """
        接收 ReplayBuffer 返回的字典 batch
        """
        # --- A. 数据搬运与类型转换 (NumPy -> Tensor) ---
        device = self.device
        
        states = torch.from_numpy(batch['states']).to(device)
        next_states = torch.from_numpy(batch['next_states']).to(device)
        rewards = torch.from_numpy(batch['rewards']).to(device)
        dones = torch.from_numpy(batch['dones']).to(device)
        
        # 处理动作 (从字典中提取并转为 Tensor)
        raw_actions = batch['actions']
        actions_for_q = {}
        
        if 'cont' in raw_actions:
            actions_for_q['cont'] = torch.from_numpy(raw_actions['cont']).to(device)
            
        if 'cat' in raw_actions:
            cat_idx = torch.from_numpy(raw_actions['cat']).to(device).long()
            # 转换为 One-hot 供 Q 网络输入
            cat_onehots = []
            for i, dim in enumerate(self.actor.action_dims['cat']):
                cat_onehots.append(F.one_hot(cat_idx[:, i], num_classes=dim).float())
            actions_for_q['cat'] = torch.cat(cat_onehots, dim=-1)
            
        if 'bern' in raw_actions:
            actions_for_q['bern'] = torch.from_numpy(raw_actions['bern']).to(device)

        # --- B. SAC 计算逻辑 (逻辑保持不变，但变量名已对齐) ---
        
        # 1. 更新 Q 网络 (Critic)
        with torch.no_grad():
            # 获取下一状态的动作 (可导采样) 和 log_prob
            next_actions_diff, next_log_probs = self.actor.sample_for_sac(next_states)
            
            # 目标 Q 值
            q1_target = self.target_critic_1(next_states, next_actions_diff)
            q2_target = self.target_critic_2(next_states, next_actions_diff)
            min_q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_log_probs
            
            # TD 目标
            y_target = rewards + self.gamma * (1 - dones) * min_q_target
            
        # 当前 Q 值预测
        q1_pred = self.critic_1(states, actions_for_q)
        q2_pred = self.critic_2(states, actions_for_q)
        
        critic_loss = F.mse_loss(q1_pred, y_target) + F.mse_loss(q2_pred, y_target)
        
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # 2. 更新 策略网络 (Actor)
        # 重新对当前状态采样
        curr_actions_diff, curr_log_probs = self.actor.sample_for_sac(states)
        
        q1_pi = self.critic_1(states, curr_actions_diff)
        q2_pi = self.critic_2(states, curr_actions_diff)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.log_alpha.exp().detach() * curr_log_probs - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. 更新 Alpha (熵系数)
        alpha_loss = -(self.log_alpha * (curr_log_probs.detach() + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 4. 目标网络软更新
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        # 保存监控指标
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_entropy = -curr_log_probs.mean().item()