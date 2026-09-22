'''
混合动作空间空间的PPO改为SAC
'''
import random
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli
import collections
import copy
import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from Algorithms.Utils import model_grad_norm, check_weights_bias_nan, compute_advantage, SquashedNormal
from Algorithms.MLP_heads import ValueNet

def sigmoid(x):
    return 1/(1+np.exp(-x))

class ReplayBufferHybrid:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def add(self, state, action_dict, reward, next_state, done, active_mask=1.0):
        # active_mask: 智能体存活=1，死亡后=0，死亡样本不参与任何反向传播
        self.buffer.append((state, action_dict, reward, next_state, done, active_mask))

    def save(self, path):
        """持久化经验池内容，支持中断续训。"""
        torch.save({'capacity': self.capacity, 'data': list(self.buffer)}, path)
        print(f"[ReplayBufferHybrid] Saved to {path}. Size: {len(self.buffer)}")

    @staticmethod
    def load(path, map_location='cpu'):
        """从磁盘读取经验池，找不到则返回 None。"""
        if not os.path.exists(path):
            return None
        ckpt = torch.load(path, map_location=map_location)
        buf = ReplayBufferHybrid(ckpt['capacity'])
        buf.buffer = collections.deque(ckpt['data'], maxlen=ckpt['capacity'])
        print(f"[ReplayBufferHybrid] Loaded from {path}. Size: {len(buf.buffer)}")
        return buf

    def sample(self, batch_size):
        # 1. 随机抽样
        transitions = random.sample(self.buffer, batch_size)
        
        # 2. 解包（兼容旧的 5 元组存档：无 active_mask 时默认 1.0=存活）
        states, actions, rewards, next_states, dones, active_masks = [], [], [], [], [], []
        for t in transitions:
            states.append(t[0])
            actions.append(t[1])
            rewards.append(t[2])
            next_states.append(t[3])
            dones.append(t[4])
            active_masks.append(t[5] if len(t) > 5 else 1.0)
        
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
            'dones': np.array(dones, dtype=np.float32).reshape(-1, 1),     # 预处理形状
            'active_masks': np.array(active_masks, dtype=np.float32).reshape(-1, 1)  # 死亡样本=0
        }
        
        return batch_dict

    def size(self):
        return len(self.buffer)


class SupervisedFireBuffer:
    """监督记忆 M_SL：每条记录只保存 fire_obs 和 hit_record。

    deque(maxlen=capacity) 保证达到容量后逐条 FIFO 覆盖，永不整体清空。
    """
    def __init__(self, capacity=10000):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def add(self, fire_obs, hit_record):
        fire_obs = np.asarray(fire_obs, dtype=np.float32).copy()
        hit_record = float(hit_record)
        if hit_record not in (0.0, 1.0):
            raise ValueError(f"fire-hit label must be 0/1, got {hit_record}")
        self.buffer.append({
            'fire_obs': fire_obs,
            'hit_record': hit_record,
        })

    def extend(self, records):
        for record in records or []:
            fire_obs = record.get('fire_obs')
            hit_record = record.get('hit_record')
            if fire_obs is None or hit_record is None:
                continue
            self.add(fire_obs, hit_record)

    def sample(self, batch_size):
        if not self.buffer:
            raise ValueError('SupervisedFireBuffer is empty.')
        samples = random.sample(list(self.buffer), min(int(batch_size), len(self.buffer)))
        return {
            'states': np.asarray([x['fire_obs'] for x in samples], dtype=np.float32),
            'labels': np.asarray([x['hit_record'] for x in samples], dtype=np.float32).reshape(-1, 1),
        }

    def size(self):
        return len(self.buffer)

    def save(self, path):
        torch.save({'capacity': self.capacity, 'data': list(self.buffer)}, path)
        print(f"[SupervisedFireBuffer] Saved to {path}. Size: {self.size()}")

    @staticmethod
    def load(path, map_location='cpu', capacity=None):
        if not os.path.exists(path):
            return None
        ckpt = torch.load(path, map_location=map_location)
        buf = SupervisedFireBuffer(int(capacity) if capacity is not None else ckpt['capacity'])
        # 兼容旧版缓存，但加载后统一收敛为两个字段。
        for item in ckpt.get('data', []):
            fire_obs = item.get('fire_obs', item.get('state'))
            hit_record = item.get('hit_record', item.get('label'))
            if fire_obs is not None and hit_record is not None:
                buf.add(fire_obs, hit_record)
        print(f"[SupervisedFireBuffer] Loaded from {path}. Size: {buf.size()}")
        return buf

# =============================================================================
# 1. 神经网络定义 (保持不变，只负责 forward 计算)
# =============================================================================


# ============================================================
# 多 cat 维度展平/还原辅助函数
# ============================================================
def _make_strides(cat_dims):
    if not cat_dims:
        return []
    m = len(cat_dims)
    strides = [1] * m
    for d in range(m - 2, -1, -1):
        strides[d] = strides[d + 1] * cat_dims[d + 1]
    return strides


def ravel_cat_indices(cat_indices, cat_dims):
    """多 dim cat 索引 (B, m) -> 展平联合索引 (B, 1)"""
    if not cat_dims:
        return torch.zeros(cat_indices.size(0), 1, dtype=torch.long, device=cat_indices.device)
    strides = _make_strides(cat_dims)
    joint = torch.zeros(cat_indices.size(0), 1, dtype=torch.long, device=cat_indices.device)
    for d, s in enumerate(strides):
        joint += cat_indices[:, d:d+1] * s
    return joint


def unravel_cat_index(joint, cat_dims):
    """展平联合索引 (B, 1) -> 多 dim cat 索引 (B, m)"""
    if not cat_dims:
        return torch.empty(joint.size(0), 0, dtype=torch.long, device=joint.device)
    strides = _make_strides(cat_dims)
    m = len(cat_dims)
    indices = torch.empty(joint.size(0), m, dtype=torch.long, device=joint.device)
    for d, (K, s) in enumerate(zip(cat_dims, strides)):
        indices[:, d] = ((joint // s) % K).squeeze(-1)
    return indices


def per_dim_onehot_to_indices(cat_onehot, cat_dims):
    """把拼接的 per-dim one-hot (B, sum K_i) 还原成索引 (B, m)"""
    if not cat_dims:
        return torch.empty(cat_onehot.size(0), 0, dtype=torch.long, device=cat_onehot.device)
    splits = torch.split(cat_onehot, cat_dims, dim=-1)
    indices = [split.argmax(dim=-1, keepdim=True) for split in splits]
    return torch.cat(indices, dim=-1)


def joint_cat_prob_from_list(probs_list):
    """由每个 cat 维度的概率 (B, K_i) 计算联合概率 (B, K1*K2*...)"""
    if not probs_list:
        return None
    prob = probs_list[0]
    for p in probs_list[1:]:
        prob = (prob.unsqueeze(-1) * p.unsqueeze(1)).view(prob.size(0), -1)
    return prob


class QNetHybrid(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dims_dict):
        super(QNetHybrid, self).__init__()
        
        # cat 展平为联合输出；cont 作为输入条件，bern/fire head 不进入 critic
        self.cont_dim = int(action_dims_dict.get('cont', 0))
        # 论文的 RL memory / discrete SAC 只优化 BFM；fire head 只由 M_SL+BCE 更新。
        # 因此 fire action 不作为 critic 输入，避免稀疏命中信号被错误地当作 SAC 动作价值学习。
        self.bern_dim = 0
        self.cat_dims = list(action_dims_dict.get('cat', []))
        self.joint_cat_dim = int(np.prod(self.cat_dims, dtype=np.int64)) if self.cat_dims else 1
        
        prev_size = state_dim + self.cont_dim + self.bern_dim
        layers = []
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_size, self.joint_cat_dim)

    def forward(self, state, action_dict):
        # 只把 cont 作为输入；cat 不同取值的 Q 由输出头给出
        parts = [state]
        if 'cont' in action_dict and action_dict['cont'] is not None:
            parts.append(action_dict['cont'])
        else:
            if self.cont_dim > 0:
                parts.append(torch.zeros(state.size(0), self.cont_dim, device=state.device, dtype=state.dtype))
        x = torch.cat(parts, dim=-1)
        return self.fc_out(self.net(x))  # (B, K1*K2*...)

from Algorithms.PPOHybrid23_0 import load_mask_config, HybridActorWrapper as PPOHybridActorWrapper

# =============================================================================
# 1. Policy 网络 (用于 SAC / TD3，支持 warning 状态下的动作 mask)
# =============================================================================

class PolicyNetHybrid(torch.nn.Module):
    """
    支持混合动作空间的策略网络 (纯 MLP)。
    引入了可学习的温度参数来控制离散和伯努利动作的熵。
    """
    def __init__(self, state_dim, hidden_dims, action_dims_dict, init_std=0.5, head_hidden_layer_num=1, Autoregressive=0, mask_cfg=None):
        super(PolicyNetHybrid, self).__init__()
        self.action_dims = action_dims_dict

        # [新增] 机动mask 开关：只在网络初始化时从
        # mask_config.json 读取一次（或被外部显式传入），永久保存为实例属性，forward() 不再重复读取磁盘。
        mask_cfg = load_mask_config(override=mask_cfg)
        self.ver_map = mask_cfg['ver']
        self.ver_mask = mask_cfg['ver']
        self.hor_map = mask_cfg['hor']
        self.hor_mask = mask_cfg['hor']

        backbone_input_dim = state_dim

        # 共享主干网络
        layers = []
        prev_size = backbone_input_dim
        for layer_size in hidden_dims:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            prev_size = layer_size
        self.net = nn.Sequential(*layers)

        # 1. 连续动作头 (Continuous)
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            cont_dim = self.action_dims['cont']
            self.log_std_cont = nn.Parameter(torch.log(torch.ones(cont_dim) * init_std))

            layers = []
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), cont_dim))
            self.fc_mu = nn.Sequential(*layers)

        # 2. 离散动作头 (Categorical)
        if 'cat' not in self.action_dims:
            self.action_dims['cat'] = []
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            self.cat_dims = list(self.action_dims['cat'])  # list, e.g., [5, 7]
            # [去串扰设计] ver_map/hor_map 分别控制是否启用 13/11 多对一映射；
            # 否则直接按外部维度 self.cat_dims (如 [5, 7] 或 [5, 6]) 构建离散头。
            self.cat_dims_internal = list(self.cat_dims)
            if self.ver_map:
                if len(self.cat_dims_internal) > 0 and self.cat_dims_internal[0] == 5:
                    self.cat_dims_internal[0] = 13
            if self.hor_map:
                if len(self.cat_dims_internal) > 1 and self.cat_dims_internal[1] in (6, 7):
                    self.cat_dims_internal[1] = 11
            total_cat_dim = sum(self.cat_dims_internal)

            layers = []
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), total_cat_dim))
            self.fc_cat = nn.Sequential(*layers)

        # 3. 伯努利动作头 (Bernoulli)
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_dim = self.action_dims['bern']
            layers = []
            layers.append(nn.Linear(prev_size, int(prev_size/2)))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(int(prev_size/2), bern_dim))
            self.fc_bern = nn.Sequential(*layers)
            nn.init.constant_(self.fc_bern[-1].bias, 0.0)

    def forward(self, x, min_std=1e-6, max_std=1.0, action_masks=None, temperature=1.0, mask_on=0):
        ver_mask = self.ver_mask
        hor_mask = self.hor_mask
        if isinstance(temperature, dict):
            temp_cat = temperature.get('cat', 1.0)
            temp_bern = temperature.get('bern', 1.0)
        else:
            temp_cat = temperature
            temp_bern = temperature

        shared_features = self.net(x)

        outputs = {'cont': None, 'cat': None, 'bern': None}

        # --- Continuous ---
        if 'cont' in self.action_dims and self.action_dims['cont'] > 0:
            mu = self.fc_mu(shared_features)
            std = torch.exp(self.log_std_cont)
            std = torch.clamp(std, min=min_std, max=max_std)
            if mu.dim() > 1:
                std = std.unsqueeze(0).expand_as(mu)
            outputs['cont'] = (mu, std)

        # --- Categorical ---
        if 'cat' in self.action_dims and sum(self.action_dims['cat']) > 0:
            cat_logits_all = self.fc_cat(shared_features)

            # 1. 切分 Logits (内部使用 13 维垂直头 + 11 维水平头切分)
            split_dims = getattr(self, 'cat_dims_internal', self.cat_dims)
            cat_logits_list = list(torch.split(cat_logits_all, split_dims, dim=-1))

            # [去串扰与机动 mask]：对 action_ver 处理 (5个动作)
            if len(cat_logits_list) > 0 and cat_logits_list[0].size(-1) == 13:
                xb_cat = x
                if xb_cat.dim() == 1:
                    xb_cat = xb_cat.unsqueeze(0)
                warning_flag_cat = xb_cat[:, 5] > 1e-6
                missile_in_mid_term_cat = xb_cat[:, 3] > 1e-6
                cond_no_warn_mid = (~warning_flag_cat) & missile_in_mid_term_cat
                cond_no_warn_no_mid = (~warning_flag_cat) & (~missile_in_mid_term_cat)

                ver_logits_all = cat_logits_list[0]  # (Batch, 13)
                v_def = ver_logits_all[:, 8:13]      # 防御：0,1,2,3,4

                if ver_mask:
                    v_off = ver_logits_all[:, 0:5]   # 进攻：0,1,2,3,4
                    v_dis = ver_logits_all[:, 5:8]   # 偏置：2,3,4
                    v_off_5 = v_off

                    B = ver_logits_all.size(0)
                    v_dis_5 = torch.full((B, 5), -1e8, dtype=ver_logits_all.dtype, device=ver_logits_all.device)
                    v_dis_5[:, 2] = v_dis[:, 0]
                    v_dis_5[:, 3] = v_dis[:, 1]
                    v_dis_5[:, 4] = v_dis[:, 2]

                    ver_logits_5 = v_def
                    ver_logits_5 = torch.where(cond_no_warn_mid.unsqueeze(1), v_dis_5, ver_logits_5)
                    ver_logits_5 = torch.where(cond_no_warn_no_mid.unsqueeze(1), v_off_5, ver_logits_5)
                    cat_logits_list[0] = ver_logits_5
                else:
                    cat_logits_list[0] = v_def

            # [去串扰与机动 mask]：对 action_hor 处理
            if len(cat_logits_list) > 1 and cat_logits_list[1].size(-1) == 11:
                xb_cat = x
                if xb_cat.dim() == 1:
                    xb_cat = xb_cat.unsqueeze(0)
                warning_flag_cat = xb_cat[:, 5] > 1e-6
                missile_in_mid_term_cat = xb_cat[:, 3] > 1e-6
                cond_no_warn_mid = (~warning_flag_cat) & missile_in_mid_term_cat
                cond_no_warn_no_mid = (~warning_flag_cat) & (~missile_in_mid_term_cat)

                hor_logits = cat_logits_list[1]  # (Batch, 11)

                if hor_mask:
                    # 11维内部头分别对应三个阶段的白名单
                    # 0:无中导-追击 1:无中导-左3 2:无中导-置尾 3:无中导-右9 4:无中导-占中
                    # 5:有中导-左偏 6:有中导-置尾 7:有中导-右偏
                    # 8:告警-左3   9:告警-置尾  10:告警-右9
                    #                              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                    m_warn_allow   = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool, device=x.device) # 告警: 8,9,10
                    m_mid_allow    = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=torch.bool, device=x.device) # 无告警+有中导: 5,6,7
                    m_no_mid_allow = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool, device=x.device) # 无告警+无中导: 0,1,2,3,4

                    legal_mask = (
                        warning_flag_cat.unsqueeze(1) & m_warn_allow |
                        cond_no_warn_mid.unsqueeze(1) & m_mid_allow |
                        cond_no_warn_no_mid.unsqueeze(1) & m_no_mid_allow
                    )
                    hor_logits = hor_logits.masked_fill(~legal_mask, -1e8)

                # 把 11 维内部头按阶段语义映射到外部动作分布
                # 外部动作: 0追击, 1左偏, 2左3, 3置尾, 4右9, 5右偏, 6占中 (7维)
                # 或      : 0追击, 1左偏, 2左3, 3置尾, 4右9, 5右偏 (6维)
                if self.cat_dims[1] == 7:
                    action_head_groups = [
                        [0],        # 0 追击
                        [5],        # 1 左偏
                        [1, 8],     # 2 左3
                        [2, 6, 9],  # 3 置尾
                        [3, 10],    # 4 右9
                        [7],        # 5 右偏
                        [4],        # 6 占中
                    ]
                elif self.cat_dims[1] == 6:
                    action_head_groups = [
                        [0],        # 0 追击
                        [5],        # 1 左偏
                        [1, 8],     # 2 左3
                        [2, 6, 9],  # 3 置尾
                        [3, 10],    # 4 右9
                        [7],        # 5 右偏
                    ]
                else:
                    # 未知维度：直接保留每个内部头
                    action_head_groups = [[i] for i in range(11)]

                cat_logits_list[1] = torch.stack([
                    torch.logsumexp(hor_logits[:, heads], dim=1)
                    for heads in action_head_groups
                ], dim=1)

            # [强制] warning=1 时，无论 hor_mask 配置如何，都屏蔽指定水平机动动作
            if len(cat_logits_list) > 1:
                xb_cat = x
                if xb_cat.dim() == 1:
                    xb_cat = xb_cat.unsqueeze(0)
                # warning时不准前进
                warning_flag_cat = xb_cat[:, 5] > 1e-6
                hor_dim = cat_logits_list[1].size(-1)
                mask_indices = [0, 1,  2,4,  5, 6] if hor_dim == 7 else ([0, 1,  2,4,  5] if hor_dim == 6 else [])
                if mask_indices:
                    in_mask = torch.zeros(hor_dim, dtype=torch.bool, device=cat_logits_list[1].device)
                    in_mask[mask_indices] = True
                    in_mask = in_mask.unsqueeze(0).expand(cat_logits_list[1].size(0), -1)
                    warning_mask = warning_flag_cat.unsqueeze(-1).expand_as(in_mask) & in_mask
                    cat_logits_list[1] = cat_logits_list[1].masked_fill(warning_mask, -1e8)
                # mid_term时不准瞄准
                missile_in_mid_term_cat = xb_cat[:, 3] > 1e-6
                cond_no_warn_mid = (~warning_flag_cat) & missile_in_mid_term_cat
                mask_indices = [0, 6] if hor_dim == 7 else ([0] if hor_dim == 6 else [])
                if mask_indices:
                    in_mask = torch.zeros(hor_dim, dtype=torch.bool, device=cat_logits_list[1].device)
                    in_mask[mask_indices] = True
                    in_mask = in_mask.unsqueeze(0).expand(cat_logits_list[1].size(0), -1)
                    mid_term_mask = cond_no_warn_mid.unsqueeze(-1).expand_as(in_mask) & in_mask
                    cat_logits_list[1] = cat_logits_list[1].masked_fill(mid_term_mask, -1e8)

            # 2. 应用温度缩放 (Logits / temperature) 并 Softmax
            final_probs_list = []
            for i, logits in enumerate(cat_logits_list):
                scaled_logits = logits / (temp_cat + 1e-8)
                final_probs_list.append(F.softmax(scaled_logits, dim=-1))

            outputs['cat'] = final_probs_list

        # --- Bernoulli ---
        if 'bern' in self.action_dims and self.action_dims['bern'] > 0:
            bern_logits = self.fc_bern(shared_features)

            xb = x
            if xb.dim() == 1:
                xb = xb.unsqueeze(0)

            cos_ata_hor = torch.clamp(xb[:, 6], -0.999999, 0.999999)
            delta_theta = xb[:, 8]
            ata = xb[:, 10]
            sin_theta = xb[:, 17]
            ammo = xb[:, 20]
            dist = xb[:, 9] * 10e3
            t_since_launch = xb[:, 21] * 120

            ammo_cond = (ammo > 0.0)
            time_const_cond = t_since_launch >= torch.clamp_min(dist/(3*340)/2, 10.0)
            ata_cond = ata < math.pi / 2
            can_fire = ammo_cond & time_const_cond & ata_cond

            delta_psi_cond = cos_ata_hor >= math.cos(np.radians(45))
            can_fire = can_fire & delta_psi_cond

            theta = torch.arcsin(sin_theta)
            elevation = theta + delta_theta
            theta_cond = theta >= elevation - np.radians(15)
            can_fire = can_fire & theta_cond

            bern_dim = self.action_dims.get('bern', 0)
            batch_size = shared_features.size(0)
            mask = torch.ones((batch_size, bern_dim), dtype=torch.bool, device=shared_features.device)
            mask[:, 0] = can_fire.to(dtype=torch.bool)

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

            bern_logits = bern_logits.masked_fill(mask == 0, -1e8)

            scaled_bern_logits = bern_logits / (temp_bern + 1e-8)
            outputs['bern'] = scaled_bern_logits

            outputs['fire_mask'] = mask.float()

        return outputs

# =============================================================================
# 2. Actor 适配器 (Wrapper) - 核心重构点
# =============================================================================

class HybridActorWrapper(PPOHybridActorWrapper):
    """
    继承自 PPOHybrid23_0 的统一接口适配器。
    为 SAC 额外提供可导采样的 sample_for_sac 方法。
    """

    def sample_for_sac(self, states, action_masks=None, gumbel_tau=1.5):
        """
        专门为 SAC 提供的采样方法。
        返回可导的 actions，以及按动作头拆分的 log_prob 字典。
        """
        actor_outputs = self.net(states, action_masks=action_masks)

        actions_differentiable = {}
        log_probs_total = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_cont = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_cat = torch.zeros(states.size(0), 1).to(self.device)
        log_probs_bern = torch.zeros(states.size(0), 1).to(self.device)
        bern_entropy = torch.zeros(states.size(0), 1).to(self.device)

        # --- Cont (连续动作，使用 rsample) ---
        if actor_outputs['cont'] is not None:
            mu, std = actor_outputs['cont']
            # 注意：此处需确保 SquashedNormal 支持 rsample 并且正确计算了 tanh 的 log_prob
            dist = Normal(mu, std)
            u = dist.rsample() # 重参数化采样
            a_norm = torch.tanh(u)
            # 计算 Squash 的 log_prob
            log_prob_cont = dist.log_prob(u) - torch.log(1 - a_norm.pow(2) + 1e-7)
            log_prob_cont_sum = log_prob_cont.sum(-1, keepdim=True)
            log_probs_cont += log_prob_cont_sum
            log_probs_total += log_prob_cont_sum

            actions_differentiable['cont'] = a_norm # 直接输出 -1~1 的范围给 Q 网络

        # --- Cat (离散动作，使用 Gumbel-Softmax) ---
        if actor_outputs['cat'] is not None:
            # PPOHybrid23_0.PolicyNetHybrid 返回的是 softmax 后的 probs，
            # 取 log 后作为 gumbel_softmax 的 logits 使用
            cat_logits_list = [torch.log(probs + 1e-8) for probs in actor_outputs['cat']]
            cat_actions = []
            log_p_cat_sum = torch.zeros(states.size(0), 1).to(self.device)
            for logits in cat_logits_list:
                # hard=True 表示前向传播输出 One-hot(例如[0,1,0])，反向传播用 softmax 的梯度
                gumbel_out = F.gumbel_softmax(logits, tau=gumbel_tau, hard=True)
                cat_actions.append(gumbel_out)

                # 计算 log_prob (近似)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs=probs)
                # 由于 hard=True 返回的是 one-hot，可以通过与 log_probs 相乘来提取选中项的 log_prob
                log_p = torch.sum(torch.log(probs + 1e-8) * gumbel_out, dim=-1, keepdim=True)
                log_p_cat_sum += log_p

            log_probs_cat += log_p_cat_sum
            log_probs_total += log_p_cat_sum
            actions_differentiable['cat'] = torch.cat(cat_actions, dim=-1)

        # --- Bern (伯努利动作，使用 Binary Gumbel-Softmax / 缓和的 Sigmoid) ---
        if actor_outputs['bern'] is not None:
            bern_logits = actor_outputs['bern']
            # 将 logits 转换为 [prob_0, prob_1] 的形式以便使用 gumbel_softmax
            logits_2d = torch.stack([torch.zeros_like(bern_logits), bern_logits], dim=-1)
            gumbel_out = F.gumbel_softmax(logits_2d, tau=1.0, hard=True)
            bern_action = gumbel_out[..., 1] # 取出代表 1(True) 的那一列

            actions_differentiable['bern'] = bern_action

            probs = torch.sigmoid(bern_logits)
            # 计算每个 bern 维度的 log_prob
            log_p_bern = torch.log(probs + 1e-8) * bern_action + torch.log(1 - probs + 1e-8) * (1 - bern_action)

            # 用 fire_mask 屏蔽被 can_fire=False 的位置：乘以 0 使其对梯度无贡献
            # 这样被 mask 的位置既不影响 actor_loss、alpha_loss，也不影响 Q 目标中的熵正则项
            fire_mask = actor_outputs.get('fire_mask', None)
            if fire_mask is not None:
                log_p_bern = log_p_bern * fire_mask  # shape: (batch, bern_dim)，masked 位置乘 0
                valid_count = fire_mask.sum(-1, keepdim=True).clamp_min(1.0)
                bern_entropy = (Bernoulli(logits=bern_logits).entropy() * fire_mask).sum(-1, keepdim=True) / valid_count
            else:
                bern_entropy = Bernoulli(logits=bern_logits).entropy().mean(-1, keepdim=True)

            log_p_bern_sum = log_p_bern.view(states.size(0), -1).sum(-1, keepdim=True)
            log_probs_bern += log_p_bern_sum
            log_probs_total += log_p_bern_sum

        log_probs = {
            'cont': log_probs_cont,
            'cat': log_probs_cat,
            'bern': log_probs_bern,
            'bern_entropy': bern_entropy,
            'total': log_probs_total,
        }
        return actions_differentiable, log_probs
# =============================================================================
# 3. SAC 算法类 (精简版)
# =============================================================================
class SACHybrid:
    def __init__(self, actor, critic_temp, critic_1, critic_2, target_critic_1, target_critic_2, 
                 actor_lr, critic_lr, alpha_lr, action_dims_dict, gamma, tau, device,
                 k_entropy={'cont':0.01, 'cat':0.005, 'bern':0.05}, critic_max_grad=2, actor_max_grad=2, max_std=0.7,
                 gumbel_tau=1.5):
        self.actor = actor
        # MARWIL_update 内部引用 self.critic，这里让其指向预训练用的 ValueNet
        self.critic = critic_temp # 仅给预训练(MARWIL)使用，在线SAC阶段弃置不用
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2

        # 保存超参，供学习率调整 / 梯度裁剪 / 重建优化器使用
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.k_entropy = k_entropy
        self.max_std = max_std
        self.actor_max_grad = actor_max_grad
        self.critic_max_grad = critic_max_grad
        self.gumbel_tau = gumbel_tau
        
        # 初始化目标网络
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.fire_head_optimizer = None
        if hasattr(self.actor.net, 'fc_bern'):
            self.fire_head_optimizer = torch.optim.Adam(self.actor.net.fc_bern.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        # 预训练 (MARWIL) 阶段优化 ValueNet 的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 自动调节温度参数 Alpha
        # 针对 Hybrid，可以设一个全局 Alpha，也可以为 cont, cat, bern 各设一个。这里用一个全局的演示。
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # target_entropy 由外部在 update() 中传入，此处不做预设
        
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, explore=True, check_obs=None, **kwargs):
        # 推理时仍然使用 get_action，用于环境交互并包含动作还原
        # 透传 mask_on / temperature 等额外参数给 wrapper
        return self.actor.get_action(state, explore=explore, check_obs=check_obs, **kwargs)

    def set_learning_rate(self, actor_lr=None, critic_lr=None):
        """动态调整学习率，兼容主训练脚本的调用接口。"""
        if actor_lr is not None:
            self.actor_lr = actor_lr
            for opt in (self.actor_optimizer, self.fire_head_optimizer):
                if opt is None:
                    continue
                for g in opt.param_groups:
                    g['lr'] = actor_lr
        if critic_lr is not None:
            self.critic_lr = critic_lr
            for opt in (self.critic_1_optimizer, self.critic_2_optimizer, self.critic_optimizer):
                for g in opt.param_groups:
                    g['lr'] = critic_lr

    def reset_optimizer(self):
        """重建优化器以清除动量（中断续训/恢复崩溃时使用）。"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        if hasattr(self.actor.net, 'fc_bern'):
            self.fire_head_optimizer = torch.optim.Adam(self.actor.net.fc_bern.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

    def save_critics(self, path):
        """保存在线SAC的Q网络与温度参数（弃置ValueNet）。"""
        torch.save({
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
            'gumbel_tau': self.gumbel_tau,
        }, path)

    def load_critics(self, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        # 兼容旧的 ValueNet critic.pt（只有 state_dict，没有Q网络键）
        if not isinstance(ckpt, dict) or 'critic_1' not in ckpt:
            print(f"[SACHybrid] {path} 不是SAC critic格式，跳过加载Q网络。")
            return
        self.critic_1.load_state_dict(ckpt['critic_1'])
        self.critic_2.load_state_dict(ckpt['critic_2'])
        self.target_critic_1.load_state_dict(ckpt['target_critic_1'])
        self.target_critic_2.load_state_dict(ckpt['target_critic_2'])
        if 'log_alpha' in ckpt:
            with torch.no_grad():
                self.log_alpha.copy_(ckpt['log_alpha'].to(self.log_alpha.device))

    def save_optimizers(self, path):
        payload = {
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }
        if self.fire_head_optimizer is not None:
            payload['fire_head_optimizer'] = self.fire_head_optimizer.state_dict()
        torch.save(payload, path)

    def load_optimizers(self, path, map_location='cpu'):
        s = torch.load(path, map_location=map_location)
        try:
            self.actor_optimizer.load_state_dict(s['actor_optimizer'])
            if self.fire_head_optimizer is not None and 'fire_head_optimizer' in s:
                self.fire_head_optimizer.load_state_dict(s['fire_head_optimizer'])
            if 'critic_1_optimizer' in s:
                self.critic_1_optimizer.load_state_dict(s['critic_1_optimizer'])
                self.critic_2_optimizer.load_state_dict(s['critic_2_optimizer'])
            if 'alpha_optimizer' in s:
                self.alpha_optimizer.load_state_dict(s['alpha_optimizer'])
        except Exception as e:
            print(f"[SACHybrid] Failed to load optimizers: {e}")

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, batch, target_entropy=1.5, alpha_clip=(0.001, 0.1), freeze_actor=False,
               actor_max_update_norm=0.05):
        """
        接收 ReplayBuffer 返回的字典 batch
        target_entropy : 目标熵（正数，由外部传入）。None 则不更新 alpha。
        alpha_clip      : (min, max) 对 alpha=exp(log_alpha) 的截断范围。
        freeze_actor   : True 时只更新 Q 网络，跳过 actor 和 alpha 更新（Q 预热阶段使用）。
        """
        # --- A. 数据搬运与类型转换 (NumPy -> Tensor) ---
        device = self.device
        
        states = torch.from_numpy(batch['states']).to(device)
        next_states = torch.from_numpy(batch['next_states']).to(device)
        rewards = torch.from_numpy(batch['rewards']).to(device)
        dones = torch.from_numpy(batch['dones']).to(device)
        if 'active_masks' in batch:
            active_masks = torch.from_numpy(np.array(batch['active_masks'])).float().to(device).view(-1, 1)
        else:
            active_masks = torch.ones(states.size(0), 1, device=device)
        
        # 处理动作 (从字典中提取并转为 Tensor)
        raw_actions = batch['actions']
        actions_for_q = {}
        
        if 'cont' in raw_actions:
            actions_for_q['cont'] = torch.from_numpy(raw_actions['cont']).to(device)
            
        # cat 只保留索引，Q 网络输出所有 cat 组合后 gather
        if 'cat' in raw_actions:
            cat_idx = torch.from_numpy(raw_actions['cat']).to(device).long()
            cat_dims = self.actor.action_dims['cat']
            joint_idx = ravel_cat_indices(cat_idx, cat_dims)
        else:
            joint_idx = None
            
        if self.actor.action_dims.get('bern', 0) > 0 and 'bern' in raw_actions:
            actions_for_q['bern'] = torch.from_numpy(raw_actions['bern']).to(device)

        # --- B. SAC 计算逻辑 (逻辑保持不变，但变量名已对齐) ---
        
        # [诊断] 统计当前 batch 中 replay buffer 存储的 bern 动作分布
        if not hasattr(self, '_diag_update_count'):
            self._diag_update_count = 0
        self._diag_update_count += 1
        
        if self._diag_update_count % 200 == 1:
            if 'bern' in raw_actions:
                bern_buf = raw_actions['bern']  # shape: (batch, bern_dim)
                n_fire = (bern_buf > 0.5).sum()
                n_no_fire = (bern_buf <= 0.5).sum()
                print(f"[SAC diag #{self._diag_update_count}] replay buffer bern: fire={n_fire}, no_fire={n_no_fire}, ratio={n_fire/(n_fire+n_no_fire+1e-8):.3f}")

        # 1. 更新 Q 网络 (Critic)
        with torch.no_grad():
            # 获取下一状态的动作 (可导采样) 和 log_prob
            next_actions_diff, next_log_probs = self.actor.sample_for_sac(next_states, gumbel_tau=self.gumbel_tau)
            
            # 把目标 actor 的 cat one-hot 还原成联合索引
            if 'cat' in next_actions_diff and next_actions_diff['cat'] is not None:
                joint_idx_next = ravel_cat_indices(
                    per_dim_onehot_to_indices(next_actions_diff['cat'], self.actor.action_dims['cat']),
                    self.actor.action_dims['cat'])
            else:
                joint_idx_next = None

            # [诊断] 统计 next_states 里被 mask 和未被 mask 的样本数
            if self._diag_update_count % 200 == 1:
                _outs_diag = self.actor.net(next_states)
                _fm = _outs_diag.get('fire_mask', None)
                if _fm is not None:
                    n_can_fire = (_fm > 0.5).sum().item()
                    n_masked = (_fm <= 0.5).sum().item()
                    bern_lp = next_log_probs['bern']
                    # 统计 sample_for_sac 采出的 bern_action 在 can_fire 位置的均值（接近1=偏开火，接近0=偏不开火）
                    valid_mask_1d = (_fm > 0.5).view(-1)
                    bern_act = next_actions_diff.get('bern', None)
                    if bern_act is not None:
                        bern_act_flat = bern_act.view(-1)
                        canfire_mean = bern_act_flat[valid_mask_1d].mean().item() if valid_mask_1d.any() else float('nan')
                    else:
                        canfire_mean = float('nan')
                    bern_logit_canfire = _outs_diag['bern'].view(-1)[valid_mask_1d]
                    logit_mean = bern_logit_canfire.mean().item() if valid_mask_1d.any() else float('nan')
                    logit_max = bern_logit_canfire.max().item() if valid_mask_1d.any() else float('nan')
                    print(f"[SAC diag #{self._diag_update_count}] next_states fire_mask: can_fire={n_can_fire}, masked={n_masked}, "
                          f"bern_logprob mean={bern_lp.mean().item():.4f}, "
                          f"bern_action[can_fire] mean={canfire_mean:.3f}, "
                          f"bern_logit[can_fire] mean={logit_mean:.3f} max={logit_max:.3f}")
            
            # 目标 Q 值
            q1_target_all = self.target_critic_1(next_states, next_actions_diff)
            q2_target_all = self.target_critic_2(next_states, next_actions_diff)
            q1_target = q1_target_all.gather(1, joint_idx_next) if joint_idx_next is not None else q1_target_all
            q2_target = q2_target_all.gather(1, joint_idx_next) if joint_idx_next is not None else q2_target_all
            alpha = self.log_alpha.exp()
            k_bern = self.k_entropy.get('bern', 0.003)
            # fire head is supervised only; it is deliberately excluded from SAC entropy/Q targets.
            entropy_reg = alpha * (next_log_probs['cont'] + next_log_probs['cat'])
            min_q_target = torch.min(q1_target, q2_target) - entropy_reg
            
            # TD 目标
            y_target = rewards + self.gamma * (1 - dones) * min_q_target
            
        # 当前 Q 值预测
        q1_pred_all = self.critic_1(states, actions_for_q)
        q2_pred_all = self.critic_2(states, actions_for_q)
        q1_pred = q1_pred_all.gather(1, joint_idx) if joint_idx is not None else q1_pred_all
        q2_pred = q2_pred_all.gather(1, joint_idx) if joint_idx is not None else q2_pred_all
        
        mask_eps = 1e-5
        active_sum = active_masks.sum()
        # 两个 Q 网络分别计算损失并独立反向传播，避免共用单一损失图
        critic_1_loss = F.mse_loss(q1_pred, y_target, reduction='mean')
        critic_2_loss = F.mse_loss(q2_pred, y_target, reduction='mean')
        critic_loss = (critic_1_loss + critic_2_loss).detach()  # 仅用于日志

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad = nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.critic_max_grad)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_grad = nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.critic_max_grad)
        self.critic_2_optimizer.step()
        critic_grad = (critic_1_grad + critic_2_grad) / 2.0

        # 2. 更新 策略网络 (Actor) —— freeze_actor=True 时跳过
        if not freeze_actor:
            # 重新对当前状态采样
            curr_actions_diff, curr_log_probs = self.actor.sample_for_sac(states, gumbel_tau=self.gumbel_tau)
            
            # 由 actor 的 per-dim cat probs 计算可微联合概率
            actor_outputs = self.actor.net(states)
            if actor_outputs['cat'] is not None:
                cat_joint_probs = joint_cat_prob_from_list(actor_outputs['cat'])
            else:
                cat_joint_probs = None
            
            # Q 网络输入 cont/bern（自动忽略 cat），输出所有 cat 的 Q 后求期望
            q1_pi_all = self.critic_1(states, curr_actions_diff)
            q2_pi_all = self.critic_2(states, curr_actions_diff)
            q1_pi = (q1_pi_all * cat_joint_probs).sum(dim=1, keepdim=True) if cat_joint_probs is not None else q1_pi_all
            q2_pi = (q2_pi_all * cat_joint_probs).sum(dim=1, keepdim=True) if cat_joint_probs is not None else q2_pi_all
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            alpha = self.log_alpha.exp().detach()
            # 机动部分 (cont+cat) 使用自适应 alpha；开火部分 (bern) 使用固定初始熵系数
            # 与原版 SAC 一致：actor_loss = -alpha * entropy - min_Q，其中 entropy = -log_prob
            mobility_log_prob = curr_log_probs['cont'] + curr_log_probs['cat']
            mobility_entropy = -mobility_log_prob
            actor_loss = ((-alpha * mobility_entropy - min_q_pi) * active_masks).sum() / (active_sum + mask_eps)

            actor_params = [p for p in self.actor.parameters() if p.requires_grad]
            actor_before = [p.detach().clone() for p in actor_params]
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(actor_params, self.actor_max_grad)
            self.actor_optimizer.step()
            if actor_max_update_norm is not None:
                with torch.no_grad():
                    update_norm = torch.sqrt(sum((p - old).pow(2).sum() for p, old in zip(actor_params, actor_before)))
                    if update_norm > actor_max_update_norm:
                        scale = actor_max_update_norm / (update_norm + 1e-12)
                        for p, old in zip(actor_params, actor_before):
                            p.copy_(old + scale * (p - old))

            # 3. 更新 Alpha (熵系数)
            if target_entropy is not None:
                self.target_entropy = target_entropy
                # alpha 只根据机动部分 (cont+cat) 的熵来调节
                # mobility_log_probs 是 log_prob（负数），熵 entropy = -log_prob（正数）
                mobility_log_probs = curr_log_probs['cont'].detach() + curr_log_probs['cat'].detach()
                mobility_entropy = -mobility_log_probs
                alpha = self.log_alpha.exp()
                alpha_loss = (alpha * (mobility_entropy - target_entropy) * active_masks).sum() / (active_sum + mask_eps)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                # 将 alpha 截断到合法范围
                if alpha_clip is not None:
                    log_alpha_min = np.log(alpha_clip[0])
                    log_alpha_max = np.log(alpha_clip[1])
                    with torch.no_grad():
                        self.log_alpha.clamp_(log_alpha_min, log_alpha_max)
            else:
                alpha_loss = torch.tensor(0.0)
        else:
            # freeze_actor=True：用零值占位，不触碰 actor/alpha 参数
            curr_log_probs = {'cont': torch.zeros(1), 'cat': torch.zeros(1), 'bern': torch.zeros(1), 'bern_entropy': torch.zeros(1), 'total': torch.zeros(1)}
            actor_loss = torch.tensor(0.0)
            actor_grad = torch.tensor(0.0)
            alpha_loss = torch.tensor(0.0)

        # 4. 目标网络软更新
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
        
        # --- 监控指标（兼容主训练脚本的 logger 字段） ---
        self.last_actor_loss = actor_loss.item()
        self.last_critic_loss = critic_loss.item()
        self.last_supervised_fire_loss = 0.0
        self.last_entropy_mobility = -(curr_log_probs['cont'] + curr_log_probs['cat']).mean().item()
        self.last_entropy_bern = curr_log_probs['bern_entropy'].mean().item()
        self.last_entropy = self.last_entropy_mobility + self.last_entropy_bern
        self.actor_loss = self.last_actor_loss
        self.critic_loss = self.last_critic_loss
        self.entropy_mean = self.last_entropy
        self.alpha = self.log_alpha.exp().item()
        self.k_bern = self.k_entropy.get('bern', 0.003)
        self.pre_clip_actor_grad = float(actor_grad)
        self.pre_clip_critic_grad = float(critic_grad)
        self.td_error_var = (y_target - q1_pred).detach().var().item()

        # 各动作头熵 / 开火概率（基于当前策略分布，便于监控）
        with torch.no_grad():
            outs = self.actor.net(states)
            self.entropy_cat = 0.0
            self.entropy_bern = 0.0
            self.max_fire_prob = 0.0
            self.min_fire_prob = 0.0
            if outs.get('cat') is not None:
                ent_c = 0.0
                for probs in outs['cat']:
                    ent_c += Categorical(probs=probs).entropy().mean().item()
                self.entropy_cat = ent_c
            if outs.get('bern') is not None:
                bern_logits = outs['bern'].clamp(min=-1e8)
                fire_mask = outs.get('fire_mask', None)
                if fire_mask is not None:
                    # fire_mask 由 PolicyNetHybrid.forward 内部计算，标记哪些样本位置允许开火（弹药充足+冷却到位+角度合理）
                    # 只统计未被 mask 的有效位置，避免被强制压成 -1e8 的位置拉低 min_fire_prob 或 entropy_bern
                    valid_mask = (fire_mask > 0.5)
                    if valid_mask.any():
                        valid_probs = torch.sigmoid(bern_logits)[valid_mask]
                        valid_logits = bern_logits[valid_mask]
                        # 熵只在有效位置内求平均，除以有效位置数
                        self.entropy_bern = Bernoulli(logits=valid_logits).entropy().sum().item() / max(valid_mask.sum().item(), 1)
                        # 最大/最小开火概率也只在有效位置内统计
                        self.max_fire_prob = valid_probs.max().item()
                        self.min_fire_prob = valid_probs.min().item()
                    else:
                        # 全 batch 都被 mask，无法开火
                        self.entropy_bern = 0.0
                        self.max_fire_prob = 0.0
                        self.min_fire_prob = 0.0
                else:
                    # 兼容性分支：如果 net 没有返回 fire_mask（理论上不应该发生），则全量统计
                    self.entropy_bern = Bernoulli(logits=bern_logits).entropy().sum(-1).mean().item()
                    fire_probs = torch.sigmoid(bern_logits)
                    self.max_fire_prob = fire_probs.max().item()
                    self.min_fire_prob = fire_probs.min().item()

    def supervised_fire_update(self, fire_batch, epochs=1, batch_size=128, max_grad_norm=2.0):
        """仅用回合结束后的 fire_obs/hit_record 更新开火头 fc_bern。

        共享 backbone 和 BFM/其它动作头不参与梯度更新；因此该更新不会改变
        SAC 学到的机动策略。fire_batch 可是 SupervisedFireBuffer.sample() 的
        字典，也可是包含 fire_obs/hit_record 的原始 record 列表。
        """
        if self.fire_head_optimizer is None:
            self.last_supervised_fire_loss = 0.0
            return 0.0

        if isinstance(fire_batch, dict):
            states = fire_batch.get('states', fire_batch.get('fire_obs'))
            labels = fire_batch.get('labels', fire_batch.get('hit_record'))
        else:
            records = list(fire_batch or [])
            states = [r.get('fire_obs') for r in records]
            labels = [r.get('hit_record') for r in records]
        if states is None or labels is None or len(states) == 0:
            self.last_supervised_fire_loss = 0.0
            return 0.0

        states = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
        labels = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=self.device).view(-1, 1)
        if states.size(0) != labels.size(0):
            raise ValueError('fire_obs and hit_record must have the same length')

        net = self.actor.net
        params = list(self.actor.parameters())
        old_requires_grad = [p.requires_grad for p in params]
        for p in params:
            p.requires_grad_(False)
        for p in net.fc_bern.parameters():
            p.requires_grad_(True)

        losses = []
        n = states.size(0)
        mb_size = max(1, min(int(batch_size), n))
        for _ in range(max(1, int(epochs))):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                with torch.no_grad():
                    features = net.net(states[idx])
                logits = net.fc_bern(features)[:, :1]
                loss = F.binary_cross_entropy_with_logits(
                    input=logits,
                    target=labels[idx],
                    reduction='mean'
                )
                self.fire_head_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.fc_bern.parameters(), max_grad_norm)
                self.fire_head_optimizer.step()
                losses.append(float(loss.detach().cpu()))

        for p, requires_grad in zip(params, old_requires_grad):
            p.requires_grad_(requires_grad)
        self.last_supervised_fire_loss = float(np.mean(losses)) if losses else 0.0
        return self.last_supervised_fire_loss

    # =========================================================================
    #  兼容旧流程：Bernoulli 开火头保护性有监督训练
    # =========================================================================
    def fire_prob_protection(self, batch, protect_epochs=4, protect_mini_batch=256):
        """
        Bern头概率范围保护器。当开火概率整体崩溃（全高或全低）时，以有监督方式
        强行拉回bern头分布，同时切断backbone和其它动作头的梯度，保护机动策略不被拖垮。

        必要条件1 (比值护栏): max_fire_prob / min_fire_prob >= 10，说明分布仍有分化空间，
                               不需要干预，直接跳过。
        必要条件2 (触发case):
          case1: max_fire_prob < 0.05  → 整体开火概率崩到极低，以0.5为监督信号，拉高熵。
          case2: min_fire_prob > 0.1   → 整体开火概率过高，以1e-3为监督信号，压低概率。

        Args:
            batch           : 与update()相同格式的经验字典（来自ReplayBuffer）。
            protect_epochs  : 保护性训练的epoch数。
            protect_mini_batch: 每个mini-batch的大小。
        """
        # ── 必要条件1：比值护栏 ──────────────────────────────────────────────────
        ratio = self.max_fire_prob / (self.min_fire_prob + 1e-12)
        if ratio >= 10.0:
            return

        # ── 必要条件2：判断触发case ──────────────────────────────────────────────
        if self.max_fire_prob < 0.05:
            target_prob = 0.5
        elif self.min_fire_prob > 0.1:
            target_prob = 1e-3
        else:
            return

        # ── 数据准备 ─────────────────────────────────────────────────────────
        def to_tensor(x, dtype):
            if isinstance(x, np.ndarray):
                return torch.tensor(x, dtype=dtype).to(self.device)
            else:
                return torch.tensor(np.array(x), dtype=dtype).to(self.device)

        if 'obs' in batch:
            actor_inputs = to_tensor(batch['obs'], torch.float)
        else:
            actor_inputs = to_tensor(batch['states'], torch.float)

        if 'active_masks' in batch:
            active_masks_p = to_tensor(batch['active_masks'], torch.float).view(-1, 1)
        else:
            active_masks_p = torch.ones(actor_inputs.size(0), 1, device=self.device)

        num_samples = actor_inputs.size(0)
        mb_size = min(protect_mini_batch, num_samples)

        # ── 冻结除bern头以外的所有actor模块 ───────────────────────────────────
        net = self.actor.net

        def set_requires_grad(module_or_param, flag):
            if isinstance(module_or_param, nn.Module):
                for p in module_or_param.parameters():
                    p.requires_grad_(flag)
            else:
                module_or_param.requires_grad_(flag)

        set_requires_grad(net.net, False)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, False)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, False)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, False)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        # ── 监督训练循环 ──────────────────────────────────────────────────────
        target_tensor = torch.tensor(target_prob, device=self.device)

        for _ in range(protect_epochs):
            perm = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mb_size):
                end = min(start + mb_size, num_samples)
                batch_idx = perm[start:end]

                mb_states = actor_inputs[batch_idx]
                mb_active = active_masks_p[batch_idx]
                active_sum_p = mb_active.sum()

                actor_out = self.actor.net(mb_states)
                if actor_out['bern'] is None:
                    break

                bern_probs = torch.sigmoid(actor_out['bern'].clamp(min=-1e8))
                target_full = target_tensor.expand_as(bern_probs)
                bern_loss_per = F.binary_cross_entropy(bern_probs, target_full, reduction='none').sum(dim=-1, keepdim=True)
                bern_loss = (bern_loss_per * mb_active).sum() / (active_sum_p + 1e-5)

                self.actor_optimizer.zero_grad()
                bern_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.actor_max_grad)
                self.actor_optimizer.step()

        # ── 恢复所有actor模块的梯度反向传播 ──────────────────────────────────
        set_requires_grad(net.net, True)
        if hasattr(net, 'fc_mu'):
            set_requires_grad(net.fc_mu, True)
        if hasattr(net, 'log_std_cont'):
            set_requires_grad(net.log_std_cont, True)
        if hasattr(net, 'fc_cat'):
            set_requires_grad(net.fc_cat, True)
        if hasattr(net, 'fc_bern'):
            set_requires_grad(net.fc_bern, True)

        return

    # --- 修改后的 MARWIL_update， 注意原先是0 ---
    def MARWIL_update(self, il_transition_dict, beta=1.0, batch_size=64, alpha=1.0, c_v=1.0, shuffled=1, label_smoothing=0.3, max_weight=100.0,
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
            raw_il_loss = self.actor.compute_il_loss(
                actor_input_batch,
                actions_batch,
                label_smoothing,
                no_bern=no_bern,
                good_samples=1,
                pre_training=1,
            )
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
    
