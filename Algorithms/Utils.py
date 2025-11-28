import os, sys
from torch.distributions import Normal
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


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


# --- 广义优势函数计算 ---
# 旧的实现
# def compute_advantage(gamma, lmbda, td_delta, dones):
#     td_delta = td_delta.detach().cpu().numpy()
#     dones = dones.detach().cpu().numpy() # [新增] 转为 numpy
#     advantage_list = []
#     advantage = 0.0
    
#     # [修改] 同时遍历 delta 和 done
#     for delta, done in zip(td_delta[::-1], dones[::-1]):
#         # 如果当前是 done，说明这是序列的最后一步（或者该步之后没有未来），
#         # 此时不应该加上一步（时间上的未来）的 advantage。
#         # 注意：这里的 advantage 变量存的是“下一步的优势”，所以要乘 (1-done)
#         advantage = delta + gamma * lmbda * advantage * (1 - done)
#         advantage_list.append(advantage)
        
#     advantage_list.reverse()
#     return torch.tensor(np.array(advantage_list), dtype=torch.float)

# --- 保持 compute_advantage 函数，但根据传入参数数量切换逻辑 ---
def compute_advantage(gamma, lmbda, td_delta, dones, truncateds=None): # truncateds 默认为 None
    # 确保输入转为 numpy
    td_delta = td_delta.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy() # 假设这里的 dones 是 terminateds (term)

    if truncateds is None:
        # --- 旧式/兼容模式：dones = term OR trunc ---
        # 此时，dones 就是 $\text{done}_t$
        advantage_list = []
        advantage = 0.0
        
        for delta, done in zip(td_delta[::-1], dones[::-1]):
            advantage = delta + gamma * lmbda * advantage * (1 - done)
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)
    
    else:
        # --- 新式模式：需要 term (dones) 和 trunc (truncateds) ---
        truncateds = truncateds.detach().cpu().numpy()
        terminateds = dones # $\text{term}_t$
        
        advantage_list = []
        advantage = 0.0
        
        for delta, term, trunc in zip(td_delta[::-1], terminateds[::-1], truncateds[::-1]):
            # 1. GAE 传递项的修正因子: $\gamma \lambda (1 - \text{term}_t) A_{t+1}$
            next_advantage_term = gamma * lmbda * advantage * (1.0 - term)
            
            # 2. 预估 A_t: $A'_t = \delta_t + \text{next\_advantage\_term}$
            advantage = delta + next_advantage_term
            
            # 3. 最终 A_t 屏蔽: $A_t = (1 - \text{trunc}_t) \cdot A'_t$
            advantage = advantage * (1.0 - trunc)
            
            advantage_list.append(advantage)
        
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)


class SquashedNormal:
    """带 tanh 压缩的高斯分布。

    采样：u ~ N(mu, std)（使用 rsample 支持 reparam），a = tanh(u)
    log_prob：基于 u 的 normal.log_prob(u) 并加上 tanh 的 Jacobian 修正项：-sum log(1 - tanh(u)^2)
    注意：外部需要把动作缩放到环境动作空间（仿射变换）。
    """

    def __init__(self, mu, std, eps=1e-6):
        self.mu = mu
        if not torch.is_tensor(std):
            std = torch.as_tensor(std, device=mu.device, dtype=mu.dtype)
        self.std = torch.clamp(std, min=float(eps))
        self.normal = Normal(mu, std)
        self.eps = eps
        self.mean = mu

    def sample(self):
        # rsample 以支持 reparameterization 重参数化采样, 结果是可导的
        u = self.normal.rsample()
        a = torch.tanh(u)
        return a, u

    def log_prob(self, a, u):
        # a: tanh(u)
        # log_prob(u) - sum log(1 - tanh(u)^2)
        # normal.log_prob 返回每个维度的 log_prob，需要 sum
        # 为数值稳定性添加小量
        log_prob_u = self.normal.log_prob(u)
        # jacobian term
        jacobian = 0 # 保存u的话就不需要该修正项
        # jacobian = 2*(np.log(2.0)-u-F.softplus(-2*u))
        # jacobian = torch.log(1 - a.pow(2) + self.eps)
        # sum over action dim, keep dims consistent: return (N, 1)
        # 取消提前求和 # return (log_prob_u - jacobian).sum(-1, keepdim=True)
        return log_prob_u - jacobian  # 返回形状为 (batch_size, action_dim)

    def entropy(self):
        # 近似：使用 base normal 的熵之和（不考虑 tanh 的修正）
        # 这在实践中通常足够，若需精确熵可用采样估计
        ent = self.normal.entropy().sum(-1)
        return ent