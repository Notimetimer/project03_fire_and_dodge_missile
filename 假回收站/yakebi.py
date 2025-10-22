import torch
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import numpy as np
import math

# 封装 SquashedNormal 类
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.base_dist = Normal(loc, scale)
        self.tanh_transform = TanhTransform(cache_size=1)
        transforms = [self.tanh_transform]
        super().__init__(self.base_dist, transforms)

# --- 实验设置 ---
mu = torch.tensor([0.0])
sigma = torch.tensor([1.0])
base_normal_dist = Normal(mu, sigma)
squashed_normal_dist = SquashedNormal(mu, sigma)

# --- 创建测试点 ---
u_original = torch.tensor([20.0])
a_original = torch.tanh(u_original) # 在 float32 下结果为 1.0

print("\n" + "="*70)
print(f"实验设定:")
print(f"  - 原始 u: {u_original.item():.32f}")
print(f"  - tanh(u) -> a: {a_original.item():.32f}")
print("  - [关键] 不再使用 a_safe，直接使用 a_original = 1.0")
print("="*70)
print("对比基于【原始 u】 vs 基于【反算 u】的 log_prob 计算结果")
print("="*70)

# ==============================================================================
# Part 1: 基于【原始 u】的正确计算 (这些方法不依赖 a，所以不受影响)
# ==============================================================================
print("--- Part 1: 基于原始 u_original = 20.0 ---")
log_prob_u_original = base_normal_dist.log_prob(u_original)

# --- 方法 2: 手动稳定雅可比 ---
jacobian_2 = 2 * (math.log(2.) - u_original - F.softplus(-2 * u_original))
log_prob_a_2 = log_prob_u_original - jacobian_2
print(f"方法 2 (手动, 稳定):     -> {log_prob_a_2.item():.8f} (正确)")

# --- 方法 4: PyTorch 底层雅可比 ---
jacobian_4 = squashed_normal_dist.tanh_transform.log_abs_det_jacobian(u_original, a_original)
log_prob_a_4 = log_prob_u_original - jacobian_4
print(f"方法 4 (PyTorch 底层):   -> {log_prob_a_4.item():.8f} (正确)")

# ==============================================================================
# Part 2: 基于【反算 u】的计算 (现在将直接面对 a=1.0)
# ==============================================================================
print("\n--- Part 2: 尝试从 a_original = 1.0 反算 u ---")
# --- 方法 5: 手动模拟 atanh 反算流程 ---
print("方法 5 (手动, atanh):")
try:
    # 1. 从 a_original 反算出 u_inferred
    u_inferred_5 = torch.atanh(a_original)
    print(f"  - 1. atanh(1.0) -> {u_inferred_5.item()}")
    # 2. 计算 log p(u_inferred)
    log_prob_u_5 = base_normal_dist.log_prob(u_inferred_5)
    print(f"  - 2. log p(inf) -> {log_prob_u_5.item()}")
    # 3. 计算雅可比项 (基于 u_inferred)
    jacobian_5 = 2 * (math.log(2.) - u_inferred_5 - F.softplus(-2 * u_inferred_5))
    print(f"  - 3. jacobian(inf) -> {jacobian_5.item()}")
    # 4. 组合
    log_prob_a_5 = log_prob_u_5 - jacobian_5
    print(f"  - 4. (-inf) - (-inf) -> {log_prob_a_5.item()} (崩溃!)")
except Exception as e:
    print(f"  - 执行失败: {e}")


# --- 方法 3: PyTorch 自动 atanh 反算流程 ---
print("方法 3 (PyTorch 自动):")
try:
    log_prob_a_3 = squashed_normal_dist.log_prob(a_original)
    print(f"  - squashed_dist.log_prob(1.0) -> {log_prob_a_3.item()} (崩溃!)")
except Exception as e:
    print(f"  - 执行失败: {e}")

print("\n" + "="*70)
print("最终结论:")
print("="*70)
print("1. 当输入为精确的 1.0 时，所有依赖 `atanh` 反算的流程 (方法3和5) 都会因 `inf - inf` 而直接崩溃，输出 `nan`。")
print("2. 只有不依赖反算、直接使用原始 `u` 的流程 (方法2和4) 才能在这种极限情况下幸存，并给出正确的结果。")
print("3. 这证明了在 RL 算法中，必须采用基于原始 `u` 的手动计算流程，才能保证数值的绝对稳健性。")
print("="*70)