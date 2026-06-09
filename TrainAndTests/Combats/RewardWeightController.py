import numpy as np
"""
1、"ema_fire_interval"[1]平均开火时间小于30s的情况下，开火惩罚内部的等待时间权重需要急剧增大，高于100s的时候权重需要急剧缩小，中间状态浮动
2、"ema_fire_delta_psi"[3]平均开火abs(delta_psi)>30的情况下，这一项权重需要急剧增大，小于30的时候可以缓慢衰减，但是不低于初始设定值
3、"ema_fire_theta"[5]平均开火瞬间的theta角度低于0的时候，这个权重需要急剧增大，高于10的时候可以缓慢衰减
4、"ema_ATA"[外部]平均ATA小于40的时候，开火惩罚需要整体减小，让crank奖励起作用，平均ATA高于50的时候缓慢恢复开火惩罚
5、"ema_delta_psi_threat"[外部]平均delta_psi小于90的时候需要减小开火权重，高于120的时候可以缓慢恢复开火权重
6、"ema_delta_theta"[外部]平均delta_theta小于0的时候开始减小开火整体权重，小于-5的时候急剧增大。大于0的时候可以慢慢恢复开火整体权重
"""

class FireRewardWeightController:
    def __init__(self, initial_fire_reward_weight=1.0, fire_internal_weights_num=5, lr_internal=0.05, lr_external=0.05):
        """
        带误差归一化与抗积分饱和的开火权重控制器
        """
        # 1. 内部 self.fire_internal_weights_num 维 Logits: [distance, time, AA, delta_psi, v, theta]
        self.fire_internal_weights_num = fire_internal_weights_num
        self.logits = np.zeros(fire_internal_weights_num)
        self.lr_in = lr_internal
        
        # 2. 外部独立权重
        self.fire_reward_weight = initial_fire_reward_weight
        self.lr_out = lr_external # 可以删了
        
        # 归一化后的全局统一增益系数 (可根据需要微调这4个核心参数)
        self.params = {
            'k_in_sq': 1.0,       # 内部二次误差增益 (替代原 alpha, beta)
            # 'k_in_decay': 0.05,   # 内部向基准回落的衰减率 (替代 lambda)
            
            # 'k_out_lin': 1.0,     # 外部线性误差增益 (替代原线性 gamma)
            # 'k_out_sq': 2.5,      # 外部二次误差增益 (替代原平方 gamma)
            
            'weight_min': 0.3, 'weight_max': 3.0  # 外部限幅
        }

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def update(self, ema_vars):
        # 前置检查：所有键必须存在且值不能为 None，任一缺失/None 都拒绝更新
        required_keys = {
            'ema_fire_interval', 'ema_fire_distance', 'ema_fire_altitude', 'ema_fire_delta_psi', 'ema_fire_theta',
            'ema_ATA', 'ema_delta_psi_threat', 'ema_delta_theta'
        }
        if not required_keys.issubset(ema_vars.keys()) or None in ema_vars.values():
            return np.ones(self.fire_internal_weights_num), self.fire_reward_weight

        t_interval   = ema_vars['ema_fire_interval']
        dist         = ema_vars['ema_fire_distance']
        altitude     = ema_vars['ema_fire_altitude']
        d_psi        = ema_vars['ema_fire_delta_psi']
        theta        = ema_vars['ema_fire_theta']
        ata          = ema_vars['ema_ATA']
        psi_threat   = ema_vars['ema_delta_psi_threat']
        delta_theta      = ema_vars['ema_delta_theta']
        
        p = self.params
        d_logits = np.zeros(self.fire_internal_weights_num)
        
        # ==========================================
        # 一、 内部 self.fire_internal_weights_num 维权重积分逻辑 (带误差缩放)
        # ==========================================
        
        # 需求 1：时间控制 -> logits[1] (缩放分母: 100)
        if t_interval < 60: # 开火太密集，应该加大等待惩罚比例
            err = (60 - t_interval)/40.0
            d_logits[0] += p['k_in_sq'] * min(err, 1.0)
        elif t_interval > 60: # 开火比较稀疏，可以降低这部分惩罚的比重
            err = (t_interval - 60) / 40.0
            d_logits[0] -= p['k_in_sq'] * min(err, 1.0) * 0.7
            
        # 需求 2：开火偏角控制 -> logits[3] (缩放分母: 180)
        abs_d_psi = abs(d_psi)
        if abs_d_psi > 30:
            err = (abs_d_psi - 30) / 30.0
            d_logits[2] += p['k_in_sq'] * 0.2 # err
        else:
            d_logits[2] -= p['k_in_sq'] * 0.2

        # 需求 3：俯仰角控制 -> logits[5] (缩放分母: 90)
        if theta < 0: # 往地上开火，加大开火的俯仰惩罚
            # err = (-theta) / 15.0
            d_logits[4] += p['k_in_sq'] * 0.1
        elif theta > 10: # 会高抛，可以给其它成分奖励机会
            d_logits[4] -= p['k_in_sq'] * 0.1 # 慢慢降下来
        
        # 学不会开火后下高，降低高抛奖励权重
        if delta_theta < 0:
            d_logits[4] -= p['k_in_sq'] * 0.1
        # 学会开火后下高，慢慢回复高抛奖励权重
        else: # 开火后知道要下高了，慢慢把开火惩罚加回来
            d_logits[4] += p['k_in_sq'] * 0.1  # 0.05

        # 初步计算下一步的 logits 并去均值
        logits_next = self.logits + self.lr_in * d_logits
        logits_next -= np.mean(logits_next)

        # Anti-Windup 反向压制: 限制开火内部相对权重在 [1/30, 5/self.fire_internal_weights_num]
        for i in range(self.fire_internal_weights_num):
            other_indices = [j for j in range(self.fire_internal_weights_num) if j != i]
            S_minus_i = np.sum(np.exp(logits_next[other_indices]))
            
            z_max = np.log(7.0) + np.log(S_minus_i)
            z_min = np.log(1.0 / (self.fire_internal_weights_num * 7.0 - 1)) + np.log(S_minus_i)
            
            logits_next[i] = np.clip(logits_next[i], z_min, z_max)

        self.logits = logits_next
        fire_inside_weight = self._softmax(self.logits) * self.fire_internal_weights_num

        # ==========================================
        # 二、 外部独立权重积分逻辑 (带误差缩放)
        # ==========================================
        multiplier = 1.0
        
        # 需求 4：ATA (缩放分母: 180)
        if ata < 55: # 开火后不知道要crank，关小开火奖励
            multiplier *= 0.99
        elif ata > 55: # 学会crank了，慢慢加回开火奖励
            multiplier *= 1.01
            
        # 需求 5：psi_threat (缩放分母: 180)
        if psi_threat < 90: # 被威胁了还不知道要躲，缩小开火奖励让机动奖励显现出来
            multiplier *= 0.995
        elif psi_threat > 120: # 学会规避了，慢慢加回开火引导
            multiplier *= 1.002
            
        # 需求 6：delta_theta (缩放分母: 90)
        if delta_theta < 0: # 开火后还在爬升，这个时候应该弱化开火惩罚，让机动奖励教它低头
            multiplier *= 0.995
        elif 0 <= delta_theta < 15: # 开火后轻微低头，没学会，依然弱化开火惩罚
            multiplier *= 0.998
        else: # 开火后知道要下高了，慢慢把开火惩罚加回来
            multiplier *= 1.002

        # 外部执行硬限幅
        self.fire_reward_weight = np.clip(self.fire_reward_weight * multiplier, 1.0, 1.0) # p['weight_min'], p['weight_max'])
        
        return fire_inside_weight, self.fire_reward_weight

    def state_dict(self):
        """返回可序列化的状态字典"""
        return {
            'logits': self.logits.tolist(),
            'fire_reward_weight': float(self.fire_reward_weight),
            'lr_in': float(self.lr_in),
            'lr_out': float(self.lr_out),
        }

    def load_state_dict(self, state_dict):
        """从字典恢复状态"""
        self.logits = np.array(state_dict['logits'], dtype=np.float64)
        self.fire_reward_weight = float(state_dict['fire_reward_weight'])
        self.lr_in = float(state_dict.get('lr_in', 0.1))
        self.lr_out = float(state_dict.get('lr_out', 0.1))

# ==========================================
# 验证脚本
# ==========================================
if __name__ == "__main__":
    controller = FireRewardWeightController(initial_fire_reward_weight=0.5, lr_internal=0.1, lr_external=0.1)
    
    mock_ema_vars = {
        'ema_fire_interval': 60,
        'ema_fire_delta_psi': 30,
        'ema_fire_theta': -4.0,
        'ema_ATA': 40,
        'ema_delta_psi_threat': 135,
        'ema_delta_theta': 7.0
    }
    
    inside_rate, reward_rate = controller.update(mock_ema_vars)
    
    print("【内部 self.fire_internal_weights_num 维权重】(distance, time, AA, d_psi, v, theta):\n", inside_rate)
    print("内部权重总和 (保持为6.0):", inside_rate.sum())
    print("\n【外部独立总权重】(fire_reward_weight):\n", reward_rate)