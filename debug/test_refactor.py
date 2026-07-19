import os, sys, math, numpy as np, torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'Algorithms'))

from Algorithms.PPOHybrid23_0GRU import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.NotMLP_heads import ValueNet

state_dim = 30
hidden_dim = [64, 64]
action_dims = {'cont': 0, 'cat': [3], 'bern': 1}
device = torch.device('cpu')

actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims).to(device)
critic_net = ValueNet(state_dim, hidden_dim).to(device)
actor = HybridActorWrapper(actor_net, action_dims, None, device).to(device)
agent = PPOHybrid(actor, critic_net, 1e-4, 5e-4, 0.95, 1, 0.2, 0.99, device)

# 1. 检查网络输出维度
s = torch.randn(1, state_dim).to(device)
out, h_out = actor_net(s, return_h=True)
print('Actor forward output keys:', list(out.keys()))
print('Actor dummy h_out shape:', h_out.shape)

v, h_c = critic_net(s, return_h=True)
print('Critic value shape:', v.shape)
print('Critic h shape:', h_c.shape, 'bidirectional:', critic_net.gru.bidirectional)

# 2. 检查评估接口
actions = {'cat': torch.tensor([[0]]), 'bern': torch.tensor([[1.0]])}
log_probs, entropy, *_ = actor.evaluate_actions(s, actions)
print('Log probs shape:', log_probs.shape)

# 3. 检查 5-step reshape + PPO update 是否可运行
B, T = 4, 5
transition_dict = {
    'states': np.random.randn(B * T, state_dim).astype(np.float32),
    'obs': np.random.randn(B * T, state_dim).astype(np.float32),
    'actions': {
        'cat': np.random.randint(0, 3, (B * T, 1)).astype(np.int64),
        'bern': np.random.randint(0, 2, (B * T, 1)).astype(np.float32),
    },
    'next_states': np.random.randn(B * T, state_dim).astype(np.float32),
    'rewards': np.random.randn(B * T).astype(np.float32),
    'dones': np.zeros(B * T, dtype=np.float32),
    'active_masks': np.ones(B * T, dtype=np.float32),
    'actor_h': [np.zeros((1, actor_net.gru_hidden_size), dtype=np.float32) for _ in range(B * T)],
    'critic_h': [np.zeros((critic_net.gru.num_layers * (2 if critic_net.gru.bidirectional else 1), critic_net.gru.hidden_size), dtype=np.float32) for _ in range(B * T)],
}
seq_dict = agent.reshape_for_rnn(transition_dict, seq_len=5)
print('Reshape keys:', list(seq_dict.keys()))
print('init_h_actor shape:', seq_dict['init_h_actor'].shape)
print('init_h_critic shape:', seq_dict['init_h_critic'].shape)
agent.update(seq_dict, adv_normed=0, mini_batch_size=4)
print('Update completed, actor_loss:', agent.actor_loss, 'critic_loss:', agent.critic_loss)

# 4. 开火概率保护仍可运行
agent.fire_prob_protection(seq_dict, protect_epochs=1)
print('fire_prob_protection completed')
