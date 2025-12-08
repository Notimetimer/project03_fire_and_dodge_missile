import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from TrainAndTests.Attacks.PPOAttack__Train import *
import re
import glob
import torch as th

from Envs.Tasks.AttackManeuverEnv import AttackTrainEnv, dt_maneuver
# 引入我们刚才写的并行 Wrapper
from Algorithms.ParallelEnv import ParallelPettingZooEnv
from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger
from Math_calculates.ScaleLearningRate import scale_learning_rate

dt_maneuver= 0.2 
action_eps = 0 # np.array([0.5, 0.8, 0]) # 0.7 # 动作平滑度

from Utilities.LocateDirAndAgents import *

# --- 用与并行训练一致的 actor/agent 初始化并加载 checkpoint ---

hidden_dim = [128, 128, 128]  # 与训练脚本一致
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 获取环境维度（复用 AttackTrainEnv 的 args，如果不存在请手动构造）
try:
    tmp_env = AttackTrainEnv(args)
except Exception:
    tmp_env = AttackTrainEnv(argparse.Namespace(max_episode_len=120, R_cage=70e3, n_envs=1, max_steps=1))
tmp_env.reset()
b_action_dim = tmp_env.b_action_spaces[0].shape[0]
tmp_obs, _ = tmp_env.attack_obs('b')
state_dim = tmp_obs.shape[0]
del tmp_env

# action 定义（与训练一致）
action_bound = np.array([[-5000, 5000], [-pi, pi], [200, 600]])
action_dims_dict = {'cont': b_action_dim, 'cat': [], 'bern': 0}

# 构建 actor_net + wrapper，并把权重加载到 wrapper 上
actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

# optional critic (用于构建 PPOHybrid，如果只做推理可以不用)
critic_net = ValueNet(state_dim, hidden_dim).to(device)
agent = PPOHybrid(actor=actor_wrapper, critic=critic_net,
                  actor_lr=1e-4, critic_lr=5e-4,
                  lmbda=0.95, epochs=10, eps=0.2, gamma=0.9, device=device)

# 找 checkpoint（兼容 actor_rein 和 actor_save 命名）
pre_log_dir = os.path.join(project_root, "logs/attack")
log_dir = get_latest_log_dir(pre_log_dir, mission_name='Attack_Parallel')
actor_path = None
if log_dir:
    actor_path = load_actor_from_log(log_dir, number=None, rein_prefix="actor_rein")
    if actor_path is None:
        actor_path = load_actor_from_log(log_dir, number=None, rein_prefix="actor_save")

if not actor_path:
    print(f"No actor checkpoint found in {log_dir}")
else:
    sd = th.load(actor_path, map_location=device, weights_only=True)
    # sd may be state_dict for actor or for agent.actor; try to load safely
    try:
        actor_wrapper.load_state_dict(sd)
    except Exception:
        # try to extract actor keys if saved full agent
        if 'actor' in sd:
            actor_wrapper.load_state_dict(sd['actor'])
        else:
            # fallback: attempt to filter matching keys
            model_sd = actor_wrapper.state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(filtered)
            actor_wrapper.load_state_dict(model_sd)
    print(f"Loaded actor for test from: {actor_path}")

t_bias = 0

try:
    env = AttackTrainEnv(args, tacview_show=1) # 是否用Tacview可视化
    for i_episode in range(3):  # 10
        r_action_list=[]
        b_action_list=[]
        # 飞机出生状态指定
        init_distance = 90e3
        red_R_ = init_distance/2 # random.uniform(20e3, 60e3)
        blue_R_ = init_distance/2
        red_beta = pi # random.uniform(0, 2*pi)
        red_psi = 0 # random.uniform(0, 2*pi)
        red_height = 8e3 # random.uniform(3e3, 10e3)
        red_N = red_R_*cos(red_beta)
        red_E = red_R_*sin(red_beta)
        blue_height = 8e3 # random.uniform(3e3, 10e3)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi
                                }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_R_, blue_height, 0.0]),
                                    'psi': np.random.choice([pi/2, -pi/2])
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        env.dt_maneuver = dt_maneuver
        step = 0
        done = False
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            r_obs_n, r_obs_check = env.attack_obs('r')
            b_obs_n, b_obs_check = env.attack_obs('b')
            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_check.copy()
            env.BUAV.obs_memory = b_obs_check.copy()
            state = np.squeeze(b_obs_n)
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1
                                    )
            # r_action_n, u_r = agent.take_action(r_obs_n, action_bounds=action_bound, explore=False)
            b_action_n, u,  _, _  = agent.take_action(b_obs_n, explore=False)
            
            # # 动作平滑（实验性）
            # b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            # hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            # ### 发射导弹，这部分不受10step约束
            # distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            # # 发射导弹判决
            # if distance <= 80e3 and distance >= 5e3:  # 在合适的距离范围内每0.2s判决一次导弹发射
            #     launch_time_count = 0
            #     launch_missile_with_basic_rules(env, side='r')
            #     launch_missile_with_basic_rules(env, side='b')

            env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.get_terminate_and_reward('b')

            step += 1
            env.render(t_bias=t_bias)
            time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")