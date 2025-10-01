import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TrainAndTests.PPOLCrankTraining import *
import re

dt_maneuver= 0.08 # 0.2 
action_eps = 0.8 # 动作平滑度

# 找出日期最大的目录
def get_latest_log_dir(pre_log_dir, mission_name=None):
    # 匹配 run-YYYYMMDD-HHMMSS 目录
    # pattern = re.compile(r"run-(\d{8})-(\d{6})")
    if mission_name:
        pattern = re.compile(rf"{re.escape(mission_name)}-run-(\d{{8}})-(\d{{6}})")
    else:
        pattern = re.compile(r"run-(\d{8})-(\d{6})")
    max_dt = None
    latest_dir = None
    for d in os.listdir(pre_log_dir):
        m = pattern.match(d)
        if m:
            dt_str = m.group(1) + m.group(2)  # 'YYYYMMDDHHMMSS'
            if max_dt is None or dt_str > max_dt:
                max_dt = dt_str
                latest_dir = d
    if latest_dir:
        return os.path.join(pre_log_dir, latest_dir)
    else:
        return None

pre_log_dir = os.path.join("./logs")
log_dir = get_latest_log_dir(pre_log_dir, mission_name=mission_name)


# 测试训练效果
agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                lmbda, epochs, eps, gamma, device)

rein_list = sorted(glob.glob(os.path.join(log_dir, "actor_rein*.pt")))
sup_list = sorted(glob.glob(os.path.join(log_dir, "actor_sup*.pt")))
latest_actor_path = rein_list[-1] if rein_list else (sup_list[-1] if sup_list else None)
if latest_actor_path:
    # 直接加载权重到现有的 agent
    sd = th.load(latest_actor_path, map_location=device)
    agent.actor.load_state_dict(sd) # , strict=False)  # 忽略缺失的键
    print(f"Loaded actor for test from: {latest_actor_path}")

t_bias = 0

try:
    env = Battle(args, tacview_show=1)
    for i_episode in range(3):  # 10
        r_action_list=[]
        b_action_list=[]
        
        # 飞机出生状态指定
        init_case = np.random.randint(initial_states.shape[0])

        red_R = random.uniform(initial_states[init_case][2], min(50e3, initial_states[init_case][3])) # 目标随机游走的话，没法使用最大攻击区的数据
        
        blue_height = initial_states[init_case][0]
        red_height = initial_states[init_case][1]

        blue_psi = random.uniform(-pi, pi)
        red_psi = sub_of_radian(blue_psi, pi)
        red_beta = blue_psi
        red_N = red_R*cos(red_beta)
        red_E = red_R*sin(red_beta)

        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                                'psi': red_psi
                                }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([0.0, blue_height, 0.0]),
                                    'psi': blue_psi
                                    }
        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=0, blue_init_ammo=1)

        done = False
        
        env.dt_maneuver = dt_maneuver
        step = 0
        hist_b_action = np.zeros(3)

        while not done:
            # print(env.t)
            # 获取观测信息
            r_obs_n = env.attack_obs('r')
            b_obs_n = env.left_crank_obs('b')
            
            # 反向转回字典方便排查
            b_check_obs = copy.deepcopy(env.state_init)
            key_order = env.key_order
            # 将扁平向量 b_obs_n 按 key_order 的顺序还原到字典 b_check_obs
            arr = np.atleast_1d(np.asarray(b_obs_n)).reshape(-1)
            idx = 0
            for k in key_order:
                if k not in b_check_obs:
                    raise KeyError(f"key '{k}' not in state_init")
                v0 = b_check_obs[k]
                # 可迭代的按长度切片，还原为 list 或 ndarray（保留原类型）
                if isinstance(v0, (list, tuple, np.ndarray)):
                    length = len(v0)
                    slice_v = arr[idx: idx + length]
                    if isinstance(v0, np.ndarray):
                        b_check_obs[k] = slice_v.copy()
                    else:
                        b_check_obs[k] = slice_v.tolist()
                    idx += length
                else:
                    # 标量
                    b_check_obs[k] = float(arr[idx])
                    idx += 1
            if idx != arr.size:
                # 长度不匹配时给出提示（便于调试）
                print(f"Warning: flattened obs length mismatch: used {idx} of {arr.size}")

            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_obs_n.copy()
            env.BUAV.obs_memory = b_obs_n.copy()
            state = np.squeeze(b_obs_n)
            distance = norm(env.RUAV.pos_ - env.BUAV.pos_)
            # 开局就发射一枚导弹
            if env.BUAV.ammo>0:
                new_missile = env.BUAV.launch_missile(env.RUAV, env.t, missile_class)
                env.BUAV.ammo -= 1
                new_missile.side = 'blue'
                env.Bmissiles.append(new_missile)
                env.missiles = env.Rmissiles + env.Bmissiles
            # 机动决策
            r_action_n = decision_rule(ego_pos_=env.RUAV.pos_, ego_psi=env.RUAV.psi,
                                    enm_pos_=env.BUAV.pos_, distance=distance,
                                    ally_missiles=env.Rmissiles, enm_missiles=env.Bmissiles,
                                    o00=o00, R_cage=env.R_cage, wander=1
                                    )
            if np.isnan(b_obs_n).any() or np.isinf(b_obs_n).any():
                print('b_obs_n', b_check_obs)
                print()
            b_action_n, u = agent.take_action(state, action_bounds=action_bound, explore=True)
            
            # 动作平滑（实验性）
            b_action_n = action_eps*hist_b_action+(1-action_eps)*b_action_n
            hist_b_action = b_action_n

            r_action_list.append(r_action_n)
            b_action_list.append(b_action_n)

            _, _, _, _, fake_terminate = env.step(r_action_n, b_action_n)  # 2、环境更新并反馈
            done, b_reward, _ = env.attack_terminate_and_reward('b')
            done = done or fake_terminate # debug 这里需要解决下仿真结束判断的问题

            if env.RUAV.dead:
                print()

            step += 1
            env.render(t_bias=t_bias)
            time.sleep(0.01)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t

except KeyboardInterrupt:
    print("验证已中断")