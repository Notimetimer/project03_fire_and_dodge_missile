import os
import sys
import numpy as np
import pickle
import torch
import argparse
import glob
import copy
import json
from datetime import datetime
# 保持您原有的引用，假设 TensorBoardLogger 在您的环境中已可用
# from tensorboardX import SummaryWriter as TensorBoardLogger 

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Envs.Tasks.ChooseStrategyEnv2 import *
from Algorithms.PPOHybrid2 import PPOHybrid, PolicyNetHybrid, HybridActorWrapper
from Algorithms.MLP_heads import ValueNet
from Visualize.tensorboard_visualize import TensorBoardLogger

def get_current_file_dir():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return os.getcwd()
        else:
            return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.dirname(os.path.abspath(__file__))

cur_dir = get_current_file_dir()

def load_il_and_transitions(folder, il_name, rl_name):
    if folder is None:
        folder = os.getcwd()
    il_path = os.path.join(folder, il_name)
    trans_path = os.path.join(folder, rl_name)
    il = None
    trans = None
    if os.path.isfile(il_path):
        with open(il_path, "rb") as f:
            il = pickle.load(f)
        print(f"Loaded IL data from: {il_path}")
    else:
        print(f"File NOT found: {il_path}")
        
    if os.path.isfile(trans_path):
        with open(trans_path, "rb") as f:
            trans = pickle.load(f)
    return il, trans

# ==========================================
# 核心修复：数据结构重组函数
# ==========================================
def restructure_actions(actions_data):
    """
    将 list of dicts [{'fly': 1, 'fire': 0}, ...] 
    转换为 dict of arrays {'cat': array([[1],...]), 'bern': array([[0],...])}
    并确保维度是 (N, 1) 以适配 PPOHybrid2 的索引操作
    """
    # 如果已经是字典格式，直接返回
    if isinstance(actions_data, dict):
        return actions_data
    
    # 如果是列表且包含字典，进行转换
    if isinstance(actions_data, list) and len(actions_data) > 0:
        print("Restructuring actions from List[Dict] to Dict[Array]...")
        
        # 初始化容器
        new_actions = {'cat': [], 'bern': []}
        
        for item in actions_data:
            # 兼容处理：item 可能是 dict，也可能是包含 dict 的 numpy array
            act = item
            if isinstance(item, np.ndarray) and item.dtype == object:
                act = item.item() # 提取 numpy 里的 dict
            
            # 映射 'fly' -> 'cat' (离散机动)
            # 映射 'fire' -> 'bern' (开关开火)
            if isinstance(act, dict):
                if 'fly' in act:
                    new_actions['cat'].append(act['fly'])
                if 'fire' in act:
                    new_actions['bern'].append(act['fire'])
            # 备用：如果数据意外变成了 list/tuple
            elif isinstance(act, (list, np.ndarray, tuple)) and len(act) >= 2:
                 new_actions['cat'].append(act[0])
                 new_actions['bern'].append(act[1])

        # 转换为 Numpy Array 并调整形状为 (Batch, 1)
        # 这一点至关重要：PPOHybrid2 里的 expert_cat[:, i] 需要 expert_cat 是二维的
        
        # 1. 'cat': 离散动作，转为 int64，Reshape 为 (N, 1)
        cat_arr = np.array(new_actions['cat'], dtype=np.int64)
        if cat_arr.ndim == 1:
            cat_arr = cat_arr.reshape(-1, 1)
        
        # 2. 'bern': 伯努利动作，转为 float32 (BCE Loss需要)，Reshape 为 (N, 1)
        bern_arr = np.array(new_actions['bern'], dtype=np.float32)
        if bern_arr.ndim == 1:
            bern_arr = bern_arr.reshape(-1, 1)

        result = {
            'cat': cat_arr,
            'bern': bern_arr
        }
        
        print(f"Structure fixed: 'cat' shape={cat_arr.shape}, 'bern' shape={bern_arr.shape}")
        return result

    return actions_data

def save_meta_once(path, state_dict):
    if os.path.exists(path):
        return
    meta = {k: list(v.shape) for k, v in state_dict.items()}
    with open(path, "w") as f:
        json.dump(meta, f)

def summarize(il_dict):
    if il_dict is None: return
    print("\nData Summary:")
    for k in ('states', 'actions', 'returns'):
        v = il_dict.get(k)
        if isinstance(v, dict):
            for sk, sv in v.items():
                print(f"  {k}['{sk}']: shape={sv.shape}")
        else:
            print(f"  {k}: shape={getattr(v, 'shape', 'Unknown')}")
            
def append_b_experience(td, state, action, reward, next_state, done):
    """
    统一把一次蓝方经验追加到 transition_dict。
    只负责按给定值追加，不处理死亡冻结逻辑（主流程中通过 dead_dict 控制何时计算 next_state/reward）。
    """
    td['states'].append(state)
    td['actions'].append(action)
    td['rewards'].append(reward)
    td['next_states'].append(next_state)
    td['dones'].append(done)
    return td

# 加载数据
il_transition_dict, transition_dict = load_il_and_transitions(
    os.path.join(cur_dir, "IL"),
    "il_transitions_combat.pkl",
    "transition_dict_combat.pkl"
)

# --- 关键步骤：执行数据重构 ---
if il_transition_dict is not None:
    # 这里完成 (Batch, Key) -> (Key, Batch) 的转换
    il_transition_dict['actions'] = restructure_actions(il_transition_dict['actions'])
    
    # 顺便确保 states 和 returns 也是标准的 float32 numpy array
    il_transition_dict['states'] = np.array(il_transition_dict['states'], dtype=np.float32)
    il_transition_dict['returns'] = np.array(il_transition_dict['returns'], dtype=np.float32)

# ------------------------------------------------------------------
# 参数与环境配置
# ------------------------------------------------------------------
parser = argparse.ArgumentParser("UAV swarm confrontation")
parser.add_argument("--max-episode-len", type=float, default=10*60, help="maximum episode time length")
parser.add_argument("--R-cage", type=float, default=55e3, help="")
args = parser.parse_args()

# 超参数
actor_lr = 1e-4
critic_lr = actor_lr * 5
max_steps = 65e4
hidden_dim = [128, 128, 128]
gamma = 0.95
lmbda = 0.95
epochs = 10
eps = 0.2
k_entropy = 0.01

env = ChooseStrategyEnv(args, tacview_show=0)
state_dim = env.obs_dim

# 动作空间定义 (需要与 BasicRules 产生的数据对应)
# cat: 离散机动动作头 (env.fly_act_dim 通常是一个列表 [n_actions])
# bern: 攻击动作头 (env.fire_dim 通常是 1)
action_dims_dict = {'cont': 0, 'cat': env.fly_act_dim, 'bern': env.fire_dim}
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
action_bound = None

# 1. 创建神经网络
actor_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
critic_net = ValueNet(state_dim, hidden_dim).to(device)

# 2. Wrapper
actor_wrapper = HybridActorWrapper(actor_net, action_dims_dict, action_bound, device).to(device)

# 3. student_agent
student_agent = PPOHybrid(
    actor=actor_wrapper, 
    critic=critic_net, 
    actor_lr=actor_lr, 
    critic_lr=critic_lr,
    lmbda=lmbda, 
    epochs=epochs, 
    eps=eps, 
    gamma=gamma, 
    device=device, 
    k_entropy=k_entropy, 
    max_std=0.3
)

adv_agent = copy.deepcopy(student_agent)



if __name__ == "__main__":
    
    # 根据参数数量缩放学习率
    from Math_calculates.ScaleLearningRate import scale_learning_rate
    actor_lr = scale_learning_rate(actor_lr, student_agent.actor)
    critic_lr = scale_learning_rate(critic_lr, student_agent.critic)
    student_agent.set_learning_rate(actor_lr=actor_lr, critic_lr=critic_lr)
    
    if il_transition_dict is None:
        print("No il_transitions_combat file found.")
        sys.exit(1)

    summarize(il_transition_dict)

    # 日志记录 (使用您自定义的 TensorBoardLogger)
    logs_dir = os.path.join(project_root, "logs/combat")
    mission_name = 'MARWIL_combat'
    log_dir = os.path.join(logs_dir, f"{mission_name}-run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    os.makedirs(log_dir, exist_ok=True)
    # --- 仅保存一次网络形状（meta json），如果已存在则跳过
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")
    
    save_meta_once(actor_meta_path, student_agent.actor.state_dict())
    save_meta_once(critic_meta_path, student_agent.critic.state_dict())
    
    # 保持您原有的 logger 初始化方式
    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True, auto_show=False)
    
    print("Start MARWIL Training...")

    # 训练循环
    # 现在 il_transition_dict['actions'] 已经是 {'cat': tensor, 'bern': tensor} 格式了
    # 能够被 MARWIL_update 里的 items() 正常遍历
    for epoch in range(80): 
        avg_actor_loss, avg_critic_loss, c = student_agent.MARWIL_update(
            il_transition_dict, 
            beta=1.0, 
            batch_size=128, # 显存如果够大可以适当调大
            label_smoothing=0.4
        )
        
        # 记录
        if epoch % 1 == 0:
            logger.add("il_train/avg_actor_loss", avg_actor_loss, epoch)
            logger.add("il_train/avg_critic_loss", avg_critic_loss, epoch)
            # logger.add("il_train/beta_c", c, epoch) # 如果 tensorboardlogger 支持的话

            print(f"Epoch {epoch}: Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")

    print("Training Finished.")

    # 第5步：开始强化学习阶段
    
    env = ChooseStrategyEnv(args, tacview_show=1)
    from BasicRules import *  # 可以直接读同一级目录
    t_bias = 0
    env.shielded = 1 # 不得不全程带上
    
    def creat_initial_state():
        # 飞机出生状态指定
        # todo: 随机出生点，确保蓝方能躲掉但不躲就会被打到
        blue_height = 9000
        red_height = 9000
        red_psi = -pi/2
        blue_psi = pi/2
        red_N = 0
        red_E = 45e3
        blue_N = red_N
        blue_E = -red_E
        DEFAULT_RED_BIRTH_STATE = {'position': np.array([red_N, red_height, red_E]),
                            'psi': red_psi
                            }
        DEFAULT_BLUE_BIRTH_STATE = {'position': np.array([blue_N, blue_height, blue_E]),
                                    'psi': blue_psi
                                    }
        return DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE

    dt_action_cycle = dt_maneuver * action_cycle_multiplier # student_agent takes action every dt_action_cycle seconds

    transition_dict_capacity = env.args.max_episode_len//dt_action_cycle + 1 # Adjusted capacity

    初始化ELO表格，为agent名称-原始ELO值-均值不变ELO值
    elo_json_path = os.path.join(log_dir, "elo_ratings.json")
    
    return_list = []

    total_steps = 0

    training_start_time = time.time()
    launch_time_count = 0

    t_bias = 0

    decide_steps_after_update = 0
    
    r_action_list = []
    b_action_list = []
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    
    while total_steps < int(max_steps):
        
        i_episode += 1
        test_run = 0
        
        根据json文件中的elo分数为红方选择一个历史策略，如果是规则那么 adv_is_rule 为TRUE，否则为FALSE
        
        
        
        episode_return = 0

        DEFAULT_RED_BIRTH_STATE, DEFAULT_BLUE_BIRTH_STATE = creat_initial_state()

        env.reset(red_birth_state=DEFAULT_RED_BIRTH_STATE, blue_birth_state=DEFAULT_BLUE_BIRTH_STATE,
                red_init_ammo=6, blue_init_ammo=6)
        r_action_label=0
        b_action_label=0
        last_decision_state = None
        current_action = None
        b_reward = None
        
        # 新增：每回合的死亡查询表（0 表示存活，1 表示已记录死亡瞬间）
        dead_dict = {'r': int(bool(env.RUAV.dead)), 'b': int(bool(env.BUAV.dead))}

        done = False

        env.dt_maneuver = dt_maneuver
        
        episode_start_time = time.time()
        
        last_r_action_label = 0
        last_b_action_label = 0
        
        # 环境运行一轮的情况
        steps_of_this_eps = -1 # 没办法了
        for count in range(round(args.max_episode_len / dt_maneuver)):
            # print(f"time: {env.t}")  # 打印当前的 count 值
            # 回合结束判断
            # print(env.running)
            current_t = count * dt_maneuver
            steps_of_this_eps += 1
            if env.running == False or done: # count == round(args.max_episode_len / dt_maneuver) - 1:
                # print('回合结束，时间为：', env.t, 's')
                break
            # 获取观测信息
            r_obs, r_check_obs = env.obs_1v1('r', pomdp=1)
            b_obs, b_check_obs = env.obs_1v1('b', pomdp=1)
            # 在这里将观测信息压入记忆
            env.RUAV.obs_memory = r_check_obs.copy()
            env.BUAV.obs_memory = b_check_obs.copy()

            # --- 智能体决策 ---
            # 判断是否到达了决策点（每 10 步）
            if steps_of_this_eps % action_cycle_multiplier == 0:
                # # **关键点 1: 完成并存储【上一个】动作周期的经验**
                # 如果这不是回合的第0步，说明一个完整的动作周期已经过去了
                if steps_of_this_eps > 0:
                    transition_dict = append_b_experience(transition_dict, last_decision_state, current_action, b_reward, b_obs, False)

                # **关键点 2: 开始【新的】一个动作周期**
                # 1. 记录新周期的起始状态
                last_decision_state = b_obs
                # 2. student_agent 产生一个动作
                
                # 红方
                r_state_check = env.unscale_state(r_check_obs)
                # 规则
                if adv_is_rule:
                    r_action_label, r_fire = basic_rules(env, 'r', r_state_check, 选择的规则智能体编号, last_action=last_r_action_label)
                    last_r_action_label = r_action_label
                # NN
                else:
                    将加载的智能体读取到adv_agent的actor上
                    r_action_exec, r_action_raw, _, r_action_check = adv_agent.take_action(r_obs, explore=1)
                    r_action_label = r_action_exec['cat'][0] # 返回可能是一个数组
                
                
                # 蓝方使用智能体                    
                # 规则
                # b_state_check = env.unscale_state(b_check_obs)
                # _, b_fire = basic_rules(env, 'b', b_state_check, 1, last_action=last_b_action_label)
                # last_b_action_label = b_action_label
                # NN
                b_action_exec, b_action_raw, _, b_action_check = student_agent.take_action(b_obs, explore=1)
                b_action_label = b_action_exec['cat'][0]
                b_fire = b_action_exec['bern'][0]
                
                # print("机动概率分布", b_action_check['cat'])
                # print("开火概率", b_action_check['bern'][0])
                
                decide_steps_after_update += 1
                
                b_action_list.append(np.array([env.t + t_bias, b_action_label]))
                current_action = {'cat': b_action_label, 'bern': b_fire}
                # current_action = np.array([b_action_label, b_fire])

            r_maneuver = env.maneuver14(env.RUAV, r_action_label)
            b_maneuver = env.maneuver14(env.BUAV, b_action_label)

            _, _, _, _, fake_terminate = env.step(r_maneuver, b_maneuver) # Environment updates every dt_maneuver
            done, b_reward, b_reward_assisted = env.combat_terminate_and_reward('b', b_action_label, b_fire)
            done = done or fake_terminate

            # Accumulate rewards between student_agent decisions
            episode_return += b_reward * env.dt_maneuver
            
            # 仅在蓝方刚被击毁（第一次检测到死亡瞬间）时，计算一次 reward 和 next_b_obs 并标记 dead_dict
            # ,未来将被alive_mask取代
            if dead_dict['b'] == 0:
                next_b_obs, next_b_check_obs = env.obs_1v1('b', pomdp=1)
                if env.BUAV.dead:
                    dead_dict['b'] = 1

            total_steps += 1
            '''显示运行轨迹'''
            # 可视化
            env.render(t_bias=t_bias)
        
        # # --- 回合结束处理 ---
        # **关键点 3: 存储【最后一个】不完整的动作周期的经验**
        # 循环结束后，最后一个动作周期因为 done=True 而中断，必须在这里手动存入
        if last_decision_state is not None:
            # # 若在回合结束前未曾在死亡瞬间计算 next_b_obs（例如超时终止或其他非击毁终止），做一次后备计算
            transition_dict = append_b_experience(transition_dict, last_decision_state, current_action, b_reward, next_b_obs, True)
            
        ELO更新
        
        # 更新agent
        if 1: # len(transition_dict['next_states']) >= transition_dict_capacity: # decide_steps_after_update >= transition_dict_capacity
            '''student_agent.update'''
            student_agent.update(transition_dict, adv_normed=False)
            decide_steps_after_update = 0
            # Clear transition_dict after update
            
            actor_grad_norm = student_agent.actor_grad
            actor_pre_clip_grad = student_agent.pre_clip_actor_grad
            critic_grad_norm = student_agent.critic_grad
            critic_pre_clip_grad = student_agent.pre_clip_critic_grad

            # 梯度监控
            logger.add("train/3 actor_grad_norm", actor_grad_norm, total_steps)
            logger.add("train/5 actor_pre_clip_grad", actor_pre_clip_grad, total_steps)
            logger.add("train/4 critic_grad_norm", critic_grad_norm, total_steps)
            logger.add("train/6 critic_pre_clip_grad", critic_pre_clip_grad, total_steps)
            # 损失函数监控
            logger.add("train/7 actor_loss", student_agent.actor_loss, total_steps)
            logger.add("train/8 critic_loss", student_agent.critic_loss, total_steps)
            # 强化学习actor特殊项监控
            logger.add("train/9 entropy", student_agent.entropy_mean, total_steps)
            logger.add("train/10 ratio", student_agent.ratio_mean, total_steps) 
            logger.add("train/11 episode/step", i_episode, total_steps)    

            transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
            
            total_steps += steps_of_this_eps + 1
            logger.add("train/1 episode_return", episode_return, total_steps)
            logger.add("train/2 win", env.win, total_steps)
            logger.add("train/2 lose", env.lose, total_steps)
            logger.add("train/2 draw", env.draw, total_steps)
        
        return_list.append(episode_return)
        
        env.clear_render(t_bias=t_bias)
        t_bias += env.t
        r_action_list = np.array(r_action_list)
        episode_end_time = time.time()  # 记录结束时间
        print(f"回合时长: {episode_end_time - episode_start_time} 秒")
        
        # --- 保存模型
        os.makedirs(log_dir, exist_ok=True)
        # critic overwrite
        critic_path = os.path.join(log_dir, "critic.pt")
        torch.save(student_agent.critic.state_dict(), critic_path)
        # actor RL snapshot
        if i_episode % 10 == 0:
            actor_name = f"actor_rein{i_episode}.pt"
            actor_path = os.path.join(log_dir, actor_name)
            torch.save(student_agent.actor.state_dict(), actor_path)

            # save elo ratings to json (atomic write)
            try:
                tmp_path = elo_json_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    # elo_ratings 已为短 key，直接保存即可
                    json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, elo_json_path)
            except Exception as e:
                print(f"Warning: failed to save elo json '{elo_json_path}': {e}")
        
        # 训练进度显示
        if (i_episode) >= 10:
            print(f"episode {i_episode}, 进度: {total_steps/max_steps:.3f}, return: {np.mean(return_list[-10:]):.3f}")
        else:
            print(f"episode {i_episode}, total_steps {total_steps}")
        

    # 训练结束处理
    
    training_end_time = time.time()  # 记录结束时间
    env.end_render() # 停止发送
    elapsed = training_end_time - training_start_time
    from datetime import timedelta
    td = timedelta(seconds=elapsed)
    d = td.days
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    print(f"总训练时长: {d}天 {h}小时 {m}分钟 {s}秒")
    
    # 保存最终 elo 到 json
    try:
        tmp_path = elo_json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(elo_ratings, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, elo_json_path)
    except Exception as e:
        print(f"Warning: failed to save final elo json '{elo_json_path}': {e}")

    # 打印最终的ELO排名
    if elo_ratings:
        print("\n--- Final ELO Rankings ---")
        sorted_elos = sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)
        for key, elo in sorted_elos:
            print(f"ELO: {elo:.0f} - {key}")
        print(f"Main Agent Final ELO: {main_agent_elo:.0f}")
        print("------------------------\n")
    logger.close()
    print(f"日志已保存到：{logger.run_dir}")
    
    