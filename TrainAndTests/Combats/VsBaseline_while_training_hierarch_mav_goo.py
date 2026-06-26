"双agent(maverick+goose)版本的并行测试Worker"

import os
import sys
import numpy as np
import torch
import random

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from BasicRules_new_hierarchical import *
from Envs.Tasks.ChooseStrategyEnv2_2_hierarchical import *
from Algorithms.PPOHybrid23_0 import PolicyNetHybrid, HybridActorWrapper


def test_worker(model_state_dict, rule_num,
                env_args, state_dim, hidden_dim,
                action_dims_dict, dt_maneuver_val,
                device_name='cpu', num_runs=1, action_cycle_multiplier=10,
                no_out=0, deterministic=False, restrict_fire=False, vertices=None, auto_regressive=0, Temperature=None):
    """
    双agent版本的并行测试Worker。
    model_state_dict 必须是含 'maverick' 和 'goose' 键的合并dict。
    maverick 负责 cat 机动动作，goose 负责 bern 开火动作。
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device(device_name)

    env_args.R_cage = 62.00e3
    env_args.max_episode_len = 15 * 60

    # 1. 局部初始化环境
    test_env = ChooseStrategyEnv(env_args, tacview_show=0, vertices=vertices)
    test_env.shielded = 1
    test_env.no_out = no_out
    test_env.dt_move = 0.04
    test_env.dt_maneuver = dt_maneuver_val

    # 2. 初始化双agent网络并加载权重
    action_dims_dict_mav = {'cont': 0, 'cat': action_dims_dict['cat'], 'bern': 0}
    action_dims_dict_goo = {'cont': 0, 'cat': 0, 'bern': action_dims_dict['bern']}

    mav_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict_mav).to(device)
    mav_actor = HybridActorWrapper(mav_net, action_dims_dict_mav, None, device).to(device)
    mav_actor.load_state_dict(model_state_dict['maverick'])
    mav_actor.eval()

    goo_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict_goo).to(device)
    goo_actor = HybridActorWrapper(goo_net, action_dims_dict_goo, None, device).to(device)
    goo_actor.load_state_dict(model_state_dict['goose'])
    goo_actor.eval()

    action_cycle = 10

    # 3. 运行对战逻辑
    result = 0
    result2 = 0
    wins = 0
    loses = 0
    draws = 0
    BVR_perish_togethers = 0

    for _ in range(num_runs):
        test_env.reset(red_init_ammo=6, blue_init_ammo=6, pomdp=1)

        steps = 0
        done = False
        b_action_label = 0
        episode_return = 0

        while not done and test_env.running:
            r_obs, r_check = test_env.obs_1v1('r', pomdp=1)
            b_obs, b_check = test_env.obs_1v1('b', pomdp=1)

            if steps % action_cycle == 0:
                # 红方 (规则)
                r_state_check = test_env.unscale_state(r_check)
                r_action_label, r_fire = basic_rules(r_state_check, rule_num)
                if r_fire:
                    test_env.RUAV.about_to_fire = 1

                # 蓝方 (双agent神经网络)
                with torch.no_grad():
                    b_mav_exec, _, _, _ = mav_actor.get_action(b_obs, explore=1, mask_on=0)
                    b_goo_exec, _, _, _ = goo_actor.get_action(b_obs, explore=1, mask_on=0)
                    b_act_exec = {'cat': b_mav_exec['cat'], 'bern': b_goo_exec['bern']}

                b_action_label = b_act_exec['cat']
                if b_act_exec['bern'][0]:
                    test_env.BUAV.about_to_fire = 1

            # 尝试发射
            if getattr(test_env.RUAV, 'about_to_fire', 0):
                r_act_label_to_pass = r_action_label if (restrict_fire or auto_regressive) else None
                tabu_fire = 1 if restrict_fire else 0
                launch_missile_immediately(test_env, 'r', action_label=r_act_label_to_pass, tabu=tabu_fire)
            b_m_id = None
            if getattr(test_env.BUAV, 'about_to_fire', 0):
                b_act_label_to_pass = b_action_label if (restrict_fire or auto_regressive) else None
                tabu_fire = 1 if restrict_fire else 0
                b_m_id = launch_missile_immediately(test_env, 'b', action_label=b_act_label_to_pass, tabu=tabu_fire)

            # 物理步
            r_maneuver = test_env.maneuver14LR(test_env.RUAV, r_action_label)
            b_maneuver = test_env.maneuver14LR(test_env.BUAV, b_action_label)
            test_env.step(r_maneuver, b_maneuver)

            # 判定
            done, b_reward1, b_reward2, b_reward3 = test_env.combat_terminate_and_reward(
                'b', b_action_label, b_m_id is not None, action_cycle)
            steps += 1

            if steps % action_cycle == 0:
                episode_return += b_reward1

            if steps * dt_maneuver_val > env_args.max_episode_len:
                break

        WVR = test_env.close_range_kill()
        BVR_perish_together = (not WVR) and test_env.draw and test_env.t <= env_args.max_episode_len - 10

        if test_env.win:
            result += 1 / num_runs
            wins += 1 / num_runs
        elif test_env.lose:
            result += 0 / num_runs
            loses += 1 / num_runs
        else:
            result += 0.5 / num_runs
            draws += 1 / num_runs
        result2 += episode_return / num_runs
        BVR_perish_togethers += BVR_perish_together / num_runs

    return rule_num, result, result2, wins, loses, draws, BVR_perish_togethers
