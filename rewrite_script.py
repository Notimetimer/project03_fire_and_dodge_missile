import os
path = r'd:\3_Machine_Learning_in_Python\project03_fire_and_dodge_missile\TrainAndTests\Controls\FlightControl_Train_dual_a_out_parallel.py'
with open(path, 'r', encoding='utf-8') as f: content = f.read()

index = content.find("if __name__=='__main__':")
if index == -1: print('Could not find main block')
else:
    new_content = content[:index] + """import torch.multiprocessing as mp
import random
import traceback

def worker_process(rank, pipe, args, state_dim, hidden_dim, action_dims_dict, action_bound, device_worker, seed):
    try:
        worker_seed = seed + rank * 1000
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        env = track_env(tacview_show=0)
        dt_decide = 0.2
        
        local_actor = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device_worker)
        from Algorithms.MLP_heads import ValueNet
        local_dummy_critic = ValueNet(state_dim, hidden_dim).to(device_worker)
        from Algorithms.PPOHybrid23_0_distil2_one_step_KL import PPOHybrid, HybridActorWrapper

        local_agent = PPOHybrid(
            actor=HybridActorWrapper(local_actor, action_dims_dict, action_bounds=action_bound, device=device_worker).to(device_worker),
            critic=local_dummy_critic,
            actor_lr=0, critic_lr=0,
            lmbda=0, eps=0, gamma=0, epochs=0,
            device=device_worker
        )
        
        while True:
            cmd, packet = pipe.recv()
            if cmd == 'EXIT':
                break
            if cmd == 'RUN_EPISODE':
                actor_weights, episode_idx = packet
                local_agent.actor.load_state_dict(actor_weights)
                
                init_height = np.random.uniform(4000, 10000)
                birth_state={'position': np.array([0.0, init_height, 0.0]),
                                'psi': np.random.uniform(-pi/6, pi/6)}
                height_req = np.clip(init_height + np.random.choice([1,-1])*(np.random.uniform(0, 1)**2)*5000 , 3000, 13000)
                psi_req = np.random.uniform(-pi, pi) * np.clip(episode_idx/1000, 0, 1)
                v_req = np.random.uniform(0.8, 2.5)*340

                env.reset(birth_state=birth_state, height_req=height_req, psi_req=psi_req, v_req=v_req, dt_report=dt_decide)
                
                obs, obs_check = env.get_obs()
                done = False
                
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
                episode_return = 0
                steps_run = 0
                
                while not done:
                    obs, obs_check = env.get_obs()
                    action, u, _, _ = local_agent.take_action(obs, explore=True)
                    steps_run += 1
                    
                    next_obs, reward, done = env.step(action)
                    
                    transition_dict['states'].append(obs)
                    transition_dict['actions'].append(u)
                    transition_dict['next_states'].append(next_obs)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    transition_dict['action_bounds'].append(action_bound)
                    
                    obs = next_obs
                    episode_return += reward * env.dt_report
                
                metrics = {
                    'return': episode_return,
                    'steps': steps_run,
                    'fail': env.fail
                }
                pipe.send({'trans': transition_dict, 'metrics': metrics})
                
    except Exception as e:
        tb = traceback.format_exc()
        try: pipe.send({'error': tb})
        except: pass

if __name__=='__main__':
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser("UAV flight control training")
    parser.add_argument("--num_workers", type=int, default=10, help="number of parallel workers")
    parser.add_argument("--max-episode-len", type=float, default=3*60, help="maximum episode time length")
    parser.add_argument("--R-cage", type=float, default=np.inf, help="")
    args = parser.parse_args()

    # 创建一个 dummy env 获取维度
    dummy_env = track_env()
    teacher_agent = UnifiedPolicyWrapper(dummy_env)
    
    action_dims_dict = {'cont': action_dim, 'cat': [], 'bern': 0}
    policy_net = PolicyNetHybrid(state_dim, hidden_dim, action_dims_dict).to(device)
    actor = HybridActorWrapper(policy_net, action_dims_dict, action_bounds=action_bound, device=device)
    from Algorithms.MLP_heads import ValueNet
    critic = ValueNet(state_dim, hidden_dim).to(device)
    
    agent = PPOHybrid(actor, critic, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
        
    os.makedirs(log_dir, exist_ok=True)
    actor_meta_path = os.path.join(log_dir, "actor.meta.json")
    critic_meta_path = os.path.join(log_dir, "critic.meta.json")

    def save_meta_once(path, state_dict):
        if os.path.exists(path):
            return
        meta = {k: list(v.shape) for k, v in state_dict.items()}
        with open(path, "w") as f:
            json.dump(meta, f)

    save_meta_once(actor_meta_path, agent.actor.state_dict())
    save_meta_once(critic_meta_path, agent.critic.state_dict())

    from Visualize.tensorboard_visualize import TensorBoardLogger

    logger = TensorBoardLogger(log_root=log_dir, host="127.0.0.1", port=6006, use_log_root=True)
    
    # 启动多进程
    workers = []
    pipes = []
    worker_device = torch.device('cpu')
    seed = 42

    print(f"Initializing {args.num_workers} training workers...")
    for i in range(args.num_workers):
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=worker_process, args=(
            i, child_conn, args, state_dim, hidden_dim, 
            action_dims_dict, action_bound, worker_device, seed
        ))
        p.start()
        workers.append(p)
        pipes.append(parent_conn)

    try:
        rl_steps = 0
        i_episode = 0
        
        while rl_steps < max_steps:
            # 1. 深度拷贝当前 Actor 权重到 CPU
            current_weights = {k: v.cpu().clone() for k, v in agent.actor.state_dict().items()}
            
            # 2. 分发任务
            for rank in range(args.num_workers):
                pipes[rank].send(('RUN_EPISODE', (current_weights, i_episode + rank)))
            
            # 3. 阻塞等待结果
            batch_results = []
            for rank in range(args.num_workers):
                try: 
                    res = pipes[rank].recv() # 阻塞等待
                except EOFError: 
                    print(f"[Error] Worker {rank} crashed silently.")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed.")
                    
                if isinstance(res, dict) and 'error' in res:
                    print(f"--- Master received error from Worker {rank}, aborting. ---")
                    for p in workers: p.terminate()
                    raise RuntimeError(f"Worker {rank} crashed with error:\\n{res['error']}")
                    
                batch_results.append(res)
            
            # 4. 汇总数据
            master_transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': [], 'action_bounds': []}
            batch_return_list = []
            batch_fail_cnt = 0
            batch_steps_run = 0
            
            for res in batch_results:
                tr = res['trans']
                metrics = res['metrics']
                
                batch_return_list.append(metrics['return'])
                if metrics['fail']: batch_fail_cnt += 1
                batch_steps_run += metrics['steps']
                
                for k in master_transition_dict:
                    master_transition_dict[k].extend(tr[k])
                    
            rl_steps += batch_steps_run
            i_episode += args.num_workers
            
            # 5. 模型更新
            agent.update(master_transition_dict)
            agent.distil(master_transition_dict, teacher_agent=teacher_agent, epochs=1, alpha=1.0)
            
            # --- 保存模型 ---
            if (i_episode // args.num_workers) % 10 == 0:
                critic_path = os.path.join(log_dir, "critic.pt")
                th.save(agent.critic.state_dict(), critic_path)
                actor_name = f"actor_rein{i_episode}.pt"
                actor_path = os.path.join(log_dir, actor_name)
                th.save(agent.actor.state_dict(), actor_path)

            # --- 日志和控制台输出 ---
            mean_return = np.mean(batch_return_list)
            survive_rate = 1.0 - (batch_fail_cnt / args.num_workers)
            print(f"Episodes {i_episode}, 进度: {rl_steps / max_steps:.3f}, batch_return: {mean_return:.3f}, survive_rate: {survive_rate:.2f}")

            logger.add("train/0 episode_return", mean_return, rl_steps)
            logger.add("train/0 survive", survive_rate, rl_steps)

            actor_grad_norm = model_grad_norm(agent.actor)
            critic_grad_norm = model_grad_norm(agent.critic)
            logger.add("train/1 actor_grad_norm", actor_grad_norm, rl_steps)
            logger.add("train/2 critic_grad_norm", critic_grad_norm, rl_steps)
            logger.add("train/3 actor_loss", agent.actor_loss, rl_steps)
            logger.add("train/4 critic_loss", agent.critic_loss, rl_steps)
            logger.add("train/5 entropy", agent.entropy_mean, rl_steps)
            logger.add("train/6 ratio", agent.ratio_mean, rl_steps)
            logger.add("train/7 steps", i_episode, rl_steps)
            if hasattr(agent, 'dis_actor_loss') and agent.dis_actor_loss != 0:
                logger.add("train/8 distil_loss", agent.dis_actor_loss, rl_steps)

    except KeyboardInterrupt:
        print("\\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        for pipe in pipes:
            try: pipe.send(('EXIT', None))
            except: pass
                
        for p in workers:
            p.join(timeout=5)
            if p.is_alive(): p.terminate()
            
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Done rewriting")
