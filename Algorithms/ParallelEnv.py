import multiprocessing as mp
import numpy as np
import pickle
import cloudpickle

# --- Worker 函数：运行在独立的进程中 ---
def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    # 实例化环境
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == 'step':
                # data 是一个字典 {'r': ..., 'b': ...}
                r_action = data.get('r')
                b_action = data.get('b')

                # --- 关键：红方规则逻辑内嵌 ---
                # 如果主进程传来的红方动作是 None，说明由环境内部规则接管
                if r_action is None:
                    # 我们将在 Env 中新增一个 helper 函数来处理这个
                    if hasattr(env, 'get_red_rule_action'):
                        r_action = env.get_red_rule_action()
                    else:
                        # 降级处理：什么都不做
                        r_action = np.array([0, 0, 300]) 

                # 执行物理步进
                # results: (r_reward, b_reward, r_done, b_done, terminate)
                results = env.step(r_action, b_action)
                
                # 获取观测
                r_obs, _ = env.attack_obs('r')
                b_obs, _ = env.attack_obs('b')
                
                # 处理自动重置
                terminate = results[4]
                done = terminate # 或者根据需要定义 done
                
                info = {'win': env.win, 'lose': env.lose}

                if done:
                    env.reset()
                    r_obs, _ = env.attack_obs('r')
                    b_obs, _ = env.attack_obs('b')

                # 发回数据
                remote.send({
                    'r_obs': r_obs,
                    'b_obs': b_obs,
                    'r_reward': results[0], # 这里假设取标量奖励
                    'b_reward': results[1],
                    'done': done,
                    'info': info
                })

            elif cmd == 'reset':
                env.reset()
                r_obs, _ = env.attack_obs('r')
                b_obs, _ = env.attack_obs('b')
                remote.send({
                    'r_obs': r_obs,
                    'b_obs': b_obs
                })

            elif cmd == 'close':
                remote.close()
                break
                
            else:
                print(f"Unknown command: {cmd}")
                
    except KeyboardInterrupt:
        print('Worker KeyboardInterrupt')
    finally:
        remote.close()

# 用于序列化 lambda 函数或局部函数
class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        self.x = pickle.loads(ob)

# --- 主类：并行环境管理器 ---
class ParallelPettingZooEnv:
    def __init__(self, env_fns):
        """
        env_fns: 一个列表，包含创建环境的函数 [fn, fn, ...]
        """
        self.closed = False # [新增] 初始化关闭状态标记
        self.n_envs = len(env_fns)
        # 创建管道
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_envs)])
        
        # 启动进程
        self.ps = [
            mp.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        for p in self.ps:
            p.daemon = True # 主进程死后子进程自动退出
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_dict):
        """
        actions_dict: {'r': [action1, action2...], 'b': [action1, action2...]}
        支持 None，表示该 Agent 由 Worker 内部托管
        """
        # 1. 分发动作
        for i, remote in enumerate(self.remotes):
            worker_action = {
                'r': actions_dict['r'][i] if actions_dict['r'] is not None else None,
                'b': actions_dict['b'][i] if actions_dict['b'] is not None else None
            }
            remote.send(('step', worker_action))

        # 2. 收集结果
        results = [remote.recv() for remote in self.remotes]
        
        # 3. 堆叠数据 (Stacking)
        # 将结果列表转换为 numpy 数组，方便神经网络批处理
        # return: {'r_obs': (N, D), 'b_obs': (N, D), ...}
        
        return {
            'r_obs': np.stack([r['r_obs'] for r in results]),
            'b_obs': np.stack([r['b_obs'] for r in results]),
            'r_reward': np.stack([r['r_reward'] for r in results]),
            'b_reward': np.stack([r['b_reward'] for r in results]),
            'dones': np.stack([r['done'] for r in results]),
            'infos': [r['info'] for r in results]
        }

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        results = [remote.recv() for remote in self.remotes]
        
        return {
            'r_obs': np.stack([r['r_obs'] for r in results]),
            'b_obs': np.stack([r['b_obs'] for r in results])
        }

    def close(self):
        # 1. 尝试发送关闭指令
        if self.closed:  # [新增] 避免重复关闭
            return

        for remote in self.remotes:
            try:
                # [新增] 检查连接状态再发送，或者捕获发送失败的异常
                remote.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                # 如果管道已经断开，说明子进程可能已经退出了，直接忽略
                pass
        
        # 2. 等待子进程退出
        for p in self.ps:
            p.join()
            
        self.closed = True # [新增] 标记为已关闭