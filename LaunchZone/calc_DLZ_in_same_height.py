'''
根据二分法计算动态可发射距离
'''
import numpy as np
from math import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LaunchZone.calc_hit_points import *
from Envs.MissileModel0910 import *

# 继续沿用导弹模型的命中/miss计算
g = 9.81
theta_limit = 85 * pi / 180

dt = 0.2
t = 0
g_ = np.array([0, -g, 0])
theta_limit = 85 * pi / 180


# 提取最后一行
def latest(vectors):
    if vectors.ndim == 1:  # 一维数组
        return vectors  # 将一维数组转换为二维数组
    elif vectors.ndim == 2:  # 二维数组
        return vectors[-1]  # 直接返回最后一行

def calc_range(a, b, epsilon, p_carrier_, v_carrier_, v_target_, move_pattern):
    calc_times = 0
    break_flag = 0
    next_b_hit = None
    while b - a >= epsilon and not break_flag:
        calc_times+=1
        lambda1 = (a+b)/2
        p_target_ = np.array([lambda1, height0, 0], dtype='float64')
        hit1 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                        show=0) # lambda
        if next_b_hit is None:
            calc_times+=1
            p_target_ = np.array([b, height0, 0], dtype='float64')
            hit2 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
                        show=0) # 运行b处计算
        else:
            hit2 = next_b_hit

        if hit2:
            break_flag = 1
        elif hit1:
            a = lambda1
            next_b_hit = hit2
        else:
            b = lambda1
            next_b_hit = hit1
    far_edge = (a+b)/2 if not break_flag else b
    return far_edge, calc_times

if __name__ == '__main__':
    # 假设我机和目标机都处在
    height0 = 10e3

    p_carrier_ = np.array([0, height0, 0], dtype='float64')
    v_carrier_ = np.array([320*1.2, 0, 0], dtype='float64')
    v_target_ = np.array([-320*1.2, 0, 0], dtype='float64')

    move_patterns = np.array(range(5)) + 1

    import time
    # # 串行计算攻击区
    # start_time = time.time()
    # far_edge = np.zeros_like(move_patterns)
    # calc_times = np.zeros_like(move_patterns)

    # for i, move_pattern in enumerate(move_patterns):
    #     b = 80e3
    #     a = 5e3
    #     epsilon = 1e3
    #     break_flag = 0

    #     next_b_hit = None
    #     while b - a >= epsilon and not break_flag:
    #         calc_times[i]+=1
    #         lambda1 = (a+b)/2
    #         p_target_ = np.array([lambda1, height0, 0], dtype='float64')
    #         hit1 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
    #                       show=0) # lambda
    #         if next_b_hit is None:
    #             calc_times[i]+=1
    #             p_target_ = np.array([b, height0, 0], dtype='float64')
    #             hit2 = sim_hit(p_carrier_, v_carrier_, p_target_, v_target_, target_move=move_pattern, datalink=1,
    #                       show=0) # 运行b处计算
    #         else:
    #             hit2 = next_b_hit

    #         if hit2:
    #             break_flag = 1
    #         elif hit1:
    #             a = lambda1
    #             next_b_hit = hit2
    #         else:
    #             b = lambda1
    #             next_b_hit = hit1
        
    #     far_edge[i] = (a+b)/2 if not break_flag else b

    # print(far_edge)
    # print(calc_times)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"串行程序运行时长: {execution_time} 秒") 

    # 并行计算
    move_patterns = np.array(range(5)) + 1
    from joblib import Parallel, delayed
    t0 = time.time()
    move_patterns = move_patterns.tolist()
    b = 80e3
    a = 5e3
    epsilon = 1e3
    parallel_tasks = [
        delayed(calc_range)(
            a, b, epsilon, p_carrier_, v_carrier_, v_target_, move_pattern)
        for move_pattern in move_patterns
    ]
    # 执行并行任务
    far_edges_and_times = Parallel(n_jobs=5)(parallel_tasks)
    t1 = time.time()
    far_edges = []
    times = []
    for far_edge, calc_time in far_edges_and_times:
        far_edges.append(far_edge)
        times.append(calc_time)
    print(far_edges)
    print(times)
    print(f"并行程序运行时长: {t1-t0} 秒")
    


