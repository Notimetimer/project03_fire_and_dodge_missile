'''
根据二分法计算动态可发射距离
'''
import numpy as np

if __name__ == '__main__':
    # 本该是可发射区远边界的解算实现
    b = 50e3
    a = 5e3
    epsilon = 1e3
    stop_flag = 0

    next_b_hit = None



    pass  # 解算限高，

    while b - a >= epsilon or not stop_flag:
        lambda1 = (a+b)/2
        hit1 = None  # 运行lambda1处计算
        if not next_b_hit:
            hit2 = None  # 运行b处计算
        else:
            hit2 = next_b_hit

        if hit2:
            stop_flag = 1
            output = b
        elif hit1:
            a = lambda1
            next_b_hit = hit2
        else:
            b = lambda1
            next_b_hit = hit1



