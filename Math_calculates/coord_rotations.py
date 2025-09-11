from math import *
import numpy as np


def active_rotation(vector, heading, theta, gamma):
    # vector是行向量，根据psi，theta，gamma的顺序旋转坐标系，最后输出行向量
    # 注意：北天东坐标
    psi = - heading
    R1 = np.array([
        [cos(psi), 0, sin(psi)],
        [0, 1, 0],
        [-sin(psi), 0, cos(psi)]
    ])
    R2 = np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cos(gamma), -sin(gamma)],
        [0, sin(gamma), cos(gamma)]
    ])
    return vector @ R1.T @ R2.T @ R3.T


def passive_rotation(vector, heading, theta, gamma):
    # vector是行向量，根据psi，theta，gamma的顺序旋转坐标系，最后输出行向量
    # 注意：北天东坐标
    psi = - heading
    R1 = np.array([
        [cos(psi), 0, -sin(psi)],
        [0, 1, 0],
        [sin(psi), 0, cos(psi)]
    ])
    R2 = np.array([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
    R3 = np.array([
        [1, 0, 0],
        [0, cos(gamma), sin(gamma)],
        [0, -sin(gamma), cos(gamma)]
    ])
    return vector @ R1.T @ R2.T @ R3.T
