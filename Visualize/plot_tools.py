import numpy as np
import matplotlib.pyplot as plt

# axis equal
def set_axes_equal(ax):
    """确保3D图的坐标轴单位长度相等。"""
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])

    # 设置等显示缩放比例
    ax.set_box_aspect([1, 1, 1])

# axis equal
def set_axes_equal_manual(ax, x_limits=None, y_limits=None, z_limits=None):
    """确保3D图的坐标轴单位长度相等。"""
    if x_limits is None:
        x_limits1 = ax.get_xlim3d()
    else:
        x_limits1 = x_limits
    if y_limits is None:
        y_limits1 = ax.get_ylim3d()
    else:
        y_limits1 = y_limits
    if z_limits is None:
        z_limits1 = ax.get_zlim3d()
    else:
        z_limits1 = z_limits
    
    x_min, x_max = x_limits1
    y_min, y_max = y_limits1
    z_min, z_max = z_limits1

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min

    max_range = max(x_range, y_range, z_range)

    x_temp = max_range if x_limits is None else x_range
    y_temp = max_range if y_limits is None else y_range
    z_temp = max_range if z_limits is None else z_range

    x_middle = (x_max + x_min) / 2
    y_middle = (y_max + y_min) / 2
    z_middle = (z_max + z_min) / 2
    
    ax.set_xlim3d([x_middle - x_temp / 2, x_middle + x_temp / 2])
    ax.set_ylim3d([y_middle - y_temp / 2, y_middle + y_temp / 2])
    ax.set_zlim3d([z_middle - z_temp / 2, z_middle + z_temp / 2])

    # 设置等显示缩放比例
    ax.set_box_aspect([x_temp, y_temp, z_temp])