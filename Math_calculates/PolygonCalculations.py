import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString

def visualize_shapely(vertices, test_pt, velocity, visualize=0):
    # 1. 计算逻辑
    poly = Polygon(vertices)
    p = Point(test_pt)
    v = np.array(velocity, dtype=float)
    v_unit = v / np.linalg.norm(v) # 归一化速度矢量
    
    is_inside = poly.contains(p)
    boundary = poly.boundary
    dist_to_boundary = boundary.distance(p) # 最短距离
    
    # 获取最近的边界点坐标 (用于验证左/右)
    nearest_p_coords = boundary.interpolate(boundary.project(p))
    nearest_p = np.array([nearest_p_coords.x, nearest_p_coords.y])
    
    # 判定左右 (叉乘)
    d_vec = nearest_p - np.array(test_pt)
    cross_product = v[0] * d_vec[1] - v[1] * d_vec[0]
    side = -1 if cross_product > 0 else 1 if cross_product < 0 else 0

    if visualize:
        # 2. 绘图
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 画多边形
        x, y = poly.exterior.xy
        ax.plot(x, y, c='#6699cc', alpha=0.7, lw=3, label='Polygon Boundary')
        ax.fill(x, y, c='#6699cc', alpha=0.3)
        
        # 画测试点
        ax.scatter([test_pt[0]], [test_pt[1]], color='red', s=100, zorder=5, label='Test Point')
        
        # 画速度矢量箭头 (原始长度)
        ax.quiver(test_pt[0], test_pt[1], v[0], v[1], color='green', 
                angles='xy', scale_units='xy', scale=1, label='Velocity Vector')
        
        # 画：沿着速度方向，长度等于最短距离的线段 (要求的功能)
        v_line_end = np.array(test_pt) + v_unit * dist_to_boundary
        ax.plot([test_pt[0], v_line_end[0]], [test_pt[1], v_line_end[1]], 
                'g--', lw=2, label='Dist-length Line (along V)')
        
        # 画：连接到最近边界点的线 (用于验证左右逻辑)
        ax.plot([test_pt[0], nearest_p[0]], [test_pt[1], nearest_p[1]], 
                'r:', lw=2, label=f'To Nearest Point ({side})')
        
        # 设置
        ax.set_title(f"Inside: {is_inside} | Side: {side} | Distance: {dist_to_boundary:.2f}")
        ax.legend()
        ax.set_aspect('equal')
        plt.grid(True)
        plt.show()
    
    return d_vec, side


if __name__=="__main__":
    # 示例数据 (非凸多边形)
    poly_verts = [(0, 0), (5, 0), (5, 5), (3, 1), (1, 5), (0, 3)]
    test_point = (2, 1)
    velocity_vec = (1, 1)

    d_vec, side = visualize_shapely(poly_verts, test_point, velocity_vec, visualize=1)
    print("距离为：", d_vec, "方向为：", side)