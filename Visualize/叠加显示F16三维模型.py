import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from math import *
import trimesh
import trimesh.transformations as tf

def check_dependencies():
    """检查所需的依赖库"""
    required_libs = {'trimesh': 'pip install trimesh', 'matplotlib': 'pip install matplotlib', 'numpy': 'pip install numpy'}
    for lib, install_cmd in required_libs.items():
        try: __import__(lib)
        except ImportError: return False
    return True

def get_transformed_mesh(base_mesh, N, U, E, psi, theta, phi, scale=2.0):
    """
    输入内容：
    N, U, E: 物理位置 (North, Up, East)
    psi, theta, phi: 航向、俯仰、滚转 (rad)
    """
    # 拷贝基础模型
    mesh = base_mesh.copy()
    
    M_scale = tf.scale_matrix(scale)

    # 1. 初始校准 (M_calib)：使原始轴对准物理“北”
    # 之前成功的参数：pi/2 绕物理 Y(上)轴
    # 这一步建立物理空间基准坐标系
    M_calib = tf.rotation_matrix(pi/2, [0, 1, 0])

    # 2. 姿态旋转 (M_pose)：依据输入欧拉角进行随体旋转
    r_x = tf.rotation_matrix(theta, [1, 0, 0])  # 俯仰
    r_y = tf.rotation_matrix(-psi,  [0, 1, 0])  # 航向
    r_z = tf.rotation_matrix(-phi,  [0, 0, 1])  # 滚转
    M_pose = tf.concatenate_matrices(r_y, r_x, r_z)
    
    # 3. 物理平移 (M_trans)
    # 根据重映射关系 E=-Phys_Z, N=-Phys_X, U=Phys_Y 逆推：
    # 为了在图中移动到 (E, N, U)，物理坐标应平移 [-N, U, -E]
    M_trans = tf.translation_matrix([-N, U, -E])
    
    # --- 最终变换顺序：先姿态旋转，再物理平移 ---
    # 在 trimesh/transformations.py 中：
    # 想要实现“旋转后再平移”，在 concatenate 列表中 M_trans 必须放在最前面。
    M_final = tf.concatenate_matrices(M_trans, M_calib, M_pose, M_scale)
    
    # 一次性应用所有线性变换
    mesh.apply_transform(M_final)
    
    return mesh.vertices, mesh.faces

def render_enu_scene(mesh_data_list):
    """
    渲染全场景。这里的输入是已经在物理 NUE 坐标系下处理好的顶点集合。
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    all_plot_verts = []

    for verts, faces in mesh_data_list:
        # --- 最终坐标映射：物理(N, U, E) -> 绘图视图(E, N, U) ---
        plot_verts = np.zeros_like(verts)
        plot_verts[:, 0] = -verts[:, 2]  # E -> Plot X
        plot_verts[:, 1] = -verts[:, 0]  # N -> Plot Y
        plot_verts[:, 2] = verts[:, 1]   # U -> Plot Z
        
        all_plot_verts.append(plot_verts)
        
        # 添加模型到图形中
        tri = Poly3DCollection(plot_verts[faces], alpha=0.8, facecolor='red', edgecolor='darkred', linewidth=0.1)
        ax.add_collection3d(tri)

    # 计算全局视野范围
    combined = np.vstack(all_plot_verts)
    max_range = np.array([combined[:,0].max()-combined[:,0].min(),
                         combined[:,1].max()-combined[:,1].min(),
                         combined[:,2].max()-combined[:,2].min()]).max() / 2.0
    mid = (combined.max(axis=0) + combined.min(axis=0)) / 2.0
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    # 绘制 E, N, U 指示轴
    axis_len = max_range * 1.5
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='blue', label='E (East)')
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='green', label='N (North)')
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='black', label='U (Up)')
    ax.text(axis_len, 0, 0, 'E', color='blue', fontsize=12, fontweight='bold')
    ax.text(0, axis_len, 0, 'N', color='green', fontsize=12, fontweight='bold')
    ax.text(0, 0, axis_len, 'U', color='black', fontsize=12, fontweight='bold')

    ax.set_xlabel('East (E)'); ax.set_ylabel('North (N)'); ax.set_zlabel('Up (U)')
    ax.set_title('UAV Snapshot Visualization (ENU System)', fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if check_dependencies():
        # 1. 初始化，加载一次基础模型
        script_dir = os.path.dirname(os.path.abspath(__file__))
        obj_path = os.path.join(script_dir, "F-16.obj")
        if not os.path.exists(obj_path):
            print(f"错误：找不到模型文件 {obj_path}")
        else:
            base = trimesh.load(obj_path)
            if hasattr(base, 'geometry'): 
                base = trimesh.util.concatenate(list(base.geometry.values()))

            # 2. 准备编队数据 (N, U, E, psi, theta, phi, scale)
            # 这里的单位是 rad，位置单位通常是米
            formation = [
                (0,  0, 0,  0, 0, 0, 1.0), 
                (-20, 20, 0, pi/4, pi/8, pi/6, 2.0),  
                (-40, 40, 0, pi/2, pi/4, pi/6, 3.0),    
                (-60, 60, 0, 3*pi/4, pi/2, pi/6, 4.0),  
            ]

            # 3. 调用函数生成物理顶点的模型
            scene_data = []
            for cfg in formation:
                verts, faces = get_transformed_mesh(base, *cfg)
                scene_data.append((verts, faces))

            # 4. 统一渲染
            render_enu_scene(scene_data)
    else:
        print("错误：缺少库，请运行 pip install trimesh matplotlib numpy")