import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 路径设置 ---
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "logs", "CombatLog_vs_Rule4.csv")

if not os.path.exists(csv_path):
    print(f"错误：在以下路径未找到文件: {csv_path}")
else:
    # --- 读取数据 ---
    df = pd.read_csv(csv_path)

    # --- 设置绘图风格 ---
    plt.style.use('default')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. 过载 (Ny) - 第一张图
    axes[0].plot(df['time'], df['r_ny'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[0].plot(df['time'], df['b_ny'], label='Blue (Rule)', color='royalblue', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Overload (g)')
    # axes[0].set_title('Normal Load Factor Comparison', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)

    # 2. 迎角 (Alpha) - 第二张图
    axes[1].plot(df['time'], df['r_alpha'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[1].plot(df['time'], df['b_alpha'], label='Blue (RL)', color='royalblue', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('AOA (deg)')
    # axes[1].set_title('Angle of Attack (Alpha) Comparison', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.4)

    # 3. 高度 (Alt) - 第三张图
    axes[2].plot(df['time'], df['r_alt'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[2].plot(df['time'], df['b_alt'], label='Blue (RL)', color='royalblue', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Altitude (m)')
    axes[2].set_xlabel('Time (s)')
    # axes[2].set_title('Altitude (Alt) Comparison', fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.4)

    # 调整布局 (增加边距)
    plt.tight_layout(pad=3.0)
    
    # 保存结果（可选）
    # save_path = os.path.join(current_dir, "Combat_VS_Rule4_Plots.png")
    # plt.savefig(save_path, dpi=300)
    # print(f"图表已保存至: {save_path}")

    plt.show()
