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

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    def add_max_min_ticks(ax, data_red, data_blue, precision=2):
        # 计算全局最大和最小
        y_min = min(data_red.min(), data_blue.min())
        y_max = max(data_red.max(), data_blue.max())
        # 仅保留最大、最小和零点（如果零在范围内）
        ticks = [y_min, y_max]
        if y_min < 0 < y_max:
            ticks.append(0)
        sorted_ticks = sorted(list(set(ticks)))
        ax.set_yticks(sorted_ticks)
        # 根据 precision 设置刻度标签格式
        if precision == 0:
            ax.set_yticklabels([f"{int(round(t))}" for t in sorted_ticks])
        else:
            ax.set_yticklabels([f"{t:.{precision}f}" for t in sorted_ticks])

    # 1. 过载 (Ny) - 第一张图
    axes[0].plot(df['time'], df['r_ny'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[0].plot(df['time'], df['b_ny'], label='Blue (Rule)', color='royalblue', linestyle='--', alpha=0.7)
    axes[0].set_ylabel('Overload (g)')
    # axes[0].set_title('Normal Load Factor Comparison', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.4)
    add_max_min_ticks(axes[0], df['r_ny'], df['b_ny'], precision=1)

    # 2. 迎角 (Alpha) - 第二张图
    axes[1].plot(df['time'], df['r_alpha'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[1].plot(df['time'], df['b_alpha'], label='Blue (Rule)', color='royalblue', linestyle='--', alpha=0.7)
    axes[1].set_ylabel('AOA (deg)')
    # axes[1].set_title('Angle of Attack (Alpha) Comparison', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.4)
    add_max_min_ticks(axes[1], df['r_alpha'], df['b_alpha'], precision=1)

    # 3. 高度 (Alt) - 第三张图
    axes[2].plot(df['time'], df['r_alt'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[2].plot(df['time'], df['b_alt'], label='Blue (Rule)', color='royalblue', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Altitude (m)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.4)
    add_max_min_ticks(axes[2], df['r_alt'], df['b_alt'], precision=0)

    # 4. 马赫数 (Mach) - 第四张图
    axes[3].plot(df['time'], df['r_mach'], label='Red (RL)', color='crimson', linewidth=1.8)
    axes[3].plot(df['time'], df['b_mach'], label='Blue (Rule)', color='royalblue', linestyle='--', alpha=0.7)
    axes[3].set_ylabel('Mach')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.4)
    add_max_min_ticks(axes[3], df['r_mach'], df['b_mach'], precision=2)

    # 调整布局 (增加边距)
    plt.tight_layout(pad=3.0)
    
    # 保存结果（可选）
    # save_path = os.path.join(current_dir, "Combat_VS_Rule4_Plots.png")
    # plt.savefig(save_path, dpi=300)
    # print(f"图表已保存至: {save_path}")
    
    plt.show()
