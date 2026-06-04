import tkinter as tk
from tkinter import ttk
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import sys
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# --- 1. 定义变量范围和初始值 ---
# 格式: '变量名': (最小值, 最大值, 初始值)
VAR_DEFS = {
    'ATA': (-pi, pi, 0),
    'theta': (-pi/2, pi/2, 0),
    'delta_psi': (-np.pi, np.pi, 0),
    'vu': (-100, 100, 0),
    'delta_theta_threat': (-pi/2, pi/2, 0),
}
zmin, zmax = -1.2, 1.2

# --- 2. 向量化的奖励函数 ---
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_reward(ATA, theta, delta_psi, vu, delta_theta_threat):
    i_can_guide =  - np.tanh(8*(abs(delta_psi)-pi/3)) # ATA
    "attack"
    inner = 4*(sigmoid(
        2-0.7*np.exp(1.2*abs(delta_psi)*2/pi) +
        1*np.clip(vu/100, -1, 1) +
        1*(theta/pi)
    )/(4))

    "crank"
    # inner = 4*(sigmoid(
    #     1.0 * i_can_guide +
    #     0.5 - 3.5*abs(abs(delta_psi)-pi/3) + 
    #     1.5 * ((-theta) / (pi/2))
    # )/(3.4))

    "escape"
    # inner = 2*sigmoid((
    #     -2 * np.exp(2*theta/(pi/2)) * np.where(delta_theta_threat>=0, 1, 0)+
    #     -5 * np.exp(1.2*(theta*2/pi)**2) * np.where(delta_theta_threat<0, 1, 0) +
    #     4 * (-1+(abs(delta_psi)/(pi/2)))
    # )/(5))

    r_event = inner
    return r_event

# --- 3. GUI 与 可视化逻辑 ---
class RewardVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("多维奖励函数切片可视化")
        self.root.geometry("1100x700")
        
        # 布局：左侧控制面板，右侧绘图区
        self.ctrl_frame = ttk.Frame(root, width=350, padding=(10, 10))
        self.ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 初始化 Matplotlib
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Prevent roll: reapply elev/azim on mouse motion so only pitch(y) and yaw(x) change
        self.canvas.mpl_connect('motion_notify_event', self._prevent_roll)
        # Ensure closing the window properly exits the app and the terminal run
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 控制面板变量
        self.var_x = tk.StringVar(value=list(VAR_DEFS.keys())[0])
        self.var_y = tk.StringVar(value=list(VAR_DEFS.keys())[1])
        self.sliders = {}
        
        self._build_control_panel()
        self.update_plot() # 初次渲染
        # Apply a one-time default Z limit; interactive zoom/pan can still change it
        try:
            self.ax.set_zlim(zmin, zmax)
        except Exception:
            pass

    def _build_control_panel(self):
        ttk.Label(self.ctrl_frame, text="X轴", font=('Arial', 10, 'bold')).grid(row=0, column=1)
        ttk.Label(self.ctrl_frame, text="Y轴", font=('Arial', 10, 'bold')).grid(row=0, column=2)
        ttk.Label(self.ctrl_frame, text="切片值调节", font=('Arial', 10, 'bold')).grid(row=0, column=3, padx=10)
        
        row = 1
        for name, (vmin, vmax, vinit) in VAR_DEFS.items():
            # 变量名标签
            ttk.Label(self.ctrl_frame, text=name).grid(row=row, column=0, sticky='w', pady=10)
            
            # X 轴和 Y 轴的选择单选框
            rx = ttk.Radiobutton(self.ctrl_frame, text="", variable=self.var_x, value=name, command=self._on_axis_change)
            rx.grid(row=row, column=1)
            ry = ttk.Radiobutton(self.ctrl_frame, text="", variable=self.var_y, value=name, command=self._on_axis_change)
            ry.grid(row=row, column=2)
            
            # 滑动条及其数值显示
            slider_frame = ttk.Frame(self.ctrl_frame)
            slider_frame.grid(row=row, column=3, padx=10, sticky='ew')
            
            val_label = ttk.Label(slider_frame, text=f"{vinit:.2f}", width=8)
            val_label.pack(side=tk.RIGHT)
            # 输入框，允许直接编辑数值
            entry_var = tk.StringVar(value=f"{vinit:.2f}")
            entry = ttk.Entry(slider_frame, textvariable=entry_var, width=8)
            entry.pack(side=tk.RIGHT, padx=(6, 0))

            slider = ttk.Scale(slider_frame, from_=vmin, to=vmax, orient='horizontal', value=vinit)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # 绑定滑动条事件（更新标签与输入框）
            slider.configure(command=lambda val, l=val_label, ev=entry_var: self._on_slider_change(val, l, ev))
            # 输入框回车或失焦时更新滑动条
            entry.bind('<Return>', lambda e, s=slider, ev=entry_var: self._on_entry_change(e, s, ev))
            entry.bind('<FocusOut>', lambda e, s=slider, ev=entry_var: self._on_entry_change(e, s, ev))
            # 鼠标滚轮改变滑动条（Windows 使用 <MouseWheel>，Linux 使用 Button-4/5）
            slider.bind('<MouseWheel>', lambda e, s=slider, n=name: self._on_mousewheel_slider(e, s, n))
            slider.bind('<Button-4>', lambda e, s=slider, n=name: self._on_mousewheel_slider(e, s, n))
            slider.bind('<Button-5>', lambda e, s=slider, n=name: self._on_mousewheel_slider(e, s, n))

            self.sliders[name] = {'slider': slider, 'label': val_label, 'entry': entry, 'entry_var': entry_var, 'vmin': vmin, 'vmax': vmax}
            row += 1
            
        self._update_slider_states()

    def _on_axis_change(self):
        # 防止 X 和 Y 选择同一个变量
        if self.var_x.get() == self.var_y.get():
            return 
        self._update_slider_states()
        self.update_plot()

    def _on_slider_change(self, val, label, entry_var=None):
        # 更新标签
        try:
            fv = float(val)
            label.configure(text=f"{fv:.2f}")
            if entry_var is not None:
                entry_var.set(f"{fv:.2f}")
        except Exception:
            pass
        # 这里使用 after 节流，防止拖动太快导致重绘卡顿
        if hasattr(self, '_after_id') and self._after_id:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(50, self.update_plot)

    def _update_slider_states(self):
        # 禁用被选为 X 和 Y 的变量的滑动条
        x_name = self.var_x.get()
        y_name = self.var_y.get()
        for name, widgets in self.sliders.items():
            if name in (x_name, y_name):
                widgets['slider'].state(['disabled'])
            else:
                widgets['slider'].state(['!disabled'])

    def _on_entry_change(self, event, slider, entry_var):
        try:
            v = float(entry_var.get())
            # clamp to slider range
            # slider options stored in self.sliders
            # find the slider name by object identity
            name = None
            for k, w in self.sliders.items():
                if w['slider'] is slider:
                    name = k
                    break
            if name is not None:
                vmin = float(self.sliders[name]['vmin'])
                vmax = float(self.sliders[name]['vmax'])
                v = max(vmin, min(vmax, v))
            slider.set(v)
        except Exception:
            # revert entry to current slider value
            try:
                entry_var.set(f"{float(slider.get()):.2f}")
            except Exception:
                pass

    def _on_mousewheel_slider(self, event, slider, name):
        # Windows: event.delta is multiple of 120; Linux: use Button-4/5
        delta = 0
        try:
            if hasattr(event, 'delta'):
                # Windows
                delta = int(event.delta / 120) # 鼠标上每个刻度的增量为120
            else:
                # Mac
                if event.num == 4:
                    delta = 1
                elif event.num == 5:
                    delta = -1
        except Exception:
            delta = 0
        if delta == 0:
            return
        vmin = float(self.sliders[name]['vmin'])
        vmax = float(self.sliders[name]['vmax'])
        step = (vmax - vmin) / 25.0
        new = slider.get() + delta * step
        new = max(vmin, min(vmax, new))
        slider.set(new)
        return 'break'

    def update_plot(self):
        x_name = self.var_x.get()
        y_name = self.var_y.get()
        
        if x_name == y_name:
            return

        # 1. 为 X 和 Y 生成网格点 (分辨率 50x50 保证流畅度)
        x_min, x_max, _ = VAR_DEFS[x_name]
        y_min, y_max, _ = VAR_DEFS[y_name]
        xx = np.linspace(x_min, x_max, 25)
        yy = np.linspace(y_min, y_max, 25)
        X, Y = np.meshgrid(xx, yy)
        
        # 2. 准备传入计算函数的参数 kwargs
        kwargs = {}
        for name in VAR_DEFS.keys():
            if name == x_name:
                kwargs[name] = X
            elif name == y_name:
                kwargs[name] = Y
            else:
                # 读取滑动条的当前值作为标量
                kwargs[name] = float(self.sliders[name]['slider'].get())
                
        # 3. 计算 Z 值
        Z = compute_reward(**kwargs)
        
        # 4. 重绘曲面
        self.ax.clear()
        self.ax.plot_surface(X, Y, Z, cmap='rainbow', rstride=1, cstride=1, alpha=0.9)
        self.ax.set_xlabel(x_name)
        self.ax.set_ylabel(y_name)
        self.ax.set_zlabel('Reward')
        self.ax.set_title(f"Z = Reward  (Fixed vars sliced via sliders)")
        # Enforce global Z limits on every redraw so user interactions or
        # switching variables won't change the displayed Z range.
        try:
            self.ax.set_zlim(zmin, zmax)
        except Exception:
            pass

        self.canvas.draw_idle()

    def _prevent_roll(self, event):
        # Only act when pointer is over our 3D axes
        if event.inaxes is not self.ax:
            return
        # Reapply current elev/azim to remove any roll introduced by default handlers
        try:
            self.ax.view_init(elev=self.ax.elev, azim=self.ax.azim)
            self.canvas.draw_idle()
        except Exception:
            pass

    def _on_close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass
        try:
            self.root.quit()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        # Force exit the process so terminal run ends
        try:
            sys.exit(0)
        except SystemExit:
            # in rare cases where SystemExit is caught, force terminate
            try:
                import os
                os._exit(0)
            except Exception:
                pass

if __name__ == '__main__':
    root = tk.Tk()
    app = RewardVisualizer(root)
    root.mainloop()