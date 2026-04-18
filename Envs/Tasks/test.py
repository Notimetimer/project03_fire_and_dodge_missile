# 左crank角度奖励
x = np.sign(delta_psi)*alpha * 180/pi
alpha_max = ego.max_radar_angle_rad*180/pi # 60
x_opt = 50
r_angle_delta_psi = np.clip(delta_psi/(self.RUAV.max_radar_angle_rad, -1, 1)
if alpha < 50*pi/180:
    r_angle_alpha = -(alpha/(50*pi/180))**2
else:
    r_angle_alpha = -(6/5)*(alpha/(self.RUAV.max_radar_angle_rad)**2

r_angle = r_angle_alpha + 0.8*r_angle_delta_psi