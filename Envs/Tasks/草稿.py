def _compute_altitude_reward(self, aircraft: Aircraft):
        """
        计算高度奖励（实际为惩罚）。
        惩罚危险的低空飞行和不经济的超高空飞行。
        """
        # 1. 获取飞机当前的高度 (y坐标) 和垂直速度
        altitude_m = aircraft.pos[1]
        v_vertical_ms = aircraft.get_velocity_vector()[1]
        # --- 低空惩罚逻辑  ---
        # 2. 计算速度惩罚 (Pv)
        Pv = 0.0
        if altitude_m <= self.SAFE_ALTITUDE_M and v_vertical_ms < 0:
            descent_speed = abs(v_vertical_ms)
            penalty_factor = (descent_speed / self.KV_MS) * ((self.SAFE_ALTITUDE_M - altitude_m) / self.SAFE_ALTITUDE_M)
            Pv = -np.clip(penalty_factor, 0.0, 1.0)
        # 3. 计算绝对高度惩罚 (PH)
        PH = 0.0
        if altitude_m <= self.DANGER_ALTITUDE_M:
            PH = np.clip(altitude_m / self.DANGER_ALTITUDE_M, 0.0, 1.0) - 1.0
        # --- 新增：超高空惩罚逻辑 ---
        # 4. 计算超高空惩罚 (P_over)
        P_over = 0.0
        if altitude_m > self.MAX_ALTITUDE_M:
            # 惩罚值与超出高度成正比
            # 例如，在13000米时，惩罚为 -(13000-12000)/1000 * 0.5 = -0.5
            # 这样设计可以平滑地增加惩罚
            P_over = -((altitude_m - self.MAX_ALTITUDE_M) / 1000.0) * self.OVER_ALTITUDE_PENALTY_FACTOR
        # 5. 合并所有惩罚项，并乘以全局缩放系数
        #    将低空惩罚和高空惩罚相加

        return Pv + PH + P_over



    def escape_terminate_and_reward(self, side): # 逃逸策略训练与奖励
        # copy了进攻的，还没改
        terminate = False
        state = self.get_state(side)
        speed = state["ego_main"][0]
        alt = state["ego_main"][1]
        target_alt = alt+state["target_information"][0]
        delta_psi = state["target_information"][1]
        delta_theta = state["target_information"][2]
        dist = state["target_information"][3]
        alpha = state["target_information"][4]
        threat_delta_psi, threat_delta_theta, threat_distance =\
            state["threat"]

        RWR = state["warning"]
        obs = self.base_obs(side)
        d_hor = obs["border"][0]

        if side == 'r':
            ego = self.RUAV
            ego_missile = self.Rmissiles[0] if self.Rmissiles else None
            enm = self.BUAV
            alive_own_missiles = self.alive_r_missiles
            alive_enm_missiles = self.alive_b_missiles
        if side == 'b':
            ego = self.BUAV
            ego_missile = self.Bmissiles[0] if self.Bmissiles else None
            enm = self.RUAV
            alive_enm_missiles = self.alive_r_missiles
            alive_own_missiles = self.alive_b_missiles
        
        '''
        逃逸机动训练
        目标机从不可逃逸区外~40km向本机发射一枚导弹并对本机做纯追踪，
        本机被导弹命中有惩罚，除此之外根据和导弹的ATA和提供密集奖励
        '''
        self.close_range_kill() # 加入近距杀

        # 被命中判为失败
        if ego.got_hit:
            terminate = True
            self.lose = 1

        # 高度出界失败
        if not self.min_alt<=alt<=self.max_alt:
            terminate = True
            self.lose = 1

        # 飞出水平边界失败
        if self.out_range(ego):
            terminate = True
            self.lose = 1
        
        # 导弹规避成功
        if self.t > self.game_time_limit \
            and not ego.dead and enm.ammo==0 and \
                len(alive_enm_missiles)==0:
            self.win = 1
            terminate = True
        
        # 水平角度奖励， 奖励和敌机在同一高度层的置尾机动(√)

        r_angle = alpha / pi
        # r_angle_h = abs(delta_psi)/pi
        # r_angle_v = 1-abs(obs["target_information"][2])/pi*2 # 不对
        r_angle_v = 1-abs(ego.theta/pi*2)
        
        L_ = enm.pos_-ego.pos_
        delta_v_ = enm.vel_-ego.vel_
        dist_dot = np.dot(delta_v_, L_)/dist
        self.dist_dot = dist_dot

        # 速度奖励
        if self.last_dist_dot is None:
            dist_dot2 = 0
        else:
            dist_dot2 = (self.dist_dot-self.last_dist_dot)/self.dt_maneuver
        self.last_dist_dot = self.dist_dot

        r_v = dist_dot2/9.8

        # temp = abs(threat_delta_psi)/pi # 远离度,对头时候最好是0.8Ma，置尾的时候越快越好
        # v_opt = (0.8+(2-0.8)*temp)*340
        # r_v = 1 - np.abs(speed-v_opt)/(2*340)

        # r_v = dist_dot/(2*340) # 远离给奖励，接近给惩罚
        
        # 高度奖励
        # 爬升下降率惩罚
        r_vu = 0
        if alt <= self.min_alt_safe:
            r_vu = np.clip(ego.vu / 100 * (self.min_alt_safe - alt) / (self.min_alt_safe-self.min_alt), 0., 1.)
        if alt >= self.max_alt_safe:
            r_vu = -np.clip(ego.vu / 100 * (alt - self.max_alt_safe) / (self.max_alt-self.max_alt_safe), 0., 1.)

        # 绝对高度惩罚
        r_abs_h = 0
        if alt <= self.min_alt_danger:
            r_abs_h = np.clip(alt / self.min_alt_danger, 0., 1.) - 1.
        if alt >= self.max_alt_danger:
            r_abs_h = np.clip((self.max_alt-alt) / (self.max_alt-self.max_alt_danger), 0., 1.) - 1.

        # r_alt = (alt<=self.min_alt_safe) * (alt-self.min_alt)/(self.min_alt_safe-self.min_alt) + \
        #         (alt>=self.max_alt_safe) * (alt-self.max_alt)/(self.max_alt_safe-self.max_alt)
        # 相对高度奖励, 关于敌机或导弹
        r_rel_h = 0

        r_alt = r_vu + r_abs_h + r_rel_h

        # r_alt = (alt<=self.min_alt_safe) * np.clip(ego.vu/100, -1, 1) + \
        #         (alt>=self.max_alt_safe) * np.clip(-ego.vu/100, -1, 1)

        # pre_alt_opt = self.min_alt_safe + 1e3 # 比最小安全高度高1000m
        # alt_opt = np.clip(pre_alt_opt, self.min_alt_safe, self.max_alt_safe)
        # r_alt = (alt<=alt_opt)*(alt-self.min_alt)/(alt_opt-self.min_alt),
        #             (alt>alt_opt)*(1-(alt-alt_opt)/(self.max_alt-alt_opt))
        
        # 距离奖励，和目标机之间的距离
        r_dist = -1+np.clip(dist/30e3, -1, 1)

        # # 水平边界奖励
        self.dhor = d_hor
        if self.last_dhor is None:
            d_hor_dot = 0
        else:
            d_hor_dot = (self.dhor-self.last_dhor)/self.dt_maneuver
        self.last_dhor = self.dhor
        r_border = d_hor_dot /340*50e3
        # r_border = 0

        # # 稀疏奖励
        # 失败惩罚
        if self.lose:
            r_event = -20
        # 取胜奖励
        elif self.win:
            r_event = 20
        else:
            r_event = 0
        # r_event = 0

        w_angle =  1 # d_hor**2
        w_border = 2 # 1-w_angle

        reward = np.sum([
            w_angle * r_angle ,
            1 * r_angle_v ,
            1 * r_v ,
            2 * r_alt ,
            1 * r_event ,
            w_border * r_border , # 10
            0.5 * r_dist,
            ])
        
        if terminate:
            self.running = False
        
        return terminate, reward, r_event