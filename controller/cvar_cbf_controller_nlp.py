import numpy as np
import casadi as ca

class DCLFCVARDCBF:
    def __init__(self, robot, obstacles, all_robots, params):
        self.type = 'cvarbf'

        self.robot = robot

        self.htype = params["htype"]
        self.S = params["S"]
        self.beta = params.get("beta", None)  
        self.gamma = 0.1

        self.w_a = 1  

        self.hlog = []
        self.cons_log = []
        self.beta_log = []
        
        self.feasible = True

        self.prev_solu = None
        self.prev_prev_solu = None

        self.update_pmf(obstacles, all_robots)  

    def update_pmf(self, obstacles, all_robots):
        self.robot_pmf = []
        self.robot_wu_samples = []
        self.robot_wx_samples = []
        for iRobot in range(len(all_robots)):
            robot_pmf, robot_wu_samples, robot_wx_samples = all_robots[iRobot].gen_pmf(all_robots[iRobot].u, 
                                                                                       all_robots[iRobot].x_curr,
                                                                                       all_robots[iRobot].noise, 
                                                                                       self.S)
            self.robot_pmf.append(robot_pmf)
            self.robot_wu_samples.append(robot_wu_samples)
            self.robot_wx_samples.append(robot_wx_samples)
            
        self.obs_pmf = []   
        self.obs_wu_samples = [] 
        self.obs_wx_samples = []
        for obs in obstacles:
            obs_pmf, obs_wu_samples, obs_wx_samples = obs.gen_pmf(obs.velocity_xy,
                                                                  obs.x_curr, 
                                                                  obs.noise,
                                                                  self.S)
            self.obs_pmf.append(obs_pmf)
            self.obs_wu_samples.append(obs_wu_samples)
            self.obs_wx_samples.append(obs_wx_samples)

    def beta_value(self, h):
        if isinstance(h, ca.MX):
            beta = ca.DM(self.beta) 
        else:
            beta = float(self.beta)  
        return beta

    def solve_opt(self, t, obstacles, all_robots, u_n = None):
        x = self.robot.x_curr
        target = self.robot.target
        u = ca.MX.sym(f'u_{t}', self.robot.m, 1)
        n_zeta = len(obstacles) + len(all_robots) - 1
        n_eta = n_zeta * self.S
        if self.robot.type == "doubleint" or self.robot.type == "singleint":
            zeta = ca.MX.sym(f'zeta_{t}', n_zeta)
            eta = ca.MX.sym(f'eta_{t}', n_eta)
        elif self.robot.type == "doubleint_v1" or self.robot.type == "unicycle_v2":
            zeta = ca.MX.sym(f'zeta_{t}', n_zeta *3 )
            eta = ca.MX.sym(f'eta_{t}', n_eta * 2 +1)

        constraints = []
        lbg = []
        ubg = []

        cost = 0

        if u_n is None:
            u_n, _ = self.robot.nominal_input(x, target)
        self.robot.un_log.append(u_n.reshape(self.robot.m,1))
        cost = ca.mtimes([(u - u_n).T, self.w_a * self.robot.R, (u - u_n)]) 


        zeta_index = 0

        # robot-obstalces constraints
        for iObs in range(len(obstacles)):
            dt = self.robot.dt
            hsk1_list = []
            offset = (iObs + len(all_robots) - 1) * self.S
            for s in range(self.S):

                wu_s = self.robot_wu_samples[self.robot.id][s].reshape(-1, 1)
                wx_s = self.robot_wx_samples[self.robot.id][s].reshape(-1, 1)
                x_k1 = self.robot.dynamics_uncertain(x, u, wu_s, wx_s)
                x_k2 = self.robot.dynamics_uncertain(x_k1, u, wu_s, wx_s)
                wu_s = self.obs_wu_samples[iObs][s].reshape(-1, 1) 
                wx_s = self.obs_wx_samples[iObs][s].reshape(-1, 1) # only position noise
                pre_obs_pos = obstacles[iObs].x_curr[:2] + (obstacles[iObs].velocity_xy[:2] + wu_s)* dt + wx_s  # next x_curr of the obstacle
                pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)

                h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, pre_obs_state, obstacles[iObs].radius + self.robot.radius, htype=self.htype )
                hs_k1 = d_h + h_k
                hsk1_list.append(hs_k1)

            x_k1 = self.robot.dynamics_uncertain(x, u)
            pre_obs_pos = obstacles[iObs].x_curr[:2] + obstacles[iObs].velocity_xy[:2] * dt
            pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)
            h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, pre_obs_state, obstacles[iObs].radius + self.robot.radius, htype=self.htype )
            
            # Constraint: -h_s - zeta[iObs] - eta[offset + s] <= 0
            constraints.append(-ca.vertcat(*hsk1_list) - zeta[zeta_index]*ca.DM.ones((self.S, 1)) - eta[offset:offset + self.S])
            lbg.extend([-ca.inf] * self.S)
            ubg.extend([0] * self.S)
            # Constraint: eta[offset + s] >= 0
            constraints.append(eta[offset:offset + self.S])
            lbg.extend([0] * self.S)
            ubg.extend([ca.inf] * self.S)
            psi1_k = -(zeta[zeta_index] + (1 / self.beta_value(h_k)) * ca.dot(self.obs_pmf[iObs], eta[offset:offset + self.S])) + (-1 + self.gamma) * h_k
            constraints.append(psi1_k)
            lbg.append(0)
            ubg.append(ca.inf)

            zeta_index += 1
            
        # Input constraints 
        constraints.append(u - self.robot.u_min)  
        lbg.extend([0] * self.robot.m)  
        ubg.extend([ca.inf] * self.robot.m)

        constraints.append(self.robot.u_max - u)  
        lbg.extend([0] * self.robot.m)
        ubg.extend([ca.inf] * self.robot.m)
        
        ## velocity constraints
        x_k1 = self.robot.dynamics(x, u)
        velocity_indices = [(2, self.robot.x_min[2, 0], self.robot.x_max[2, 0]),  # vx
                            (3, self.robot.x_min[3, 0], self.robot.x_max[3, 0])]  # vy
        for idx, v_min, v_max in velocity_indices:
            # Lower bound: v >= v_min
            H_l = np.zeros((4, 1))
            H_l[idx, 0] = 1
            L_l = -v_min
            h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, H=H_l, L=L_l, htype='linear')
            constraints.append(d_h + self.gamma * h_k)
            lbg.append(0)
            ubg.append(ca.inf)

            # Upper bound: v <= v_max
            H_u = np.zeros((4, 1))
            H_u[idx, 0] = -1
            L_u = v_max
            h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, H=H_u, L=L_u, htype='linear')
            constraints.append(d_h + self.gamma * h_k)
            lbg.append(0)
            ubg.append(ca.inf)

        # # add environment bounds
        # x_k1 = self.robot.dynamics(x, u)
        # for i in range(2):
        #     constraints.append(x_k1[i] - self.robot.x_min[i, 0])
        #     lbg.append(0)
        #     ubg.append(ca.inf)
        #     constraints.append(self.robot.x_max[i, 0] - x_k1[i])
        #     lbg.append(0)
        #     ubg.append(ca.inf)


        opt_vars = ca.vertcat(u, zeta, eta)
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*constraints)
        }
        opts = {"ipopt.print_level": 0, "print_time": 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = opt_vars.numel()
        # --- Construct initial guess ---
        x0 = np.zeros((n_vars, 1))
        if self.prev_solu is not None and len(self.prev_solu) >= n_vars:
            x0[:n_vars,0] = self.prev_solu[:n_vars,0]
        else:
            x0[0:self.robot.m, 0] = u_n[:,0]   
        lbx = -ca.inf * np.ones((n_vars, 1))
        ubx = ca.inf * np.ones((n_vars, 1))

        
        solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        status = solver.stats()['return_status']

        if status in ['Solve_Succeeded']:
            sol = solution['x'].full().flatten()
            sol_g = solution["g"].full().flatten()
            sol_cost = solution["f"].full().flatten()
            
            self.prev_prev_solu = self.prev_solu
            self.prev_solu = sol.reshape(-1, 1)
            
            u_opt = sol[0:self.robot.m].reshape(-1, 1)
                                    
            self.update_logs(sol, sol_g, sol_cost, obstacles, all_robots)
            x_k1 = None
        else:
            x_k1 = None
            u_opt = None
        return x_k1, u_opt, status

    def update_logs(self, sol, sol_g, sol_cost, obstacles, all_robots):
        x      = self.robot.x_curr
        uopt   = sol[0:self.robot.m].reshape(-1, 1)
        x_k1   = self.robot.dynamics(x, uopt)

        h_list     = []
        beta_list  = []
        cons_list  = []
  

        c_idx = 0      
        for iObs in range(len(obstacles)):
            pre_obs_pos   = obstacles[iObs].x_curr
            pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2]))  # equal to obstacle[iObs].trajectory
            h_k, _ = self.robot.agent_barrier_dt(
                x, x_k1, 
                pre_obs_state, 
                obstacles[iObs].radius + self.robot.radius, 
                htype=self.htype
            )
            h_list.append(h_k)
            beta_list.append(self.beta_value(h_k))

            psi1_k_val = sol_g[c_idx + 2*self.S]
            cons_list.append(psi1_k_val)

            c_idx += (2*self.S + 1)
            # zeta_index += 1


        self.hlog.append(np.array(h_list).reshape(-1, 1))
        self.beta_log.append(np.array(beta_list).reshape(-1, 1))
        self.cons_log.append(np.array(cons_list).reshape(-1, 1))


