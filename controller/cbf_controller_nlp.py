import numpy as np
import casadi as ca
from scipy.stats import norm

class DCLFDCBF:
    def __init__(self, robot,  obstacles, all_robots, params):
        self.type = 'cbf'
        self.robot = robot
        self.htype = params["htype"]
        self.S = params["S"]
        self.gamma =0.1

        self.w_a = 1  

        self.hlog = []
        self.cons_log = []
        
        self.feasible = True

        self.prev_solu = None
        self.prev_prev_solu = None
 

    
    def solve_opt(self, t, obstacles, all_robots, u_n = None):
        x = self.robot.x_curr
        target = self.robot.target
        u = ca.MX.sym(f'u_{t}', self.robot.m, 1)

        constraints = []
        lbg = []
        ubg = []

        if u_n is None:
            u_n, _ = self.robot.nominal_input(x, target)
        cost = ca.mtimes([(u - u_n).T, self.w_a * self.robot.R, (u - u_n)]) 
        self.robot.un_log.append(u_n.reshape(self.robot.m,1))

                
        x_k1 = self.robot.dynamics(x, u)
    
        ## robot-obstalces constraints
        for iObs in range(len(obstacles)):
            dt = self.robot.dt
            pre_obs_pos = obstacles[iObs].x_curr[:2] + obstacles[iObs].velocity_xy[:2] * dt
            pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)

            h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, pre_obs_state, obstacles[iObs].radius+ self.robot.radius, htype=self.htype)
            constraints.append(d_h + self.gamma * h_k)
                
            lbg.append(0)
            ubg.append(ca.inf)
        
        ## Input constraints
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
        
        # ## boundary contraints
        # x_k1 = self.robot.dynamics(x, u)
        # for i in range(2):
        #     constraints.append(x_k1[i] - self.robot.x_min[i, 0])
        #     lbg.append(0)
        #     ubg.append(ca.inf)
        #     constraints.append(self.robot.x_max[i, 0] - x_k1[i])
        #     lbg.append(0)
        #     ubg.append(ca.inf)
            
        opt_vars = ca.vertcat(u)
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*constraints)
        }
        opts = {"ipopt.print_level": 0, "print_time": 0}

        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        n_vars = opt_vars.numel()
        x0 = np.zeros((n_vars, 1)) 
        if self.prev_solu is not None and len(self.prev_solu) >= n_vars:
            # x0[0:self.robot.m, 0] = self.prev_solu[self.robot.m, 0]  # shape (n_vars, 1)
            x0[:n_vars,0] = self.prev_solu[:n_vars,0]
        else:
            x0[0:self.robot.m, 0] = u_n[:,0]   
        lbx = -ca.inf * np.ones((n_vars, 1))
        ubx = ca.inf * np.ones((n_vars, 1))
        
        solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        status = solver.stats()['return_status']   
        if status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level']:
            sol = solution['x'].full().flatten()
            sol_g = solution['g'].full().flatten()
            sol_cost = solution['f'].full().flatten()
            
            self.prev_prev_solu = self.prev_solu
            self.prev_solu = sol.reshape(-1, 1)

            u_opt = sol[:self.robot.m].reshape(-1, 1)

            self.update_logs(sol, sol_g, sol_cost, obstacles, all_robots)
            x_k1 = None
        else:
            x_k1 = None
            u_opt = None
        return x_k1, u_opt, status

    def update_logs(self, sol, sol_g, sol_cost, obstacles, all_robots):
        x = self.robot.x_curr
        uopt = sol[:self.robot.m].reshape(-1, 1)
        x_k1 = self.robot.dynamics(x, uopt) 
        x_k2 = self.robot.dynamics(x_k1, uopt)
        h_list = []
        cons_list = []
        
        c_idx = 0

        for iObs in range(len(obstacles)):
            pre_obs_pos = obstacles[iObs].x_curr
            pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)

            h_k, d_h = self.robot.agent_barrier_dt(x, uopt, pre_obs_state, obstacles[iObs].radius + self.robot.radius)
            h_list.append(h_k)
            
            cons_list.append(sol_g[c_idx])
            c_idx += 1
            
        self.hlog.append(np.array(h_list).reshape(-1, 1))
        self.cons_log.append(np.array(cons_list).reshape(-1, 1))
