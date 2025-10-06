import numpy as np
import casadi as ca
from controller.cvar_cbf_controller_nlp_beta_dt import DCLFCVARDCBF as BaseDCLFCVARDCBF
from controller.interaction_aware_conformal_prediction import InteractionAwareConformalPrediction

class ConformalDCLFCVARDCBF(BaseDCLFCVARDCBF):
    """
    Interaction-Aware Conformal Prediction enhanced CVaR Barrier Function Controller
    
    Integrates ICP (Interaction-Aware Conformal Prediction) with CVaR-CBF for enhanced safety:
    - CVaR-CBF (base): Risk-aware collision avoidance with uncertainty sampling
    - ICP module: Dynamic safety radius adaptation based on prediction accuracy
    
    Key features:
    1. Real-time trajectory prediction for moving obstacles
    2. Conformal prediction for uncertainty quantification  
    3. Dynamic safety margins that adapt to prediction quality
    4. Individual radius tracking per obstacle
    """
    
    def __init__(self, robot, obstacles, all_robots, params):
        # Initialize base controller
        super().__init__(robot, obstacles, all_robots, params)
        
        self.type = 'conformal_cvarbf'
        
        # Slack variable penalty weight for constraint relaxation
        self.w_slack = params.get("w_slack", 1000.0)
        
        # Logging for analysis
        self.icp_radius_log = []
        self.step_counter = 0  # Simple step counter for debugging
        
        
        # Initialize ICP module for interaction-aware trajectory prediction
        icp_params = {
            "prediction_horizon": params.get("prediction_horizon", 5),
            "failure_probability": params.get("failure_probability", 0.05),
            "calibration_size": params.get("calibration_size", 8),
            "num_episodes": params.get("num_simulation_episodes", 4),
            "observation_window": params.get("observation_window", 3),
            "min_radius": params.get("icp_min_radius", 0.05),  
            "max_radius": params.get("icp_max_radius", 0.30),
            "ema":        params.get("icp_ema", 0.60),
            "rate_limit": params.get("icp_rate_limit", 0.10),  
        }
        self.icp_module = InteractionAwareConformalPrediction(icp_params)
        
        # Store ICP radius history for visualization
        self.icp_radius_history = {}  # {obstacle_id: [radius_at_step_0, radius_at_step_1, ...]}
        
    
    def solve_opt_one_iter(self, t, obstacles, all_robots, u_n=None):
        """
        Enhanced QP with CDT-RA complementary integration:
        
        ROLE SEPARATION:
        - RA-CVaR-BF: Instantaneous safety constraints (hard bounds)
        - CDT: Long-term statistical safety (soft guidance + safety margin scaling)
        
        ANTI-REDUNDANCY MEASURES:
        - Optional RA β-adaptation disable (to isolate CDT contribution)
        - CDT safety margin scaling (acts on constraint parameters) 
        - Different risk metrics (collision vs near-miss frequencies)
        """
        x = self.robot.x_curr
        target = self.robot.target
        u = ca.MX.sym(f'u_{t}', self.robot.m, 1)
        n_zeta = len(obstacles) + len(all_robots) - 1
        n_eta = n_zeta * self.S
        zeta = ca.MX.sym(f'zeta_{t}', n_zeta)
        eta = ca.MX.sym(f'eta_{t}', n_eta)
        
        n_slack = len(obstacles)  
        slack = ca.MX.sym(f'slack_{t}', n_slack, 1)

        constraints = []
        lbg = []
        ubg = []

        # Base cost (same as original)
        if u_n is None:
            u_n, _ = self.robot.nominal_input(x, target)
        self.robot.un_log.append(u_n.reshape(self.robot.m, 1))
        
        
        # Build current state dict for ICP
        current_state_dict = {
            "robot_position": np.array(self.robot.x_curr[:2]).flatten(),
            "robot_velocity": np.array(self.robot.x_curr[2:4]).flatten(),
            "human_positions": [np.array(obs.x_curr[:2]).flatten() for obs in obstacles],
            "human_velocities": [np.array(obs.velocity_xy[:2]).flatten() for obs in obstacles],
        }

        # Simple robot forward plan (length = prediction_horizon)
        robot_plan = []
        for i in range(self.icp_module.prediction_horizon):
            future_pos = self.robot.x_curr[:2] + self.robot.x_curr[2:4] * (i * self.robot.dt)
            robot_plan.append([float(future_pos[0]), float(future_pos[1]), float(self.robot.x_curr[2]), float(self.robot.x_curr[3])])
        robot_plan = np.array(robot_plan, dtype=float)


        try:
            self.icp_module.tick(current_state_dict, robot_plan)
        except Exception as e:
            if self.total_steps % 50 == 0:
                print(f"⚠️ ICP outer-loop tick failed: {e}")

                # 取每个行人的 k=0 半径
        step1_radii = self.icp_module.get_step1_radii(list(range(len(obstacles))))
        
        # Store ICP radius history for visualization
        for obs_id, radius in step1_radii.items():
            if obs_id not in self.icp_radius_history:
                self.icp_radius_history[obs_id] = []
            self.icp_radius_history[obs_id].append(radius)
            
            # Debug: Print radius changes for troubleshooting
            if len(self.icp_radius_history[obs_id]) > 1:
                prev_radius = self.icp_radius_history[obs_id][-2]
                if abs(radius - prev_radius) > 0.001:
                    print(f"📊 Controller: Obs{obs_id} radius {prev_radius:.4f}→{radius:.4f}")

        
        # ----- Optional: diagnostic printing -----
        if self.step_counter % 20 == 0:
            avg_r = np.mean(list(step1_radii.values())) if len(step1_radii) > 0 else 0.0
            print(f"🟣 ICP(A1) radii@k=0 avg={avg_r:.3f} over {len(step1_radii)} humans")
            
        
        self.step_counter += 1

        # Planning objective: J = Goal distance + λₜ · Human avoidance (unchanged philosophy)
        goal_cost = ca.mtimes([(u - u_n).T, self.w_a * self.robot.R, (u - u_n)])
        
        # Human avoidance cost: distance-based penalty for being near obstacles  
        avoidance_cost = 0.0
        for iObs, obs in enumerate(obstacles):
            obs_pos = obs.x_curr[:2]
            robot_pos = x[:2]
            distance = ca.norm_2(robot_pos - obs_pos)
            safety_distance = obs.radius + self.robot.radius + float(step1_radii.get(iObs, self.icp_module.min_radius))
            
            # Penalty increases as we get closer to the obstacle
            if_near = ca.if_else(distance < 2.0 * safety_distance, 
                               1.0 / (distance + 0.1), 0.0)
            avoidance_cost += if_near
    
        # Simplified objective with slack penalty
        slack_penalty = self.w_slack * ca.dot(slack, slack)  
        cost = goal_cost + slack_penalty
        
        
        # Same constraints as base controller (CVaR constraints)
        zeta_index = 0
        # robot-obstacles constraints
        for iObs in range(len(obstacles)):
            dt = self.robot.dt
            hsk1_list = []
            offset = (iObs + len(all_robots) - 1) * self.S
            for s in range(self.S):

                wu_s = self.robot_wu_samples[self.robot.id][s].reshape(-1, 1)
                wx_s = self.robot_wx_samples[self.robot.id][s].reshape(-1, 1)
                x_k1 = self.robot.dynamics_uncertain(x, u, wu_s, wx_s)
                wu_s = self.obs_wu_samples[iObs][s].reshape(-1, 1) 
                wx_s = self.obs_wx_samples[iObs][s].reshape(-1, 1)  
                pre_obs_pos = obstacles[iObs].x_curr[:2] + (obstacles[iObs].velocity_xy[:2] + wu_s)* dt + wx_s   
                pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)

                # Use ICP safety radius to enhance the safety margin (with upper limit to avoid over-conservative)
                icp_radius = min(float(step1_radii.get(iObs, self.icp_module.min_radius)), self.icp_module.max_radius)  # Cap ICP contribution
                safety_margin = obstacles[iObs].radius + self.robot.radius + icp_radius
                h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, pre_obs_state, safety_margin, htype=self.htype )
                hs_k1 = d_h + h_k
                hsk1_list.append(hs_k1)


            x_k1 = self.robot.dynamics_uncertain(x, u)
            pre_obs_pos = obstacles[iObs].x_curr[:2] + obstacles[iObs].velocity_xy[:2] * dt
            pre_obs_state = np.vstack((pre_obs_pos, obstacles[iObs].velocity_xy[:2])).reshape(-1, 1)
            # Apply ICP safety radius to the nominal constraint as well (with upper limit)
            icp_radius = min(float(step1_radii.get(iObs, self.icp_module.min_radius)), self.icp_module.max_radius)  # Cap ICP contribution
            safety_margin = obstacles[iObs].radius + self.robot.radius + icp_radius
            h_k, d_h = self.robot.agent_barrier_dt(x, x_k1, pre_obs_state, safety_margin, htype=self.htype )
            
            # Constraint: -h_s - zeta[iObs] - eta[offset + s] <= 0
            constraints.append(-ca.vertcat(*hsk1_list) - zeta[zeta_index]*ca.DM.ones((self.S, 1)) - eta[offset:offset + self.S])
            lbg.extend([-ca.inf] * self.S)
            ubg.extend([0] * self.S)
            # Constraint: eta[offset + s] >= 0
            constraints.append(eta[offset:offset + self.S])
            lbg.extend([0] * self.S)
            ubg.extend([ca.inf] * self.S)
            psi1_k = -(zeta[zeta_index] + (1/self.beta) * ca.dot(self.obs_pmf[iObs], eta[offset:offset + self.S])) + (-1 + self.gamma) * h_k

            constraints.append(psi1_k + slack[iObs])
            lbg.append(0)
            ubg.append(ca.inf)
            if np.isnan(np.array(h_k)).any():
                print("DEBUG pre_obs_state:", pre_obs_state)
                print("DEBUG type:", type(pre_obs_state))
                print("x_curr", obstacles[iObs].x_curr)
                print("velocity_xy", obstacles[iObs].velocity_xy)
                print("wu_s", wu_s)
                print("wx_s", wx_s)
                print("dt", dt)
                print("pre_obs_pos", pre_obs_pos)
                print("pre_obs_state", pre_obs_state)
                assert not np.isnan(x).any(), "NaN in robot state x"
                if isinstance(x_k1, ca.MX) or isinstance(x_k1, ca.SX):

                    pass
                else:
                    assert not np.isnan(x_k1).any(), "NaN in robot future x_k1"


                if isinstance(pre_obs_state, ca.MX) or isinstance(pre_obs_state, ca.SX):
                    pass  
                else:
                    assert not np.isnan(pre_obs_state).any(), "NaN in obstacle predicted state"

                input("NaN in h_k")

            zeta_index += 1
            
        # Input constraints 
        constraints.append(u - self.robot.u_min)  
        lbg.extend([0] * self.robot.m)  
        ubg.extend([ca.inf] * self.robot.m)

        constraints.append(self.robot.u_max - u)  
        lbg.extend([0] * self.robot.m)
        ubg.extend([ca.inf] * self.robot.m)
        
        # Velocity constraints (same as base)
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


        constraints.append(slack)
        lbg.extend([0] * n_slack)
        ubg.extend([ca.inf] * n_slack)

        opt_vars = ca.vertcat(u, zeta, eta, slack)
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*constraints)
        }
        opts = {"ipopt.print_level": 0, "print_time": 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = opt_vars.numel()
        # Construct initial guess
        x0 = np.zeros((n_vars, 1))
        if self.prev_solu is not None and len(self.prev_solu) >= (self.robot.m + n_zeta + n_eta):

            x0[:self.robot.m + n_zeta + n_eta, 0] = self.prev_solu[:self.robot.m + n_zeta + n_eta, 0]

            x0[self.robot.m + n_zeta + n_eta:, 0] = 0
        else:
            x0[0:self.robot.m, 0] = u_n[:,0]

        lbx = -ca.inf * np.ones((n_vars, 1))
        ubx = ca.inf * np.ones((n_vars, 1))

        solution = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        status = solver.stats()['return_status']
        
        return solution, status

    def solve_opt(self, t, obstacles, all_robots, u_n=None):
        """
        Conformal controller main solve loop with lambda update
        """
        # Use baseline beta strategy (same as adap_cvarbf)
        for beta_val in self.beta_candidates:
            self.beta = beta_val

            solution, status = self.solve_opt_one_iter(t, obstacles, all_robots, u_n)

            if status in ['Solve_Succeeded']:
                sol = solution['x'].full().flatten()
                sol_g = solution["g"].full().flatten()
                sol_cost = solution['f'].full().flatten()

                self.prev_prev_solu = self.prev_solu
                self.prev_solu = sol.reshape(-1, 1)

                u_opt = sol[0:self.robot.m].reshape(-1, 1)
                            
                self.update_logs(sol, sol_g, sol_cost, obstacles, all_robots)
                
                
                # Log average ICP radius for analysis
                step1_radii_for_log = self.icp_module.get_step1_radii(list(range(len(obstacles))))
                avg_icp_radius = np.mean(list(step1_radii_for_log.values())) if step1_radii_for_log else 0.1
                self.icp_radius_log.append(avg_icp_radius)
                
                x_k1 = None
                break
            else:
                continue
                
        if status not in ['Solve_Succeeded']:
            x_k1 = None
            u_opt = None
                        
        return x_k1, u_opt, status
        
    def update_logs(self, sol, sol_g, sol_cost, obstacles, all_robots):
        """
        Extended logging to include conformal controller metrics
        """
        # Call base class logging
        super().update_logs(sol, sol_g, sol_cost, obstacles, all_robots)
