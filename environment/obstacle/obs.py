import numpy as np
import casadi as ca
from ..object_base import ObjectBase
class Obstacle(ObjectBase):
    def __init__(self, x0, u_max, u_min, mapsize, radius, dt, noise, id, target, seed, color, behavior, factor=1, behavior_type='all_obj'):
        super().__init__(x0, radius, dt, noise, id, target, seed)
        self.type = "singleint_obs"

        self.color = color
        self.behavior = behavior
        self.factor = factor
        self.behavior_type = behavior_type
        self.color = 'k' 

        
        self.A = np.array([
            [1, 0],
            [0, 1]
        ])
        self.B = np.array([
            [dt, 0],
            [0, dt]
        ])

        self.m = 2
        self.n = 2

        self.u_min = np.array([u_min[0], u_min[1]]).reshape(-1, 1)
        self.u_max = np.array([u_max[0], u_max[1]]).reshape(-1, 1)
        self.x_min = np.array([mapsize[0,0],
                               mapsize[2,0]]).reshape(-1, 1)
        self.x_max = np.array([mapsize[1,0],
                               mapsize[3,0]]).reshape(-1, 1)

        self.xlog = [self.x0] # position
        self.ulog = [np.zeros((self.m, 1))]

        self.u = np.zeros((self.m, 1)).reshape(-1, 1)

        self.trajectory = [] # position and velocity(control input)
        self.trajectory.append(np.vstack([self.x0, self.u]))
        self.velocity_xy = np.zeros((self.m, 1))

       
    def dynamics(self, x, u):
        if max(self.noise[1]) > 1e-5:
            u_noise = self.u_disturbance(u)  
        else:
            u_noise = np.zeros((self.m, 1))
        if max(self.noise[0]) > 1e-5:
            x_noise = self.x_disturbance(x)
        else:
            x_noise = np.zeros((self.n, 1))
        return self.A @ x + self.B @ (u + u_noise) + x_noise 

    def dynamics_uncertain(self, x, u, wu=np.zeros((2,1)), wx = np.zeros((2,1))):
        return self.A @ x + self.B @ (u + wu) + wx

    def nominal_input(self, X, G, d_min=0.05, k_v=0.5):
        G = np.copy(G.reshape(-1, 1))
        v_max = np.sqrt(self.u_max[0]**2 + self.u_max[1]**2) # TODO
        pos_errors = G[0:2,0] - X[0:2,0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag
        return None, v_des.reshape(-1, 1)
    
    def step(self, x_k1=None, uk=None):
        if x_k1 is None:
            x_k1 = self.dynamics(self.x_curr, uk)
        self.x_curr = x_k1
        self.xlog.append(self.x_curr)
        self.ulog.append(uk)

        self.velocity_xy = uk
        self.trajectory.append(np.vstack([self.x_curr, self.velocity_xy]))

    # def step(self, x_k1=None, uk=None):
    #     # 如果没有输入，直接做一次动力学
    #     if x_k1 is None:
    #         x_k1 = self.dynamics(self.x_curr, uk)
    #     # --- NaN保护 ---
    #     if np.isnan(x_k1).any():
    #         print("[WARNING] Obstacle x_k1 contains NaN! Using last valid state.")
    #         x_k1 = np.nan_to_num(x_k1, nan=0.0)  # 或者直接用 self.x_curr，不更新
    #     self.x_curr = x_k1

    #     # uk 也防NaN
    #     if uk is not None and np.isnan(uk).any():
    #         print("[WARNING] Obstacle uk contains NaN! Using zero control.")
    #         uk = np.nan_to_num(uk, nan=0.0)

    #     self.xlog.append(self.x_curr)
    #     self.ulog.append(uk)

    #     self.velocity_xy = uk if uk is not None else np.zeros_like(self.x_curr[:2])
    #     # 也保护一下velocity_xy
    #     if np.isnan(self.velocity_xy).any():
    #         print("[WARNING] Obstacle velocity_xy contains NaN! Forcing to zero.")
    #         self.velocity_xy = np.zeros_like(self.velocity_xy)

    #     # trajectory长度保护
    #     self.trajectory.append(np.vstack([self.x_curr, self.velocity_xy]))



