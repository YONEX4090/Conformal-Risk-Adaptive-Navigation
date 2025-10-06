import casadi as ca
from ..object_base import ObjectBase
import numpy as np

def is_casadi(x,u=None):
    if u is None:
        return isinstance(x, ca.SX) or isinstance(x, ca.MX)
    else:
        return isinstance(x, ca.SX) or isinstance(x, ca.MX) or isinstance(u, ca.SX) or isinstance(u, ca.MX)



class DoubleIntegratorRobot(ObjectBase):
    def __init__(self, x0, u_max, u_min, mapsize, radius, dt, noise, id, target, seed):
        assert len(x0) == 4, "DoubleIntegratorRobot requires a 4-dimensional x0"
        super().__init__(x0, radius, dt, noise, id, target, seed)
        self.type = "doubleint"
        self.behavior_type = 'only_obs'
        
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [0.5 * dt ** 2, 0],
            [0, 0.5 * dt ** 2],
            [dt, 0],
            [0, dt]
        ])

        self.n = 4
        self.m = 2

        self.u_min = np.array([u_min[0], u_min[1]]).reshape(-1, 1)
        self.u_max = np.array([u_max[0], u_max[1]]).reshape(-1, 1)
    
        self.x_min = np.array([mapsize[0,0], 
                               mapsize[2,0], 
                               -2, -2]).reshape(-1, 1)
        self.x_max = np.array([mapsize[1,0],
                               mapsize[3,0],
                               2, 2]).reshape(-1, 1)  

        self.xlog = [self.x0]
        self.ulog = [np.zeros((self.m, 1))]
        self.un_log = []  # 添加nominal input日志

        self.u = np.zeros((self.m, 1)).reshape(-1, 1)
        
        self.P = np.eye(self.n) # state cost
        self.Q = np.eye(2) 
        self.R = np.eye(self.m) # control cost
        
        # 添加v_pref属性（用于ORCA控制器）
        # 基于最大速度计算偏好速度
        self.v_pref = min(np.sqrt(self.x_max[2]**2 + self.x_max[3]**2), 1.5)
        
        # 添加v_obs_est属性（用于动画兼容性）
        self.v_obs_est = 1.0
        
        # 添加rate属性（用于动画兼容性）
        self.rate = 10.0



    def step(self, x_k1=None, uk=None):
        if x_k1 is None:
            x_k1 = self.dynamics(self.x_curr, uk)
        self.x_curr = x_k1
        self.xlog.append(self.x_curr)
        self.ulog.append(uk)
        self.velocity_xy = self.x_curr[2:4].reshape(-1, 1)
                
    def reset(self):
        self.x_curr = self.x0
        self.xlog = [self.x0]
        self.ulog = [np.zeros((self.m, 1))]
        self.un_log = []
        self.u_cost = 0

    def dynamics(self, x, u): 
        assert x.shape == (self.n, 1), f"x shape: {x.shape}"
        assert u.shape == (self.m, 1), f"u shape: {u.shape}"
        if max(self.noise[1]) > 1e-5:
            u_noise = self.u_disturbance(u) 
        else:
            u_noise = np.zeros((self.m,1))
        if max(self.noise[0]) > 1e-5: 
            x_noise = self.x_disturbance(x)
        else:
            x_noise = np.zeros((self.n,1))
        return self.A @ x + self.B @ (u + u_noise) + x_noise  
    
    
    def dynamics_uncertain(self, x, u, wu =np.zeros((2,1)), wx= np.zeros((4,1))): 
        return self.A @ x + self.B @ (u + wu) + wx
    

    def nominal_input(self, X, G, d_min=0.05, k_v=0.5, k_a=2.5):
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = np.sqrt(self.x_max[2]**2 + self.x_max[3]**2)
        a_max = np.sqrt(self.u_max[0]**2 + self.u_max[1]**2)

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[2:4, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1), v_des.reshape(-1, 1)
    

    def agent_barrier_dt(self, x_k, x_k1, obs_state = None, radius=0.1, H=np.zeros((4, 1)), L=-np.inf, htype='dist'):
        if H is None:
            H = ca.MX.zeros(4, 1)
        # if L is None:
        #     L = -1e10  # CasADi 不支持 -np.inf，这里用一个极小值代替
        if htype == 'dist':
            obs_pos = obs_state[0:2].reshape(-1, 1)
            h_k1 = self.h_dist(x_k1, obs_pos, radius)
            h_k = self.h_dist(x_k, obs_pos, radius)
        elif htype == 'linear':
            h_k1 = self.h_linear(x_k1, H = H, L = L)
            h_k = self.h_linear(x_k, H=H, L=L)
        elif htype == 'vel':
            obs_pos = obs_state[0:2].reshape(-1, 1)
            h_k1 = self.h_vel(x_k1, obs_pos, radius)
            h_k = self.h_vel(x_k, obs_pos, radius)
        elif htype == 'dist_cone':
            obs_state = obs_state.reshape(-1,1)
            h_k1 = self.h_dist_cone(x_k1, obs_state, radius)
            h_k = self.h_dist_cone(x_k, obs_state, radius)
        else:
            raise ValueError(f"Unknown htype: {htype}")

        d_h = h_k1 - h_k

        return h_k, d_h

    
    def h_dist(self, x_k, obs_pos, radius, beta=1.05):
        '''Computes CBF h(x) = ||x-x_obs||^2 - d_min^2'''
        if is_casadi(x_k):
            h = ca.mtimes((x_k[0:2] - obs_pos[0:2]).T, (x_k[0:2] - obs_pos[0:2])) - beta*radius**2
        else:
            h = (x_k[0, 0] - obs_pos[0, 0])**2 + (x_k[1, 0] - obs_pos[1, 0])**2 - beta*radius**2
        return h
    
    def h_linear(self, x_k, H = np.array([0,0,0,1]).reshape(-1,1), L=-1):
        if is_casadi(x_k):
            h = ca.mtimes(x_k.T, H) + L 
        else:
            h = x_k.T @ H + L
        return h
    
    def h_vel(self, x_k, obs_pos, radius):
        if is_casadi(x_k):
            vel_norm_sq  = ca.fmax(x_k[2]**2 + x_k[3]**2, 0)
            h = ca.mtimes((x_k[0:2] - obs_pos[0:2]).T, (x_k[0:2] - obs_pos[0:2])) - radius**2 - vel_norm_sq /(2* self.u_max[0])
        else:
            vel_norm_sq = np.maximum(x_k[2]**2 + x_k[3]**2, 0)
            h = (x_k[0, 0] - obs_pos[0, 0])**2 + (x_k[1, 0] - obs_pos[1, 0])**2 - radius**2 - vel_norm_sq /(2* self.u_max[0])
        return h
        
    
    def collision_cone_value(self, robot_state, obs_state):
        p_rel = obs_state[0:2] - robot_state[0:2]
        v_rel = obs_state[2:4] - robot_state[2:4]
        if is_casadi(robot_state):
            norm_p_rel = ca.norm_2(p_rel)
            norm_v_rel = ca.norm_2(v_rel)
            dot_product = ca.dot(p_rel, v_rel) / (norm_p_rel * norm_v_rel)
        else:
            norm_p_rel = np.linalg.norm(p_rel)
            norm_v_rel = np.linalg.norm(v_rel)
            if norm_p_rel < 1e-5 or norm_v_rel < 1e-5:
                dot_product = 0.0
            else:
                dot_product = np.dot(p_rel.T, v_rel) / (norm_p_rel * norm_v_rel)
                dot_product = float(dot_product)
        return dot_product

        
        
    def h_dist_cone(self, x_k, obs_state, radius, v_obs_est=1.0, rate=10.0):
        self.v_obs_est = v_obs_est
        self.rate = rate

        if is_casadi(x_k):
            dot_product = self.collision_cone_value(x_k, obs_state)
            h_dist = ca.mtimes((x_k[0:2] - obs_state[0:2]).T, (x_k[0:2] - obs_state[0:2])) - radius**2
            w = ca.norm_2(obs_state[2:4]) / self.v_obs_est
            theta_val =  (1.0 / rate) * ca.log(1.0 + ca.exp(-rate * dot_product))
  
        else:
            dot_product = self.collision_cone_value(x_k, obs_state)
            h_dist = (x_k[0, 0] - obs_state[0, 0])**2 + (x_k[1, 0] - obs_state[1, 0])**2 - radius**2
            w = np.linalg.norm(obs_state[2:4]) / self.v_obs_est
            theta_val = (1.0 / rate) * np.log(1 + np.exp(-rate * dot_product)) 
            theta_val = float(theta_val)
            
        h_vel = radius**2 * w * theta_val
        return  h_dist - h_vel
    