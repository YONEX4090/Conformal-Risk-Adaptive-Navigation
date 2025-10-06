"""
    Reference:

    This policy has been adapted from the repository of "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph" in ICRA 2023. The repository can be found at: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph
"""
import numpy as np
from environment.policy import Policy

class SFM(Policy):
    def __init__(self, obstacle, all_obstacles, all_robots):
        super().__init__()
        self.type = 'sfm'

        self.obstacle = obstacle
        # self.obstacle.color = 'g'
        
        self.is_bottleneck = False
        self.configure()

    def configure(self, section='sfm'):
 
        self.A = 2.0 # my setting
        self.B = 0.5
        self.KI = 1.0 # 2
        
        self.A_static = 1.0
        self.B_static = 0.03
        self.A_bottleneck = 1.0 
        self.B_bottleneck = 0.03
        
        self.time_step = self.obstacle.dt
        self.radius = 0.01
        
        self.static_obs = [] 
        self.feasible = True
        
    def solve_opt(self, ob):

        self_state = ob[0]
        human_states = ob[1]
        
        
        delta_x = self_state.gx - self_state.px
        delta_y = self_state.gy - self_state.py
        dist_to_goal = np.sqrt(delta_x**2 + delta_y**2)
        dist_to_goal = 1.0 if dist_to_goal < 1e-3 else dist_to_goal
        desired_vx = (delta_x / dist_to_goal) * self_state.v_pref
        desired_vy = (delta_y / dist_to_goal) * self_state.v_pref
        KI = self.KI # Inverse of relocation time K_i
        curr_delta_vx = KI * (desired_vx - self_state.vx)
        curr_delta_vy = KI * (desired_vy - self_state.vy)

        A = self.A # Other observations' interaction strength: 1.5
        B = self.B # Other observations' interaction range: 1.0
        interaction_vx = 0
        interaction_vy = 0
        for other_human_state in human_states:
            delta_x = self_state.px - other_human_state.px
            delta_y = self_state.py - other_human_state.py
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)
            if dist_to_human < 1e-5 or np.isnan(dist_to_human):
                continue
            
            interaction_vx += A * np.exp((self_state.radius + other_human_state.radius  - dist_to_human) / B) * (delta_x / dist_to_human)
            interaction_vy += A * np.exp((self_state.radius + other_human_state.radius  - dist_to_human) / B) * (delta_y / dist_to_human)

        # Sum of push & pull forces
        total_delta_vx = (curr_delta_vx + interaction_vx) * self.time_step
        total_delta_vy = (curr_delta_vy + interaction_vy) * self.time_step

        # clip the speed so that sqrt(vx^2 + vy^2) <= v_pref
        new_vx = self_state.vx + total_delta_vx
        new_vy = self_state.vy + total_delta_vy
        
        act_norm = np.linalg.norm([new_vx, new_vy])

        if act_norm > self_state.v_pref: # TODO hear, each axis should be u_max
            # return (2,1) action
            ux = new_vx / act_norm * self_state.v_pref
            uy = new_vy / act_norm * self_state.v_pref
            u = np.array([ux, uy]).reshape(-1, 1)
        else:
            u = np.array([new_vx, new_vy]).reshape(-1, 1)
        x = None
        return x, u