from environment.state_plus import ObservableState, FullState
import numpy as np
from scipy.stats import norm
import casadi as ca
def _pdf_or_ones(values, std):
    if std == 0:
        return np.ones((len(values), 1))
    else:
        return norm.pdf(values, 0, std).reshape(-1,1)
        

class ObjectBase:
    def __init__(self, x0, radius, dt, noise, id, target, seed =1):
        """
        Initialize the base robot and obstacle class with general attributes.
        """
        seed_obj = seed+id
        self.rng = np.random.default_rng(seed_obj)  
        
        self.dt = dt           # Time step
        self.id = id  # Assign a unique ID
        
        self.target = np.array(target, dtype=float).reshape(-1, 1)
        self.x0 = np.array(x0).reshape(-1, 1)  # Initial state vector
        self.noise = noise
        self.radius = radius

        self.m = None             # Control input dimension
        self.n = None             # State dimension
        
        self.u = None          # Control input vector
    
        self.x_min = None     
        self.x_max = None
        self.u_min = None         
        self.u_max = None
        
        self.xlog = []
        self.ulog = []
        self.un_log = []
        

        # self.u_cost = 0
        self.x_curr = self.x0.astype(float)  # Current state vector
        self.velocity_xy = np.zeros((2, 1))

    
    def dynamics(self):
        raise NotImplementedError("Dynamics method must be implemented in subclass.")
                                  
    def reset(self):
        raise NotImplementedError("Reset method must be implemented in subclass.")

    def nominal_input(self):
        raise NotImplementedError("Nominal input method must be implemented in subclass.")

    def step(self, x_k1=None, uk=None):
        raise NotImplementedError("Step method must be implemented in subclass.")

    def u_disturbance(self, control_input):
        if max(self.noise[1]) <= 1e-5 or isinstance(control_input, ca.MX):  
            return np.zeros((self.m, 1))
        else:
            std_ux = np.sqrt(self.noise[1][0]*(control_input[0, 0]**2) + self.noise[1][1]*(control_input[1,0]**2))
            std_uy = np.sqrt( self.noise[1][2]*(control_input[0, 0]**2) +  self.noise[1][3]*(control_input[1,0]**2))
            sample = self.rng.normal( [[0], [0]], scale=[[std_ux], [std_uy]])
            return  sample.reshape(-1,1)

    def x_disturbance(self, state):
        if max(self.noise[0]) <= 1e-5 or isinstance(state, ca.MX):  
            return np.zeros((self.n, 1))
        else:
            if state.shape[0] == 2:
                std_px = self.noise[0][0]
                std_py = self.noise[0][1]
                sample = self.rng.normal( [[0], [0]], scale=[[std_px], [std_py]])
            else:
                std_px = self.noise[0][0]
                std_py = self.noise[0][1]
                std_vx = self.noise[0][2]
                std_vy = self.noise[0][3]
                sample = self.rng.normal( [[0], [0], [0], [0]], scale=[[std_px], [std_py], [std_vx], [std_vy]]) 
            return sample.reshape(-1,1)

    
    def gen_pmf(self, control_input, state, noise, S):
        std_ux_ux, std_ux_uy, std_uy_ux, std_uy_uy = noise[1]
        
        std_ux = np.sqrt(std_ux_ux*(control_input[0,0]**2) +
                        std_ux_uy*(control_input[1,0]**2))
        std_uy = np.sqrt(std_uy_ux*(control_input[0,0]**2) +
                        std_uy_uy*(control_input[1,0]**2))
        samples_u = np.random.normal([[0],[0]], [[std_ux], [std_uy]], size=(S,2,1))
        pdf_ux = _pdf_or_ones(samples_u[:,0], std_ux)
        pdf_uy = _pdf_or_ones(samples_u[:,1], std_uy)
            
        if state.shape[0] == 2:
            std_px, std_py = noise[0]
            if std_px <= 1e-5 and std_py <= 1e-5:
                std_px = 0.001
                std_py = 0.001
            samples_x = np.random.normal([[0],[0]],
                                    [[std_px], [std_py]],
                                    size=(S,2,1))
            pdf_px = _pdf_or_ones(samples_x[:,0], std_px)
            pdf_py = _pdf_or_ones(samples_x[:,1], std_py)
            
            joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py 

        else:
            std_px, std_py, std_vx, std_vy = noise[0]

            samples_x = np.random.normal([[0],[0],[0],[0]],
                                    [[std_px], [std_py], [std_vx], [std_vy]],
                                    size=(S,4,1))
            pdf_px = _pdf_or_ones(samples_x[:,0], std_px)
            pdf_py = _pdf_or_ones(samples_x[:,1], std_py)
            pdf_vx = _pdf_or_ones(samples_x[:,2], std_vx)
            pdf_vy = _pdf_or_ones(samples_x[:,3], std_vy)
        
            joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py * pdf_vx * pdf_vy
        
        pmf = joint_pdf / np.sum(joint_pdf)

        return pmf, samples_u, samples_x


    def sfm_obstacles(self, obstacles):
        return [
            obj.sfm_obstacle_state() for obj in obstacles
        ]
        
    def sfm_obstacle_state(self):
        return ObservableState(
            px=self.x_curr[0, 0],
            py=self.x_curr[1, 0],
            vx=self.velocity_xy[0, 0],
            vy=self.velocity_xy[1, 0],
            radius=self.radius,
        )
        
    def sfm_state(self):
        self.update_target()
        
        # _, v_des = self.nominal_input(self.x_curr, self.target)
        v_des = 2.0 
            
        theta = np.arctan2(self.velocity_xy[1, 0], self.velocity_xy[0, 0])
        return FullState(
            px=self.x_curr[0, 0],
            py=self.x_curr[1, 0],
            vx=self.velocity_xy[0, 0],
            vy=self.velocity_xy[1, 0],
            radius=self.radius,
            v_pref=np.linalg.norm(v_des),
            gx=self.target[0, 0],
            gy=self.target[1, 0],
            theta=theta,
        )
        
    def update_target(self, proximity_threshold=0.2):
        current_pos = self.x_curr[:2]
        target_pos = self.target[:2]
        distance = np.linalg.norm(current_pos - target_pos)
        if distance < proximity_threshold:
            self.target = self.x0.copy()
        
    def update_local_objects(self, obstacles, robots, R_sensing=20,max_num_obs=5,use_one_obs=True):
        local_objects = []
        obj_pos = self.x_curr[:2]  

        obstacles_in_range = []
        for obs in obstacles:
            if self.type == 'singleint_obs' and obs.id == self.id:
                continue
            obs_pos = obs.x_curr[:2]
            distance = np.linalg.norm(obj_pos - obs_pos)
            if distance <= R_sensing:
                obstacles_in_range.append((obs, distance))

        if not obstacles_in_range and use_one_obs:
            nearest_obs = min(obstacles, key=lambda o: np.linalg.norm(obj_pos - o.x_curr[:2]))
            obstacles_in_range.append((nearest_obs, np.linalg.norm(obj_pos - nearest_obs.x_curr[:2])))

        obstacles_in_range.sort(key=lambda tup: tup[1])
        if len(obstacles_in_range) > max_num_obs:
            closest_obstacles = [obs for obs, _ in obstacles_in_range[:5]]
        else:   
            closest_obstacles = [obs for obs, _ in obstacles_in_range]

        if self.behavior_type == 'all_obj':
            for robot in robots:
                if self.type in ['singleint','doubleint']:
                    if robot.id == self.id:
                        continue
                robot_pos = robot.x_curr[:2]
                distance = np.linalg.norm(obj_pos - robot_pos)
                if distance <= R_sensing:
                    local_objects.append(robot)

        local_objects.extend(closest_obstacles)

        return local_objects
      

    def observation(self, all_obstacles, all_robots):            
        self_state = self.sfm_state()
        loc_objects = self.update_local_objects(all_obstacles, all_robots, R_sensing=5)
        loc_obj_state_list = self.sfm_obstacles(loc_objects)

        return [self_state, loc_obj_state_list]

