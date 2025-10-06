import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'controller', 'crowdnav'))

from controller.crowdnav.rl.networks.model import Policy


class CrowdNavController:

    
    def __init__(self, robot, obstacles, robots, params=None):

        self.robot = robot
        self.obstacles = obstacles 
        self.robots = robots
        self.params = params if params else {}

        self.max_human_num = self.params.get('max_human_num', 20)
        self.predict_steps = self.params.get('predict_steps', 5)  
        self.spatial_edge_dim = int(2 * (self.predict_steps + 1))
        

        self.robot_radius = getattr(robot, 'radius', 0.3)
        self.v_pref = self.params.get('v_pref', 1.0)
        self.max_speed = self.params.get('max_speed', 1.5)
        

        self.safety_margin = self.params.get('safety_margin', 0.2)  
        

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = self._load_policy()
        

        self.last_action = np.array([0.0, 0.0])
        self.last_velocity = np.array([0.0, 0.0])  
        
        self.crowdnav_dt = 0.25  
        self.cached_action = np.array([0.0, 0.0])  
        self.last_crowdnav_time = -1.0  
        
        self.htype = "neural_policy"  
        self.type = "crowdnav"  
        

        self.velocity_to_acceleration = True  
        self.acceleration_gain = self.params.get('acceleration_gain', 2.0)  

        self.rnn_hxs = None
        self._initialize_rnn_states()
        

        self.hlog = []
        self.cons_log = []  
        self.feasible = True
        
    def _load_policy(self):


        model_dir = self.params.get('model_dir', 
                                   'controller/crowdnav/trained_models/GST_predictor_non_rand')
        model_file = self.params.get('model_file', '41200.pt')
        
        model_path = os.path.join(model_dir, 'checkpoints', model_file)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CrowdNav model not found at: {model_path}")
            

        observation_space = self._create_observation_space()
        action_space = self._create_action_space()
        

        try:
            sys.path.append(model_dir)
            from arguments import get_args
            algo_args = get_args()
            algo_args.env_name = 'CrowdSimPredRealGST-v0'
        except:
            from controller.crowdnav.arguments import get_args
            algo_args = get_args()
            algo_args.env_name = 'CrowdSimPredRealGST-v0'
        
        policy = Policy(
            observation_space.spaces,
            action_space, 
            base_kwargs=algo_args,
            base='selfAttn_merge_srnn'
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        policy.load_state_dict(state_dict)
        policy.eval()
        policy.to(self.device)
        
        print(f"Successfully loaded CrowdNav policy from {model_path}")
        return policy
    
    def _initialize_rnn_states(self):
        if self.policy is None:
            return
            
        node_num = 1
        edge_num = self.max_human_num + 1  # human_num + robot
        
        self.rnn_hxs = {}
        
        human_node_rnn_size = getattr(self.policy.base, 'human_node_rnn_size', 128)
        human_human_edge_rnn_size = getattr(self.policy.base, 'human_human_edge_rnn_size', 256)
        
        self.rnn_hxs['human_node_rnn'] = torch.zeros(
            1, node_num, human_node_rnn_size, device=self.device
        )
        self.rnn_hxs['human_human_edge_rnn'] = torch.zeros(
            1, edge_num, human_human_edge_rnn_size, device=self.device
        )
    
    def _create_observation_space(self):
        import gym
        
        d = {}
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf,
                                           shape=(self.max_human_num, self.spatial_edge_dim), dtype=np.float32)
        d['visible_masks'] = gym.spaces.Box(low=0, high=1, shape=(self.max_human_num,), dtype=bool)
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        return gym.spaces.Dict(d)
    
    def _create_action_space(self):
        import gym
        high = np.inf * np.ones([2])
        return gym.spaces.Box(-high, high, dtype=np.float32)
    
    def solve_opt(self, t_curr, loc_obstacles, all_robots):

        try:

            if self.robot is None:
                print("Error: robot is None")
                return None, None, "Failed"
            
            if self.robot.x_curr is None:
                print("Error: robot.x_curr is None")
                return None, None, "Failed"
                
            if self.robot.target is None:
                print("Error: robot.target is None") 
                return None, None, "Failed"
            

            time_since_last_call = t_curr - self.last_crowdnav_time
            should_call_network = (self.last_crowdnav_time < 0) or (time_since_last_call >= self.crowdnav_dt)
            
            
            if should_call_network:
                print("🔄 Calling the CrowdNav neural network...")
                obs = self._convert_state_to_crowdnav_format(t_curr, loc_obstacles, all_robots)
                
                if obs is None:
                    print("Error: observation conversion failed")
                    return None, None, "Failed"
                

                with torch.no_grad():

                    obs_tensor = {}
                    for key, value in obs.items():
                        if isinstance(value, np.ndarray):
                            obs_tensor[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
                        else:
                            obs_tensor[key] = torch.tensor(value).float().unsqueeze(0).to(self.device)
                    

                    masks = torch.ones(1, 1, device=self.device)  
                    value, action, action_log_probs, self.rnn_hxs = self.policy.act(
                        obs_tensor, self.rnn_hxs, masks, deterministic=True
                    )
                    
                    action = action.cpu().numpy().flatten()
                    self.cached_action = action.copy()  
                    self.last_crowdnav_time = t_curr    
            else:
                action = self.cached_action.copy()
            
            desired_velocity = np.clip(action, -self.max_speed, self.max_speed)
            
            
            if self.velocity_to_acceleration:

                current_velocity = self.robot.x_curr[2:4].flatten()  # [vx, vy]


                velocity_error = desired_velocity - current_velocity

                uk = (self.acceleration_gain * velocity_error).reshape(-1, 1)


                max_acceleration = 1.5  
                uk_before = uk.copy()
                uk = np.clip(uk, -max_acceleration, max_acceleration)

                

                dt = self.robot.dt if hasattr(self.robot, 'dt') else 0.1
                xk = self.robot.x_curr.copy()
                
                xk[0] += current_velocity[0] * dt + 0.5 * uk[0] * dt**2  # px
                xk[1] += current_velocity[1] * dt + 0.5 * uk[1] * dt**2  # py
                xk[2] += uk[0] * dt  # vx  
                xk[3] += uk[1] * dt  # vy
                

                for i, obs_obj in enumerate(loc_obstacles):
                    future_distance = np.linalg.norm(xk[:2,0] - obs_obj.x_curr[:2,0])
                    collision_distance = self.robot_radius + obs_obj.radius
                    if future_distance < collision_distance:

                

                self.last_velocity = desired_velocity
                
            else:

                uk = desired_velocity.reshape(-1, 1)
                dt = 0.1
                xk = self.robot.x_curr.copy()
                xk[0] += uk[0] * dt  
                xk[1] += uk[1] * dt  
                xk[2] = uk[0]        
                xk[3] = uk[1]        
            
            self.last_action = action
            

            h_value = self._compute_safety_metric(loc_obstacles)
            self.hlog.append(h_value)
            
            
            self.cons_log.append([0.1])  
            
            return xk, uk, "Optimal"
            
        except Exception as e:
            import traceback
            print(f"CrowdNav controller error: {e}")
            print(f"Detailed traceback: {traceback.format_exc()}")
            self.feasible = False
            return None, None, "Failed"
    
    def _convert_state_to_crowdnav_format(self, t_curr, loc_obstacles, all_robots):

        obs = {}
        
        # Robot node: [visible_humans_num, px, py, r, gx, gy, v_pref, theta]
        visible_humans = float(len(loc_obstacles))
        
        vx = float(self.robot.x_curr[2]) if len(self.robot.x_curr) > 2 else 0.0
        vy = float(self.robot.x_curr[3]) if len(self.robot.x_curr) > 3 else 0.0
        theta = np.arctan2(vy, vx)
        
        obs['robot_node'] = np.array([[
            float(self.robot.x_curr[0]),      
            float(self.robot.x_curr[1]),      
            float(self.robot_radius),         
            float(self.robot.target[0]),      
            float(self.robot.target[1]),      
            float(self.v_pref),               
            float(theta)                      
        ]], dtype=np.float32)
        

        obs['temporal_edges'] = np.array([[vx, vy]], dtype=np.float32)
        

        spatial_edges = np.ones((self.max_human_num, self.spatial_edge_dim), dtype=np.float32) * np.inf
        
        robot_pos = self.robot.x_curr[:2].flatten()
        
        for i, obstacle in enumerate(loc_obstacles[:self.max_human_num]):
            try:

                obs_pos = obstacle.x_curr[:2].flatten()
                relative_pos = obs_pos - robot_pos
                

                crowdnav_human_radius = 0.3  
                actual_obstacle_radius = getattr(obstacle, 'radius', 0.3)
                

                scale_factor = actual_obstacle_radius / crowdnav_human_radius
                
 
                if np.linalg.norm(relative_pos) > 0:
                    relative_pos = relative_pos * scale_factor
                
                print(f"🔧 Obstacle {i} scale adaptation: actual radius {actual_obstacle_radius:.1f}m → equivalent distance scaled {scale_factor:.1f} times")
                

                trajectory = []
                

                trajectory.extend([relative_pos[0], relative_pos[1]])
                

                obs_vx = float(obstacle.x_curr[2]) if len(obstacle.x_curr) > 2 else 0.0
                obs_vy = float(obstacle.x_curr[3]) if len(obstacle.x_curr) > 3 else 0.0
                
                for step in range(1, self.predict_steps + 1):
                    dt_pred = 0.25 * step  
                    future_x = relative_pos[0] + obs_vx * dt_pred * scale_factor
                    future_y = relative_pos[1] + obs_vy * dt_pred * scale_factor
                    trajectory.extend([future_x, future_y])

                spatial_edges[i, :len(trajectory)] = trajectory
                

                distance = np.linalg.norm(relative_pos)
                safe_distance = self.robot_radius + obstacle.radius + self.safety_margin
                if distance < safe_distance:
                    print(f"⚠️ distance:{distance:.3f}, safe_distance:{safe_distance:.3f}")
                        
            except Exception as e:
                print(f"Error processing obstacle {i}: {e}")

                

        spatial_edges[np.isinf(spatial_edges)] = 15.0
        obs['spatial_edges'] = spatial_edges
        

        visible_masks = np.zeros(self.max_human_num, dtype=bool)
        visible_masks[:len(loc_obstacles)] = True
        obs['visible_masks'] = visible_masks
        

        num_detected = len(loc_obstacles) if len(loc_obstacles) > 0 else 1
        obs['detected_human_num'] = np.array([num_detected], dtype=np.float32)
        
        return obs
    
    def _compute_safety_metric(self, loc_obstacles):
        if not loc_obstacles:
            return np.array([1.0])  
            
        min_distance = float('inf')
        for obs in loc_obstacles:
            distance = np.linalg.norm(self.robot.x_curr[:2] - obs.x_curr[:2])
            safety_distance = distance - self.robot_radius - obs.radius - self.safety_margin
            min_distance = min(min_distance, safety_distance)
            

        return np.array([max(min_distance, -1.0)])
