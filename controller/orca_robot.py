"""
ORCA (Optimal Reciprocal Collision Avoidance) Controller for Robots

Reference:
van den Berg, J., Lin, M., & Manocha, D. (2008). Reciprocal velocity obstacles for 
real-time multi-agent navigation. In 2008 IEEE International Conference on Robotics 
and Automation (pp. 1928-1935). IEEE.

This implementation is specifically designed for robot control in the current project.
"""
import numpy as np
import rvo2

class ORCARobot:
    def __init__(self, robot, obstacles, all_robots, params):
        self.type = 'orca_robot'
        self.robot = robot
        self.obstacles = obstacles 
        self.all_robots = all_robots
        self.params = params
        
        # RVO2 simulator
        self.sim = None
        

        self.feasible = True
        self.htype = params.get('htype', 'dist')
        self.hlog = []  
        self.cons_log = []  
        
        self.configure()
        print(f"🔄 ORCA-Robot initialized for robot {self.robot.id}")

    def configure(self):

        self.neighbor_dist = 15.0      
        self.max_neighbors = 10        
        self.time_horizon = 2.0        
        self.time_horizon_obst = 2.0   
        self.safety_space = 0.3        
        self.max_speed = 1.5           
        
        # 其他属性
        self.time_step = self.robot.dt
        
        print(f"   neighbor_dist: {self.neighbor_dist}")
        print(f"   time_horizon: {self.time_horizon}")
        print(f"   safety_space: {self.safety_space}")

    def solve_opt(self, t, obstacles, all_robots, u_n=None):

        try:

            class SelfState:
                pass

            self_state = SelfState()
            self_state.px = self.robot.x_curr[0, 0]
            self_state.py = self.robot.x_curr[1, 0] 
            if self.robot.x_curr.shape[0] >= 4:
                self_state.vx = self.robot.x_curr[2, 0]
                self_state.vy = self.robot.x_curr[3, 0]
            else:
                self_state.vx = 0.0
                self_state.vy = 0.0
            self_state.gx = self.robot.target[0, 0]
            self_state.gy = self.robot.target[1, 0]
            self_state.radius = self.robot.radius
            self_state.v_pref = self.robot.v_pref
            

            human_states = []
            

            class HumanState:
                pass
                
            for other_robot in all_robots:
                if other_robot.id != self.robot.id:
                    state = HumanState()
                    state.px = other_robot.x_curr[0, 0]
                    state.py = other_robot.x_curr[1, 0]
                    if other_robot.x_curr.shape[0] >= 4:
                        state.vx = other_robot.x_curr[2, 0]
                        state.vy = other_robot.x_curr[3, 0]
                    else:
                        state.vx = 0.0
                        state.vy = 0.0
                    state.radius = other_robot.radius
                    human_states.append(state)
            
            # 添加obstacles
            for obs in obstacles:
                state = HumanState()
                state.px = obs.x_curr[0, 0]
                state.py = obs.x_curr[1, 0]
                if hasattr(obs, 'velocity_xy') and obs.velocity_xy is not None:
                    state.vx = obs.velocity_xy[0, 0]
                    state.vy = obs.velocity_xy[1, 0]
                else:
                    state.vx = 0.0
                    state.vy = 0.0
                state.radius = obs.radius
                human_states.append(state)




            
            # max number of humans = current number of humans
            self.max_neighbors = len(human_states)
            self.radius = self_state.radius
            params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
            if self.sim is not None and self.sim.getNumAgents() != len(human_states) + 1:
                del self.sim
                self.sim = None
                
            if self.sim is None:
                self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
                self.sim.addAgent((self_state.px, self_state.py), *params, self_state.radius + 0.01 + self.safety_space,
                                  self_state.v_pref, (self_state.vx, self_state.vy))
                for human_state in human_states:
                    self.sim.addAgent((human_state.px, human_state.py), *params, human_state.radius + 0.01 + self.safety_space,
                                      self.max_speed, (human_state.vx, human_state.vy))
            else:
                self.sim.setAgentPosition(0, (self_state.px, self_state.py))
                self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
                for i, human_state in enumerate(human_states):
                    self.sim.setAgentPosition(i + 1, (human_state.px, human_state.py))
                    self.sim.setAgentVelocity(i + 1, (human_state.vx, human_state.vy))

            # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
            velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
            speed = np.linalg.norm(velocity)
            

            if speed > 1e-3:

                direction = velocity / speed
                desired_speed = min(self_state.v_pref, speed, 1.0)  
                pref_vel = direction * desired_speed
            else:
                pref_vel = np.array([0.0, 0.0])



            self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
            for i, human_state in enumerate(human_states):

                self.sim.setAgentPrefVelocity(i + 1, (0, 0))

            self.sim.doStep()
            computed_velocity = np.array(self.sim.getAgentVelocity(0), dtype=float)
            computed_velocity = np.nan_to_num(computed_velocity, nan=0.0)


            

            current_velocity = np.array([self_state.vx, self_state.vy])
            desired_velocity = computed_velocity
            

            dt = self.time_step
            acceleration = (desired_velocity - current_velocity) / dt
            
 
            max_acc = 2.0  
            acc_magnitude = np.linalg.norm(acceleration)
            if acc_magnitude > max_acc:
                acceleration = acceleration / acc_magnitude * max_acc

            
            u = acceleration.reshape(-1, 1)
            

            robot_pos = (self_state.px, self_state.py)
            min_dist = self._compute_min_distance(robot_pos, self.robot.radius, obstacles, all_robots)
            self.hlog.append(np.array([max(0.0, min_dist)]))
            
            return None, u, "Solve_Succeeded"
            
        except Exception as e:
            print(f"❌ ORCA robot solve error: {e}")
            import traceback
            traceback.print_exc()
            self.feasible = False
            

            if u_n is not None:
                u = u_n
            else:
                _, u = self.robot.nominal_input(self.robot.x_curr, self.robot.target)
            
            self.hlog.append(np.array([0.0]))
            return None, u, "Solve_Failed"
    
    def _compute_min_distance(self, robot_pos, robot_radius, obstacles, all_robots):

        min_dist = float('inf')
        

        for obs in obstacles:
            obs_pos = (obs.x_curr[0, 0], obs.x_curr[1, 0])
            dist = np.linalg.norm(np.array(robot_pos) - np.array(obs_pos)) - robot_radius - obs.radius
            min_dist = min(min_dist, dist)
        

        for other_robot in all_robots:
            if other_robot.id != self.robot.id:
                other_pos = (other_robot.x_curr[0, 0], other_robot.x_curr[1, 0])
                dist = np.linalg.norm(np.array(robot_pos) - np.array(other_pos)) - robot_radius - other_robot.radius
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    def reset(self):

        if self.sim is not None:
            del self.sim
            self.sim = None
        self.feasible = True
        print(f"🔄 ORCA-Robot Controller reset for robot {self.robot.id}")
    
    def __del__(self):

        if hasattr(self, 'sim') and self.sim is not None:
            del self.sim
