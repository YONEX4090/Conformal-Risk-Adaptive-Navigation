import json
import yaml
import numpy as np
from .robot.double_int import DoubleIntegratorRobot as DoubleIntegratorRobotV2
from .obstacle.obs import Obstacle as ObstacleInt
from util.util import *

class EnvironmentBase:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        mapsize = config.get("mapsize", [0, 20, 0, 20])
        dt = config.get("dt", 0.1)
        goal_thershold = config.get("goal_thershold", 0.3)
        
        seed = config.get("random_seed", 1)  # Default seed if not provided
        np.random.seed(seed)  # This will globally set the random state

        self.mapsize = np.array(mapsize).reshape(-1, 1)
        self.dt = dt
        self.goal_thershold = goal_thershold
        self.seed = seed
        
        self.t_curr = 0.0

        # Initialize robots
        self._init_robots(config)
        # Initialize obstacles
        self.x0s = []
        self.targets = []
        self._init_obstacles(config)


    def _init_robots(self, config):
        # Initialize multiple robots
        self.robots = []
        for id, robot_config in enumerate(config.get("robots", [])):
            robot_type = robot_config.get("type", "doubleint").lower()
            x0 = robot_config.get("x0", [0, 0, 0, 0])
            radius = robot_config.get("radius", 0.1)
            noise = robot_config.get("noise", [0.0, 0.0, 0.0, 0.0])
            target = robot_config.get("target", [0,0,0,0])
            if robot_type == "doubleint":
                u_max = robot_config.get("u_max", [3, 3])
                u_min = robot_config.get("u_min", [-3, -3]) # [ax, ay]
                robot = DoubleIntegratorRobotV2(x0,
                                                u_max, u_min,
                                                self.mapsize, 
                                                radius,
                                                self.dt, noise,
                                                id,
                                                target,
                                                self.seed)

            else:
                raise ValueError(f"Unknown robot type: {robot_type}")

            self.robots.append(robot)




    def _init_obstacles(self, config):
        if "obstacles" not in config or config.get("obstacles") is None:
            return
        self.obstacles = []
        self.existing_points = np.empty((0, 2)) 
        self.existing_radii = np.empty((0, 1)) 

        for obs_block in config.get("obstacles", []):
            number = obs_block.get("number", 1)
            distribution = obs_block.get("distribution", {})
            init_type = distribution.get("name", "manual")

            u_max = obs_block.get("u_max", [0.7, 0.7]) 
            u_min = obs_block.get("u_min", [-0.7, -0.7])
            radius = obs_block.get("radius", 0.5)
            color = obs_block.get("color", "grey")
            noise_obs = obs_block.get("noise", [0.0, 0.0, 0.0, 0.0])
            kinematics = obs_block.get("kinematics", "int") 
            behavior = obs_block.get("behavior",{}) 
            behavior_mode = behavior.get("mode", "rvo") 
            behavior_factor = behavior.get("factor", 1.0)  
            behavior_type = behavior.get("type", "all_obj") 

            if init_type == "random":
                range_low = distribution.get("range_low", [self.mapsize[0], self.mapsize[2], 0.0])
                range_high = distribution.get("range_high", [self.mapsize[1], self.mapsize[3], 3.14])
       
                mode = distribution.get("mode", "y_upper_lower")

                x0s, targets = generate_crowd_positions(
                    range_low, range_high, number, radius, mode, existing_points=np.array(self.existing_points), existing_radii=self.existing_radii
                )
                
                self.existing_points = np.vstack((self.existing_points, x0s, targets))
                self.existing_radii = np.vstack((self.existing_radii, radius * np.ones((2 * len(x0s), 1))))

                    
            elif init_type == "manual":
                x0s = obs_block.get("x0", [])
                targets = obs_block.get("target", x0s)
                self.existing_points = np.vstack((self.existing_points, x0s, targets))
                self.existing_radii = np.vstack((self.existing_radii, radius * np.ones((2 * len(x0s), 1))))
                

            self.x0s.append(x0s)  
            self.targets.append(targets)

            for i, (x0, target) in enumerate(zip(x0s, targets)):
                obs_id = i
                if kinematics == "int":
                    self.obstacles.append(
                        ObstacleInt(
                            x0=x0,
                            u_max=u_max,
                            u_min=u_min,
                            mapsize = self.mapsize,
                            radius=radius,
                            dt=self.dt,
                            noise=noise_obs,
                            id=obs_id,
                            target = target,
                            seed = self.seed,
                            color=color,
                            behavior = behavior_mode,
                            factor=behavior_factor,
                            behavior_type=behavior_type
                        )
                    )
                
    

    def done(self, t, time_total, feasible=True):

        info = {
            "feasible": feasible,
            "collision": False,
            "frozen": False,
            "reached_goal": False,
        }

        if not feasible:
            # message = "Infeasible solution."
            return True, info

        if t >= time_total:
            # message = "Simulation time reached."
            return True, info

        for robot in self.robots:
            for obs in self.obstacles:
                if np.linalg.norm(robot.x_curr[0:2] - obs.x_curr[0:2]) < robot.radius + obs.radius:
                    info["collision"] = True
                    # message = f"Robot {robot.id} collided with obstacle {obs.id}."
                    return True, info

        all_reached = True
        for robot in self.robots:
            if np.linalg.norm(robot.x_curr[0:2] - robot.target[0:2]) < self.goal_thershold:
                continue
            else:
                all_reached = False
                break
        if all_reached:
            info["reached_goal"] = True
            # message = "All robots reached the goal."
            return True, info

        return False, info
    
