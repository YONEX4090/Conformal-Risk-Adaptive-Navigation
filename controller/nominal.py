import numpy as np
    
    
class Nominal:
    def __init__(self, obstacle, all_obstacles, all_robots, d_min = 0.1, k_v = 0.5):
        self.type = 'nominal'

        self.obstacle = obstacle
        self.d_min = 0.1 # goal_thershold
        self.k_v = 0.5  

        self.feasible = True


    def solve_opt(self, ob):
        self.obstacle.update_target()
        _, u = self.obstacle.nominal_input(self.obstacle.x_curr, self.obstacle.target)
        x = None
        assert u.shape == (2, 1), f"u shape: {u.shape}"
        return x, u

       