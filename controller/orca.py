"""
ORCA (Optimal Reciprocal Collision Avoidance) Controller

Reference:
van den Berg, J., Lin, M., & Manocha, D. (2008). Reciprocal velocity obstacles for 
real-time multi-agent navigation. In 2008 IEEE International Conference on Robotics 
and Automation (pp. 1928-1935). IEEE.

This implementation is based on the CrowdNav reference and adapted for the current project.
"""
import numpy as np
import rvo2
from environment.policy import Policy

class ORCA(Policy):
    def __init__(self, obstacle, all_obstacles, all_robots):
        super().__init__()
        self.type = 'orca'

        self.obstacle = obstacle
        # self.obstacle.color = 'g'
        
        self.is_bottleneck = False
        self.configure()

    def configure(self, section='orca'):
        self.neighbor_dist = 10
        self.safety_space = 0.15
        self.time_horizon = 5
        self.time_horizon_obst = 5
        self.time_step = self.obstacle.dt
        self.radius = 0.01
        self.sim = None
        self.max_speed = 1.0
        
        self.static_obs = [] 
        self.feasible = True
    
    def solve_opt(self, ob):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = ob[0]
        human_states = ob[1]

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
        pref_vel = velocity / speed if speed > 1 else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        v = np.array(self.sim.getAgentVelocity(0), dtype=float)   # shape (2,)
        v = np.nan_to_num(v, nan=0.0)
        u = v.reshape(2, 1)
        x = None
        return x, u
