
import numpy as np
from typing import List, Tuple, Dict, Optional, DefaultDict
from collections import defaultdict

class InteractionAwareConformalPrediction:
    def __init__(self, params: Dict):

        self.prediction_horizon: int = params.get("prediction_horizon", 5)  
        self.alpha: float = params.get("failure_probability", 0.05)         
        self.tau: float = 1.0 - self.alpha                                  
        self.num_simulation_episodes: int = params.get("num_episodes", 4)
        self.dt: float = params.get("dt", 0.25)
        self.update_frequency: int = params.get("icp_update_frequency", 10)  
        self.max_radius: float = params.get("max_radius", 0.35)
        self.min_radius: float = params.get("min_radius", 0.01)
        self.ema: float = params.get("ema", 0.6)                             
        self.rate_limit: float = params.get("rate_limit", 0.15)               
        self.batch_weight: float = params.get("batch_weight", 0.5)           


        self.M: int = params.get("M", 10)
        base_gamma = params.get("base_gamma", 0.25)                           
        self.gammas = np.array([base_gamma / (2**m) for m in range(self.M)], dtype=float)
        self.eta_hedge: float = params.get("eta_hedge", 2.0)                  
        self.sigma_mix: float = params.get("sigma_mix", 0.05)                


        self.q_est: DefaultDict[int, np.ndarray] = defaultdict(lambda: self._get_initial_radius())  
        self.q_bank: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.tile(np.linspace(0.02, 0.12, self.M)[:,None], (1,self.prediction_horizon)))  # [M, K]
        self.w_bank: DefaultDict[int, np.ndarray] = defaultdict(lambda: np.full((self.M, self.prediction_horizon), 1.0/self.M, dtype=float))


        self.step_counter: int = 0
        self.pred_hist = []
        self.last_outer_update: int = -9999
        

        self._initial_radius_counter = 0
    
    def _get_initial_radius(self) -> np.ndarray:

        base_values = [0.02, 0.05, 0.08, 0.12, 0.15]
        initial_value = base_values[self._initial_radius_counter % len(base_values)]
        self._initial_radius_counter += 1
        return np.full((self.prediction_horizon,), initial_value, dtype=float)


    def _dtaci_update_one(self, human_id: int, k: int, err: float, lr_scale: float = 1.0):


        qs = self.q_bank[human_id][:, k]    # shape [M]
        ws = self.w_bank[human_id][:, k]    # shape [M]


        indicators = (err < qs).astype(float)
        # Robbins–Monro for quantile: q ← q + γ*(τ - I)
        qs_new = qs + (self.gammas * lr_scale) * (self.tau - indicators)

        # pinball loss: ρ_τ(e - q) = (τ - 1[e<q])*(e - q)
        pinball = (self.tau - indicators) * (err - qs)
        # Hedge（softmax over negative loss）

        stabilized = -self.eta_hedge * (pinball - np.mean(pinball))

        expw = np.exp(stabilized)
        new_ws = (1 - self.sigma_mix) * ws * expw + self.sigma_mix * (1.0 / self.M)
        new_ws = new_ws / (np.sum(new_ws) + 1e-12)


        q_comb = float(np.dot(new_ws, qs_new))


        q_prev = float(self.q_est[human_id][k])

        q_clamped = np.clip(q_comb, self.min_radius, self.max_radius)
        # EMA
        q_smooth = self.ema * q_clamped + (1 - self.ema) * q_prev

        dq = np.clip(q_smooth - q_prev, -self.rate_limit, self.rate_limit)
        q_final = q_prev + dq


        self.q_bank[human_id][:, k] = qs_new
        self.w_bank[human_id][:, k] = new_ws
        self.q_est[human_id][k] = q_final


    def apply_measurements(self, current_state: Dict):
 

        human_positions = current_state.get("human_positions", [])
        if not human_positions:
            return

        curr = self.step_counter
        new_hist = []
        updates_count = 0
        
        for item in self.pred_hist:
            s = item["step"]
            preds = item["preds"]  # {h: [K,2]}
            keep = True
            for h, traj in preds.items():

                if h >= len(human_positions):
                    continue
                z = np.array(human_positions[h])
                K = min(len(traj), self.prediction_horizon)
                for k in range(K):
                    if s + (k + 1) == curr:
                        err = float(np.linalg.norm(traj[k] - z))
                        old_radius = float(self.q_est[h][k])
                        self._dtaci_update_one(h, k, err, lr_scale=1.0)
                        new_radius = float(self.q_est[h][k])
                        updates_count += 1
                        
                        # Debug: Show radius updates
                        if abs(new_radius - old_radius) > 0.001:
                            print(f"🔄 ICP Update: H{h} k={k} err={err:.3f} r: {old_radius:.3f}→{new_radius:.3f}")
                        
                if any(s + (k + 1) > curr for k in range(K)):
                    keep = True
                else:
                    keep = False
            if keep:
                new_hist.append(item)
        
        # Debug: Track update frequency
        if self.step_counter % 20 == 0:
            print(f"🔍 ICP Apply: Step {curr}, {updates_count} radius updates from {len(self.pred_hist)} history items")
            
        self.pred_hist = [it for it in new_hist if it["step"] >= curr - (self.prediction_horizon + 1)]


    def online_update_from_measurement(self, errors_by_human: Dict[int, List[float]]):
        """
        errors_by_human: {human_id: [e_k for k=0..K-1]} 
        """
        for h, e_list in errors_by_human.items():
            K = min(self.prediction_horizon, len(e_list))
            for k in range(K):
                err = float(max(0.0, e_list[k]))
                self._dtaci_update_one(h, k, err, lr_scale=1.0)


    def batch_update_from_simulation(self, current_state: Dict, robot_plan: np.ndarray, num_episodes: Optional[int] = None):
        """

        """
        if num_episodes is None:
            num_episodes = self.num_simulation_episodes


        preds = self._predict_all(current_state, robot_plan)  # {h: [K,2]}

        sims = self._simulate_all(current_state, robot_plan, num_episodes)  # List[{h: [K,2]}]

        # Debug: Track batch updates
        total_updates = 0
        radius_changes = {}
        

        for epi_idx, epi in enumerate(sims):
            for h, pred_traj in preds.items():
                if h not in epi:
                    continue
                sim_traj = epi[h]
                K = min(len(pred_traj), len(sim_traj), self.prediction_horizon)
                
                if h not in radius_changes:
                    radius_changes[h] = []
                    
                for k in range(K):
                    old_radius = float(self.q_est[h][k])
                    err = float(np.linalg.norm(pred_traj[k] - sim_traj[k]))
                    self._dtaci_update_one(h, k, err, lr_scale=self.batch_weight)
                    new_radius = float(self.q_est[h][k])
                    total_updates += 1
                    
                    # Track k=0 radius changes for debugging
                    if k == 0 and abs(new_radius - old_radius) > 0.001:
                        radius_changes[h].append((epi_idx, err, old_radius, new_radius))

        # Debug output
        if total_updates > 0:
            print(f"🔄 ICP Batch: {total_updates} updates from {num_episodes} episodes")
            for h, changes in radius_changes.items():
                if changes:
                    final_change = changes[-1]
                    print(f"   H{h}: {len(changes)} changes, final k=0: {final_change[2]:.3f}→{final_change[3]:.3f} (err={final_change[1]:.3f})")

        self.last_outer_update = self.step_counter


    def get_step1_radii(self, human_ids: List[int]) -> Dict[int, float]:
        out = {}
        for h in human_ids:
            q = float(self.q_est[h][0])  # k=0
            clipped_q = np.clip(q, self.min_radius, self.max_radius)
            out[h] = clipped_q
            
        # Debug: Print radius changes every 10 steps to see if they're updating
        if self.step_counter % 10 == 0 and len(out) > 0:
            radius_str = ", ".join([f"H{h}:{r:.3f}" for h, r in out.items()])
            print(f"🔍 ICP Step {self.step_counter}: Dynamic radii = {radius_str}")
            
        return out


    def tick(self, current_state: Dict, robot_plan: np.ndarray):


        self.apply_measurements(current_state)


        preds_now = self._predict_all(current_state, robot_plan)
        self.pred_hist.append({'step': self.step_counter, 'preds': preds_now})

        self.step_counter += 1
        if (self.step_counter - self.last_outer_update) >= self.update_frequency or self._should_trigger_update(current_state):
            try:

                self.batch_update_from_simulation(current_state, robot_plan, self.num_simulation_episodes)
            except Exception as e:

                pass


    def _predict_all(self, current_state: Dict, robot_plan: np.ndarray) -> Dict[int, np.ndarray]:
        robot_pos = np.array(current_state.get("robot_position", np.zeros(2)))
        humans = list(zip(current_state.get("human_positions", []),
                          current_state.get("human_velocities", [])))
        preds = {}
        for h_id, (h_pos, h_vel) in enumerate(humans):
            preds[h_id] = self._predict_human_trajectory(np.array(h_pos), np.array(h_vel), robot_pos, robot_plan)
        return preds

    def _simulate_all(self, current_state: Dict, robot_plan: np.ndarray, num_episodes: int) -> List[Dict[int, np.ndarray]]:
        sims = []
        human_positions = current_state.get("human_positions", [])
        human_velocities = current_state.get("human_velocities", [])
        for epi in range(num_episodes):
            episode = {}
            for h_id, (h_pos, h_vel) in enumerate(zip(human_positions, human_velocities)):
                episode[h_id] = self._simulate_human_trajectory(np.array(h_pos), np.array(h_vel), robot_plan, epi)
            sims.append(episode)
        return sims

    def _predict_human_trajectory(self, h_pos: np.ndarray, h_vel: np.ndarray, robot_pos: np.ndarray, robot_plan: np.ndarray) -> np.ndarray:

        dt = self.dt
        pos = h_pos.copy()
        vel = h_vel.copy()
        out = []
        for t in range(self.prediction_horizon):
            rpos = robot_plan[min(t, len(robot_plan)-1)][:2]
            to_robot = rpos - pos
            d = np.linalg.norm(to_robot)
            avoid = -0.4 * to_robot / (d**2 + 0.5) if d > 0 and d < 3.0 else np.zeros(2)
            vel = 0.9 * vel + 0.1 * avoid * dt
            vel = np.clip(vel, -1.0, 1.0)
            pos = pos + vel * dt
            out.append(pos.copy())
        return np.array(out)

    def _simulate_human_trajectory(self, h_pos: np.ndarray, h_vel: np.ndarray, robot_plan: np.ndarray, epi_idx: int) -> np.ndarray:

        dt = self.dt
        noise_scale = 0.1 + 0.08 * epi_idx
        pos = h_pos + np.random.normal(0, noise_scale, 2)
        vel = h_vel + np.random.normal(0, noise_scale*0.5, 2)
        out = []
        for t in range(self.prediction_horizon):
            rpos = robot_plan[min(t, len(robot_plan)-1)][:2]
            to_robot = rpos - pos
            d = np.linalg.norm(to_robot)
            avoid = -2.0 * to_robot / (d**2 + 0.1) if d > 0 else np.zeros(2)
            avoid += np.random.normal(0, noise_scale*0.3, 2)
            avoid = np.clip(avoid, -1.2, 1.2)
            vel = np.clip(vel + avoid * dt, -1.5, 1.5)
            pos = pos + vel * dt
            out.append(pos.copy())
        return np.array(out)


    def _should_trigger_update(self, current_state: Dict) -> bool:
        robot_pos = np.array(current_state.get("robot_position", np.zeros(2)))
        human_positions = current_state.get("human_positions", [])
        if not human_positions:
            return False
        dmin = min(float(np.linalg.norm(robot_pos - np.array(hp))) for hp in human_positions)
        return dmin < 2.0  

