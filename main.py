import os
import time
import click

from environment.env_base import EnvironmentBase
from controller.cbf_controller_nlp import DCLFDCBF
from controller.cvar_cbf_controller_nlp import DCLFCVARDCBF
from controller.cvar_cbf_controller_nlp_beta_dt import DCLFCVARDCBF as DCLFCVARDCBFMPCBETADT
from controller.sfm import SFM
from controller.nominal import Nominal
import numpy as np

from util.util import *
from util.animation import *

# Get the absolute path of the current script directory for relative path management
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))



@click.command()
@click.option(
    "--htype", "htype", type=str, default='dist_cone', show_default=True,
    help="Type of h function (vel, dist, dist_cone). Determines safety constraint formulation."
)
@click.option(
    "--S", "S", type=int, default=15, show_default=True,
    help="Number of samples in the uncertainty distribution. Higher S increases risk accuracy."
)
@click.option(
    "--beta", "beta", type=float, default=0.99, show_default=True,
    help="Risk aversion parameter. 0.1 for risk averse, 0.9 for risk neutral."
)
@click.option(
    "--save-ani", "save_ani", type=bool, default=True, show_default=True,
    help="If True, saves the animation to disk."
)
@click.option(
    "--time-total", "time_total", type=float, default=50.0, show_default=True,
    help="Total simulation time in seconds."
)
@click.option(
    "--ctrl-type", "ctrl_type", type=str, default="adap_cvarbf", show_default=True,
    help="Controller type (cbf, cvarbf, adap_cvarbf)."
)
def main(htype, S, beta, save_ani, time_total, ctrl_type, run_id):
    np.random.seed(run_id)
    """
    Main entry point for safe navigation simulation in uncertain environments.

    Initializes the environment and controllers, runs the main simulation loop,
    and optionally saves an animation and performance metrics.
    """
    # Gather simulation parameters in a dictionary for easy passing
    params = {
        "htype": htype,
        "beta": beta,
        "S": S,
        "ctrl_type": ctrl_type,
    }

    # Choose the config folder (edit as needed to test different scenarios)
    # config_folder = SCRIPT_DIR + '/config/' + '20obs/'
    config_folder = SCRIPT_DIR + '/config/' + 'one_obs/'

    # Folder to store simulation figures and animations
    figure = 'figures/'
    figures_folder = os.path.join(config_folder, figure)
    os.makedirs(figures_folder, exist_ok=True)

    # Load the environment configuration YAML file
    file = 'config'
    env = EnvironmentBase(config_file=config_folder + file + '.yaml')

    # Create robot and obstacle controllers
    controllers = create_controllers(env, ctrl_type, params)
    obs_controllers = create_obs_controller(env)

    done = False
    # Main simulation loop
    while not done:
        start_time = time.time()
        for robot, controller in zip(env.robots, controllers):
            # Update perceived obstacles for each robot
            loc_obstacles = robot.update_local_objects(env.obstacles, env.robots, R_sensing=5)
            xk, uk, status = controller.solve_opt(env.t_curr, loc_obstacles, env.robots)

            # Logging and error handling
            log_info(f"{status} at time {env.t_curr}")
            if uk is not None:
                robot.step(xk, uk)
                print(f"Robot {robot.id} - Time Step {env.t_curr}:")
                print(f"  x_t = {robot.x_curr.flatten()}")
                print(f"  u_t = {uk.flatten()}")
                print(f"  h_t = {controller.hlog[-1].flatten()}")
            else:
                log_error(f"Optimization failed for Robot {robot.id}. Exiting simulation.")
                controller.feasible = False

            end_time = time.time()
            computation_time = end_time - start_time
            log_info(f"Solver Computation Time: {computation_time:.6f} seconds")

        # Update all obstacles' behaviors
        for obs, obs_controller in zip(env.obstacles, obs_controllers):
            ob_obs = obs.observation(env.obstacles, env.robots)
            xok, uok = obs_controller.solve_opt(ob_obs)
            obs.step(uk=uok)

        # Advance simulation time
        env.t_curr += env.dt

        # Check for simulation termination conditions (goal reached, collision, time exceeded, etc.)
        done, info = env.done(env.t_curr, time_total, controller.feasible)
        if done:
            print(info)

    # Save the animation and metrics after simulation is complete (if enabled)
    if save_ani:
        filename = os.path.join(
            figures_folder, f"{ctrl_type}_beta{beta}_h{htype}"
        )
        animate(env, controllers, filename, save_ani=save_ani)
        performance_metrics(env, controllers, info, computation_time, filename)
    return info


def create_controllers(env, ctrl_type, params):
    """
    Factory for creating the correct type of robot controllers based on the argument.

    Parameters
    ----------
    env : EnvironmentBase
        The simulation environment containing robots, obstacles, etc.
    ctrl_type : str
        The type of controller ('cbf', 'cvarbf', 'adap_cvarbf').
    params : dict
        Dictionary of controller and simulation parameters.

    Returns
    -------
    list
        List of controller instances (one per robot).
    """
    controller_mapping = {
        "cvarbf": DCLFCVARDCBF,  #  CVaR-CBF control 
        "cbf": DCLFDCBF,         # Deterministic CBF (no uncertainty)
        "adap_cvarbf": DCLFCVARDCBFMPCBETADT,  # Adaptive CVaR-CBF 
    }

    if ctrl_type not in controller_mapping:
        raise ValueError(f"Unknown controller type: {ctrl_type}")

    controllers = [
        controller_mapping[ctrl_type](robot, env.obstacles, env.robots, params)
        for robot in env.robots
    ]
    return controllers


def create_obs_controller(env):
    """
    Factory for creating controllers for obstacles, based on their behavior.

    Parameters
    ----------
    env : EnvironmentBase
        The simulation environment containing obstacle objects.

    Returns
    -------
    list
        List of obstacle controller instances (one per obstacle).
    """
    controller_mapping = {
        "sfm": SFM,
        "nominal": Nominal,
    }
    obs_controllers = []
    for obs in env.obstacles:
        if obs.behavior not in controller_mapping:
            raise ValueError(f"Unknown controller type: {obs.behavior}")
        obs_controllers.append(
            controller_mapping[obs.behavior](obs, env.obstacles, env.robots)
        )
    return obs_controllers


if __name__ == "__main__":
    main()
