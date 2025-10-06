#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from matplotlib.animation import FuncAnimation


def performance_metrics(env, controllers, info, computation_time, filename):
        
    (state_histories, control_inputs_list, h_values_list, cons_values_list,
     _, obs_state_list, _, velocities_list, cost_list) = load_data_from_controller(env, controllers)
    
    metrics = []
    time_step = env.robots[0].dt  
    robot_info = info

    for robot_idx, state_history in enumerate(state_histories):
            
        feasible = robot_info.get("feasible", controllers[robot_idx].feasible)
        
        metric = {
            "robot_id": robot_idx,
            "feasible": feasible,
            "collision": robot_info.get("collision", False),
            "frozen": robot_info.get("frozen", False),
            "reached_goal": robot_info.get("reached_goal", False),
        }

            
        success = feasible and robot_info.get("reached_goal", False) \
                  and (not robot_info.get("collision", False)) and (not robot_info.get("frozen", False))
        metric["success"] = success

        if success:
            x_positions = state_history[0, :]
            y_positions = state_history[1, :]
            robot_radius = env.robots[robot_idx].radius

            diff_x = np.diff(x_positions)
            diff_y = np.diff(y_positions)
            distances = np.sqrt(diff_x**2 + diff_y**2)
            trajectory_length = float(np.sum(distances))

            trajectory_time = float((state_history.shape[1] - 1) * time_step)

            min_safety_margin = float('inf')
            for t in range(state_history.shape[1]):
                for obs_idx, obs in enumerate(env.obstacles):
                    if t < obs_state_list[obs_idx].shape[1]:
                        ox = obs_state_list[obs_idx][0, t]
                        oy = obs_state_list[obs_idx][1, t]
                        r_obs = obs.radius
                        dist = np.sqrt((x_positions[t] - ox)**2 + (y_positions[t] - oy)**2)
                        margin = dist - (robot_radius + r_obs)
                        if margin < min_safety_margin:
                            min_safety_margin = float(margin)

            if len(h_values_list[robot_idx]) > 0:
                min_h_value = float(np.min(h_values_list[robot_idx]))
            else:
                min_h_value = None

            if len(cons_values_list[robot_idx]) > 0:
                min_cons_value = float(np.min(cons_values_list[robot_idx]))
            else:
                min_cons_value = None

            mean_cost = float(np.mean(cost_list[robot_idx]))

        else:
            trajectory_length = None
            trajectory_time = None
            min_safety_margin = None
            min_h_value = None
            min_cons_value = None
            mean_cost = None

            
        if computation_time is not None:
            metric["computation_time"] = computation_time
        else:
            metric["computation_time"] = None
            
        metric.update({
            "trajectory_length": trajectory_length,
            "trajectory_time": trajectory_time,
            "min_safety_margin": min_safety_margin,
            "min_h_value": min_h_value,
            "min_cons_value": min_cons_value,
            "cost": mean_cost
        })

        metrics.append(metric)

    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4, default=lambda o: float(o) if isinstance(o, (np.float32, np.float64)) else o)

    print(f"Metrics saved to {filename}")


def compute_safety_margins(env, state_histories, obs_state_list):
    """
    Compute the safety margin for each robot over time, ensuring the time steps
    match the robot trajectories.
    """
    safety_margin_list = []

    for i, robot_states in enumerate(state_histories):
        T = robot_states.shape[1]  # Get the number of time steps for this robot
        r_robot = env.robots[i].radius

        margins = np.zeros(T)
        for t in range(T):
            # Compute the minimum safety margin for this robot at time t
            min_dist_minus_radii = np.inf
            for j, obs in enumerate(env.obstacles):
                if t < obs_state_list[j].shape[1]:  # Ensure obstacle data exists
                    ox = obs_state_list[j][0, t]
                    oy = obs_state_list[j][1, t]
                    r_obs = obs.radius

                    dist = np.sqrt((robot_states[0, t] - ox)**2 + (robot_states[1, t] - oy)**2)
                    dist_minus_radii = dist - (r_robot + r_obs)

                    if dist_minus_radii < min_dist_minus_radii:
                        min_dist_minus_radii = dist_minus_radii

            margins[t] = min_dist_minus_radii

        safety_margin_list.append(margins)

    return safety_margin_list



def load_data_from_controller(env, controllers = []):
    """
    Returns:
        state_histories, control_inputs_list, h_values_list, beta_values_list,
        obs_state_list, nominal_inputs_list, velocities_list
    """
    state_histories = [np.hstack(robot.xlog) for robot in env.robots]

    control_inputs_list = [np.hstack(robot.ulog) for robot in env.robots]
            
    h_values_list = []
    for controller in controllers:
        # Compute the minimum for each entry in hlog independently
        h_values = [np.min(entry) for entry in controller.hlog]
        h_values_list.append(h_values)
    cons_values_list = []
    for controller in controllers:
        cons_values = [np.min(entry) for entry in controller.cons_log]
        cons_values_list.append(cons_values)
            
    if hasattr(controllers[0], "beta_log") and len(controllers[0].beta_log) > 0:
        beta_values_list = []
        for controller in controllers:
            beta_values = [np.min(entry) for entry in controller.beta_log]
            beta_values_list.append(beta_values)
    else:
        beta_values_list = [np.zeros((len(controller.hlog))) for controller in controllers]


    obs_state_list = [np.hstack(obs.trajectory) for obs in env.obstacles]

    if len(env.robots[0].un_log) > 0:
        nominal_inputs_list = [np.hstack(robot.un_log) for robot in env.robots]
    else:
        nominal_inputs_list = [np.zeros((2, 1)) for _ in env.robots]

    velocities_list = []
    if env.robots[0].type == "doubleint":
        for state_history in state_histories:
            velocities = state_history[2:4, :] # vx, vy
            velocities_list.append(velocities)
    elif env.robots[0].type == "unicycle_v2":
        for state_history in state_histories:
            velocities = state_history[3, :] # v
            thetas = state_history[2, :] # theta
            vx = velocities * np.cos(thetas)
            vy = velocities * np.sin(thetas)
            velocities_list.append(np.vstack((vx,vy)))
        
    if hasattr(controllers[0], "cost_log") and len(controllers[0].cost_log) > 0:
        cost_list = [np.hstack(controller.cost_log) for controller in controllers]
    else:
        cost_list = [np.zeros((1, 1)) for _ in env.robots]
    
    return (state_histories, control_inputs_list, h_values_list, beta_values_list, cons_values_list,
            obs_state_list, nominal_inputs_list, velocities_list, cost_list)

def init_obstacles(ax_left, obs_state_list, env):
    obstacle_patches = []
    obs_patches_original = []
    obstacle_texts = []

    for i, obs in enumerate(env.obstacles):
        initial_x = obs_state_list[i][0, 0]
        initial_y = obs_state_list[i][1, 0]
        radius = obs.radius

        obstacle_circle = plt.Circle((initial_x, initial_y), radius, 
                                     color='grey', alpha=0.3)
        ax_left.add_patch(obstacle_circle)
        obstacle_patches.append(obstacle_circle)

        obs_orig = plt.Circle(
            (initial_x, initial_y),
            obs.radius,
            color=obs.color,
            alpha=0.3,
            fill=True
        )
        ax_left.add_patch(obs_orig)
        obs_patches_original.append(obs_orig)

        obstacle_texts= []
        
    return obstacle_patches, obs_patches_original, obstacle_texts


def init_robots(ax_left, env):
    scatters = []
    circles = []
    circles_original = []

    for i, robot in enumerate(env.robots):
        # initial and target positions
        ax_left.plot(robot.x0[0], robot.x0[1], 'yo', label=f"Start",markersize=8, alpha=0.6)
        ax_left.plot(robot.target[0], robot.target[1], 'y^', 
                     label=f"Goal", markersize=8, alpha=0.6)

        scat = ax_left.scatter([], [], s=20, color='blue', alpha =0.4)  # Use 'color' instead of 'c'

        scatters.append(scat)

        # dynamic radius circle (initially same as original radius, but lighter color)
        radius_circle = plt.Circle((robot.x0[0], robot.x0[1]), robot.radius, 
                                   color='lightblue', alpha=0.5)
        ax_left.add_patch(radius_circle)
        circles.append(radius_circle)

        # original circle
        circle_orig = plt.Circle((robot.x0[0], robot.x0[1]), robot.radius, 
                                 color='gray', alpha=0.3, fill=True)
        ax_left.add_patch(circle_orig)
        circles_original.append(circle_orig)

    return scatters, circles, circles_original

def init_lines(ax_h, ax_margin, ax_u, ax_beta, env):
    """
    Initialize the lines for h, velocity, control inputs, and beta values.
    """
    lines_h = []
    lines_margin = []
    lines_u1 = []
    lines_u2 = []
    lines_beta = []
    lines_beta_u = []
    lines_beta_u_bar = []

    lines_nom_u1 = []
    lines_nom_u2 = []
    lines_cost = []

    for i in range(len(env.robots)):
        colors = ['#ef8a62', '#67a9cf']

        # min h
        line_h, = ax_h.plot([], [],linewidth=2,color = 'grey',alpha= 0.9)
        lines_h.append(line_h)
        # min beta
        line_beta, = ax_beta.plot([], [],linewidth= 2,color = 'orange',alpha= 0.9, label = r"$\beta_t$")
        lines_beta.append(line_beta)
        # min beta_u
        line_beta_u, = ax_beta.plot([], [],linewidth= 1,color = 'green',alpha= 0.9, linestyle='--', label = r"$\beta_u$")
        lines_beta_u.append(line_beta_u)
        # min beta_u_bar
        # line_beta_u_bar, = ax_beta.plot([], [],linewidth= 1,color = 'red',alpha= 0.9, linestyle='--', label = r"$\bar{\beta}_u$")
        line_beta_u_bar, = ax_beta.plot([], [],linewidth= 1,color = 'red',alpha= 0.9, linestyle='--')
        lines_beta_u_bar.append(line_beta_u_bar)

        # velocity
        line_margin, = ax_margin.plot([], [],linewidth= 2,color = 'grey',alpha= 0.9)
        lines_margin.append(line_margin)

        # control inputs
        line_u1, = ax_u.plot([], [], linewidth= 2, label=f'$u_x$',color=colors[0],alpha= 0.9)
        line_u2, = ax_u.plot([], [], linewidth= 2, label=f'$u_y$',color=colors[1],alpha= 0.9)
        
        lines_u1.append(line_u1)
        lines_u2.append(line_u2)
    
        line_nom_u1, = ax_u.plot([], [], '--')
        line_nom_u2, = ax_u.plot([], [], '--')
        lines_nom_u1.append(line_nom_u1)
        lines_nom_u2.append(line_nom_u2)
        
        
        ax_h.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
        ax_margin.axhline(y=0, color='r', linestyle='--', linewidth=0.8)
        ax_u.axhline(y=env.robots[0].u_min[0], color='r', linestyle='--', linewidth=0.8)
        ax_u.axhline(y=env.robots[0].u_max[0], color='r',linestyle='--',linewidth=0.8)
 
    
    return (lines_h, lines_margin, lines_u1, lines_u2,
            lines_beta, lines_beta_u, lines_beta_u_bar, lines_nom_u1, lines_nom_u2, lines_cost)


def set_y_axis_limits(ax_h, ax_margin, ax_u, ax_beta):
    ax_h.set_ylim(-0.5, 15.0)
    ax_margin.set_ylim(-0.5, 15.0)
    ax_u.set_ylim(-4, 4)
    # ax_beta.set_ylim(-0.1, 0.5)
    ax_beta.set_ylim(-0.1, 1.1)


def compute_closest_obstacle_factors(env, num, compute_factor_and_params, state_histories, obs_state_list):
    time_index = num - 1 if num > 0 else 0

    # For each robot, find the obstacle that is closest (minimum Euclidean distance)
    robot_closest_factors = [1.0] * len(env.robots)
    robot_closest_obs = [-1] * len(env.robots)
    robot_closest_distances = [float('inf')] * len(env.robots)
    robot_delta = [0.0] * len(env.robots)

    for i in range(len(env.robots)):
        # Get the current position of robot i.
        robot_x = state_histories[i][0, time_index]
        robot_y = state_histories[i][1, time_index]
        for j in range(len(env.obstacles)):
            # Get the current position of obstacle j.
            obs_x = obs_state_list[j][0, time_index]
            obs_y = obs_state_list[j][1, time_index]
            # Compute Euclidean distance between robot and obstacle.
            distance = np.sqrt((robot_x - obs_x) ** 2 + (robot_y - obs_y) ** 2)
            if distance < robot_closest_distances[i]:
                robot_closest_distances[i] = distance
                robot_closest_obs[i] = j
                # Compute the factor for this robot-obstacle pair.
                factor, _, _, Delta = compute_factor_and_params(i, j, time_index)
                robot_closest_factors[i] = factor
                robot_delta[i] = Delta

    # Similarly, for each obstacle, find the closest robot.
    obstacle_closest_factors = [1.0] * len(env.obstacles)
    obstacle_closest_robot = [-1] * len(env.obstacles)
    obstacle_closest_distances = [float('inf')] * len(env.obstacles)

    for j in range(len(env.obstacles)):
        obs_x = obs_state_list[j][0, time_index]
        obs_y = obs_state_list[j][1, time_index]
        for i in range(len(env.robots)):
            robot_x = state_histories[i][0, time_index]
            robot_y = state_histories[i][1, time_index]
            distance = np.sqrt((robot_x - obs_x) ** 2 + (robot_y - obs_y) ** 2)
            if distance < obstacle_closest_distances[j]:
                obstacle_closest_distances[j] = distance
                obstacle_closest_robot[j] = i
                factor, _, _, _ = compute_factor_and_params(i, j, time_index)
                obstacle_closest_factors[j] = factor

    return (robot_closest_factors, robot_closest_obs,
            obstacle_closest_factors, obstacle_closest_robot, robot_delta)
    


def animate(env, controllers, filename='traj', save_ani=True):
    (state_histories, control_inputs_list, h_values_list, beta_values_list, cons_values_list,
        obs_state_list, nominal_inputs_list, velocities_list, cost_list) = load_data_from_controller(env, controllers)
 
        
    # Compute safety margins
    safety_margin_list = compute_safety_margins(env, state_histories, obs_state_list)
    
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1])  
    ax_left   = fig.add_subplot(gs[:, 0])
    ax_beta   = fig.add_subplot(gs[0, 1])   # row: 0, col: 1
    ax_u      = fig.add_subplot(gs[1, 1])   # row: 1, col: 1
    ax_h      = fig.add_subplot(gs[0, 2])   # row: 0, col: 2
    ax_margin = fig.add_subplot(gs[1, 2])   # row: 1, col: 2

    plt.subplots_adjust(
        left=0.08,   
        right=0.95,  
        top=0.90,    
        bottom=0.15, 
        wspace=0.25, 
        hspace=0.35  
    )

    obstacle_patches, obs_patches_original, obstacle_texts = init_obstacles(ax_left, obs_state_list, env)
    scatters, circles, circles_original = init_robots(ax_left, env)
    
    # 🟣 Initialize ICP boundary visualization (showing enhanced obstacle boundary)
    icp_circles = []
    icp_texts = []
    for i in range(len(env.obstacles)):
        # Create ICP boundary circle (showing the enhanced boundary) - make it very visible
        icp_circle = Circle((0, 0), 0, fill=False, edgecolor='cyan', 
                          linewidth=4, linestyle='--', alpha=1.0)
        ax_left.add_patch(icp_circle)
        icp_circles.append(icp_circle)
        
        # Add text to show ICP enhancement value - make it large and visible
        icp_text = ax_left.text(0, 0, '', fontsize=12, ha='center', va='center',
                               color='cyan', weight='bold', alpha=1.0,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        icp_texts.append(icp_text)
    
    # Add observation range visualization around robot
    observation_range_circle = Circle((0, 0), 3.0, fill=False, edgecolor='orange', 
                                    linewidth=2, linestyle=':', alpha=0.6)
    ax_left.add_patch(observation_range_circle)
    
    # Add direction indicator dot on the observation range circle
    direction_dot = ax_left.scatter([], [], s=40, color='green', marker='o', 
                                   alpha=0.8, linewidth=2, zorder=10)
    direction_dots = [direction_dot]
    
    (lines_h, lines_margin, lines_u1, lines_u2, 
     lines_beta, lines_beta_u, lines_beta_u_bar, lines_nom_u1, lines_nom_u2,
     lines_cost) = init_lines(ax_h, ax_margin, ax_u, ax_beta,env)
     
    ax_left.set_aspect('equal', adjustable='box')
    ax_left.set_xlim(env.mapsize[0]-0.2, env.mapsize[1]+0.2)
    ax_left.set_ylim(env.mapsize[2], env.mapsize[3])
    ax_left.set_xlabel("x (m)", labelpad=0, fontsize=12)
    ax_left.set_ylabel("y (m)", labelpad=0, fontsize=12)
    
    # Add only Start and Goal to legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='yellow', linestyle='None', markersize=8, label='Start'),
        Line2D([0], [0], marker='^', color='yellow', linestyle='None', markersize=8, label='Goal')
    ]
    ax_left.legend(legend_elements, ['Start', 'Goal'], loc='upper right', fontsize=10)
    
    # # Optional: Add ICP and observation range legend entries (commented out)
    # legend_elements = ax_left.get_legend().get_legend_handles() if ax_left.get_legend() else []
    # legend_labels = [t.get_text() for t in ax_left.get_legend().get_texts()] if ax_left.get_legend() else []
    # legend_elements.extend([
    #     Line2D([0], [0], color='cyan', linestyle='--', linewidth=4, label='ICP Dynamic Zone'),
    #     Line2D([0], [0], color='orange', linestyle=':', linewidth=2, label='Observation Range'),
    #     Line2D([0], [0], marker='o', color='green', linestyle='None', markersize=8, label='Selected waypoint')
    # ])
    # legend_labels.extend(['ICP Dynamic Zone', 'Observation Range', 'Selected waypoint'])
    # ax_left.legend(legend_elements, legend_labels, loc='upper right', fontsize=10)
    ax_left.grid(False)

    # ax_h.set_xlabel('Time Step', labelpad=0,fontsize=12)
    ax_h.set_ylabel(r'CBF Value $h_t$', fontsize=12)
    # ax_h.legend()
    ax_h.grid(False)

    # ax_margin.set_title('Closest Distance to obstacle', fontsize=12)
    ax_margin.set_xlabel('Time Step', fontsize=12)
    ax_margin.set_ylabel('Closest Dist. to Obs.', fontsize=12)
    # ax_margin.legend()
    ax_margin.grid(False)
    

    # ax_u.set_title('Control Inputs Over Time', fontsize=12)
    ax_u.set_xlabel('Time Step', fontsize=12)
    ax_u.set_ylabel('Control Input', fontsize=12)
    ax_u.legend()
    ax_u.grid(False)

    # ax_beta.set_title('Risk Level', fontsize=12)
    # ax_beta.set_xlabel('Time Step', labelpad=0, fontsize=12)
    ax_beta.set_ylabel(r'Risk Level $\beta_t$',fontsize=12)
    ax_beta.legend()
    ax_beta.grid(False)
    
    set_y_axis_limits(ax_h, ax_margin, ax_u, ax_beta)

    def init_func():
        for scat, circle in zip(scatters, circles):
            scat.set_offsets(np.empty((0, 2)))
            scat.set_array([])
            circle.set_center((0, 0))
        
        for line_h in lines_h:
            line_h.set_data([], [])
        for line_beta, line_beta_u, line_beta_u_bar in zip(lines_beta, lines_beta_u, lines_beta_u_bar):
            line_beta.set_data([], [])
            line_beta_u.set_data([], [])
            line_beta_u_bar.set_data([], [])
            
        for line_margin_obj in lines_margin:
            line_margin_obj.set_data([], [])
            
        for line_u1_obj, line_u2_obj in zip(lines_u1, lines_u2):
            line_u1_obj.set_data([], [])
            line_u2_obj.set_data([], [])

        for ln_u1, ln_u2 in zip(lines_nom_u1, lines_nom_u2):
            ln_u1.set_data([], [])
            ln_u2.set_data([], [])

        for idx, obs_patch in enumerate(obstacle_patches):
            obs_x = obs_state_list[idx][0, 0]
            obs_y = obs_state_list[idx][1, 0]
            obs_patch.center = (obs_x, obs_y)
            obs_patches_original[idx].center = (obs_x, obs_y)
            # obstacle_texts[idx].set_position((obs_x, obs_y))
            
        for line_cost in lines_cost:
            line_cost.set_data([], [])
        
        return ( lines_h + lines_margin + lines_u1 + lines_u2   
                + lines_beta + lines_beta_u + lines_beta_u_bar + scatters + circles
                + obstacle_patches + circles_original
                + obs_patches_original + lines_nom_u1 + lines_nom_u2
                + obstacle_texts + lines_cost + icp_circles + icp_texts 
                + [observation_range_circle, direction_dot] )
        
    def update_frame(num):
        max_num = max([s.shape[1] for s in state_histories])

        def compute_factor_and_params(robot_idx, obs_idx, t):
            if (t < state_histories[robot_idx].shape[1] and 
                t < obs_state_list[obs_idx].shape[1]):
                if controllers[robot_idx].htype == "dist_cone":
                    dot_product = env.robots[robot_idx].collision_cone_value(state_histories[robot_idx][:, t], obs_state_list[obs_idx][:, t])
                    w = np.linalg.norm(obs_state_list[obs_idx][2:4, t]) / env.robots[robot_idx].v_obs_est
                    theta_value = (1.0 / env.robots[robot_idx].rate) * np.log(1 + np.exp(-env.robots[robot_idx].rate * dot_product)) 
                    factor = np.sqrt(1.0 + w * theta_value)
                    return factor, dot_product, 0.0, w * theta_value
                else:
                    return (1.0, 0.0, 0.0, 0.0)
            else:
                return (1.0, 0.0, 0.0, 0.0)

        (robot_factors, robot_best_obs,
         obstacle_factors, obstacle_best_robot, robot_delta) = compute_closest_obstacle_factors(
            env, num, compute_factor_and_params, state_histories, obs_state_list)
        
        for i in range(len(env.robots)):
            x = state_histories[i][0, :]
            y = state_histories[i][1, :]
            offsets = np.column_stack((x[:num], y[:num]))
            scatters[i].set_offsets(offsets)
            scatters[i].set_array(np.linspace(0, 1, num))

            if num > 0:
                robot_current_pos = (x[num - 1], y[num - 1])
                circles[i].set_center(robot_current_pos)
                circles_original[i].set_center(robot_current_pos)
                
                # Update observation range circle to follow robot
                if i == 0:  # Only for first robot
                    observation_range_circle.set_center(robot_current_pos)
                    
                    # Calculate robot direction and update direction dot
                    # Show direction dot every 1 second (assuming dt=0.25, every 4 frames = 1 second)
                    dt = env.robots[0].dt if hasattr(env.robots[0], 'dt') else 0.25
                    frames_per_second = int(1.0 / dt)  # frames needed for 1 second
                    show_dot = (num % frames_per_second) < (frames_per_second // 4)  # Show for 1/4 of the second
                    
                    if num > 1 and show_dot:
                        # Calculate direction from velocity (current pos - previous pos)
                        prev_pos = (x[num - 2], y[num - 2])
                        dx = robot_current_pos[0] - prev_pos[0]
                        dy = robot_current_pos[1] - prev_pos[1]
                        
                        # If robot is moving, show direction
                        if dx**2 + dy**2 > 0.001:  # Only if robot moved significantly
                            # Normalize direction vector
                            direction_magnitude = np.sqrt(dx**2 + dy**2)
                            dx_norm = dx / direction_magnitude
                            dy_norm = dy / direction_magnitude
                            
                            # Place dot on observation circle in direction of movement
                            observation_radius = 3.0
                            dot_x = robot_current_pos[0] + dx_norm * observation_radius
                            dot_y = robot_current_pos[1] + dy_norm * observation_radius
                            
                            direction_dot.set_offsets([(dot_x, dot_y)])
                            direction_dot.set_alpha(0.8)
                        else:
                            # Hide dot if robot is not moving
                            direction_dot.set_alpha(0.0)
                    else:
                        # Hide dot for other frames
                        direction_dot.set_alpha(0.0)
                
                if robot_best_obs[i] != -1:
                    r_eff_robot = env.robots[i].radius + (env.robots[i].radius + env.obstacles[robot_best_obs[i]].radius) * (robot_factors[i] -1)
                else:
                    r_eff_robot = env.robots[i].radius
                circles[i].set_radius(r_eff_robot)
                
        for idx, obs_patch in enumerate(obstacle_patches):
            obs_patch.set_facecolor(env.obstacles[idx].color)
            # obstacle_texts[idx].set_color('black') 
            
            if num < obs_state_list[idx].shape[1]:
                obs_x = obs_state_list[idx][0, num]
                obs_y = obs_state_list[idx][1, num]
                obs_patch.center = (obs_x, obs_y)
                obs_patches_original[idx].center = (obs_x, obs_y)
                # obstacle_texts[idx].set_position((obs_x, obs_y))
            
            # Note: Don't reset radius here - it will be set by ICP logic below
            
        # 🟣 Update obstacle radii with ICP enhancement (make obstacles themselves grow/shrink)
        for obs_idx in range(len(env.obstacles)):
            if (len(controllers) > 0 and 
                hasattr(controllers[0], 'icp_module') and 
                num > 0 and obs_idx < len(obs_state_list) and
                num <= obs_state_list[obs_idx].shape[1]):
                
                # Get obstacle position first to check observation range
                obs_pos = (obs_state_list[obs_idx][0, num-1],
                          obs_state_list[obs_idx][1, num-1])
                
                # Get robot position at this time step
                robot_pos = (state_histories[0][0, num-1] if num > 0 else state_histories[0][0, 0],
                            state_histories[0][1, num-1] if num > 0 else state_histories[0][1, 0])
                
                # Check if obstacle is within observation range (3 meters)
                observation_range = 3.0
                distance_to_robot = np.sqrt((obs_pos[0] - robot_pos[0])**2 + (obs_pos[1] - robot_pos[1])**2)
                
                if distance_to_robot > observation_range:
                    # Obstacle is outside observation range, reset to original state
                    obstacle_patches[obs_idx].set_radius(env.obstacles[obs_idx].radius)
                    obstacle_patches[obs_idx].set_facecolor(env.obstacles[obs_idx].color)
                    icp_circles[obs_idx].set_visible(False)
                    icp_texts[obs_idx].set_visible(False)
                    continue
                
                # Get individual ICP safety radius for this specific obstacle
                try:
                    # First try to get from history for the exact time step
                    history = getattr(controllers[0], 'icp_radius_history', {}).get(obs_idx, [])
                    if len(history) > 0 and num-1 < len(history):
                        # Use exact historical value for this time step
                        individual_radius = history[num-1]
                    else:
                        # Fall back to current radius if no history for this time step
                        step1_radii = controllers[0].icp_module.get_step1_radii([obs_idx])
                        individual_radius = step1_radii.get(obs_idx, 0.1)
                    
                    # Use actual ICP radius without scaling to show real dynamics
                    enhanced_radius = env.obstacles[obs_idx].radius + individual_radius
                    
                    # Update the obstacle patch itself to show dynamic radius
                    old_radius = obstacle_patches[obs_idx].get_radius()
                    obstacle_patches[obs_idx].set_radius(enhanced_radius)
                    
                    # Make the obstacle color change with ICP radius for extra visibility
                    original_color = env.obstacles[obs_idx].color
                    if isinstance(original_color, str):
                        # If color is a string, keep it as is
                        dynamic_color = original_color
                    else:
                        # If color is RGB, make it more intense with higher ICP radius
                        intensity = min(1.0, 0.5 + individual_radius)
                        dynamic_color = (original_color[0] * intensity, original_color[1] * intensity, original_color[2] * intensity)
                    obstacle_patches[obs_idx].set_facecolor(dynamic_color)
                    
                    # Create ICP boundary ring to show the enhancement clearly
                    # Comment out ICP circle display to avoid visual clutter
                    # icp_circles[obs_idx].set_center(obs_pos)
                    # icp_circles[obs_idx].set_radius(enhanced_radius)
                    icp_circles[obs_idx].set_visible(False)
                    
                    # Update text showing ICP enhancement value (show original value, not scaled)
                    # Comment out text display to avoid cluttering trajectory view
                    # icp_texts[obs_idx].set_position((obs_pos[0], obs_pos[1] + enhanced_radius + 0.1))
                    # icp_texts[obs_idx].set_text(f'ICP:{individual_radius:.4f}')  # 增加精度显示
                    icp_texts[obs_idx].set_visible(False)
                    
                    # Debug: Check if ICP value is changing
                    if num % 5 == 0 and obs_idx < 2:  # Check first 2 obstacles every 5 frames
                        history_len = len(history) if len(history) > 0 else 0
                        if history_len > 0:
                            recent_values = history[-3:] if history_len >= 3 else history
                            print(f"🎬 Frame {num}: Obs{obs_idx} current={individual_radius:.4f}, recent={[f'{v:.4f}' for v in recent_values]}, len={history_len}")
                        else:
                            print(f"🎬 Frame {num}: Obs{obs_idx} NO HISTORY, using fallback={individual_radius:.4f}")
                    
                    # Debug radius update - show all significant changes
                    if abs(enhanced_radius - old_radius) > 0.005:  # 降低检测阈值
                        direction = "📈" if enhanced_radius > old_radius else "📉"
                        print(f"{direction} Frame {num}: Obs{obs_idx} {old_radius:.3f}→{enhanced_radius:.3f} (ICP={individual_radius:.3f})")
                    
                    # Debug: Print radius values every 20 frames
                    if num % 20 == 0:
                        print(f"🎬 Frame {num}: Obs{obs_idx} base={env.obstacles[obs_idx].radius:.2f} + ICP={individual_radius:.3f} = total={enhanced_radius:.3f}")
                    
                except Exception:
                    # Reset to original radius if error occurs
                    obstacle_patches[obs_idx].set_radius(env.obstacles[obs_idx].radius)
                    icp_circles[obs_idx].set_visible(False)
                    icp_texts[obs_idx].set_visible(False)
            else:
                # Reset to original radius if no ICP module or out of observation range
                obstacle_patches[obs_idx].set_radius(env.obstacles[obs_idx].radius)
                obstacle_patches[obs_idx].set_facecolor(env.obstacles[obs_idx].color)  # Reset color too
                icp_circles[obs_idx].set_visible(False)
                icp_texts[obs_idx].set_visible(False)

        for i in range(len(env.robots)):
            best_j = robot_best_obs[i]
            if best_j != -1:
                obstacle_patches[best_j].set_facecolor('red')
                obstacle_patches[best_j].set_alpha(0.3)
                # obstacle_texts[best_j].set_color('red')
                
            xdata = np.arange(num)
            h_vals = h_values_list[i][:num]
            lines_h[i].set_data(xdata, h_vals)
            beta_vals = beta_values_list[i][:num]
            lines_beta[i].set_data(xdata, beta_vals)
            margin = safety_margin_list[i][:num]
            lines_margin[i].set_data(xdata, margin)

            if controllers[i].type == "cvar" or controllers[i].type == "adap_cvarbf":
                if controllers[i].htype in ["dist_cone"]:
                    beta_u = np.full(num, 0.5) 
                    lines_beta_u[i].set_data(xdata, beta_u)
                
                elif controllers[i].htype in ["dist"]:
                    beta_u = np.full(num, 0.5) 
                    lines_beta_u[i].set_data(xdata, beta_u)


            u1 = control_inputs_list[i][0, :num]
            u2 = control_inputs_list[i][1, :num]
            lines_u1[i].set_data(xdata, u1)
            lines_u2[i].set_data(xdata, u2)


        ax_h.set_xlim(0, max_num)
        ax_margin.set_xlim(0, max_num)
        ax_u.set_xlim(0, max_num)
        ax_beta.set_xlim(0, max_num)
        ax_beta.legend()   

        
        # ICP circles and texts are now per obstacle
            
        return (lines_h + lines_margin + lines_u1 + lines_u2
                + lines_beta + lines_beta_u + lines_beta_u_bar + scatters + circles + circles_original
                + obstacle_patches + obs_patches_original
                + lines_nom_u1 + lines_nom_u2 + obstacle_texts
                + lines_cost + icp_circles + icp_texts + [observation_range_circle, direction_dot])
        
    num_frames = min([s.shape[1] for s in state_histories])


    ani = FuncAnimation(
        fig,
        update_frame,
        frames=num_frames ,
        init_func=init_func,
        blit=False,
        interval=50,
        repeat=False
    )

    if save_ani and filename:
        filename = filename + ".gif"
        ani.save(filename, writer='pillow', fps=25)
        png_filename = filename.replace('.gif', '_lastframe.png')
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')  
        # png_filename = filename.replace('.gif', '_lastframe.jpg')
        # plt.savefig(png_filename, dpi=300)  

    # plt.close(fig)
    plt.show()



