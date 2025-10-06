
import numpy as np

def log_error(message):
    print(f"\033[31m[ERROR] {message}\033[0m")  

def log_debug(message):
    print(f"\033[34m[DEBUG] {message}\033[0m")  

def log_info(message):
    print(f"\033[32m[INFO] {message}\033[0m")  
    

def generate_crowd_positions(range_low, range_high, number, radii, mode="y_upper_lower", existing_points=None, existing_radii=None):
    if mode == "y_upper_lower":
        range_low_initial = [range_low[0], (range_low[1] + range_high[1]) / 2]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [range_high[0], (range_low[1] + range_high[1]) / 2]

    elif mode == "x_left_right":
        range_low_initial = [range_low[0], range_low[1]]
        range_high_initial = [(range_low[0] + range_high[0]) / 2, range_high[1]]
        range_low_target = [(range_low[0] + range_high[0]) / 2, range_low[1]]
        range_high_target = [range_high[0], range_high[1]]

    elif mode == "xy_left_up_right_down":
        range_low_initial = [range_low[0], (range_low[1] + range_high[1]) / 2]
        range_high_initial = [(range_low[0] + range_high[0]) / 2, range_high[1]]
        range_low_target = [(range_low[0] + range_high[0]) / 2, range_low[1]]
        range_high_target = [range_high[0], (range_low[1] + range_high[1]) / 2]

    elif mode == "xy_right_up_left_down":
        range_low_initial = [(range_low[0] + range_high[0]) / 2, (range_low[1] + range_high[1]) / 2]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [(range_low[0] + range_high[0]) / 2, (range_low[1] + range_high[1]) / 2]

    elif mode == "whole_space":
        range_low_initial = [range_low[0], range_low[1]]
        range_high_initial = [range_high[0], range_high[1]]
        range_low_target = [range_low[0], range_low[1]]
        range_high_target = [range_high[0], range_high[1]]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Generate initial positions
    initial_positions = generate_points_with_radius(
        range_low_initial, range_high_initial, number, radii, existing_points, existing_radii
    )

    # Combine existing points and radii with newly generated initial positions
    combined_points = (
        np.vstack((existing_points, initial_positions)) if existing_points is not None else initial_positions
    )
    combined_radii = (
        np.vstack((existing_radii, radii * np.ones((len(initial_positions), 1)))) if existing_radii is not None else radii * np.ones((len(initial_positions), 1))
    )

    # Generate target positions
    target_positions = generate_points_with_radius(
        range_low_target, range_high_target, number, radii, combined_points, combined_radii
    )

    return initial_positions, target_positions


def generate_points_with_radius(range_low, range_high, number, radii, existing_points=None, existing_radii=None):
    points = []  # Initialize list to store new points
    for _ in range(number):
        while True:
            # Generate a random point within the specified range
            point = np.random.uniform(range_low[:2], range_high[:2])

            # Check distance from this point to all existing points
            valid = True
            if existing_points is not None and existing_radii is not None:
                for i, existing_point in enumerate(existing_points):
                    existing_radius = existing_radii[i, 0]  # Extract radius as scalar
                    distance = np.linalg.norm(point - existing_point)
                    if distance < (radii + existing_radius):  # Sum of radii
                        valid = False
                        break

            # If the point is valid, add it to the list and break the loop
            if valid:
                points.append(point)
                break

    return np.array(points)


