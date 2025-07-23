import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import datetime
from pathlib import Path
import logging

# Platform-specific imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

# Local imports
from neurapy.robot import neurapy_logger
from config import default_config
from .utils import (
    # General functions
    validate_circle_points,
    compute_circle_point,
    pen_lift,
    # Force control functions
    ForceControlConfig,
    TrajectoryRecorder,
    visualize_trajectory,
    calibrate_force_sensor,
    detect_contact,
    emergency_force_stop,
    move_to_contact,
    force_controlled_approach,
    auto_calibrate_force_threshold,
    test_force_detection,
    analyze_circle_quality
)

# Import the centralized logging setup
from .utils import setup_drawing_logger

# Drawing functions logging
drawing_logger = setup_drawing_logger("drawing_functions", "drawing")

# =================================================================
# BASIC DRAWING FUNCTIONS
# =================================================================

def draw_circle_xz(robot, radius, center_pose, num_points=12):
    """Basic circle drawing in XZ plane"""
    drawing_logger.info(f"Drawing basic circle with radius: {radius:.3f} m")
    print(f"\nTrying radius: {radius:.3f} m")

    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    points = []
    for angle in angles:
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]
        point = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
        points.append(point)

    reference_joint = robot.get_current_joint_angles()
    for i, point in enumerate(points):
        try:
            robot.compute_inverse_kinematics(point, reference_joint)
        except Exception as e:
            drawing_logger.error(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f}): {e}")
            print(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f})")
            return False

    try:
        drawing_logger.info("Executing circular motion...")
        print("Executing circular motion...")
        robot.move_circular(
            speed=0.05,
            acceleration=0.02,
            jerk=30,
            rotation_speed=0.25,
            rotation_acceleration=0.5,
            rotation_jerk=30,
            blending_mode=1,
            target_pose=points,
            current_joint_angles=reference_joint,
            weaving=False,
            controller_parameters={
                "control_mode": "position",
                "force_vector": {k: {"is_active": False, "value": 0} for k in ["Fx", "Fy", "Fz"]},
                "torque_vector": {k: {"is_active": False, "value": 0} for k in ["Mx", "My", "Mz"]}
            }
        )
        drawing_logger.info("Circle completed successfully!")
        print("Circle completed successfully.")
        return True
    except Exception as e:
        drawing_logger.error(f"Error during circular motion: {e}")
        print(f"Error during circular motion: {e}")
        return False


def draw_circle_xz_manual(robot, radius, center_pose, num_points=12):
    """Manual circle drawing in XZ plane with individual joint movements"""
    drawing_logger.info(f"Drawing manual circle with radius: {radius:.3f} m")
    print(f"\nTrying radius: {radius:.3f} m")

    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    points = []
    for angle in angles:
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]
        point = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
        points.append(point)

    reference_joint = robot.get_current_joint_angles()
    for i, point in enumerate(points):
        try:
            robot.compute_inverse_kinematics(point, reference_joint)
        except Exception as e:
            drawing_logger.error(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f}): {e}")
            print(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f})")
            return False

    try:
        drawing_logger.info("Executing manual circular motion...")
        print("Executing circular motion...")
        for point in points:
            joint_ref = robot.get_current_joint_angles()
            point_robot = robot.compute_inverse_kinematics(point, joint_ref)
            print(point_robot)
            robot.move_joint(point_robot, speed=5)
        
        time.sleep(0.1)
        drawing_logger.info("Manual circle completed successfully!")
        print("Circle completed successfully.")
        return True
    except Exception as e:
        drawing_logger.error(f"Error during manual circular motion: {e}")
        print(f"Error during circular motion: {e}")
        return False


def draw_square(robot, square_size=0.2, center_pose=None, steps_per_side=10):
    """
    Draw a square of given size in the XZ plane (like on a wall), centered at center_pose, with Y fixed.

    Parameters:
    - robot: instance of neurapy.robot.Robot
    - square_size: length of the side of the square in meters
    - center_pose: list [X, Y, Z, R, P, Yaw] or None
    - steps_per_side: number of segments per side

    Returns:
    - None
    """
    drawing_logger.info(f"Drawing square with size: {square_size:.3f} m")
    
    if center_pose is None:
        center_pose = robot.get_tcp_pose_quaternion()
        center_pose = robot.convert_quaternion_to_euler_pose(center_pose)

    half = square_size / 2
    # Define corners in XZ plane, Y fixed
    square_corners = [
        [center_pose[0] + half, center_pose[2] + half],
        [center_pose[0] - half, center_pose[2] + half],
        [center_pose[0] - half, center_pose[2] - half],
        [center_pose[0] + half, center_pose[2] - half],
        [center_pose[0] + half, center_pose[2] + half],  # close the square
    ]

    y = center_pose[1]
    r, p, yw = center_pose[3], center_pose[4], center_pose[5]

    for i in range(4):
        start = square_corners[i]
        end = square_corners[i+1]
        for step in range(steps_per_side + 1):
            alpha = step / steps_per_side
            x = (1 - alpha) * start[0] + alpha * end[0]
            z = (1 - alpha) * start[1] + alpha * end[1]
            target_pose = [x, y, z, r, p, yw]
            try:
                reference_joint = robot.get_current_joint_angles()
                joint_angles = robot.compute_inverse_kinematics(
                    target_pose=target_pose,
                    reference_joint=reference_joint,
                )
                robot.move_joint(joint_angles)
            except Exception as e:
                drawing_logger.error(f"IK error at step {step} on side {i}: {e}")
                print(f"IK error at step {step} on side {i}: {e}")

# =================================================================
# ADVANCED DRAWING FUNCTIONS (moved from drawer.py)
# =================================================================

def draw_circle_trajectory(robot, radius, center_pose, num_points=12, speed=80, acceleration=5):
    """Advanced circle drawing with trajectory planning"""
    drawing_logger.info(f"Drawing circle with radius: {radius:.3f} m")
    
    # Pre-validate all points first
    if not validate_circle_points(robot, radius, center_pose, num_points):
        drawing_logger.error(f"Circle validation failed for radius {radius:.3f}m")
        return False
    
    # If validation passes, build and execute trajectory
    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    joint_trajectory = []
    reference_joint = robot.get_current_joint_angles()
    
    # Compute joint angles for all circle points (already validated)
    for i, angle in enumerate(angles):
        target_pose = compute_circle_point(center_pose, radius, angle, pen_lift_offset=0.0, scaling_factor=1.0)
        
        try:
            joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
            joint_trajectory.append(joint_angles)
            reference_joint = joint_angles
        except Exception as e:
            # This shouldn't happen since we pre-validated, but handle just in case
            drawing_logger.error(f"Unexpected IK error at point {i+1}: {e}")
            return False
    
    if len(joint_trajectory) < 3:
        drawing_logger.error("Not enough reachable points")
        return False
    
    try:
        # Execute trajectory using move_joint with correct parameter structure
        joint_property = {
            "speed": speed,
            "acceleration": acceleration,
            "safety_toggle": True,
            "target_joint": joint_trajectory,
            "current_joint_angles": robot.get_current_joint_angles()
        }
        
        robot.move_joint(**joint_property)
        drawing_logger.info("Circle completed successfully!")
        return True
        
    except Exception as e:
        drawing_logger.error(f"Error during trajectory: {e}")
        return False


def draw_circle_trajectory_lifting(robot, radius, center_pose, num_points=None, speed=None, acceleration=None, enable_visualization=False, selected_position_key=None):
    """Advanced circle drawing with pen lifting and visualization"""
    
    # TODO: check move in first point offset
    
    drawing_logger.info(f"Drawing circle with radius: {radius:.3f} m" + (" (with visualization)" if enable_visualization else ""))
    if num_points is None:
        num_points = default_config.circle_num_points
    if speed is None:
        speed = default_config.circle_drawing_speed
    if acceleration is None:
        acceleration = default_config.circle_drawing_acceleration
    pen_lift_offset = default_config.circle_pen_lift_offset
    
    # Apply row-wise/column-wise scaling factor
    scaling_factor = 1.0  # manual correction of pen lift offset
    if selected_position_key is ["upper_right"]:
        scaling_factor = 1.05
    elif selected_position_key in ["upper_left", "upper_center"]:
        scaling_factor = 1.15
        drawing_logger.info(f"Applying scaling factor {scaling_factor} for position {selected_position_key}")
    elif selected_position_key in ["lower_left", "lower_right"]:
        scaling_factor = 0.65
    elif selected_position_key in ["lower_center"]:
        scaling_factor = 0.80
        drawing_logger.info(f"Applying scaling factor {scaling_factor} for position {selected_position_key}")
    
    # Pre-validate all points including lifted poses
    if not validate_circle_points(robot, radius, center_pose, num_points, check_lifting=True, pen_lift_offset=pen_lift_offset):
        drawing_logger.error(f"Circle validation with lifting failed for radius {radius:.3f}m")
        return False
    
    num_angles = max(8, num_points)
    angles = np.linspace(0, 2*np.pi, num_angles + 1)[:-1]
    
    # Build complete trajectory before any robot movement
    joint_trajectory = []
    reference_joint = robot.get_current_joint_angles()
    
    # First, compute the lifted pose for initial movement (without pen lift offset)
    first_angle = angles[0]
    first_pose_lifted = compute_circle_point(center_pose, radius, first_angle, pen_lift_offset=0.0, scaling_factor=1.0)
    drawing_logger.debug(f"First point (lifted): {[round(x, 3) for x in first_pose_lifted]}")
    
    try:
        first_joint_lifted = robot.compute_inverse_kinematics(first_pose_lifted, reference_joint)
        joint_trajectory.append(first_joint_lifted)
        drawing_logger.debug("First point (lifted) IK computed successfully")
    except Exception as e:
        drawing_logger.error(f"Failed to compute IK for first lifted point: {e}")
        return False
    
    # Now compute all remaining points with pen down
    for i, angle in enumerate(angles[1:]):  # Start from second point
        target_pose = compute_circle_point(center_pose, radius, angle, pen_lift_offset, scaling_factor)
        try:
            joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
            joint_trajectory.append(joint_angles)
            reference_joint = joint_angles
            drawing_logger.debug(f"Point {i+2} (angle {angle:.1f}°) IK computed successfully")
        except Exception as e:
            drawing_logger.error(f"Unexpected IK error at point {i+2} (angle {angle:.1f}°): {e}")
            drawing_logger.error(f"Target pose: {[round(x, 3) for x in target_pose]}")
            return False
    
    # Complete the circle by connecting back to the first point (with pen down)
    first_angle = angles[0]
    first_pose = compute_circle_point(center_pose, radius, first_angle, pen_lift_offset, scaling_factor)
    
    try:
        first_joint_angles = robot.compute_inverse_kinematics(first_pose, reference_joint)
        joint_trajectory.append(first_joint_angles)
        drawing_logger.debug(f"Circle completion point (angle {first_angle:.1f}°) IK computed successfully")
    except Exception as e:
        drawing_logger.error(f"Unexpected IK error at circle completion point: {e}")
        drawing_logger.error(f"Target pose: {[round(x, 3) for x in first_pose]}")
        return False
    
    if len(joint_trajectory) < 3:
        drawing_logger.error("Not enough reachable points")
        return False
    
    # Now that we have validated the entire trajectory, start robot movement
    try:
        # Move to first circle point with pen lifted
        current_joints = robot.get_current_joint_angles()
        drawing_logger.debug(f"Using reference joints: {[round(x, 3) for x in current_joints]}")
        robot.move_joint(joint_trajectory[0], speed=speed)
        drawing_logger.debug("Moved to first point with pen lifted")
    except Exception as e:
        drawing_logger.error(f"Failed to move to first point: {e}")
        return False
    
    # Pen down
    pen_lift(robot, pen_lift_offset, 
             scaling_factor, speed=1, 
             acceleration=1, to_white_board=True)

    recorder = None
    if enable_visualization:
        recorder = TrajectoryRecorder(robot)
        recorder.start_recording()

    # Execute the remaining trajectory (starting from second point)
    try:
        joint_property = {
            "speed": 100,
            "acceleration": acceleration,
            "safety_toggle": True,
            "target_joint": joint_trajectory[1:],  # Skip first point since we're already there
            "current_joint_angles": robot.get_current_joint_angles()
        }
        robot.move_joint(**joint_property)

        """ 
        # test alternative approach using move_trajectory -> returns kinematic/dynamic error for any timestamps
        traj_property = {
            "current_joint_angles": robot.get_current_joint_angles(),
            "target_joint": joint_trajectory[1:],  # Skip first point since we're already there
            "timestamps": [10 * i for i in range(len(joint_trajectory) - 1)],
        }
        print("Executing trajectory with move_trajectory...")
        robot.move_trajectory(**traj_property)
        """



        if enable_visualization:
            recorder.stop_recording()
            positions, timestamps = recorder.get_trajectory_data()
            if len(positions) > 0:
                visualize_trajectory(positions, center_pose, radius, 
                                    f"Circle Drawing - Radius {radius:.3f}m")
        drawing_logger.info("Circle completed successfully!")
        return True
    except Exception as e:
        if recorder:
            recorder.stop_recording()
        drawing_logger.error(f"Error during trajectory: {e}")
        return False


def draw_circle_trajectory_with_force_control(robot, radius, center_pose, config, num_points=None, speed=None, acceleration=None, enable_visualization=True):
    """Advanced circle drawing with force control and contact detection"""
    
    # Use config values if not provided
    if num_points is None:
        num_points = default_config.circle_num_points
    if speed is None:
        speed = default_config.circle_force_control_speed
    if acceleration is None:
        acceleration = default_config.circle_drawing_acceleration
    
    drawing_logger.info(f"Drawing circle with force control - radius: {radius:.3f} m")
    
    # Step 1: Force-controlled approach to find contact level
    contact_success, adjusted_center_pose = force_controlled_approach(
        robot, center_pose, config
    )
    
    if not contact_success:
        drawing_logger.warning("Force control failed, using original pose")
        adjusted_center_pose = center_pose
    
    # Step 2: Pre-validate all points at the detected/original level
    if not validate_circle_points(robot, radius, adjusted_center_pose, num_points):
        drawing_logger.error(f"Circle validation failed for radius {radius:.3f}m at contact level")
        return False
    
    # Step 3: Initialize trajectory recorder if visualization enabled
    recorder = None
    if enable_visualization:
        recorder = TrajectoryRecorder(robot)
        recorder.start_recording(monitor_forces=True)
    
    try:
        # Build trajectory at contact level
        angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
        joint_trajectory = []
        reference_joint = robot.get_current_joint_angles()
        
        for i, angle in enumerate(angles):
            target_pose = compute_circle_point(adjusted_center_pose, radius, angle, pen_lift_offset=0.0, scaling_factor=1.0)
            
            try:
                joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
                joint_trajectory.append(joint_angles)
                reference_joint = joint_angles
            except Exception as e:
                drawing_logger.error(f"Unexpected IK error at point {i+1}: {e}")
                if recorder:
                    recorder.stop_recording()
                return False
        
        # Enable force mode for drawing
        robot.set_go_till_forces_mode_for_next_spline()
        joint_property = {
            "speed": speed,
            "acceleration": acceleration,
            "safety_toggle": True,
            "target_joint": joint_trajectory,
            "current_joint_angles": robot.get_current_joint_angles()
        }
        
        robot.move_joint(**joint_property)
        
        if recorder:
            recorder.stop_recording()
            positions, timestamps = recorder.get_trajectory_data()
            
            if len(positions) > 0:
                # Analyze circle quality
                quality_stats = analyze_circle_quality(positions, adjusted_center_pose, radius)
                
                # Visualize trajectory
                visualize_trajectory(positions, adjusted_center_pose, radius, 
                                   f"Force-Controlled Circle - Radius {radius:.3f}m")
                
                # Save data
                recorder.save_data()
        
        drawing_logger.info("Force-controlled circle completed successfully!")
        return True
        
    except Exception as e:
        if recorder:
            recorder.stop_recording()
        drawing_logger.error(f"Error during force-controlled trajectory: {e}")
        return False

# =================================================================
# GRID DRAWING FUNCTIONS
# =================================================================

def interpolate_line(start_pose, end_pose, num_points=10):
    """Linear interpolation between start and end pose (including both endpoints)."""
    return [
        [
            start_pose[j] + (end_pose[j] - start_pose[j]) * t / (num_points - 1)
            for j in range(6)
        ]
        for t in range(num_points)
    ]

def find_reachable_pose(robot, original_pose, reference_joint, fallback_strategies):
    """
    Try original_pose and then fallback_strategies. Return the first reachable pose, or None if all fail.
    """
    # Try original pose
    try:
        robot.compute_inverse_kinematics(target_pose=original_pose, reference_joint=reference_joint)
        return original_pose
    except Exception:
        pass
    # Try all fallback strategies
    for j, adjust_func in enumerate(fallback_strategies):
        candidate_pose = adjust_func(original_pose)
        try:
            robot.compute_inverse_kinematics(target_pose=candidate_pose, reference_joint=reference_joint)
            drawing_logger.info(f"Pose adjusted successfully (strategy {j+1})")
            print(f"Pose adjusted successfully (strategy {j+1})")
            return candidate_pose
        except Exception:
            continue
    return None

def draw_tictactoe_grid(robot, grid_positions, speed=None, acceleration=None, num_line_points=10):
    """
    Draw the tic-tac-toe grid lines by moving between grid separator points.
    Validates and applies fallback for each movement step using the robot's current configuration.
    """
    if speed is None:
        speed = default_config.circle_drawing_speed
    if acceleration is None:
        acceleration = default_config.circle_drawing_acceleration
    
    drawing_logger.info("Starting tic-tac-toe grid drawing...")
    print("Drawing tic-tac-toe grid...")
    
    movement_plan = [
        ("vertical_left_top", "start"),
        ("vertical_left_bottom", "line"),
        ("vertical_right_bottom", "transition"),
        ("vertical_right_top", "line"),
        ("horizontal_top_right", "transition"),
        ("horizontal_top_left", "line"),
        ("horizontal_bottom_left", "transition"),
        ("horizontal_bottom_right", "line"),
    ]
    
    missing_positions = []
    for position_name, _ in movement_plan:
        if position_name not in grid_positions:
            missing_positions.append(position_name)
    if missing_positions:
        drawing_logger.error(f"Missing grid positions: {missing_positions}")
        print(f"Error: Missing grid positions: {missing_positions}")
        return False
    
    pen_lift_offset = default_config.circle_pen_lift_offset
    prev_pose = None
    for i, (position_name, action) in enumerate(movement_plan):
        position_data = grid_positions[position_name]
        original_tcp_pose = position_data["tcp_pose"]
        fallback_strategies = []
        # Down the line for start and transitions (finer and more aggressive)
        if (action == "start" or action == "transition") and i + 1 < len(movement_plan):
            next_position_name, _ = movement_plan[i + 1]
            next_pose = grid_positions[next_position_name]["tcp_pose"]
            direction = [next_pose[j] - original_tcp_pose[j] for j in range(6)]
            down_line_adjustments = [
                lambda pose, f=f: [pose[j] + direction[j] * f for j in range(6)]
                for f in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            ]
            fallback_strategies += down_line_adjustments
        # More aggressive and finer Z/X adjustments
        if "vertical" in position_name:
            fallback_strategies += [
                lambda pose, dz=dz: [pose[0], pose[1], pose[2] + dz, pose[3], pose[4], pose[5]]
                for dz in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1]
            ]
        elif "horizontal" in position_name:
            if "right" in position_name:
                fallback_strategies += [
                    lambda pose, dx=dx: [pose[0] + dx, pose[1], pose[2], pose[3], pose[4], pose[5]]
                    for dx in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1]
                ]
            else:
                fallback_strategies += [
                    lambda pose, dx=dx: [pose[0] - dx, pose[1], pose[2], pose[3], pose[4], pose[5]]
                    for dx in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, -0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.09, -0.1]
                ]
        # For transitions, also try moving closer to previous point
        if action == "transition" and prev_pose is not None:
            direction_x = original_tcp_pose[0] - prev_pose[0]
            direction_z = original_tcp_pose[2] - prev_pose[2]
            transition_adjustments = [
                lambda pose, f=f: [pose[0] - direction_x * f, pose[1], pose[2] - direction_z * f, pose[3], pose[4], pose[5]]
                for f in [0.1, 0.2, 0.3, 0.5]
            ]
            fallback_strategies = transition_adjustments + fallback_strategies
        # Use the robot's current configuration as reference
        reference_joint = robot.get_current_joint_angles()
        reachable_pose = find_reachable_pose(robot, original_tcp_pose, reference_joint, fallback_strategies)
        if reachable_pose is None:
            drawing_logger.error(f"Failed to find reachable pose for {position_name} after all adjustments")
            print(f"Error: Failed to find reachable pose for {position_name} after all adjustments")
            return False
        # Execute the movement
        if action == "start":
            robot.move_joint(robot.compute_inverse_kinematics(reachable_pose, reference_joint), speed=speed)
            drawing_logger.debug("Moved to first grid point with pen lifted")
        elif action == "line":
            pen_lift(robot, pen_lift_offset, 0.1, speed=1, acceleration=1, to_white_board=True)
            points = interpolate_line(prev_pose, reachable_pose, num_points=num_line_points)
            try:
                robot.move_linear(target_pose=points, speed=speed, acceleration=acceleration, current_joint_angles=robot.get_current_joint_angles())
                drawing_logger.debug(f"Drew line to {position_name}")
            except Exception as e:
                drawing_logger.warning(f"move_linear failed for line to {position_name}, falling back to move_joint: {e}")
                print(f"move_linear failed for line to {position_name}, falling back to move_joint")
                robot.move_joint(robot.compute_inverse_kinematics(reachable_pose, reference_joint), speed=speed)
        elif action == "transition":
            pen_lift(robot, pen_lift_offset, 1.0, speed=1, acceleration=1, to_white_board=False)
            robot.move_joint(robot.compute_inverse_kinematics(reachable_pose, reference_joint), speed=speed)
            drawing_logger.debug(f"Transitioned (pen up) to {position_name}")
        prev_pose = reachable_pose
    drawing_logger.info("Grid drawing completed successfully!")
    print("Grid drawing completed successfully!")
    return True

# =================================================================
# UTILITY FUNCTIONS FOR DRAWING
# =================================================================

def ask_grid_drawing_mode():
    """Interactive prompt for grid drawing mode"""
    print("\n=== Grid Drawing Mode ===")
    print("1. No grid drawing")
    print("2. Draw tic-tac-toe grid")
    
    while True:
        choice = input("Select grid drawing mode (1-2): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            print("Invalid choice. Please enter 1 or 2.")

def ask_visualization_mode():
    """Interactive prompt for visualization mode"""
    print("\n=== Visualization Mode ===")
    print("1. No visualization")
    print("2. Visualization")
    
    while True:
        choice = input("Select visualization mode (1-3): ").strip()
        if choice == "1":
            return False
        elif choice == "2":
            return True
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def ask_force_control_mode():
    """Interactive prompt for force control mode"""
    print("\n=== Force Control Mode ===")
    print("1. No force control")
    print("2. Force control")
    
    while True:
        choice = input("Select force control mode (1-3): ").strip()
        if choice == "1":
            return None
        elif choice == "2":
            return "basic"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def ask_force_threshold(default=2.0):
    """Interactive prompt for force threshold"""
    print(f"\n=== Force Threshold Configuration ===")
    print(f"Current default: {default}N")
    print("1. Use default")
    print("2. Manual input")
    print("3. Auto-calibration")
    
    while True:
        choice = input("Select threshold method (1-3): ").strip()
        if choice == "1":
            return default
        elif choice == "2":
            try:
                threshold = float(input("Enter force threshold (N): "))
                return threshold
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == "3":
            return "auto"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def draw_circle_with_precomputed_trajectory(robot, radius, center_pose, precomputed_data, speed=None, acceleration=None, enable_visualization=False):
    """Draw circle using precomputed trajectory data for faster execution"""
    
    if speed is None:
        speed = default_config.circle_drawing_speed
    if acceleration is None:
        acceleration = default_config.circle_drawing_acceleration
    
    drawing_logger.info(f"Drawing circle with precomputed trajectory - radius: {radius:.3f} m")
    
    # Check if we have valid precomputed data for this radius
    if radius not in precomputed_data or not precomputed_data[radius]['valid']:
        error_msg = precomputed_data.get(radius, {}).get('error', 'No precomputed data available')
        drawing_logger.error(f"Invalid precomputed trajectory for radius {radius:.3f}m: {error_msg}")
        return False
    
    joint_trajectory = precomputed_data[radius]['joint_trajectory']
    
    if len(joint_trajectory) < 3:
        drawing_logger.error("Precomputed trajectory has insufficient points")
        return False
    
    # Initialize trajectory recorder if visualization enabled
    recorder = None
    if enable_visualization:
        recorder = TrajectoryRecorder(robot)
        recorder.start_recording()
    
    try:
        # Move to first circle point with pen lifted
        current_joints = robot.get_current_joint_angles()
        drawing_logger.debug(f"Using reference joints: {[round(x, 3) for x in current_joints]}")
        robot.move_joint(joint_trajectory[0], speed=speed)
        drawing_logger.debug("Moved to first point with pen lifted")
    except Exception as e:
        drawing_logger.error(f"Failed to move to first point: {e}")
        return False
    
    # Pen down
    pen_lift_offset = default_config.circle_pen_lift_offset
    pen_lift(robot, pen_lift_offset, 1.0, speed=1, acceleration=1, to_white_board=True)
    
    # Execute the remaining trajectory (starting from second point)
    try:
        joint_property = {
            "speed": speed,
            "acceleration": acceleration,
            "safety_toggle": True,
            "target_joint": joint_trajectory[1:],  # Skip first point since we're already there
            "current_joint_angles": robot.get_current_joint_angles()
        }
        robot.move_joint(**joint_property)
        
        if enable_visualization and recorder:
            recorder.stop_recording()
            positions, timestamps = recorder.get_trajectory_data()
            if len(positions) > 0:
                visualize_trajectory(positions, center_pose, radius, 
                                    f"Precomputed Circle - Radius {radius:.3f}m")
        
        drawing_logger.info("Precomputed circle completed successfully!")
        return True
        
    except Exception as e:
        if recorder:
            recorder.stop_recording()
        drawing_logger.error(f"Error during precomputed trajectory: {e}")
        return False

