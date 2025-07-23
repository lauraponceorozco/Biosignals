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

# Add NeuraPy to path for running from root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
neuraPy_dir = os.path.join(current_dir, "..")
sys.path.insert(0, neuraPy_dir)

# Platform-specific imports
if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

# Local imports
from neurapy.robot import neurapy_logger
from config import default_config #! Global config for drawing
from drawing.utils import (
    # General functions
    load_tictactoe_positions,
    load_grid_positions,
    select_position,
    validate_circle_points,
    wait_for_motion_completion,
    is_in_safe_position,
    move_to_safe_position,
    find_best_radius,
    pen_lift,
    compute_circle_point,
    move_to_position_with_pen_lifting,
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
    analyze_circle_quality,
    # Logging setup
    setup_drawing_logger
)
from drawing.drawing_functions import (
    # Advanced drawing functions
    draw_circle_trajectory,
    draw_circle_trajectory_lifting,
    draw_circle_trajectory_with_force_control,
    draw_tictactoe_grid,
    ask_grid_drawing_mode,
    # Utility functions
    ask_visualization_mode,
    ask_force_control_mode,
    ask_force_threshold
)

circle_logger = setup_drawing_logger("drawer", "drawing")

def main():
    robot = None
    try:
        # Initialize robot
        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        circle_logger.info("Robot initialized and ready.")
        time.sleep(1)

        # update tool position -> is NOT considered in the inverse kinematics!
        #robot.update_current_tool_parameters(offsetZ=0.1285)
        #robot.update_current_tool_parameters(offsetZ=0.5) #testing

        # Initialize grid positions
        positions = load_tictactoe_positions(default_config.positions_file)
        if not positions:
            circle_logger.error("No positions found.")
            return

        # Initialize grid separator positions
        grid_positions = load_grid_positions(default_config.grid_positions_file)
        if not grid_positions:
            circle_logger.warning("No grid positions found. Grid drawing will be disabled.")
            grid_positions = {}

        # Move to / Check safe position
        safe_joint_angles = default_config.safe_joint_angles
        if not is_in_safe_position(robot, safe_joint_angles):
            move_to_safe_position(robot, safe_joint_angles, speed=default_config.circle_safe_movement_speed)

        # Prompt drawing modes
        enable_visualization = ask_visualization_mode()
        enable_force_control = ask_force_control_mode()
        # enable_grid_drawing = ask_grid_drawing_mode() if grid_positions else False
        enable_grid_drawing = False # too dangerous

        # Initialize force control config
        config = ForceControlConfig()
        if enable_force_control:
            use_auto_threshold = input("Use auto-calibration for force threshold? (y/n): ").strip().lower()
            if use_auto_threshold in ["y", "yes"]:
                config.force_threshold = auto_calibrate_force_threshold(robot)
            else:
                config.force_threshold = ask_force_threshold(config.force_threshold)
            config.update_from_user_input()

        # Draw grid if requested
        if enable_grid_drawing:
            circle_logger.info("Drawing tic-tac-toe grid...")
            grid_success = draw_tictactoe_grid(robot, grid_positions)
            if grid_success:
                circle_logger.info("Grid drawing completed successfully!")
                print("Grid drawing completed successfully!")
            else:
                circle_logger.warning("Grid drawing failed!")
                print("Grid drawing failed!")
            
            # Return to safe position after grid drawing
            move_to_safe_position(robot, safe_joint_angles, speed=default_config.circle_safe_movement_speed)

        # Main drawing loop
        while True:

            # Ask for position
            selected_key = select_position()
            if selected_key == "quit":
                print("Exiting...")
                break
            selected_joint_angles = positions[selected_key]["joint_angles"]

            # Move to position with pen lifting
            circle_logger.debug(f"Moving to selected position with pen lifting: {selected_key}")
            success_center_offset = move_to_position_with_pen_lifting(robot, selected_joint_angles, logger=circle_logger)

            if not success_center_offset:
                circle_logger.warning("Pen lifting movement failed, trying direct movement")
                robot.move_joint(selected_joint_angles, speed=default_config.circle_safe_movement_speed)

            # Get current pose
            selected_pose = robot.get_tcp_pose()
            #selected_pose[3] += -0.1 # test with different orientation
            circle_logger.debug(f"Selected: {selected_key}, Pose: {np.round(selected_pose, 3).tolist()}")

            # Create [radius_lower_limit, radius_upper_limit] around the selected radius
            radii_try = default_config.circle_drawing_radii
            radius = radii_try[0]
            radius_lower_limit = radius - 0.001
            radius_upper_limit = radius + 0.001
            radii = np.linspace(radius_lower_limit, radius_upper_limit, 25)
            circle_logger.debug(f"Radii to try: {radii}")
            print(f"Radii to try: {radii}")

            # Define escape radii and sort them by distance to target radius
            escape_radii_base = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
            escape_radii = sorted(escape_radii_base, key=lambda x: abs(x - radius))

            # Try different radii
            success = False
            for radius in radii:
                if enable_force_control:
                    success = draw_circle_trajectory_with_force_control(
                        robot, radius, selected_pose, config, enable_visualization=enable_visualization
                    )
                else:
                    success = draw_circle_trajectory_lifting(
                        robot, radius, selected_pose, selected_position_key=selected_key, speed=90
                    )
                if success:
                    break

            if not success:
                circle_logger.warning("Main radii failed, trying escape radii...")
                for radius in escape_radii:
                    if enable_force_control:
                        success = draw_circle_trajectory_with_force_control(
                            robot, radius, selected_pose, config, enable_visualization=enable_visualization
                        )
                    else:
                        success = draw_circle_trajectory_lifting(
                            robot, radius, selected_pose, selected_position_key=selected_key
                        )
                    if success:
                        break

            if not success:
                circle_logger.warning("Trying adaptive radius...")
                best_radius = find_best_radius(robot, selected_pose, max_radius=0.04, min_radius=0.01)
                if best_radius:
                    if enable_force_control:
                        draw_circle_trajectory_with_force_control(
                            robot, best_radius, selected_pose, config, enable_visualization=enable_visualization
                        )
                    else:
                        draw_circle_trajectory_lifting(
                            robot, best_radius, selected_pose, selected_position_key=selected_key
                        )

            # Return to safe position
            move_to_safe_position(robot, safe_joint_angles, speed=default_config.circle_safe_movement_speed)

    except Exception as e:
        circle_logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if robot:
            try:
                if not is_in_safe_position(robot, safe_joint_angles):
                    move_to_safe_position(robot, safe_joint_angles, speed=3)
                robot.stop()
            except Exception:
                pass

if __name__ == "__main__":
    main()
