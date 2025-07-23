# authors: Christian Ritter, Phillip Wagner, Esther Utasá

import socket
import time
import json
import sys
import os
import numpy as np
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

from neurapy.robot import neurapy_logger


def parse_field_command(data_str):
    """Parse fieldId from JSON message and return field number"""
    try:
        data = json.loads(data_str)
        field_id = data.get("fieldId")
        if field_id is not None:
            return int(field_id)
        else:
            print(f"Warning: No fieldId found in message: {data_str}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None
    except ValueError as e:
        print(f"Error converting fieldId to int: {e}")
        return None


# Import drawing functions
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
    # Precomputation functions
    precompute_circle_trajectories,
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
    draw_circle_with_precomputed_trajectory,
    draw_tictactoe_grid,
    ask_grid_drawing_mode,
    # Utility functions
    ask_visualization_mode,
    ask_force_control_mode,
    ask_force_threshold
)


circle_logger = setup_drawing_logger("drawer", "drawing")

# Initialize robot
robot = Robot()
robot.switch_to_automatic_mode()
robot.enable_collision_detection()
circle_logger.info("Robot initialized and ready.")
time.sleep(1)
# robot.reset_fault()

safe_joint_angles = [
    1.7261193717088836, -0.545651122616592, -0.004781705737237272,
    1.1377811889934681, -0.06824117266732665, 0.9290174891702697,
    -0.06739890636069706
]


# Load positions and setup
positions = load_tictactoe_positions(default_config.positions_file)
if not positions:
    circle_logger.error("No positions found.")
    exit(1)

# Initialize grid separator positions
grid_positions = load_grid_positions(default_config.grid_positions_file)
if not grid_positions:
    circle_logger.warning("No grid positions found. Grid drawing will be disabled.")
    grid_positions = {}

# Move to / Check safe position
safe_joint_angles = default_config.safe_joint_angles
if not is_in_safe_position(robot, safe_joint_angles):
    move_to_safe_position(robot, safe_joint_angles, speed=default_config.circle_safe_movement_speed)

# Default configuration (no user interaction for UDP mode)
enable_visualization = False
enable_force_control = False
config = ForceControlConfig()
radii_try = [0.04, 0.036, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005]
field_to_position = { # Field ID to position mapping
    1: "upper_left", 2: "upper_center", 3: "upper_right",
    4: "middle_left", 5: "middle_center", 6: "middle_right", 
    7: "lower_left", 8: "lower_center", 9: "lower_right"
}

# Initialize timing statistics
timing_stats = {
    "total_operations": 0,
    "successful_operations": 0,
    "failed_operations": 0,
    "position_times": {},
    "average_times": {},
    "total_time": 0.0
}

def get_field_id():
    while True:
        user_input = input("Enter a field ID (1–9): ")
        if user_input in [str(i) for i in range(1, 10)]:
            print(f"You selected field ID: {user_input}")
            return int(user_input)
        else:
            #print("Invalid input. Please enter a number between 1 and 9.")
            break

def log_timing_stats(position_key, duration, success, radius_used=None):
    """Log timing statistics for circle drawing operations"""
    if position_key not in timing_stats["position_times"]:
        timing_stats["position_times"][position_key] = {
            "times": [],
            "successful_times": [],
            "failed_times": [],
            "radii_used": []
        }
    
    timing_stats["position_times"][position_key]["times"].append(duration)
    timing_stats["position_times"][position_key]["radii_used"].append(radius_used)
    
    if success:
        timing_stats["position_times"][position_key]["successful_times"].append(duration)
        timing_stats["successful_operations"] += 1
    else:
        timing_stats["position_times"][position_key]["failed_times"].append(duration)
        timing_stats["failed_operations"] += 1
    
    timing_stats["total_operations"] += 1
    timing_stats["total_time"] += duration
    
    # Calculate and log statistics
    times = timing_stats["position_times"][position_key]["times"]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    circle_logger.info(f"Position {position_key} - Duration: {duration:.2f}s, "
                      f"Success: {success}, Radius: {radius_used}, "
                      f"Avg: {avg_time:.2f}s, Min: {min_time:.2f}s, Max: {max_time:.2f}s")
    
    print(f"⏱️  Position {position_key}: {duration:.2f}s ({'SUCCESS' if success else 'FAILED'}) "
          f"| Radius: {radius_used} | Avg: {avg_time:.2f}s")

def print_timing_summary():
    """Print summary of all timing statistics"""
    print("\n" + "="*60)
    print("📊 CIRCLE DRAWING TIMING SUMMARY")
    print("="*60)
    
    for position_key, stats in timing_stats["position_times"].items():
        if stats["times"]:
            avg_time = sum(stats["times"]) / len(stats["times"])
            success_rate = len(stats["successful_times"]) / len(stats["times"]) * 100
            print(f"📍 {position_key}:")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Total attempts: {len(stats['times'])}")
            print(f"   Successful: {len(stats['successful_times'])}")
            print(f"   Failed: {len(stats['failed_times'])}")
            print()
    
    if timing_stats["total_operations"] > 0:
        overall_avg = timing_stats["total_time"] / timing_stats["total_operations"]
        overall_success_rate = timing_stats["successful_operations"] / timing_stats["total_operations"] * 100
        print(f"📈 OVERALL STATISTICS:")
        print(f"   Total operations: {timing_stats['total_operations']}")
        print(f"   Successful: {timing_stats['successful_operations']}")
        print(f"   Failed: {timing_stats['failed_operations']}")
        print(f"   Success rate: {overall_success_rate:.1f}%")
        print(f"   Average time: {overall_avg:.2f}s")
        print(f"   Total time: {timing_stats['total_time']:.2f}s")
    print("="*60)


while True:
    try:
        # 

        # Ask for position
        field_id = get_field_id()
        print("selected field ID: ", field_id)
        if field_id < 1 or field_id > 9:
            print("Exiting...")
            break
        
        if field_id is not None :
            print(f"Processing field ID: {field_id}")
            
            # Map field ID to position key
            position_key = field_to_position.get(field_id)
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

            if position_key and position_key in positions:
                selected_joint_angles = positions[position_key]["joint_angles"]

                # Start timing for this operation
                operation_start_time = time.time()
                circle_drawing_success = False
                radius_used = None

                # Precompute trajectories for all radii in parallel during movement
                circle_logger.debug(f"Starting precomputation for {len(radii)} radii during movement")
                
                # Start precomputation in a separate thread
                import threading
                precomputation_complete = threading.Event()
                precomputed_data = {}
                
                def precompute_during_movement():
                    try:
                        # Wait for robot to reach position
                        # time.sleep(0.5)  # Give robot time to start moving
                        
                        # Precompute trajectories for all radii
                        main_precomputed = precompute_circle_trajectories(
                            robot, selected_pose, radii, selected_position_key=position_key
                        )
                        
                        # Also precompute escape radii
                        escape_precomputed = precompute_circle_trajectories(
                            robot, selected_pose, escape_radii, selected_position_key=position_key
                        )
                        
                        # Update the shared variable
                        precomputed_data.update(main_precomputed)
                        precomputed_data.update(escape_precomputed)
                        
                        circle_logger.info(f"Precomputation completed: {sum(1 for r in precomputed_data.values() if r['valid'])} valid trajectories")
                        precomputation_complete.set()
                    except Exception as e:
                        circle_logger.error(f"Precomputation failed: {e}")
                        precomputation_complete.set()
                
                # Start precomputation thread
                precompute_thread = threading.Thread(target=precompute_during_movement)
                precompute_thread.daemon = True
                precompute_thread.start()
                
                # Move to position with pen lifting
                circle_logger.debug(f"Moving to selected position with pen lifting: {position_key}")
                success_center_offset = move_to_position_with_pen_lifting(robot, selected_joint_angles, logger=circle_logger)

                if not success_center_offset:
                    circle_logger.warning("Pen lifting movement failed, trying direct movement")
                    robot.move_joint(selected_joint_angles, speed=default_config.circle_safe_movement_speed)

                # Get current pose
                selected_pose = robot.get_tcp_pose()
                circle_logger.debug(f"Selected: {position_key}, Pose: {np.round(selected_pose, 3).tolist()}")
                
                # Wait for precomputation to complete (with timeout)
                precomputation_complete.wait(timeout=5.0)
                
                if not precomputation_complete.is_set():
                    circle_logger.warning("Precomputation timeout, falling back to original method")
                    precomputed_data = {}

                # Try different radii using precomputed trajectories
                success = False
                for radius in radii:
                    if radius in precomputed_data and precomputed_data[radius]['valid']:
                        # Use precomputed trajectory
                        if enable_force_control:
                            success = draw_circle_trajectory_with_force_control(
                                robot, radius, selected_pose, config, enable_visualization=enable_visualization
                            )
                        else:
                            success = draw_circle_with_precomputed_trajectory(
                                robot, radius, selected_pose, precomputed_data, speed=100, enable_visualization=enable_visualization
                            )
                    else:
                        # Fallback to original method
                        if enable_force_control:
                            success = draw_circle_trajectory_with_force_control(
                                robot, radius, selected_pose, config, enable_visualization=enable_visualization
                            )
                        else:
                            success = draw_circle_trajectory_lifting(
                                robot, radius, selected_pose, selected_position_key=position_key, speed=100
                            )
                    
                    if success:
                        radius_used = radius
                        circle_drawing_success = True
                        break

                if not success:
                    circle_logger.warning("Main radii failed, trying escape radii...")
                    for radius in escape_radii:
                        if radius in precomputed_data and precomputed_data[radius]['valid']:
                            # Use precomputed trajectory
                            if enable_force_control:
                                success = draw_circle_trajectory_with_force_control(
                                    robot, radius, selected_pose, config, enable_visualization=enable_visualization
                                )
                            else:
                                success = draw_circle_with_precomputed_trajectory(
                                    robot, radius, selected_pose, precomputed_data, enable_visualization=enable_visualization
                                )
                        else:
                            # Fallback to original method
                            if enable_force_control:
                                success = draw_circle_trajectory_with_force_control(
                                    robot, radius, selected_pose, config, enable_visualization=enable_visualization
                                )
                            else:
                                success = draw_circle_trajectory_lifting(
                                    robot, radius, selected_pose, selected_position_key=position_key
                                )
                        
                        if success:
                            radius_used = radius
                            circle_drawing_success = True
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
                                robot, best_radius, selected_pose, selected_position_key=position_key
                            )
                        radius_used = best_radius
                        circle_drawing_success = True

                # Return to safe position
                move_to_safe_position(robot, safe_joint_angles, speed=default_config.circle_safe_movement_speed)

                # Calculate total operation time
                operation_duration = time.time() - operation_start_time
                
                # Log timing statistics
                log_timing_stats(position_key, operation_duration, circle_drawing_success, radius_used)

                print(f"Robot action completed for field {field_id}")
            else:
                print(f"Position not found for field {field_id}")
        else:
            print(f"Invalid field ID: {field_id}")
        

        
    except KeyboardInterrupt:
        print("\nShutting down...")
        print_timing_summary()
        break
    except Exception as e:
        circle_logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        if robot:
            try:
                robot.reset_fault()
            except:
                pass

# Cleanup
if robot:
    try:
        if not is_in_safe_position(robot, safe_joint_angles):
            move_to_safe_position(robot, safe_joint_angles, speed=3)
        robot.stop()
    except:
        pass

# Print final timing summary
print_timing_summary()