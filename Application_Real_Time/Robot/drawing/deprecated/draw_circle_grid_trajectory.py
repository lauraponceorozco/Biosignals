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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

from neurapy.robot import neurapy_logger

# Import utility functions
from utils import (
    load_tictactoe_positions,
    select_position,
    validate_circle_points,
    wait_for_motion_completion,
    is_in_safe_position,
    move_to_safe_position,
    find_best_radius
)

# Logging
circle_logger = neurapy_logger.getChild("circle_drawing")
logs_dir = Path("NeuraPy/data/logs")
logs_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"circle_drawing_{timestamp}.log"
log_filepath = logs_dir / log_filename
file_handler = logging.FileHandler(log_filepath)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s] : %(message)s :(%(filename)s:%(lineno)d)",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
circle_logger.addHandler(file_handler)
circle_logger.info(f"Logging to file: {log_filepath}")

class TrajectoryRecorder:
    """Records end-effector positions during robot motion"""
    
    def __init__(self, robot):
        self.robot = robot
        self.positions = []
        self.timestamps = []
        self.recording = False
        self.recording_thread = None
        self.logger = circle_logger.getChild("trajectory_recorder")
    
    def start_recording(self):
        """Start recording end-effector positions"""
        self.positions = []
        self.timestamps = []
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        self.logger.info("Started recording end-effector trajectory...")
    
    def stop_recording(self):
        """Stop recording end-effector positions"""
        self.recording = False
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        self.logger.info(f"Stopped recording. Captured {len(self.positions)} positions.")
    
    def _record_loop(self):
        """Background thread that continuously records positions"""
        start_time = time.time()
        while self.recording:
            try:
                pose = self.robot.get_tcp_pose()
                self.positions.append(pose[:3])  # Only x, y, z
                self.timestamps.append(time.time() - start_time)
                time.sleep(0.05)  # 20 Hz recording rate
            except Exception as e:
                self.logger.error(f"Error recording position: {e}")
                break
    
    def get_trajectory_data(self):
        """Get recorded trajectory data"""
        return np.array(self.positions), np.array(self.timestamps)

def visualize_trajectory(positions, center_pose, radius, title="End-Effector Trajectory"):
    """Visualize the recorded end-effector trajectory"""
    
    if len(positions) < 3:
        circle_logger.warning("Not enough trajectory data to visualize")
        return
    
    positions = np.array(positions)
    
    # Create figure with 3D plot
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='o', label='End')
    ax1.scatter(center_pose[0], center_pose[1], center_pose[2], c='orange', s=100, marker='*', label='Center')
    
    # Draw ideal circle for comparison
    angles = np.linspace(0, 2*np.pi, 100)
    ideal_x = center_pose[0] + radius * np.cos(angles)
    ideal_y = center_pose[1] * np.ones_like(angles)  # Y should be constant
    ideal_z = center_pose[2] + radius * np.sin(angles)
    ax1.plot(ideal_x, ideal_y, ideal_z, 'r--', linewidth=1, alpha=0.7, label='Ideal Circle')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'{title} - 3D View')
    ax1.legend()
    ax1.grid(True)
    
    # XY projection
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Actual Trajectory')
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='o', label='End')
    ax2.scatter(center_pose[0], center_pose[1], c='orange', s=100, marker='*', label='Center')
    ax2.plot(ideal_x, ideal_y, 'r--', linewidth=1, alpha=0.7, label='Ideal Circle')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # XZ projection (the circle plane)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, label='Actual Trajectory')
    ax3.scatter(positions[0, 0], positions[0, 2], c='g', s=100, marker='o', label='Start')
    ax3.scatter(positions[-1, 0], positions[-1, 2], c='r', s=100, marker='o', label='End')
    ax3.scatter(center_pose[0], center_pose[2], c='orange', s=100, marker='*', label='Center')
    ax3.plot(ideal_x, ideal_z, 'r--', linewidth=1, alpha=0.7, label='Ideal Circle')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('XZ Projection (Circle Plane)')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')
    
    # Trajectory statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics
    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    ideal_circumference = 2 * np.pi * radius
    start_end_distance = np.linalg.norm(positions[-1] - positions[0])
    
    # Calculate deviation from ideal circle
    deviations = []
    for pos in positions:
        # Project point onto ideal circle plane
        ideal_pos = np.array([pos[0], center_pose[1], pos[2]])
        center_to_point = ideal_pos - np.array(center_pose[:3])
        distance_from_center = np.linalg.norm(center_to_point)
        deviation = abs(distance_from_center - radius)
        deviations.append(deviation)
    
    max_deviation = np.max(deviations)
    avg_deviation = np.mean(deviations)
    
    stats_text = f"""
    Trajectory Statistics:
    =====================
    Target Radius: {radius:.3f} m
    Total Distance: {total_distance:.3f} m
    Ideal Circumference: {ideal_circumference:.3f} m
    Start-End Distance: {start_end_distance:.3f} m
    Max Deviation: {max_deviation:.3f} m
    Avg Deviation: {avg_deviation:.3f} m
    Circle Closure: {'Good' if start_end_distance < 0.01 else 'Poor'}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Create directory for saving plots
    viz_dir = Path("NeuraPy/data/drawing_visualization")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"circle_trajectory_{timestamp}.png"
    filepath = viz_dir / filename
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    circle_logger.info(f"Trajectory visualization saved to: {filepath}")
    
    # Show the plot
    # plt.show()

def draw_circle_trajectory(robot, radius, center_pose, num_points=12, speed=6, acceleration=5, enable_visualization=False):
    """Draw circle using joint trajectory approach to avoid IK errors, with optional visualization."""
    
    circle_logger.info(f"Drawing circle with radius: {radius:.3f} m" + (" (with visualization)" if enable_visualization else ""))
    pen_lift_offset = 0.01  # [m]
    
    # Pre-validate all points first
    if not validate_circle_points(robot, radius, center_pose, num_points):
        circle_logger.error(f"Circle validation failed for radius {radius:.3f}m")
        return False
    
    # Calculate first point of the circle
    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    first_angle = angles[0]
    first_x = center_pose[0] + radius * np.cos(first_angle)
    first_z = center_pose[2] + radius * np.sin(first_angle)
    first_y = center_pose[1]
    first_pose = [first_x, first_y, first_z, center_pose[3], center_pose[4], center_pose[5]]
    first_pose_lifted = [first_x, first_y - pen_lift_offset, first_z, center_pose[3], center_pose[4], center_pose[5]]

    # Move (pen up) to first circle point, but with Y offset using move_joint
    first_joint_lifted = robot.compute_inverse_kinematics(first_pose_lifted, robot.get_current_joint_angles())
    robot.move_joint(first_joint_lifted, speed=speed)
    # Pen down (move forward in +Y at first point)
    robot.move_linear_relative(
        cartesian_offset=[0.0, pen_lift_offset + 0.0075, 0.0, 0.0, 0.0, 0.0], 
        speed=speed, acceleration=acceleration)

    # If visualization is enabled, set up recorder
    recorder = None
    if enable_visualization:
        recorder = TrajectoryRecorder(robot)
        recorder.start_recording()

    # Build and execute trajectory
    # Start from the second point, since we are already at the first point after pen down
    joint_trajectory = []
    reference_joint = robot.get_current_joint_angles()
    for i, angle in enumerate(angles[1:]):  # Start from second point
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]
        target_pose = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
        try:
            joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
            joint_trajectory.append(joint_angles)
            reference_joint = joint_angles
            circle_logger.debug(f"Point {i+2} (angle {angle:.1f}°) IK computed successfully")
        except Exception as e:
            circle_logger.error(f"Unexpected IK error at point {i+2} (angle {angle:.1f}°): {e}")
            circle_logger.error(f"Target pose: {[round(x, 3) for x in target_pose]}")
            if recorder:
                recorder.stop_recording()
            return False
    if len(joint_trajectory) < 2:
        circle_logger.error("Not enough reachable points")
        if recorder:
            recorder.stop_recording()
        return False
    try:
        joint_property = {
            "speed": speed,
            "acceleration": acceleration,
            "safety_toggle": True,
            "target_joint": joint_trajectory,
            "current_joint_angles": robot.get_current_joint_angles()
        }
        robot.move_joint(**joint_property)
        # Pen up (move back from the board)
        robot.move_linear_relative(cartesian_offset=[0.0, -pen_lift_offset, 0.0, 0.0, 0.0, 0.0], 
                                   speed=speed, acceleration=acceleration)
        if enable_visualization:
            recorder.stop_recording()
            positions, timestamps = recorder.get_trajectory_data()
            if len(positions) > 0:
                visualize_trajectory(positions, center_pose, radius, 
                                    f"Circle Drawing - Radius {radius:.3f}m")
        circle_logger.info("Circle completed successfully!")
        return True
    except Exception as e:
        if recorder:
            recorder.stop_recording()
        circle_logger.error(f"Error during trajectory: {e}")
        return False

def ask_visualization_mode():
    """Ask user if they want visualization"""
    while True:
        choice = input("\nEnable trajectory visualization? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    robot = None
    
    try:
        print("=== Circle Drawing with Trajectory Approach ===")
        circle_logger.info("Starting circle drawing session")
        print("Initializing robot...")
        
        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        time.sleep(1)
        circle_logger.info("Robot initialized successfully.")

        try:
            robot.reset_fault()
            circle_logger.info("Reset any fault state.")
        except Exception:
            pass

        # Joint angles for neutral/safe starting position
        safe_joint_angles = [
            1.7261193717088836, -0.545651122616592, -0.004781705737237272,
            1.1377811889934681, -0.06824117266732665, 0.9290174891702697,
            -0.06739890636069706
        ]

        # Load tic-tac-toe positions
        json_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "tictactoe_positions.json")
        )
        circle_logger.info(f"Loading positions from: {json_path}")
        positions = load_tictactoe_positions(json_path)

        if not positions:
            circle_logger.error("No positions loaded. Exiting.")
            sys.exit(1)

        grid_2_safe_speed = 9
        grid_2_safe_acceleration = 9
        grid_2_safe_time = 0.5 # [s]

        circle_points = 16

        # Check if robot is in safe position, if not move there
        if not is_in_safe_position(robot, safe_joint_angles):
            circle_logger.info("Robot not in safe position. Moving to safe position first...")
            if not move_to_safe_position(robot, safe_joint_angles, speed=grid_2_safe_speed):
                circle_logger.error("Failed to move to safe position. Exiting.")
                sys.exit(grid_2_safe_time)
        else:
            circle_logger.info("Robot already in safe position.")

        print("\nStarting interactive circle drawing session...")
        
        # Ask for visualization mode
        enable_visualization = ask_visualization_mode()
        
        while True:
            selected_key = select_position()
            if selected_key == "quit":
                circle_logger.info("User requested to quit")
                print("Exiting.")
                break

            # Speed and acceleration parameters
            safe_2_grid_speed = 9
            safe_2_grid_acceleration = 9
            safe_2_grid_time = 0 # [s]
            
            circle_speed = 50
            circle_acceleration = 6

            # 0.036 works for all except 1, 5, 7, 8
            # radii_old = [0.02, 0.018, 0.015] # [m]
            radii_new = [0.036, 0.03, 0.025, 0.02, 0.018, 0.015] # [m]

            circle_logger.info(f"Moving to joint position of selected cell: {selected_key}")
            selected_joint_angles = positions[selected_key]["joint_angles"]
            robot.move_joint(selected_joint_angles, speed=safe_2_grid_speed)
            time.sleep(safe_2_grid_time)

            # TODO: Pen up (lift off after moving to cell position)
            pen_lift_offset = 0.01  # [m]
            robot.move_linear_relative(
                cartesian_offset=[0.0, -pen_lift_offset, 0.0, 0.0, 0.0, 0.0], 
                speed=safe_2_grid_speed, acceleration=safe_2_grid_acceleration)

            # Get actual TCP pose from the robot
            selected_pose = robot.get_tcp_pose()
            circle_logger.info(f"Current TCP pose: {np.round(selected_pose, 3).tolist()}")
            print(f"Selected position: {selected_key.replace('_', ' ').title()}")

            # Try drawing circle with adaptive radius finding
            success = False
            
            # Method 1: Try predefined radii first (faster)
            for radius in radii_new:
                circle_logger.info(f"Trying radius: {radius:.3f}m")
                success = draw_circle_trajectory(robot, radius, selected_pose, 
                                                  speed=circle_speed, acceleration=circle_acceleration, enable_visualization=enable_visualization)
                if success:
                    circle_logger.info(f"Successfully drew circle with radius: {radius:.3f}m")
                    break
                else:
                    circle_logger.warning(f"Failed to draw circle with radius: {radius:.3f}m")
            
            # Method 2: If predefined radii fail, find best radius automatically
            if not success:
                circle_logger.info("Predefined radii failed, finding best radius automatically...")
                best_radius = find_best_radius(robot, selected_pose, max_radius=0.06, min_radius=0.01)
                if best_radius:
                    success = draw_circle_trajectory(robot, best_radius, selected_pose,
                                                   speed=circle_speed, acceleration=circle_acceleration, enable_visualization=enable_visualization)
            
            if not success:
                circle_logger.error("All radius attempts failed")
            
            # Always return to safe position after drawing (successful or not)
            circle_logger.info("Returning to safe position...")
            if not move_to_safe_position(robot, safe_joint_angles, speed=6):
                circle_logger.warning("Failed to return to safe position")

    except TimeoutError as e:
        circle_logger.error(f"Connection timeout error: {e}")
        print("Please check if the robot is powered on and connected to the network.")
    except ConnectionError as e:
        circle_logger.error(f"Connection error: {e}")
        print("Please check the robot connection and network settings.")
    except Exception as e:
        circle_logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            try:
                robot.reset_fault()
                circle_logger.info("Fault reset executed after failure.")
            except:
                circle_logger.error("Could not reset fault state.")
    finally:
        try:
            if robot is not None:
                # Final safety check - try to move to safe position before stopping
                try:
                    if not is_in_safe_position(robot, safe_joint_angles):
                        circle_logger.info("Moving to safe position before stopping...")
                        move_to_safe_position(robot, safe_joint_angles, speed=3)
                except:
                    pass
                robot.stop()
                circle_logger.info("Robot stopped safely.")
        except:
            pass 