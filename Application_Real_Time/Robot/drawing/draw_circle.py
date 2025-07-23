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

# =================================================================
# CONFIGURATION CLASSES
# =================================================================

class ForceControlConfig:
    """Configuration class for force control parameters"""
    def __init__(self):
        self.force_threshold = 1.0  # [V]
        self.analog_channel = 1  # input channel for analog sensor
        self.approach_height = 0.02  # [m] - 2cm above target
        self.contact_step_size = 0.001  # [m] - 1mm steps
        self.contact_speed = 2  # [units] - slow speed for contact detection
        self.max_contact_attempts = 50  # maximum steps to find contact
        self.contact_timeout = 5.0  # [s] - max time for contact detection
        self.emergency_force_limit = 10.0  # [N] - emergency stop threshold
    
    def update_from_user_input(self):
        """Interactive configuration"""
        print("\n=== Force Control Configuration ===")
        
        # Force threshold
        try:
            threshold = input(f"Force threshold [{self.force_threshold}N]: ").strip()
            if threshold:
                self.force_threshold = float(threshold)
        except ValueError:
            print("Invalid input, using default")
        
        # Approach height
        try:
            height = input(f"Approach height [{self.approach_height*1000:.0f}mm]: ").strip()
            if height:
                self.approach_height = float(height) / 1000.0  # Convert mm to m
        except ValueError:
            print("Invalid input, using default")
        
        # Contact step size
        try:
            step = input(f"Contact step size [{self.contact_step_size*1000:.1f}mm]: ").strip()
            if step:
                self.contact_step_size = float(step) / 1000.0  # Convert mm to m
        except ValueError:
            print("Invalid input, using default")
        
        print(f"Configuration: {self.force_threshold}N threshold, {self.approach_height*1000:.0f}mm approach")

class TrajectoryRecorder:
    """Records end-effector positions during robot motion"""
    
    def __init__(self, robot):
        self.robot = robot
        self.positions = []
        self.timestamps = []
        self.force_data = []
        self.recording = False
        self.recording_thread = None
        self.logger = circle_logger.getChild("trajectory_recorder")
    
    def start_recording(self, monitor_forces=False):
        """Start recording end-effector positions"""
        self.positions = []
        self.timestamps = []
        self.force_data = []
        self.recording = True
        self.monitor_forces = monitor_forces
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
                current_time = time.time() - start_time
                self.timestamps.append(current_time)
                
                # Record forces if enabled
                if self.monitor_forces:
                    try:
                        torques = self.robot.get_current_joint_torques()
                        self.force_data.append((current_time, torques))
                    except:
                        pass  # Continue without force data if unavailable
                
                time.sleep(0.05)  # 20 Hz recording rate
            except Exception as e:
                self.logger.error(f"Error recording position: {e}")
                break
    
    def get_trajectory_data(self):
        """Get recorded trajectory data"""
        return np.array(self.positions), np.array(self.timestamps)
    
    def get_force_data(self):
        """Get recorded force data"""
        return self.force_data
    
    def save_data(self):
        """Save recorded data to files"""
        data_dir = Path("NeuraPy/data/trajectory_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Save trajectory data
        if len(self.positions) > 0:
            trajectory_file = data_dir / f"trajectory_{timestamp}.json"
            trajectory_data = {
                'positions': [pos.tolist() for pos in self.positions],
                'timestamps': self.timestamps
            }
            with open(trajectory_file, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            self.logger.info(f"Trajectory data saved to: {trajectory_file}")
        
        # Save force data
        if len(self.force_data) > 0:
            force_file = data_dir / f"forces_{timestamp}.json"
            with open(force_file, 'w') as f:
                json.dump(self.force_data, f, indent=2)
            self.logger.info(f"Force data saved to: {force_file}")

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

# =================================================================
# FORCE CONTROL FUNCTIONS
# =================================================================

def calibrate_force_sensor(robot):
    """Calibrate the force/torque sensor before drawing"""
    try:
        robot.tare_fts()
        circle_logger.info("Force sensor calibrated successfully")
        return True
    except Exception as e:
        circle_logger.error(f"Force sensor calibration failed: {e}")
        return False

def detect_contact(robot, analog_channel=0, voltage_threshold=1.0):
    """
    Detect contact based on analog input from a pressure sensor or FSR.

    Args:
        robot: The robot instance.
        analog_channel (int): Analog input channel (usually 0 or 1).
        voltage_threshold (float): Voltage threshold indicating contact (e.g., 1.0V).
    
    Returns:
        contact_detected (bool), analog_value (float)
    """
    try:
        analog_value = robot.get_analog_input(analog_channel)
        contact_detected = analog_value >= voltage_threshold

        if contact_detected:
            circle_logger.debug(f"Contact detected! Analog input: {analog_value:.3f}V")
        
        return contact_detected, analog_value
    except Exception as e:
        circle_logger.error(f"Error reading analog input: {e}")
        return False, None

def emergency_force_stop(robot, max_force=10.0):
    """Emergency stop if forces exceed safe limits"""
    try:
        torques = robot.get_current_joint_torques()
        max_torque = max(abs(t) for t in torques)
        
        if max_torque > max_force:
            circle_logger.error(f"EMERGENCY STOP: Force {max_torque:.1f}N exceeds limit {max_force}N")
            robot.stop()
            return True
        return False
    except Exception as e:
        circle_logger.error(f"Error in emergency force check: {e}")
        return False

def move_to_contact(robot, target_pose, config):
    """Move along -Y axis until contact is detected (via analog voltage)"""
    circle_logger.info("Moving along -Y axis until contact is detected...")

    current_pose = robot.get_tcp_pose()
    original_y = current_pose[1]
    start_time = time.time()

    for attempt in range(config.max_contact_attempts):
        if emergency_force_stop(robot, config.emergency_force_limit):
            circle_logger.error("Emergency stop triggered")
            return False, None

        if time.time() - start_time > config.contact_timeout:
            circle_logger.warning("Contact detection timeout")
            return False, None

        contact, val = detect_contact(robot, analog_channel=config.analog_channel,
                                      voltage_threshold=config.force_threshold)
        if contact:
            final_pose = robot.get_tcp_pose()
            distance_moved = original_y - final_pose[1]
            circle_logger.info(f"Contact detected at attempt {attempt+1}, moved {distance_moved*1000:.1f}mm in Y")
            return True, final_pose

        current_pose[1] -= config.contact_step_size  # Move -Y

        try:
            robot.set_go_till_forces_mode_for_next_spline()
            robot.move_linear(current_pose, speed=config.contact_speed, acceleration=config.contact_speed)
            time.sleep(0.1)
        except Exception as e:
            circle_logger.error(f"Error during Y-contact movement: {e}")
            return False, None

    circle_logger.warning("Max attempts reached without contact detection")
    return False, None


def force_controlled_approach(robot, target_pose, config):
    """Complete force-controlled approach sequence"""
    circle_logger.info("Starting force-controlled approach...")
    
    # Step 1: Calibrate force sensor (marker should be in air)
    if not calibrate_force_sensor(robot):
        circle_logger.warning("Force calibration failed, continuing without force control")
        return False, target_pose
    
    # Step 2: Move to position above target (add safety margin)
    approach_pose = target_pose.copy()
    approach_pose[2] += config.approach_height  # Default 2cm above target
    
    try:
        robot.move_linear(approach_pose, speed=6, acceleration=6)
        circle_logger.info(f"Moved to approach position, {config.approach_height*1000:.0f}mm above target")
    except Exception as e:
        circle_logger.error(f"Failed to move to approach position: {e}")
        return False, target_pose
    
    # Step 3: Detect contact with board
    contact_detected, contact_pose = move_to_contact(robot, target_pose, config)
    if not contact_detected:
        circle_logger.error("Could not detect contact with board")
        return False, target_pose
    
    # Update target pose to detected contact level
    adjusted_pose = target_pose.copy()
    adjusted_pose[2] = contact_pose[2]  # Use detected Z level
    
    circle_logger.info(f"Adjusted target pose to contact level: Z = {adjusted_pose[2]:.4f}m")
    return True, adjusted_pose

def auto_calibrate_force_threshold(robot, num_samples=5):
    """Automatically determine optimal force threshold"""
    circle_logger.info("Auto-calibrating force threshold...")
    
    print(f"Please touch and release the marker on the board {num_samples} times")
    contact_forces = []
    air_forces = []
    
    for i in range(num_samples):
        input(f"Sample {i+1}: Touch marker to board and press Enter...")
        if not calibrate_force_sensor(robot):
            print("Warning: Force calibration failed for this sample")
            continue
            
        time.sleep(0.5)
        
        # Sample forces in contact
        contact_torques = []
        for _ in range(10):  # 1 second of sampling
            _, torques = detect_contact(robot, threshold=0.1)  # Very low threshold
            if torques:
                contact_torques.extend([abs(t) for t in torques])
            time.sleep(0.1)
        
        if contact_torques:
            contact_forces.append(max(contact_torques))
        
        input("Lift marker and press Enter...")
        time.sleep(0.5)
        
        # Sample forces in air
        air_torques = []
        for _ in range(10):
            _, torques = detect_contact(robot, threshold=0.1)
            if torques:
                air_torques.extend([abs(t) for t in torques])
            time.sleep(0.1)
        
        if air_torques:
            air_forces.append(max(air_torques))
    
    if contact_forces and air_forces:
        avg_contact = np.mean(contact_forces)
        avg_air = np.mean(air_forces)
        
        # Set threshold to 30% above average air force, but below average contact
        optimal_threshold = avg_air + (avg_contact - avg_air) * 0.3
        
        circle_logger.info(f"Auto-calibration results:")
        circle_logger.info(f"  Average air force: {avg_air:.2f}N")
        circle_logger.info(f"  Average contact force: {avg_contact:.2f}N")
        circle_logger.info(f"  Recommended threshold: {optimal_threshold:.2f}N")
        
        return optimal_threshold
    
    circle_logger.warning("Auto-calibration failed, using default 2.0N")
    return 2.0

def test_force_detection(robot, config, test_positions=3):
    """Test force detection at multiple board positions"""
    circle_logger.info("Starting force detection test...")
    
    for i in range(test_positions):
        print(f"\nTest {i+1}/{test_positions}")
        input("Position marker above board and press Enter...")
        
        # Test with different thresholds
        thresholds = [1.0, 2.0, 3.0, 4.0]
        
        for threshold in thresholds:
            print(f"Testing threshold: {threshold}N")
            calibrate_force_sensor(robot)
            
            contact, torques = detect_contact(robot, threshold)
            print(f"  Contact detected: {contact}")
            if torques:
                print(f"  Torques: {[round(t, 3) for t in torques]}")
            
            time.sleep(1)

def analyze_circle_quality(positions, center_pose, radius):
    """Analyze the quality of the drawn circle"""
    if len(positions) < 5:
        return None
    
    positions = np.array(positions)
    
    # Calculate actual center
    actual_center = np.mean(positions, axis=0)
    
    # Calculate distances from actual center
    distances = [np.linalg.norm(pos - actual_center) for pos in positions]
    
    # Calculate statistics
    stats = {
        'target_radius': radius,
        'actual_radius_mean': np.mean(distances),
        'actual_radius_std': np.std(distances),
        'radius_error': abs(np.mean(distances) - radius),
        'circularity': 1 - (np.std(distances) / np.mean(distances)),  # 1 = perfect circle
        'center_offset': np.linalg.norm(actual_center - center_pose[:3]),
        'closure_error': np.linalg.norm(positions[-1] - positions[0])
    }
    
    circle_logger.info(f"Circle quality analysis:")
    circle_logger.info(f"  Target radius: {stats['target_radius']:.3f}m")
    circle_logger.info(f"  Actual radius: {stats['actual_radius_mean']:.3f}±{stats['actual_radius_std']:.3f}m")
    circle_logger.info(f"  Radius error: {stats['radius_error']:.3f}m")
    circle_logger.info(f"  Circularity: {stats['circularity']:.3f} (1.0 = perfect)")
    circle_logger.info(f"  Center offset: {stats['center_offset']:.3f}m")
    circle_logger.info(f"  Closure error: {stats['closure_error']:.3f}m")
    
    return stats

# =================================================================
# ENHANCED DRAWING FUNCTIONS
# =================================================================

def draw_circle_moveJoint(robot, radius, center_pose, num_points=12, speed=6, acceleration=5):
    """Draw circle using joint trajectory approach to avoid IK errors"""
    
    circle_logger.info(f"Drawing circle with radius: {radius:.3f} m")
    
    # Pre-validate all points first
    if not validate_circle_points(robot, radius, center_pose, num_points):
        circle_logger.error(f"Circle validation failed for radius {radius:.3f}m")
        return False
    
    # If validation passes, build and execute trajectory
    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    joint_trajectory = []
    reference_joint = robot.get_current_joint_angles()
    
    # Compute joint angles for all circle points (already validated)
    for i, angle in enumerate(angles):
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]
        target_pose = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
        
        try:
            joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
            joint_trajectory.append(joint_angles)
            reference_joint = joint_angles
        except Exception as e:
            # This shouldn't happen since we pre-validated, but handle just in case
            circle_logger.error(f"Unexpected IK error at point {i+1}: {e}")
            return False
    
    if len(joint_trajectory) < 3:
        circle_logger.error("Not enough reachable points")
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
        circle_logger.info("Circle completed successfully!")
        return True
        
    except Exception as e:
        circle_logger.error(f"Error during trajectory: {e}")
        return False


def draw_circle_trajectory(robot, radius, center_pose, num_points=12, speed=6, acceleration=5):
    """Draw circle using joint trajectory approach to avoid IK errors"""
    
    circle_logger.info(f"Drawing circle with radius: {radius:.3f} m")
    
    # Pre-validate all points first
    if not validate_circle_points(robot, radius, center_pose, num_points):
        circle_logger.error(f"Circle validation failed for radius {radius:.3f}m")
        return False
    
    # If validation passes, build and execute trajectory
    angles = np.linspace(0, 2*np.pi, max(8, num_points) + 1)[:-1]
    joint_trajectory = []
    reference_joint = robot.get_current_joint_angles()
    
    # Compute joint angles for all circle points (already validated)
    for i, angle in enumerate(angles):
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]
        target_pose = [x, y, z, center_pose[3], center_pose[4], center_pose[5]]
        
        try:
            joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
            joint_trajectory.append(joint_angles)
            reference_joint = joint_angles
        except Exception as e:
            # This shouldn't happen since we pre-validated, but handle just in case
            circle_logger.error(f"Unexpected IK error at point {i+1}: {e}")
            return False
    
    if len(joint_trajectory) < 3:
        circle_logger.error("Not enough reachable points")
        return False
    
    try:
        # Execute trajectory using move_joint with correct parameter structure
        joint_property = {
            "timestamps": [0.1 * i for i in range(len(joint_trajectory))],
            "target_joint": joint_trajectory,
            "current_joint_angles": robot.get_current_joint_angles()
        }
        
        robot.move_trajectory(**joint_property)
        circle_logger.info("Circle completed successfully!")
        return True
        
    except Exception as e:
        circle_logger.error(f"Error during trajectory: {e}")
        return False

def draw_circle_trajectory_lifting(robot, radius, center_pose, num_points=12, speed=6, acceleration=5, enable_visualization=False):
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



def draw_circle_trajectory_with_force_control(robot, radius, center_pose, config, num_points=12, speed=60, acceleration=5, enable_visualization=True):
    """Draw circle with force control and optional visualization"""
    
    circle_logger.info(f"Drawing circle with force control - radius: {radius:.3f} m")
    
    # Step 1: Force-controlled approach to find contact level
    contact_success, adjusted_center_pose = force_controlled_approach(
        robot, center_pose, config
    )
    
    if not contact_success:
        circle_logger.warning("Force control failed, using original pose")
        adjusted_center_pose = center_pose
    
    # Step 2: Pre-validate all points at the detected/original level
    if not validate_circle_points(robot, radius, adjusted_center_pose, num_points):
        circle_logger.error(f"Circle validation failed for radius {radius:.3f}m at contact level")
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
            x = adjusted_center_pose[0] + radius * np.cos(angle)
            z = adjusted_center_pose[2] + radius * np.sin(angle)
            y = adjusted_center_pose[1]
            target_pose = [x, y, z, adjusted_center_pose[3], adjusted_center_pose[4], adjusted_center_pose[5]]
            
            try:
                joint_angles = robot.compute_inverse_kinematics(target_pose, reference_joint)
                joint_trajectory.append(joint_angles)
                reference_joint = joint_angles
            except Exception as e:
                circle_logger.error(f"Unexpected IK error at point {i+1}: {e}")
                if recorder:
                    recorder.stop_recording()
                return False
        
        # Enable force mode for drawing
        robot.set_go_till_forces_mode_for_next_spline()
        
        # Execute trajectory
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
        
        circle_logger.info("Force-controlled circle completed successfully!")
        return True
        
    except Exception as e:
        if recorder:
            recorder.stop_recording()
        circle_logger.error(f"Error during force-controlled trajectory: {e}")
        return False

def draw_circle_trajectory_with_visualization(robot, radius, center_pose, num_points=12, speed=60, acceleration=5):
    """Legacy function - now redirects to force control version with default config"""
    default_config = ForceControlConfig()
    return draw_circle_trajectory_with_force_control(
        robot, radius, center_pose, default_config, num_points, speed, acceleration, enable_visualization=True
    )

# =================================================================
# USER INTERFACE FUNCTIONS
# =================================================================

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

def ask_force_control_mode():
    """Ask user if they want force control"""
    while True:
        choice = input("Enable force control for contact detection? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def ask_force_threshold(default=2.0):
    """Ask user for force threshold"""
    while True:
        try:
            threshold = input(f"Enter force threshold (Newtons, default {default}): ").strip()
            if threshold == "":
                return default
            threshold = float(threshold)
            if 0.5 <= threshold <= 10.0:
                return threshold
            else:
                print("Please enter a value between 0.5 and 10.0")
        except ValueError:
            print("Please enter a valid number")

def main():
    robot = None
    try:
        print("=== Enhanced Force-Controlled Circle Drawing ===")
        circle_logger.info("Starting session with configurable force control")

        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        time.sleep(1)
        # robot.reset_fault()
        circle_logger.info("Robot initialized and ready.")

        safe_joint_angles = [
            1.7261193717088836, -0.545651122616592, -0.004781705737237272,
            1.1377811889934681, -0.06824117266732665, 0.9290174891702697,
            -0.06739890636069706
        ]

        positions = load_tictactoe_positions("NeuraPy/data/tictactoe_positions_new.json")
        if not positions:
            circle_logger.error("No positions found.")
            return

        # Check and move to safe position
        if not is_in_safe_position(robot, safe_joint_angles):
            move_to_safe_position(robot, safe_joint_angles, speed=6)

        enable_visualization = ask_visualization_mode()
        enable_force_control = ask_force_control_mode()
        config = ForceControlConfig()

        if enable_force_control:
            use_auto_threshold = input("Use auto-calibration for force threshold? (y/n): ").strip().lower()
            if use_auto_threshold in ["y", "yes"]:
                config.force_threshold = auto_calibrate_force_threshold(robot)
            else:
                config.force_threshold = ask_force_threshold(config.force_threshold)
            config.update_from_user_input()

        radii_try = [0.08, 0.06, 0.04, 0.036, 0.03]

        while True:
            selected_key = select_position()
            if selected_key == "quit":
                print("Exiting...")
                break

            selected_joint_angles = positions[selected_key]["joint_angles"]
            robot.move_joint(selected_joint_angles, speed=6)
            time.sleep(0.5)
            selected_pose = robot.get_tcp_pose()
            print(f"Selected: {selected_key}, Pose: {np.round(selected_pose, 3).tolist()}")

            success = False
            for radius in radii_try:
                if enable_force_control:
                    success = draw_circle_trajectory_with_force_control(
                        robot, radius, selected_pose, config, enable_visualization=enable_visualization
                    )
                else:
                    if enable_visualization:
                        success = draw_circle_trajectory_with_visualization(
                            robot, radius, selected_pose
                        )
                    else:
                        # success = draw_circle_trajectory(
                        #     robot, radius, selected_pose
                        # )
                        success = draw_circle_trajectory_lifting(
                            robot, radius, selected_pose
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
                        draw_circle_moveJoint(
                            robot, best_radius, selected_pose
                        )

            # Return to safe position
            move_to_safe_position(robot, safe_joint_angles, speed=6)

    except Exception as e:
        circle_logger.error(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        # if robot:
        #     try:
        #         robot.reset_fault()
        #     except:
        #         pass
    finally:
        if robot:
            try:
                if not is_in_safe_position(robot, safe_joint_angles):
                    move_to_safe_position(robot, safe_joint_angles, speed=3)
                robot.stop()
            except:
                pass

if __name__ == "__main__":
    main()
