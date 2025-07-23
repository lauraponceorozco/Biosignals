import sys
import os
import time
import json
import numpy as np
from neurapy.robot import Robot


def draw_circle(robot, radius=0.05, center_pose=None, num_points=8):
    """
    Draw a circle using the robot's move_circular function.
    
    Args:
        robot: Robot instance
        radius: Radius of the circle in meters (default: 0.05m = 5cm)
        center_pose: Center position and orientation of the circle [x, y, z, roll, pitch, yaw]
        num_points: Number of points to use for the circular motion (minimum 3)
    """
    # Use current pose if center_pose not provided
    if center_pose is None:
        center_pose = robot.get_tcp_pose_quaternion()
        center_pose = robot.convert_quaternion_to_euler_pose(center_pose)

    # Ensure we have at least 3 points for circular motion
    num_points = max(3, num_points)
    
    # Generate points around the circle
    angles = np.linspace(0, 2*np.pi, num_points)
    
    # Create three points for the circular motion
    # We'll use the first three points to define the circle
    points = []
    for i in range(3):
        angle = angles[i]
        x = center_pose[0] + radius * np.cos(angle)
        y = center_pose[1] + radius * np.sin(angle)
        z = center_pose[2]  # Keep Z constant
        
        # Keep the same orientation as the center pose
        point = [
            float(x),  # Convert numpy float to Python float
            float(y),
            float(z),
            float(center_pose[3]),  # Roll
            float(center_pose[4]),  # Pitch
            float(center_pose[5])   # Yaw
        ]
        points.append(point)

    # Configure circular motion properties with more conservative values
    circular_properties = {
        "speed": 0.1,  # Reduced speed
        "acceleration": 0.05,  # Reduced acceleration
        "jerk": 50,  # Reduced jerk
        "rotation_speed": 0.5,  # Reduced rotation speed
        "rotation_acceleration": 1.0,  # Reduced rotation acceleration
        "rotation_jerk": 50,  # Reduced rotation jerk
        "blending_mode": 1,  # DYNAMIC_BLENDING
        "target_pose": points,
        "current_joint_angles": robot.get_current_joint_angles(),
        "weaving": False,
        "controller_parameters": {
            "control_mode": "position",
            "force_vector": {
                "Fx": {"is_active": False, "value": 0},
                "Fy": {"is_active": False, "value": 0},
                "Fz": {"is_active": False, "value": 0}
            },
            "torque_vector": {
                "Mx": {"is_active": False, "value": 0},
                "My": {"is_active": False, "value": 0},
                "Mz": {"is_active": False, "value": 0}
            }
        }
    }

    try:
        # First verify that we can reach the target poses
        for point in points:
            try:
                robot.compute_inverse_kinematics(
                    target_pose=point,
                    reference_joint=robot.get_current_joint_angles()
                )
            except Exception as e:
                print(f"Warning: Cannot reach point {point}: {e}")
                return False

        # If all points are reachable, execute the motion
        robot.move_circular(**circular_properties)
        return True
    except Exception as e:
        print(f"Error during circular motion: {e}")
        return False


if __name__ == "__main__":
    try:
        # Initialize robot and set to automatic mode
        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        time.sleep(1)        

        # Define start pose (quaternion format)
        start_pose = [1.5964657875281139, -0.21498875564747447, 0.04914206323847121, 
                     0.7993526697786512, -0.006428787647667216, 0.9474057842594262, -0.4232397553489405]

        # Draw circle using the start pose as center
        success = draw_circle(robot, radius=0.05, center_pose=None, num_points=8)
        if not success:
            print("Failed to draw circle - please check robot configuration and workspace")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        robot.stop()
