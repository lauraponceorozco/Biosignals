import sys
import os
import time
import numpy as np

# append parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from neurapy.robot import Robot

def move_to_pose(robot, target_pose):
    """
    Move the robot directly to the target pose.

    Parameters:
    - robot: instance of neurapy.robot.Robot
    - target_pose: list [X, Y, Z, qw, qx, qy, qz] for the target position and orientation in quaternions

    Returns:
    - None
    """
    try:
        # Convert quaternion pose to Euler angles
        position = target_pose[:3]  # [X, Y, Z]
        quaternion = target_pose[3:]  # [qw, qx, qy, qz]
        euler_pose = robot.convert_quaternion_to_euler_pose([*position, *quaternion])
        
        # Use current joint angles as reference
        reference_joint = robot.get_current_joint_angles()
        joint_angles = robot.compute_inverse_kinematics(
            target_pose=euler_pose,
            reference_joint=reference_joint,
        )
        robot.move_joint(joint_angles)
    except Exception as e:
        print(f"Error moving to target pose: {e}")
        raise

if __name__ == "__main__":
    try:
        # Initialize robot and set to automatic mode
        robot = Robot()
        robot.set_mode("Automatic")
        time.sleep(1)  # Give time for mode change to take effect

        # Enable collision detection
        robot.enable_collision_detection()
        
        # Target position with quaternion orientation
        target_pose = [1.5964657875281139, -0.21498875564747447, 0.04914206323847121, 
                      0.7993526697786512, -0.006428787647667216, 0.9474057842594262, -0.4232397553489405]
        print("Moving to target position...")
        move_to_pose(robot, target_pose)
        print("Movement completed!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Ensure robot is stopped
        robot.stop()
