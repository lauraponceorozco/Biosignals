import sys
import os
import time
import json
import numpy as np
from neurapy.robot import Robot


def draw_cross(robot, cross_size=0.2, center_pose=None, steps_per_line=5):
    # Use current pose if center_pose not provided
    if center_pose is None:
        center_pose = robot.get_tcp_pose_quaternion()
        center_pose = robot.convert_quaternion_to_euler_pose(center_pose)

    # Define cross points in local XY for continuous drawing
    half = cross_size / 2
    # Points ordered for continuous drawing: left -> center -> right -> center -> top -> center -> bottom
    cross_points = [
        [-half, 0],  # Left point
        [0, 0],      # Center
        [half, 0],   # Right point
        [0, 0],      # Center
        [0, half],   # Top point
        [0, 0],      # Center
        [0, -half]   # Bottom point
    ]

    # Draw the continuous line
    for i in range(len(cross_points) - 1):
        start = cross_points[i]
        end = cross_points[i + 1]
        
        for step in range(steps_per_line + 1):
            alpha = step / steps_per_line  # interpolation parameter for step size
            x = center_pose[0] + (1 - alpha) * start[0] + alpha * end[0]
            y = center_pose[1] + (1 - alpha) * start[1] + alpha * end[1]
            z = center_pose[2]  # Keep Z fixed
            
            target_pose = [
                x,
                y,
                z,
                center_pose[3],       # Roll
                center_pose[4],       # Pitch
                center_pose[5]        # Yaw
            ]

            try:
                reference_joint = robot.get_current_joint_angles()  # reference joint angles
                joint_angles = robot.compute_inverse_kinematics(
                    target_pose=target_pose,
                    reference_joint=reference_joint,
                )
                robot.move_joint(joint_angles)
            except Exception as e:
                print(f"IK error at step {step} on segment {i}: {e}")

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

        # Draw cross on the whiteboard using the start pose as center
        draw_cross(robot, cross_size=0.2, center_pose=None, steps_per_line=10)
        
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        robot.stop()
