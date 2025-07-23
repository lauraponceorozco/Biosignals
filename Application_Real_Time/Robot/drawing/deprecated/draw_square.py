import sys
import os
import json
import time
import pyttsx3
import math
import numpy as np

# append parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from neurapy.robot import Robot

# error: could not find IK

r = Robot()
r.set_mode("Automatic")

#load plane points
with open("drawing/planePoints_old.json", "r") as f:
    rob_points = json.load(f)
#print(rob_points)

"""
print(rob_points[0])
print(rob_points[1])
print(rob_points[2])
"""

""""
Transformation matrix
get the rotation matrix that transforms from the workspace of the whiteboard to the robot workspace
assumption: the xz plane of the new workspace (whiteboard) is vertical 
-> we only perform a z-rotation, only the first two points are considered
-> if the whiteboard is tilted (not vertical), we need to consider the third point as well
"""

# get relevant points in the robot workspace
p1_rob = np.array(rob_points[0][:3])
p2_rob = np.array(rob_points[1][:3])
print(rob_points[0])

# copute the related vector pointing from p1 to p2
v_rob = [p2_rob[0] - p1_rob[0], p2_rob[1] - p1_rob[1], p2_rob[2] - p1_rob[2]]

# define the aligned unit vector in the whiteboard workspace
v_wb = np.array([1.0, 0.0, 1.0])

# compute the angle about z: FROM whiteboard TO robot coordinate system
theta = math.atan2(v_wb[0]*v_rob[1] - v_wb[1]*v_rob[0], v_wb[0]*v_rob[0] + v_wb[1]*v_rob[1])
print("Angle in deg: ", math.degrees(theta))

# transformation matrix FROM robot TO whiteboard 
T_rob_wb = np.array([[math.cos(theta), -math.sin(theta), 0, -p1_rob[0]],
     [math.sin(theta), math.cos(theta), 0, -p1_rob[1]],
     [0, 0, 1, -p1_rob[2]],
     [0, 0, 0, 1]])
print("Transformation matrix: ", T_rob_wb)



"""
Draw a square based on point 1 and 2
"""
#1. move to lower left corner
#start_point = [-0.24563976996402628, 0.3354423750787085, 1.1144532947066266, 1.575580228516363, 0.8955091228283477, 3.095210624587609]
#middle_point = rob_points[0]
#middle_point[1] = rob_points[0][1] - 0.1 #middle point ~10cm away from the whiteboard (x axes are more or less aligned)
linear_property = {
"speed": 0.9,
"acceleration": 0.2,
"target_pose": [rob_points[0]],
"current_joint_angles":r.robot_status("jointAngles")
}

linear_property = {
"speed": 0.9,
"acceleration": 0.2,
"target_pose": [rob_points[1]],
"current_joint_angles":r.robot_status("jointAngles")
}

r.move_linear_from_current_position(**linear_property)
