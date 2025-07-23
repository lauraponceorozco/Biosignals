import sys
import os
import time
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

def draw_circle_xz(robot, radius=0.05, center_pose=None, num_points=12, use_home=False):
    """
    Draw a circle in the XZ plane (like on a vertical wall) using continuous motion.
    
    Args:
        robot: Robot instance
        radius: Radius of the circle in meters (default: 0.05m = 5cm)
        center_pose: Center position and orientation [x, y, z, roll, pitch, yaw]
        num_points: Number of points to define the circle (minimum 8, recommended 12)
        use_home: If True, move to home position first
    """
    
    # Move to home position first if requested
    """
    if use_home:
        print("Moving to home position...")
        try:
            robot.move_joint("Home", speed=5)
            time.sleep(3)
            print("✓ Moved to Home position")
        except Exception as e:
            print(f"Failed to move to home: {e}")
            print("Using current position instead...")
    """
    # Use current pose if center_pose not provided
    if center_pose is None:
        print("Getting current pose as circle center...")
        center_pose = robot.get_tcp_pose_quaternion()
        center_pose = robot.convert_quaternion_to_euler_pose(center_pose)
    
    print(f"Circle center: X={center_pose[0]:.3f}, Y={center_pose[1]:.3f}, Z={center_pose[2]:.3f}")
    print(f"Circle radius: {radius}m in XZ plane")

    # Ensure we have enough points for a smooth circle
    num_points = max(8, num_points)
    
    # Generate points around the circle in XZ plane (Y is fixed)
    angles = np.linspace(0, 2*np.pi, num_points + 1)[:-1]  # Don't repeat the first point
    
    # Create points for the circular motion
    points = []
    for angle in angles:
        # Circle in XZ plane: X varies with cos, Z varies with sin, Y stays constant
        x = center_pose[0] + radius * np.cos(angle)
        z = center_pose[2] + radius * np.sin(angle)
        y = center_pose[1]  # Y stays constant (depth into the wall)
        
        # Keep the same orientation as the center pose
        point = [
            float(x),
            float(y),
            float(z),
            float(center_pose[3]),  # Roll
            float(center_pose[4]),  # Pitch
            float(center_pose[5])   # Yaw
        ]
        points.append(point)

    print(f"Generated {len(points)} points for the circle")
    
    # First verify that we can reach all target poses
    print("Checking if all circle points are reachable...")
    reference_joint = robot.get_current_joint_angles()
    
    for i, point in enumerate(points):
        try:
            robot.compute_inverse_kinematics(
                target_pose=point,
                reference_joint=reference_joint
            )
            if i % 3 == 0:  # Print every 3rd point to avoid spam
                print(f"✓ Point {i+1} reachable: X={point[0]:.3f}, Z={point[2]:.3f}")
        except Exception as e:
            print(f"Point {i+1} NOT reachable: X={point[0]:.3f}, Z={point[2]:.3f}")
            print(f"Error: {e}")
            print(f"Try smaller radius (current: {radius}m)")
            return False

    print("All points are reachable! Starting circular motion...")

    # Configure circular motion with simple, conservative parameters
    circular_properties = {
        "speed": 0.1,  # Linear speed in m/s
        "acceleration": 0.05,  # Linear acceleration in m/s²
        "jerk": 50,  # Linear jerk
        "rotation_speed": 0.5,  # Rotational speed in rad/s
        "rotation_acceleration": 1.0,  # Rotational acceleration in rad/s²
        "rotation_jerk": 50,  # Rotational jerk
        "blending_mode": 1,  # DYNAMIC_BLENDING for smooth motion
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
        print("Executing circular motion...")
        robot.move_circular(**circular_properties)
        print("Circle completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during circular motion: {e}")
        print("Try reducing the radius or number of points")
        return False

def check_reachable_pose(robot, pose):
    """Check if a pose is reachable by the robot"""
    try:
        reference_joint = robot.get_current_joint_angles()
        robot.compute_inverse_kinematics(
            target_pose=pose,
            reference_joint=reference_joint,
        )
        return True
    except:
        return False

if __name__ == "__main__":
    try:
        # Initialize robot and set to automatic mode
        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        time.sleep(1)
        
        # # Load tic-tac-toe positions from JSON file
        # tictactoe_file = os.path.join(os.path.dirname(__file__), "tictactoe_positions.json")
        # with open(tictactoe_file, 'r') as f:
        #     tictactoe_positions = json.load(f)
        
        # # Get the center grid position (index 4 - "Middle Center")
        # center_grid = None
        # for position in tictactoe_positions:
        #     if position["index"] == 4:  # Middle Center
        #         center_grid = position
        #         break
        
        # if center_grid is None:
        #     print("Error: Could not find center grid position in tic-tac-toe positions")
        #     exit(1)
        
        # print(f"Using center grid position: {center_grid['name']}")
        # print(f"TCP Pose: {center_grid['tcp_pose']}")
        
        # # Use the center grid's TCP pose as the center pose for the circle
        # center_pose = center_grid['tcp_pose']  # Already in euler format [x, y, z, roll, pitch, yaw]

        success = draw_circle_xz(
            robot, 
            radius=0.05,      # 5cm radius
            center_pose=None, # Use center grid position
            num_points=12,    # 12 points for smooth circle
            use_home=True     # Go to home first
        )
        
        if success:
            print("Circle drawn successfully!")
        else:
            print("Failed to draw circle")
            print("Try:")
            print("- Smaller radius (e.g., 0.03 or 0.02)")
            print("- Move robot to more central position manually")
            print("- Fewer points (e.g., 8 instead of 12)")
        
    except Exception as e:
        print(f"General error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            robot.stop()
        except:
            pass
