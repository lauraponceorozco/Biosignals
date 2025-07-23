import sys
import os
import time
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
6
if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

POSITION_NAMES = [
    "Upper Left", "Upper Center", "Upper Right",
    "Middle Left", "Middle Center", "Middle Right",
    "Lower Left", "Lower Center", "Lower Right"
]

def draw_circle_xz(robot, radius, center_pose, num_points=12):
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
            print(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f})")
            return False

    try:
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
        print("Circle completed successfully.")
        return True
    except Exception as e:
        print(f"Error during circular motion: {e}")
        return False

def draw_circle_xz_manual(robot, radius, center_pose, num_points=12):
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
            print(f"Point {i+1} not reachable (X={point[0]:.3f}, Z={point[2]:.3f})")
            return False

    try:
        print("Executing circular motion...")
        traj = []
        for point in points:
            joint_ref = robot.get_current_joint_angles()
            point_robot = robot.compute_inverse_kinematics(point, joint_ref)
            print(point_robot)
            robot.move_joint(point_robot, speed=5)
        
        time.sleep(0.1)
        print("Circle completed successfully.")
        return True
    except Exception as e:
        print(f"Error during circular motion: {e}")
        return False


if __name__ == "__main__":
    robot = None
    try:
        robot = Robot()
        robot.switch_to_automatic_mode()
        robot.enable_collision_detection()
        time.sleep(1)
        print("Robot initialized.")

        try:
            robot.reset_fault()
            print("Resetting any fault state.")
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
        print(f"Loading positions from: {json_path}")
        positions = load_tictactoe_positions(json_path)

        while True:
            selected_key = select_position()
            if selected_key == "quit":
                print("Exiting.")
                break

            # Move to safe starting position before drawing
            print("Moving to safe joint position before drawing...")
            robot.move_joint(safe_joint_angles, speed=10)
            time.sleep(3)

            # Move to the joint angles of the selected position
            selected_joint_angles = positions[selected_key]["joint_angles"]
            print("Moving to joint position of selected cell...")
            robot.move_joint(selected_joint_angles, speed=10)
            time.sleep(3)

            # Get actual TCP pose from the robot
            selected_pose = robot.get_tcp_pose()
            print(f"Pose from forward kinematics: {np.round(selected_pose, 3).tolist()}")
            print(f"Selected position: {selected_key.replace('_', ' ').title()}")

            # Optional: neutral orientation
            # selected_pose[3:] = [0.0, np.pi, 0.0]

            success = False
            for radius in [0.02, 0.018, 0.015]:
                if draw_circle_xz_manual(robot, radius, selected_pose):
                    success = True
                    break

            if not success:
                print("All radii failed. Try adjusting robot pose or selecting another position.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        if robot is not None:
            try:
                robot.reset_fault()
                print("Fault reset executed after failure.")
            except:
                print("Could not reset fault state.")
    finally:
        try:
            if robot is not None:
                robot.stop()
        except:
            pass
