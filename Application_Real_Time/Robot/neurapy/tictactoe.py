import time
import sys
import os
#from drawing.draw_circle_grid import draw_circle_xz

# append parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from drawing.drawing_functions import draw_circle_xz_manual

if sys.platform == "darwin":
    from neurapy.robot_mac import Robot
else:
    from neurapy.robot import Robot

class TicTacToeField:
    def __init__(self, field_id: int, center_pose: list):
        """
        field_id: integer ID of the field (1-9)
        center_pose: target pose for robot movement (e.g. [X, Y, Z, roll, pitch, yaw])
        """
        self.field_id = field_id
        self.center_pose = center_pose

class TicTacToeGame:
    def __init__(self, robot: Robot):
        self.robot = robot
        # Initialize 9 fields with hardcoded positions.
        # Later, you can replace these with positions from camera calibration.
        self.fields = self._init_fields()

    def _init_fields(self):
        # Hardcoded positions for each of the 9 fields.
        # TODO: Replace with actual (calibrated) positions.
        # pay attention to order of fields (see GUI also for reference)
        positions = {
            1: [0.5, 0.1, 0.2, 0, 0, 0],
            2: [1.047442603628777,
            -0.15552752575143466,
            0.20252328921076215,
            0.555267714089084,
            0.5492205491032468,
            1.1535052410833524,
            -0.10522112016200241],
            3: [0.5, 0.5, 0.2, 0, 0, 0],
            4: [0.7, 0.1, 0.2, 0, 0, 0],
            5: [0.7, 0.3, 0.2, 0, 0, 0],
            6: [0.7, 0.5, 0.2, 0, 0, 0],
            7: [0.9, 0.1, 0.2, 0, 0, 0],
            8: [0.9, 0.3, 0.2, 0, 0, 0],
            9: [0.9, 0.5, 0.2, 0, 0, 0],
        }
        fields = []
        for fid, pose in positions.items():
            fields.append(TicTacToeField(fid, pose))
        return fields

    def draw_field(self, field_id: int):
        """
        Perform a move on the given field_id -> draw circle
        """
        # Retrieve the matching field based on field_id.
        field = next((f for f in self.fields if f.field_id == field_id), None)
        if field is None:
            print(f"Field {field_id} not found!")
            return False

        print(f"Robot drawing circle at field {field_id} ...")
        # success = 1
        # TODO comment back in when draw_circle is implemented correctly

        # Move to safe starting position before drawing
        print("Moving to safe joint position before drawing...")
        # Joint angles for neutral/safe starting position
        safe_joint_angles = [
            1.7261193717088836, -0.545651122616592, -0.004781705737237272,
            1.1377811889934681, -0.06824117266732665, 0.9290174891702697,
            -0.06739890636069706
        ]
        self.robot.move_joint(safe_joint_angles, speed=5)
        time.sleep(3)

        # Move to the joint angles of the selected position
        selected_joint_angles = field.center_pose
        print("Moving to joint position of selected cell...")
        self.robot.move_joint(selected_joint_angles, speed=3)
        time.sleep(3)

        success = False
        for radius in [0.02, 0.018, 0.015]:
            if draw_circle_xz_manual(self.robot, radius, field.center_pose):
                success = True
                break

        if not success:
            print("All radii failed. Try adjusting robot pose or selecting another position.")
        return success
    