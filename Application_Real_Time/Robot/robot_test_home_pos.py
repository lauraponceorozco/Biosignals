import sys
import os

# append parent folder to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)



from neurapy.robot import Robot
r = Robot()
r.set_mode("Automatic")

r.move_joint("Home", speed=5)