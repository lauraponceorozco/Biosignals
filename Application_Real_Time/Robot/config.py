"""
Configuration file for GUI-Robot Connection System
Modify these settings to customize the behavior
"""

# UDP Communication Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 9001  # Port for enhanced UDP listener
GUI_UDP_PORT = 9000  # Port for GUI to receive predictions

# Robot Configuration
SAFE_JOINT_ANGLES_OLD = [
    1.7261193717088836,  # Joint 1
    -0.545651122616592,  # Joint 2
    -0.004781705737237272,  # Joint 3
    1.1377811889934681,  # Joint 4
    -0.06824117266732665,  # Joint 5
    0.9290174891702697,  # Joint 6
    -0.06739890636069706  # Joint 7
]
SAFE_JOINT_ANGLES = [
    1.641027629810569,  # Joint 1
    -0.8986273673454269,  # Joint 2
    -0.029094327506951123,  # Joint 3
    1.2282935456192867,  # Joint 4
    0.03312552115949044,  # Joint 5
    1.2537768763594426,  # Joint 6
    0.07379623541798772  # Joint 7
]

# Drawing Configuration
DEFAULT_RADII = [0.05]  # [m]
DEFAULT_CIRCLE_SPEED = 100
DEFAULT_CIRCLE_ACCELERATION = 6

# Circle Drawing Specific Configuration
CIRCLE_DRAWING_RADII = DEFAULT_RADII  # [m] - radii to try in order
CIRCLE_DRAWING_SPEED = 100
CIRCLE_DRAWING_ACCELERATION = 5
CIRCLE_NUM_POINTS = 12
CIRCLE_PEN_LIFT_OFFSET = 0.02  # [m] - additional lift for pen up/down
CIRCLE_Y_OFFSET = 0.02  # [m] - 2cm above the drawing surface for safe approach
CIRCLE_SAFE_MOVEMENT_SPEED = 6
CIRCLE_FORCE_CONTROL_SPEED = 60

# Force Control Configuration
DEFAULT_FORCE_THRESHOLD = 2.0  # [N]
DEFAULT_APPROACH_HEIGHT = 0.02  # [m]
DEFAULT_CONTACT_STEP_SIZE = 0.001  # [m]
DEFAULT_MAX_CONTACT_ATTEMPTS = 50
DEFAULT_CONTACT_TIMEOUT = 5.0  # [s]
DEFAULT_EMERGENCY_FORCE_LIMIT = 10.0  # [N]

# GUI Configuration
GUI_OPTIONS_COUNT = 9  # Number of options in GUI (3x3 grid)
GUI_FIELD_MAPPING = {
    # Maps GUI option index to robot field ID (1-9)
    # Standard tic-tac-toe numbering:
    # 1 2 3
    # 4 5 6  
    # 7 8 9
    0: 1, 1: 2, 2: 3,  # Top row
    3: 4, 4: 5, 5: 6,  # Middle row
    6: 7, 7: 8, 8: 9   # Bottom row
}

# Navigation Commands (from SSVEP predictions)
NAVIGATION_COMMANDS = {
    "9.0": "move_up",
    "11.0": "move_down", 
    "13.0": "select"
}

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] : %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# File Paths
POSITIONS_FILE = "Application_Real_Time/Robot/data/tictactoe_positions_green.json"
GRID_POSITIONS_FILE = "Application_Real_Time/Robot/data/grid_positions_green.json"
LOGS_DIR = "Application_Real_Time/Robot/data/logs"
VISUALIZATION_DIR = "Application_Real_Time/Robot/data/drawing_visualization"
TRAJECTORY_DATA_DIR = "Application_Real_Time/Robot/data/trajectory_data"
DRAWING_VISUALIZATION_DIR = "Application_Real_Time/Robot/data/drawing_visualization"

# Safety Settings
ENABLE_COLLISION_DETECTION = True
ENABLE_EMERGENCY_STOP = True
SAFE_POSITION_THRESHOLD = 0.1  # [rad]

# Visualization Settings
ENABLE_TRAJECTORY_VISUALIZATION = False
ENABLE_FORCE_VISUALIZATION = False
SAVE_TRAJECTORY_DATA = True

# Performance Settings
RECORDING_RATE = 20  # Hz
MOTION_TIMEOUT = 30.0  # [s]
JOINT_ANGLE_THRESHOLD = 0.01  # [rad]

# Network Settings
SOCKET_TIMEOUT = 1.0  # [s]
MAX_RECV_SIZE = 1024  # bytes

# Drawing Quality Settings
MIN_CIRCLE_POINTS = 8
MAX_CIRCLE_POINTS = 16
CIRCLE_CLOSURE_THRESHOLD = 0.01  # [m]
RADIUS_SEARCH_STEP = 0.002  # [m]

# Default Configuration Class
class DefaultConfig:
    """Default configuration class for easy access"""
    
    def __init__(self):
        # UDP Settings
        self.udp_ip = UDP_IP
        self.udp_port = UDP_PORT
        self.gui_udp_port = GUI_UDP_PORT
        
        # Robot Settings
        self.safe_joint_angles = SAFE_JOINT_ANGLES
        self.enable_collision_detection = ENABLE_COLLISION_DETECTION
        self.enable_emergency_stop = ENABLE_EMERGENCY_STOP
        self.safe_position_threshold = SAFE_POSITION_THRESHOLD
        
        # Drawing Settings
        self.radii_to_try = DEFAULT_RADII
        self.circle_speed = DEFAULT_CIRCLE_SPEED
        self.circle_acceleration = DEFAULT_CIRCLE_ACCELERATION
        self.min_circle_points = MIN_CIRCLE_POINTS
        self.max_circle_points = MAX_CIRCLE_POINTS
        self.circle_closure_threshold = CIRCLE_CLOSURE_THRESHOLD
        self.radius_search_step = RADIUS_SEARCH_STEP
        
        # Circle Drawing Specific Settings
        self.circle_drawing_radii = CIRCLE_DRAWING_RADII
        self.circle_drawing_speed = CIRCLE_DRAWING_SPEED
        self.circle_drawing_acceleration = CIRCLE_DRAWING_ACCELERATION
        self.circle_num_points = CIRCLE_NUM_POINTS
        self.circle_pen_lift_offset = CIRCLE_PEN_LIFT_OFFSET
        self.circle_y_offset = CIRCLE_Y_OFFSET
        self.circle_safe_movement_speed = CIRCLE_SAFE_MOVEMENT_SPEED
        self.circle_force_control_speed = CIRCLE_FORCE_CONTROL_SPEED
        
        # Force Control Settings
        self.force_threshold = DEFAULT_FORCE_THRESHOLD
        self.approach_height = DEFAULT_APPROACH_HEIGHT
        self.contact_step_size = DEFAULT_CONTACT_STEP_SIZE
        self.max_contact_attempts = DEFAULT_MAX_CONTACT_ATTEMPTS
        self.contact_timeout = DEFAULT_CONTACT_TIMEOUT
        self.emergency_force_limit = DEFAULT_EMERGENCY_FORCE_LIMIT
        
        # GUI Settings
        self.gui_options_count = GUI_OPTIONS_COUNT
        self.gui_field_mapping = GUI_FIELD_MAPPING.copy()
        self.navigation_commands = NAVIGATION_COMMANDS.copy()
        
        # Visualization Settings
        self.enable_trajectory_visualization = ENABLE_TRAJECTORY_VISUALIZATION
        self.enable_force_visualization = ENABLE_FORCE_VISUALIZATION
        self.save_trajectory_data = SAVE_TRAJECTORY_DATA
        self.recording_rate = RECORDING_RATE
        
        # Performance Settings
        self.motion_timeout = MOTION_TIMEOUT
        self.joint_angle_threshold = JOINT_ANGLE_THRESHOLD
        self.socket_timeout = SOCKET_TIMEOUT
        self.max_recv_size = MAX_RECV_SIZE
        
        # File Paths
        self.positions_file = POSITIONS_FILE
        self.grid_positions_file = GRID_POSITIONS_FILE
        self.logs_dir = LOGS_DIR
        self.visualization_dir = VISUALIZATION_DIR
        self.trajectory_data_dir = TRAJECTORY_DATA_DIR
        self.drawing_visualization_dir = DRAWING_VISUALIZATION_DIR
        
        # Logging Settings
        self.log_level = LOG_LEVEL
        self.log_format = LOG_FORMAT
        self.log_date_format = LOG_DATE_FORMAT

# Create default config instance
default_config = DefaultConfig() 