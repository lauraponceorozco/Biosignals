"""
Drawing module for NeuraPy robot control system.

This module provides various drawing functions for the robot, including:
- Basic circle and square drawing
- Advanced trajectory-based drawing with pen lifting
- Force-controlled drawing with contact detection
- Visualization and analysis tools
"""

# Import basic drawing functions
from .drawing_functions import (
    # Basic drawing functions
    draw_circle_xz,
    draw_circle_xz_manual,
    draw_square,
    
    # Advanced drawing functions
    draw_circle_trajectory,
    draw_circle_trajectory_lifting,
    draw_circle_trajectory_with_force_control,
    
    # Utility functions
    ask_visualization_mode,
    ask_force_control_mode,
    ask_force_threshold
)

# Import utility functions
from .utils import (
    # Position management
    load_tictactoe_positions,
    select_position,
    
    # Validation and safety
    validate_circle_points,
    wait_for_motion_completion,
    is_in_safe_position,
    move_to_safe_position,
    find_best_radius,
    
    # Drawing utilities
    pen_lift,
    compute_circle_point,
    move_to_position_with_pen_lifting,
    
    # Force control
    ForceControlConfig,
    TrajectoryRecorder,
    visualize_trajectory,
    calibrate_force_sensor,
    detect_contact,
    emergency_force_stop,
    move_to_contact,
    force_controlled_approach,
    auto_calibrate_force_threshold,
    test_force_detection,
    analyze_circle_quality,
    
    # Logging setup
    setup_drawing_logger
)

# Import main drawer interface
from .drawer import main as drawer_main

__all__ = [
    # Basic drawing functions
    "draw_circle_xz",
    "draw_circle_xz_manual", 
    "draw_square",
    
    # Advanced drawing functions
    "draw_circle_trajectory",
    "draw_circle_trajectory_lifting",
    "draw_circle_trajectory_with_force_control",
    "draw_circle_with_precomputed_trajectory",
    
    # Utility functions
    "ask_visualization_mode",
    "ask_force_control_mode", 
    "ask_force_threshold",
    
    # Position management
    "load_tictactoe_positions",
    "select_position",
    
    # Validation and safety
    "validate_circle_points",
    "wait_for_motion_completion",
    "is_in_safe_position",
    "move_to_safe_position",
    "find_best_radius",
    
    # Drawing utilities
    "pen_lift",
    "compute_circle_point",
    "move_to_position_with_pen_lifting",
    "precompute_circle_trajectories",
    
    # Force control
    "ForceControlConfig",
    "TrajectoryRecorder",
    "visualize_trajectory",
    "calibrate_force_sensor",
    "detect_contact",
    "emergency_force_stop",
    "move_to_contact",
    "force_controlled_approach",
    "auto_calibrate_force_threshold",
    "test_force_detection",
    "analyze_circle_quality",
    
    # Logging setup
    "setup_drawing_logger",
    
    # Main interface
    "drawer_main"
]
