# Import drawing functions
from neurapy.robot import neurapy_logger
from config import default_config #! Global config for drawing
from drawing.utils import (
    # General functions
    load_tictactoe_positions,
    load_grid_positions,
    select_position,
    validate_circle_points,
    wait_for_motion_completion,
    is_in_safe_position,
    move_to_safe_position,
    find_best_radius,
    pen_lift,
    compute_circle_point,
    move_to_position_with_pen_lifting,
    # Force control functions
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
from drawing.drawing_functions import (
    # Advanced drawing functions
    draw_circle_trajectory,
    draw_circle_trajectory_lifting,
    draw_circle_trajectory_with_force_control,
    draw_tictactoe_grid,
    ask_grid_drawing_mode,
    # Utility functions
    ask_visualization_mode,
    ask_force_control_mode,
    ask_force_threshold
)




# Create [radius_lower_limit, radius_upper_limit] around the selected radius
radii_try = default_config.circle_drawing_radii
radius = radii_try[0]
radius_lower_limit = radius - 0.001
radius_upper_limit = radius + 0.001
radii = np.linspace(radius_lower_limit, radius_upper_limit, 25)
print(f"Radii to try: {radii}")


# Define escape radii and sort them by distance to target radius
escape_radii_base = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
escape_radii = sorted(escape_radii_base, key=lambda x: abs(x - radius))


# Try different radii
success = False
for radius in radii:
        success = draw_circle_trajectory_lifting(
            robot, radius, selected_pose, selected_position_key=position_key, speed=90
        )


if not success:
    circle_logger.warning("Main radii failed, trying escape radii...")
    for radius in escape_radii:
        if enable_force_control:
            success = draw_circle_trajectory_with_force_control(
                robot, radius, selected_pose, config, enable_visualization=enable_visualization
            )
        else:
            success = draw_circle_trajectory_lifting(
                robot, radius, selected_pose, selected_position_key=position_key
            )
        if success:
            break

if not success:
    circle_logger.warning("Trying adaptive radius...")
    best_radius = find_best_radius(robot, selected_pose, max_radius=0.04, min_radius=0.01)
    if best_radius:
        if enable_force_control:
            draw_circle_trajectory_with_force_control(
                robot, best_radius, selected_pose, config, enable_visualization=enable_visualization
            )
        else:
            draw_circle_trajectory_lifting(
                robot, best_radius, selected_pose, selected_position_key=position_key
            )