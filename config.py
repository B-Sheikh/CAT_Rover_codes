"""
config.py
=====================================================================
Centralized configuration parameters for the Autonomous Car project.
Covers:
  - Robot physical dimensions & kinematics
  - Serial communication & Arduino pin mappings
  - Camera & HSV color detection thresholds (Green start, Red goal)
  - 2.5D Terrain, Elevation & Obstacle Costmap parameters
  - Reinforcement Learning & Trajectory Optimization hyperparameters
  - Navigation speeds, thresholds, and multi-round mission settings
=====================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class RobotConfig:
    # Kinematic & Physical Constants
    wheel_diameter_m: float = 0.065       # 65 mm wheel diameter
    wheel_base_m: float = 0.100           # 160 mm track width (between left/right contact patches)
    ticks_per_rev: int = 20               # 20 slots on encoder disk
    max_rpm: float = 200.0                # Approximate free-run RPM at 12V
    max_linear_speed_mps: float = 0.68    # ~ (pi * 0.065 * 200) / 60 m/s
    min_pwm_to_move: int = 70             # Minimum PWM to overcome static friction
    nominal_mass_kg: float = 1.20         # Total vehicle mass in kg


@dataclass
class SerialConfig:
    port: str = "/dev/ttyACM0"
    baud_rate: int = 115200
    timeout_sec: float = 1.0
    command_timeout_ms: int = 500         # Arduino fail-safe stop timeout
    sensor_stream_hz: float = 20.0        # Sensor packet send rate from Arduino


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    framerate: int = 30
    horizontal_fov_deg: float = 62.2      # Pi Camera Module 3 horizontal FOV
    focal_length_px: float = 550.0        # Estimated fx for 640x480 resolution
    backend: str = "auto"                 # "picamera", "usb", "sim", "auto"
    usb_device_id: int = 0


@dataclass
class VisionConfig:
    # Physical dimensions of color target dots
    dot_real_diameter_m: float = 0.08     # 8 cm real-world diameter of colored circular dot

    # HSV Color Ranges (H: 0-180, S: 0-255, V: 0-255 in OpenCV)
    # Green Dot (Start Point)
    green_hsv_lower: Tuple[int, int, int] = (35, 60, 50)
    green_hsv_upper: Tuple[int, int, int] = (85, 255, 255)

    # Red Dot (Goal / Ending Point) - Red wraps around 0 and 180 in HSV
    red_hsv_lower1: Tuple[int, int, int] = (0, 70, 50)
    red_hsv_upper1: Tuple[int, int, int] = (10, 255, 255)
    red_hsv_lower2: Tuple[int, int, int] = (170, 70, 50)
    red_hsv_upper2: Tuple[int, int, int] = (180, 255, 255)

    # Contour filtering
    min_contour_area: float = 80.0        # Minimum pixel area to filter noise
    max_contour_area: float = 120000.0    # Maximum pixel area to ignore full-screen glare
    min_circularity: float = 0.40         # Circularity metric: 4 * pi * Area / (Perimeter^2)

    # ArUco Marker fallback (optional)
    aruco_dict_id: int = 0                # cv2.aruco.DICT_4X4_50
    aruco_real_size_m: float = 0.10


@dataclass
class TerrainConfig:
    # 2.5D Spatial Grid Map Parameters
    map_size_x_m: float = 6.0             # 6m x 6m mapping arena
    map_size_y_m: float = 6.0
    grid_resolution_m: float = 0.05       # 5 cm high-resolution spatial cells
    obstacle_inflation_radius_m: float = 0.15  # Safety buffer around obstacles

    # Elevation & Slope Parameters
    max_negotiable_slope_deg: float = 20.0     # Maximum climbable slope angle
    slope_gravity_pwm_gain: float = 4.5        # PWM compensation per degree of incline
    incline_load_sensitivity: float = 0.35     # Load drop sensitivity for encoder-based slope estimation
    slip_penalty_weight: float = 15.0          # Reward penalty for wheel slip on loose/uneven terrain
    elevation_cost_weight: float = 8.0         # Cost penalty for traversing steep grades vs flatter paths


@dataclass
class RLConfig:
    # Spatial State Discretization
    grid_resolution_m: float = 0.10       # 10 cm spatial grid cell size
    heading_bins: int = 16                # 16 orientation bins (22.5 deg each)
    obstacle_bins: int = 3                # 0: clear (>45cm), 1: caution (20-45cm), 2: danger (<20cm)
    elevation_bins: int = 3               # 0: flat, 1: uphill incline, 2: downhill decline

    # Q-Learning Hyperparameters
    learning_rate: float = 0.15           # Alpha
    discount_factor: float = 0.95         # Gamma
    initial_epsilon: float = 0.35         # Initial exploration rate in Round 1
    min_epsilon: float = 0.05             # Minimum exploration rate
    epsilon_decay_per_round: float = 0.60 # Epsilon decay factor across exploration rounds

    # Reward Function Weights
    reward_goal_reached: float = 500.0    # Huge positive reward for reaching goal
    reward_progress_weight: float = 15.0  # Reward for reducing distance to goal (per meter)
    reward_step_penalty: float = -0.5     # Time penalty per control step
    reward_collision: float = -300.0      # Severe penalty for collision / danger zone
    reward_obstacle_proximity_weight: float = -20.0 # Penalty per cm inside obstacle caution zone
    reward_jerk_penalty_weight: float = -0.05       # Penalty for abrupt steering changes
    reward_vision_alignment_weight: float = 2.5     # Reward for keeping goal in camera frame center
    reward_slope_penalty_weight: float = -5.0       # Penalty for excessive elevation changes / steep bumps

    # Cross-Entropy / Trajectory Optimization
    max_trajectory_steps: int = 800       # Max steps per round before timeout
    best_route_file: str = "best_route.json"
    trajectory_smoothing_factor: float = 0.2


@dataclass
class NavConfig:
    # Obstacle avoidance thresholds
    obstacle_stop_cm: float = 20.0        # Hard stop & pivot threshold
    obstacle_slow_cm: float = 45.0        # Slow down / veer threshold
    goal_tolerance_m: float = 0.18        # Distance within which goal is considered reached
    start_tolerance_m: float = 0.20       # Distance within which start is reached when looping

    # Speeds (PWM values 0..255)
    base_pwm: int = 160                   # Standard forward cruise speed
    fast_pwm: int = 210                   # Fast forward speed for optimal routes
    slow_pwm: int = 110                   # Slow maneuvering speed
    turn_gain: float = 180.0              # Heading error proportional gain
    max_pwm: int = 255

    # Visual Servoing
    vision_blend_weight: float = 0.65     # Blend between visual bearing and odometry heading

    # Multi-Round Mission Settings
    exploration_rounds: int = 3           # Number of sensor-driven learning rounds (e.g. 2-3 rounds)
    sensorless_replay_laps: int = 5       # Number of sensorless optimal route replay laps
    control_loop_hz: float = 20.0         # Main control loop frequency (20 Hz = 50 ms)


@dataclass
class AppConfig:
    robot: RobotConfig = field(default_factory=RobotConfig)
    serial: SerialConfig = field(default_factory=SerialConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    nav: NavConfig = field(default_factory=NavConfig)


# Global default instance
DEFAULT_CONFIG = AppConfig()
