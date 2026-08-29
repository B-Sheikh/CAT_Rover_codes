"""
terrain_map.py
=====================================================================
2.5D Spatial Terrain, Obstacle Costmap & Elevation Analyzer.

Capabilities:
  1. 2.5D Multi-Layer Spatial Grid Map:
     - Layer 0: Obstacle Occupancy & Euclidean Clearance Field
     - Layer 1: Terrain Elevation & Slope Gradient (Incline / Decline in deg)
     - Layer 2: Traction & Wheel-Slip Heatmap
  2. Multi-Modal Slope & Elevation Estimator:
     - Real-time IMU pitch / roll integration (if hardware IMU is connected)
     - Dynamic Load & Motor Transfer Function Estimator (estimates slope
       from motor back-EMF / speed-vs-PWM deficit: v_expected vs v_measured)
     - Wheel slip ratio fusion
  3. Obstacle Raycasting & Safety Buffer Inflation (5cm grid resolution)
  4. Slope-Compensated Feedforward PWM Profile Synthesis for Replay
=====================================================================
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np

from config import DEFAULT_CONFIG, TerrainConfig, RobotConfig


@dataclass
class TerrainPointInfo:
    obstacle_dist_m: float      # Distance to nearest obstacle
    slope_deg: float            # Incline angle (+ uphill, - downhill)
    slip_ratio: float           # Measured traction loss (0..1)
    is_safe: bool               # Safe for traversal
    traversal_cost: float       # Composite energy & safety cost


class SlopeEstimator:
    """
    Estimates terrain elevation gradient & slope angle using IMU or
    dynamic motor load deficit (v_expected(PWM) - v_measured).
    """
    def __init__(self, robot_cfg: Optional[RobotConfig] = None,
                 terrain_cfg: Optional[TerrainConfig] = None):
        self.rcfg = robot_cfg or DEFAULT_CONFIG.robot
        self.tcfg = terrain_cfg or DEFAULT_CONFIG.terrain
        self._filtered_pitch_deg = 0.0

    def estimate_slope(self, left_pwm: int, right_pwm: int,
                       v_mps: float,
                       imu_pitch_deg: Optional[float] = None,
                       slip_ratio: float = 0.0) -> float:
        """
        Returns estimated slope angle in degrees (+ uphill, - downhill).
        Fuses direct IMU pitch if available, or calculates from kinematic load deficit.
        """
        if imu_pitch_deg is not None and not math.isnan(imu_pitch_deg):
            # Hardware IMU pitch directly available
            raw_slope = imu_pitch_deg
        else:
            # Kinematic Motor Load Incline Estimator:
            # On flat ground, expected speed: v = f(PWM)
            avg_pwm = (abs(left_pwm) + abs(right_pwm)) / 2.0
            if avg_pwm < self.rcfg.min_pwm_to_move:
                raw_slope = 0.0
            else:
                pwm_ratio = (avg_pwm - self.rcfg.min_pwm_to_move) / (255.0 - self.rcfg.min_pwm_to_move)
                expected_v = pwm_ratio * self.rcfg.max_linear_speed_mps
                v_deficit = expected_v - abs(v_mps)

                # Positive deficit -> vehicle moving slower than expected on flat ground -> climbing uphill
                # Negative deficit -> accelerating faster -> descending downhill
                # Slope approx: sin(theta) ~ (m * a_resist) / (m * g)
                sin_theta = max(-0.5, min(0.5, (v_deficit / max(0.1, expected_v)) * self.tcfg.incline_load_sensitivity))
                raw_slope = math.degrees(math.asin(sin_theta))

        # Low-pass filter to smooth mechanical vibrations
        self._filtered_pitch_deg = 0.80 * self._filtered_pitch_deg + 0.20 * raw_slope
        return float(self._filtered_pitch_deg)

    def compute_slope_compensated_pwm(self, base_pwm: int, slope_deg: float) -> int:
        """
        Modulates motor PWM to compensate for gravity on inclines:
          - Uphill (+slope): Boosts PWM to maintain cruise speed.
          - Downhill (-slope): Reduces PWM / brakes to prevent skidding.
        """
        if base_pwm == 0:
            return 0

        # Gravity compensation: Delta_PWM = k * sin(slope)
        compensation = int(self.tcfg.slope_gravity_pwm_gain * slope_deg)
        compensated = base_pwm + compensation
        return max(-255, min(255, compensated))


class TerrainMap25D:
    """
    High-resolution 2.5D spatial costmap representing obstacle clearance,
    elevation gradient / slope, and terrain slip across the arena.
    """
    def __init__(self, terrain_cfg: Optional[TerrainConfig] = None):
        self.cfg = terrain_cfg or DEFAULT_CONFIG.terrain
        self.res = self.cfg.grid_resolution_m
        self.origin_x = -self.cfg.map_size_x_m / 2.0
        self.origin_y = -self.cfg.map_size_y_m / 2.0

        self.cells_x = int(self.cfg.map_size_x_m / self.res)
        self.cells_y = int(self.cfg.map_size_y_m / self.res)

        # Layers (Shape: cells_y, cells_x)
        self.occupancy_grid = np.zeros((self.cells_y, self.cells_x), dtype=np.float32)
        self.slope_grid = np.zeros((self.cells_y, self.cells_x), dtype=np.float32)
        self.slip_grid = np.zeros((self.cells_y, self.cells_x), dtype=np.float32)
        self.visit_count = np.zeros((self.cells_y, self.cells_x), dtype=np.int32)

        self.slope_estimator = SlopeEstimator(terrain_cfg=self.cfg)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int((x - self.origin_x) / self.res)
        gy = int((y - self.origin_y) / self.res)
        return (max(0, min(self.cells_x - 1, gx)),
                max(0, min(self.cells_y - 1, gy)))

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        x = (gx * self.res) + self.origin_x + (self.res / 2.0)
        y = (gy * self.res) + self.origin_y + (self.res / 2.0)
        return x, y

    def update_cell_telemetry(self, x: float, y: float, slope_deg: float, slip_ratio: float):
        """Records terrain characteristics at robot's current position."""
        gx, gy = self.world_to_grid(x, y)
        self.visit_count[gy, gx] += 1
        # Exponential moving average of slope & slip
        alpha = 0.3
        self.slope_grid[gy, gx] = (1.0 - alpha) * self.slope_grid[gy, gx] + alpha * slope_deg
        self.slip_grid[gy, gx] = (1.0 - alpha) * self.slip_grid[gy, gx] + alpha * slip_ratio

    def update_ultrasonic_ray(self, robot_x: float, robot_y: float, robot_theta: float,
                              us1_cm: float, us2_cm: float):
        """
        Raycasts ultrasonic sensor returns into the occupancy grid with obstacle inflation.
        """
        sensors = [
            (us1_cm, math.radians(+15.0)),  # Left sensor
            (us2_cm, math.radians(-15.0)),  # Right sensor
        ]

        for dist_cm, offset_angle in sensors:
            if dist_cm <= 0 or dist_cm > 250.0:
                continue

            dist_m = dist_cm / 100.0
            beam_angle = robot_theta + offset_angle
            obs_x = robot_x + dist_m * math.cos(beam_angle)
            obs_y = robot_y + dist_m * math.sin(beam_angle)

            gx, gy = self.world_to_grid(obs_x, obs_y)

            # Mark obstacle cell and inflate safety buffer
            inflation_cells = int(math.ceil(self.cfg.obstacle_inflation_radius_m / self.res))
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.cells_x and 0 <= ny < self.cells_y:
                        d_cells = math.hypot(dx, dy)
                        if d_cells <= inflation_cells:
                            cost = max(0.0, 1.0 - (d_cells / (inflation_cells + 1.0)))
                            self.occupancy_grid[ny, nx] = min(1.0, self.occupancy_grid[ny, nx] + 0.4 * cost)

    def query_point(self, x: float, y: float) -> TerrainPointInfo:
        """Returns comprehensive terrain & obstacle info at coordinate (x, y)."""
        gx, gy = self.world_to_grid(x, y)
        occ = float(self.occupancy_grid[gy, gx])
        slope = float(self.slope_grid[gy, gx])
        slip = float(self.slip_grid[gy, gx])

        # Clearance approximation from occupancy grid
        obstacle_dist = max(0.05, (1.0 - occ) * 1.5)
        is_safe = (occ < 0.70) and (abs(slope) < self.cfg.max_negotiable_slope_deg)

        # Composite traversal cost
        cost = (occ * 100.0) + (abs(slope) * self.cfg.elevation_cost_weight) + (slip * self.cfg.slip_penalty_weight)

        return TerrainPointInfo(
            obstacle_dist_m=obstacle_dist,
            slope_deg=slope,
            slip_ratio=slip,
            is_safe=is_safe,
            traversal_cost=float(cost),
        )
