"""
odometry.py
=====================================================================
Dead-reckoning pose estimate (x, y, theta) from 4-wheel encoder ticks
using differential-drive kinematics and midpoint integration.

Features:
  - Midpoint Runge-Kutta integration for high arc accuracy
  - 4-wheel slip detection (compares front vs rear on each side)
  - Linear velocity (m/s), angular velocity (rad/s), and curvature estimation
  - Goal distance, bearing, and heading error computation helpers
  - Calibration parameters pulled from config.py
=====================================================================
"""

import math
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

from config import DEFAULT_CONFIG, RobotConfig


@dataclass
class Pose2D:
    x: float = 0.0          # meters
    y: float = 0.0          # meters
    theta: float = 0.0      # radians, 0 = facing +x, counter-clockwise positive


class Odometry:
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                 robot_cfg: Optional[RobotConfig] = None):
        self.cfg = robot_cfg or DEFAULT_CONFIG.robot
        self.x = x
        self.y = y
        self.theta = theta

        # Precompute constants
        self.wheel_circ = math.pi * self.cfg.wheel_diameter_m
        self.m_per_tick = self.wheel_circ / float(self.cfg.ticks_per_rev)
        self.wheel_base = self.cfg.wheel_base_m

        # Previous raw tick counts from Arduino
        self._last_fl: Optional[int] = None
        self._last_rl: Optional[int] = None
        self._last_fr: Optional[int] = None
        self._last_rr: Optional[int] = None
        self._last_time: float = time.time()

        # Telemetry & Diagnostics
        self.last_slip_ratio_left: float = 0.0   # |FL delta - RL delta| / max(delta)
        self.last_slip_ratio_right: float = 0.0  # |FR delta - RR delta| / max(delta)
        self.total_distance_traveled_m: float = 0.0
        self.current_v_mps: float = 0.0
        self.current_omega_radps: float = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """Resets the internal pose state and tick baselines."""
        self.x = x
        self.y = y
        self.theta = math.atan2(math.sin(theta), math.cos(theta))
        self._last_fl = None
        self._last_rl = None
        self._last_fr = None
        self._last_rr = None
        self._last_time = time.time()
        self.last_slip_ratio_left = 0.0
        self.last_slip_ratio_right = 0.0
        self.total_distance_traveled_m = 0.0
        self.current_v_mps = 0.0
        self.current_omega_radps = 0.0

    def update(self, enc_fl_ticks: int, enc_rl_ticks: int,
               enc_fr_ticks: int, enc_rr_ticks: int,
               left_forward: bool, right_forward: bool,
               timestamp: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Updates odometry from raw cumulative tick counts received from Arduino.
        Returns current pose tuple: (x, y, theta).
        """
        now = timestamp or time.time()
        dt = max(1e-4, now - self._last_time)
        self._last_time = now

        # Initialize tick baselines on first packet
        if self._last_fl is None:
            self._last_fl, self._last_rl = enc_fl_ticks, enc_rl_ticks
            self._last_fr, self._last_rr = enc_fr_ticks, enc_rr_ticks
            return self.x, self.y, self.theta

        # Compute raw tick deltas
        d_fl = max(0, enc_fl_ticks - self._last_fl)
        d_rl = max(0, enc_rl_ticks - self._last_rl)
        d_fr = max(0, enc_fr_ticks - self._last_fr)
        d_rr = max(0, enc_rr_ticks - self._last_rr)

        self._last_fl, self._last_rl = enc_fl_ticks, enc_rl_ticks
        self._last_fr, self._last_rr = enc_fr_ticks, enc_rr_ticks

        # Wheel slip diagnostics (mismatch between front and rear on same side)
        max_l = max(d_fl, d_rl, 1)
        max_r = max(d_fr, d_rr, 1)
        self.last_slip_ratio_left = abs(d_fl - d_rl) / float(max_l)
        self.last_slip_ratio_right = abs(d_fr - d_rr) / float(max_r)

        # Average tick deltas for left and right sides
        d_l_ticks = (d_fl + d_rl) / 2.0
        d_r_ticks = (d_fr + d_rr) / 2.0

        # Convert to signed displacement in meters
        d_l_m = d_l_ticks * self.m_per_tick * (1.0 if left_forward else -1.0)
        d_r_m = d_r_ticks * self.m_per_tick * (1.0 if right_forward else -1.0)

        # Differential drive center displacement and angular rotation
        d_center = (d_l_m + d_r_m) / 2.0
        d_theta = (d_r_m - d_l_m) / self.wheel_base

        # Midpoint Runge-Kutta integration for higher precision along arcs
        mid_theta = self.theta + (d_theta / 2.0)
        self.x += d_center * math.cos(mid_theta)
        self.y += d_center * math.sin(mid_theta)
        self.theta = math.atan2(math.sin(self.theta + d_theta), math.cos(self.theta + d_theta))

        self.total_distance_traveled_m += abs(d_center)
        self.current_v_mps = d_center / dt
        self.current_omega_radps = d_theta / dt

        return self.x, self.y, self.theta

    def pose(self) -> Tuple[float, float, float]:
        """Returns (x, y, theta) in meters and radians."""
        return self.x, self.y, self.theta

    def pose_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "theta_deg": math.degrees(self.theta),
            "v_mps": self.current_v_mps,
            "omega_radps": self.current_omega_radps,
            "total_dist_m": self.total_distance_traveled_m,
            "slip_l": self.last_slip_ratio_left,
            "slip_r": self.last_slip_ratio_right,
        }

    def distance_to(self, target_x: float, target_y: float) -> float:
        """Euclidean distance from current position to target coordinates."""
        return math.hypot(target_x - self.x, target_y - self.y)

    def bearing_to(self, target_x: float, target_y: float) -> float:
        """Relative angle (radians) from current heading to target point."""
        target_heading = math.atan2(target_y - self.y, target_x - self.x)
        err = target_heading - self.theta
        return math.atan2(math.sin(err), math.cos(err))
