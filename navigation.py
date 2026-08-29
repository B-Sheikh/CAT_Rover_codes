"""
navigation.py
=====================================================================
Multi-Modal Navigation Controller & Multi-Round Mission Manager.

Features:
  1. 2.5D Terrain & Elevation Costmap Integration
  2. Exploration Mode:
     - RL Action Selection (with Terrain Slope & Obstacle State)
     - Visual Target Guidance (Green Start, Red Goal)
     - Dynamic Obstacle Avoidance Override
     - Online Elevation & Slope Gradient Analysis
  3. Sensorless Replay Mode:
     - Pure Feedforward Time-Indexed Replay with Slope PWM Compensation
     - Zero Sensor Dependency (0 Camera, 0 Ultrasonic Calls)
  4. MultiRoundMissionManager Automating Full Lifecycle
=====================================================================
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List

from config import DEFAULT_CONFIG, NavConfig, RLConfig, RobotConfig, TerrainConfig
from rl_agent import RLTrajectoryOptimizer, ACTION_PRIMITIVES, EpisodeSummary


class MissionState(Enum):
    INIT = "INIT"
    EXPLORE_TO_RED_GOAL = "EXPLORE_TO_RED_GOAL"
    REACHED_RED_GOAL = "REACHED_RED_GOAL"
    RETURN_TO_GREEN_START = "RETURN_TO_GREEN_START"
    LAP_COMPLETED = "LAP_COMPLETED"
    OPTIMIZATION_PHASE = "OPTIMIZATION_PHASE"
    SENSORLESS_REPLAY = "SENSORLESS_REPLAY"
    MISSION_FINISHED = "MISSION_FINISHED"


@dataclass
class NavCommand:
    left_speed: int                 # 0..255 PWM magnitude
    right_speed: int                # 0..255 PWM magnitude
    left_forward: bool              # True for forward, False for reverse
    right_forward: bool             # True for forward, False for reverse
    reached: bool                   # True if goal or replay phase completed
    mode: str = "EXPLORATION"       # "EXPLORATION", "OBSTACLE_AVOID", "SENSORLESS_REPLAY"
    action_idx: int = 0
    action_name: str = "CRUISE"
    slope_deg: float = 0.0          # Estimated terrain slope in degrees


def normalize_angle(rad: float) -> float:
    """Normalizes angle to [-pi, pi]."""
    return math.atan2(math.sin(rad), math.cos(rad))


def compute_heading_error(current_theta: float, target_x: float, target_y: float,
                          cur_x: float, cur_y: float) -> float:
    """Computes relative heading error to target coordinates."""
    desired_theta = math.atan2(target_y - cur_y, target_x - cur_x)
    return normalize_angle(desired_theta - current_theta)


class NavigationController:
    """
    Computes low-level motor commands for Exploration, Obstacle Avoidance,
    and Sensorless Replay modes.
    """
    def __init__(self, nav_cfg: Optional[NavConfig] = None,
                 robot_cfg: Optional[RobotConfig] = None,
                 terrain_cfg: Optional[TerrainConfig] = None):
        self.cfg = nav_cfg or DEFAULT_CONFIG.nav
        self.rcfg = robot_cfg or DEFAULT_CONFIG.robot
        self.tcfg = terrain_cfg or DEFAULT_CONFIG.terrain

    def compute_exploration_command(self, cur_x: float, cur_y: float, cur_theta: float,
                                    target_x: float, target_y: float,
                                    us1_cm: float, us2_cm: float,
                                    slope_deg: float = 0.0,
                                    vision_bearing: Optional[float] = None,
                                    rl_action_idx: int = 0) -> NavCommand:
        """
        Computes motor command during exploration:
          1. Obstacle avoidance takes absolute precedence.
          2. Blends RL action with visual servoing bearing, distance scaling, and slope compensation.
        """
        dist_to_target = math.hypot(target_x - cur_x, target_y - cur_y)
        if dist_to_target <= self.cfg.goal_tolerance_m:
            return NavCommand(0, 0, True, True, reached=True, mode="TARGET_REACHED", slope_deg=slope_deg)

        # 1. Reactive Obstacle Avoidance Override
        front_min = min(v for v in (us1_cm, us2_cm) if v > 0) if (us1_cm > 0 or us2_cm > 0) else 999.0

        if front_min < self.cfg.obstacle_stop_cm:
            # Pivot away from the more-blocked sensor
            left_clearer = (us1_cm < 0 or us1_cm > us2_cm)
            if left_clearer:
                return NavCommand(
                    left_speed=130, right_speed=130,
                    left_forward=False, right_forward=True,
                    reached=False, mode="OBSTACLE_AVOID",
                    action_idx=4, action_name="PIVOT_LEFT",
                    slope_deg=slope_deg
                )
            else:
                return NavCommand(
                    left_speed=130, right_speed=130,
                    left_forward=True, right_forward=False,
                    reached=False, mode="OBSTACLE_AVOID",
                    action_idx=5, action_name="PIVOT_RIGHT",
                    slope_deg=slope_deg
                )

        # 2. RL Action Base Speeds
        act_l, act_r, act_name = ACTION_PRIMITIVES[rl_action_idx]

        # 3. Heading & Visual Servoing Correction
        odometry_heading_err = compute_heading_error(cur_theta, target_x, target_y, cur_x, cur_y)
        if vision_bearing is not None:
            effective_err = ((1.0 - self.cfg.vision_blend_weight) * odometry_heading_err +
                             self.cfg.vision_blend_weight * vision_bearing)
        else:
            effective_err = odometry_heading_err

        turn_offset = self.cfg.turn_gain * effective_err

        # Speed scaling
        speed_scale = 1.0
        if front_min < self.cfg.obstacle_slow_cm:
            speed_scale = min(speed_scale, (front_min - self.cfg.obstacle_stop_cm) /
                              (self.cfg.obstacle_slow_cm - self.cfg.obstacle_stop_cm))
        if dist_to_target < 0.45:
            speed_scale = min(speed_scale, max(0.4, dist_to_target / 0.45))
        speed_scale = max(0.35, min(1.0, speed_scale))

        # Slope Gravity Compensation
        slope_boost = int(self.tcfg.slope_gravity_pwm_gain * slope_deg)

        cmd_l = (act_l * speed_scale) - (turn_offset * 0.5) + slope_boost
        cmd_r = (act_r * speed_scale) + (turn_offset * 0.5) + slope_boost

        cmd_l = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, cmd_l))
        cmd_r = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, cmd_r))

        return NavCommand(
            left_speed=int(abs(cmd_l)),
            right_speed=int(abs(cmd_r)),
            left_forward=(cmd_l >= 0),
            right_forward=(cmd_r >= 0),
            reached=False,
            mode="RL_EXPLORATION",
            action_idx=rl_action_idx,
            action_name=act_name,
            slope_deg=slope_deg,
        )

    def compute_sensorless_replay_step(self, elapsed_sec: float,
                                       best_route: Dict[str, Any]) -> NavCommand:
        """
        Executes purely time-synchronized open-loop feedforward commands
        from best_route.json with zero sensor dependency.
        """
        commands = best_route.get("commands", [])
        if not commands:
            return NavCommand(0, 0, True, True, reached=True, mode="SENSORLESS_REPLAY")

        total_duration = commands[-1]["time_sec"]
        if elapsed_sec >= total_duration:
            return NavCommand(0, 0, True, True, reached=True, mode="REPLAY_LAP_COMPLETE")

        idx = 0
        while idx < len(commands) - 1 and commands[idx + 1]["time_sec"] <= elapsed_sec:
            idx += 1

        c0 = commands[idx]
        if idx == len(commands) - 1:
            cmd_l = c0["left_pwm"]
            cmd_r = c0["right_pwm"]
            slope = c0.get("slope_deg", 0.0)
        else:
            c1 = commands[idx + 1]
            dt = max(1e-4, c1["time_sec"] - c0["time_sec"])
            alpha = max(0.0, min(1.0, (elapsed_sec - c0["time_sec"]) / dt))
            cmd_l = int(c0["left_pwm"] + alpha * (c1["left_pwm"] - c0["left_pwm"]))
            cmd_r = int(c0["right_pwm"] + alpha * (c1["right_pwm"] - c0["right_pwm"]))
            slope = float(c0.get("slope_deg", 0.0) + alpha * (c1.get("slope_deg", 0.0) - c0.get("slope_deg", 0.0)))

        cmd_l = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, cmd_l))
        cmd_r = max(-self.cfg.max_pwm, min(self.cfg.max_pwm, cmd_r))

        return NavCommand(
            left_speed=int(abs(cmd_l)),
            right_speed=int(abs(cmd_r)),
            left_forward=(cmd_l >= 0),
            right_forward=(cmd_r >= 0),
            reached=False,
            mode="SENSORLESS_REPLAY",
            action_name="BLIND_FEEDFORWARD",
            slope_deg=slope,
        )


class MultiRoundMissionManager:
    """
    High-level state machine automating the complete mission:
      - Rounds 1..N: Exploration with Green/Red dot tracking + 2.5D elevation/obstacle analysis.
      - Trajectory Optimization: Synthesizes slope-compensated 'best_route.json'.
      - Sensorless Replay: Infinite or N-lap blind replay of the optimal route.
    """
    def __init__(self, rl_agent: RLTrajectoryOptimizer,
                 nav_controller: NavigationController,
                 green_start_pos: Tuple[float, float] = (0.0, 0.0),
                 red_goal_pos: Tuple[float, float] = (2.0, 0.0),
                 exploration_rounds: int = 3,
                 replay_laps: int = 5):
        self.rl = rl_agent
        self.nav = nav_controller
        self.green_pos = green_start_pos
        self.red_pos = red_goal_pos
        self.total_exploration_rounds = exploration_rounds
        self.total_replay_laps = replay_laps

        self.current_round = 1
        self.current_replay_lap = 1
        self.state = MissionState.INIT
        self.round_start_time = time.time()
        self.replay_start_time = 0.0
        self.best_route_data: Optional[Dict[str, Any]] = None

    def start_mission(self, start_time: Optional[float] = None):
        """Initializes Round 1 of exploration."""
        self.current_round = 1
        self.current_replay_lap = 1
        self.state = MissionState.EXPLORE_TO_RED_GOAL
        self.round_start_time = start_time if start_time is not None else time.time()
        self.rl.start_round(self.current_round, start_time=self.round_start_time)
        print(f"\n{'='*60}")
        print(f"  STARTING AUTONOMOUS MISSION: ROUND 1 / {self.total_exploration_rounds}")
        print(f"  Target: RED Goal at {self.red_pos} | Start: GREEN at {self.green_pos}")
        print(f"{'='*60}\n")

    def step(self, cur_x: float, cur_y: float, cur_theta: float,
             v_mps: float, omega_radps: float,
             us1_cm: float, us2_cm: float,
             slip_ratio: float = 0.0,
             imu_pitch_deg: Optional[float] = None,
             green_bearing: Optional[float] = None,
             red_bearing: Optional[float] = None,
             current_time: Optional[float] = None) -> NavCommand:
        """
        Executes one control step in the current mission state.
        """
        now = current_time if current_time is not None else time.time()
        elapsed_in_round = now - self.round_start_time

        # ----------------------------------------------------
        # 1. EXPLORATION PHASES (Rounds 1 .. N)
        # ----------------------------------------------------
        if self.state == MissionState.EXPLORE_TO_RED_GOAL:
            target_x, target_y = self.red_pos
            vision_bearing = red_bearing

            # Estimate terrain slope / elevation gradient
            prev_l, prev_r = self.rl.prev_action_pwm
            slope_deg = self.rl.slope_estimator.estimate_slope(
                left_pwm=prev_l, right_pwm=prev_r,
                v_mps=v_mps, imu_pitch_deg=imu_pitch_deg,
                slip_ratio=slip_ratio
            )

            # Update 2.5D obstacle map from ultrasonic sensors
            self.rl.terrain_map.update_ultrasonic_ray(cur_x, cur_y, cur_theta, us1_cm, us2_cm)

            # Discretize state for RL (including slope/elevation state)
            rl_state = self.rl.discretize_state(
                cur_x, cur_y, cur_theta,
                us1_cm, us2_cm,
                slope_deg=slope_deg,
                vision_bearing=vision_bearing
            )
            action_idx = self.rl.select_action(rl_state, explore=True)

            cmd = self.nav.compute_exploration_command(
                cur_x, cur_y, cur_theta,
                target_x, target_y,
                us1_cm, us2_cm,
                slope_deg=slope_deg,
                vision_bearing=vision_bearing,
                rl_action_idx=action_idx,
            )

            # Compute RL reward and record telemetry
            min_front = min(v for v in (us1_cm, us2_cm) if v > 0) if (us1_cm > 0 or us2_cm > 0) else -1.0
            reward = self.rl.compute_reward(
                cur_x, cur_y, target_x, target_y,
                us1_cm, us2_cm, cmd.reached,
                action_pwm=(cmd.left_speed, cmd.right_speed),
                slope_deg=slope_deg,
                slip_ratio=slip_ratio,
                vision_bearing=vision_bearing,
            )
            self.rl.record_step(
                t_sec=now, x=cur_x, y=cur_y, theta=cur_theta,
                left_pwm=cmd.left_speed if cmd.left_forward else -cmd.left_speed,
                right_pwm=cmd.right_speed if cmd.right_forward else -cmd.right_speed,
                v_mps=v_mps, omega_radps=omega_radps,
                reward=reward, obstacle_min_cm=min_front,
                slope_deg=slope_deg, slip_ratio=slip_ratio,
                vision_bearing=vision_bearing,
            )

            if cmd.reached:
                print(f"[Mission] RED GOAL REACHED in Round {self.current_round} at ({cur_x:.2f}, {cur_y:.2f})!")
                self.rl.end_round(self.current_round, goal_reached=True, end_time=now)
                self._advance_to_next_round_or_replay(current_time=now)
                return NavCommand(0, 0, True, True, reached=True, mode="GOAL_REACHED", slope_deg=slope_deg)

            # Check timeout
            if len(self.rl.current_trajectory) >= self.rl.cfg.max_trajectory_steps:
                print(f"[Mission] Round {self.current_round} reached max step limit. Wrapping up round.")
                self.rl.end_round(self.current_round, goal_reached=False, end_time=now)
                self._advance_to_next_round_or_replay(current_time=now)
                return NavCommand(0, 0, True, True, reached=True, mode="ROUND_TIMEOUT", slope_deg=slope_deg)

            return cmd

        # ----------------------------------------------------
        # 2. SENSORLESS REPLAY PHASE
        # ----------------------------------------------------
        elif self.state == MissionState.SENSORLESS_REPLAY:
            elapsed_replay = now - self.replay_start_time
            cmd = self.nav.compute_sensorless_replay_step(elapsed_replay, self.best_route_data)

            if cmd.reached:
                print(f"\n[Mission] SENSORLESS REPLAY LAP {self.current_replay_lap} / {self.total_replay_laps} COMPLETED!")
                if self.current_replay_lap < self.total_replay_laps:
                    self.current_replay_lap += 1
                    self.replay_start_time = now
                    print(f"[Mission] Starting Sensorless Replay Lap {self.current_replay_lap} ...")
                    return NavCommand(0, 0, True, True, reached=False, mode="REPLAY_START_NEXT_LAP")
                else:
                    print(f"\n{'='*60}")
                    print("  ALL SENSORLESS REPLAY LAPS COMPLETED SUCCESSFULLY!")
                    print(f"{'='*60}\n")
                    self.state = MissionState.MISSION_FINISHED
                    return NavCommand(0, 0, True, True, reached=True, mode="MISSION_FINISHED")

            return cmd

        return NavCommand(0, 0, True, True, reached=True, mode="IDLE")

    def _advance_to_next_round_or_replay(self, current_time: Optional[float] = None):
        """Advances round count or triggers the sensorless replay phase."""
        now = current_time if current_time is not None else time.time()
        if self.current_round < self.total_exploration_rounds:
            self.current_round += 1
            self.round_start_time = now
            self.state = MissionState.EXPLORE_TO_RED_GOAL
            self.rl.start_round(self.current_round, start_time=now)
            print(f"\n{'='*60}")
            print(f"  STARTING EXPLORATION ROUND {self.current_round} / {self.total_exploration_rounds}")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("  ALL EXPLORATION ROUNDS COMPLETE! SYNTHESIZING OPTIMAL ROUTE...")
            print(f"{'='*60}\n")
            self.state = MissionState.OPTIMIZATION_PHASE
            self.best_route_data = self.rl.synthesize_best_route()

            if self.best_route_data is not None:
                self.state = MissionState.SENSORLESS_REPLAY
                self.current_replay_lap = 1
                self.replay_start_time = now
                print(f"\n{'*'*60}")
                print("  SWITCHING TO SENSORLESS BLIND REPLAY MODE (0 SENSOR DEPENDENCY)")
                print(f"  Executing {self.total_replay_laps} Laps of the Optimal Route...")
                print(f"{'*'*60}\n")
            else:
                print("[Mission] Error: Failed to synthesize best route. Mission stopped.")
                self.state = MissionState.MISSION_FINISHED
