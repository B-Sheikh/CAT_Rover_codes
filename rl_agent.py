"""
rl_agent.py
=====================================================================
Reinforcement Learning & Trajectory Optimization Engine for the Pi 5.

Key Components:
  1. 2.5D Terrain & Elevation Costmap Integration
  2. Spatial Discretization & State-Action Feature Map
  3. Q-Learning with Epsilon-Greedy Exploration & Experience Replay
  4. Multi-objective Reward Shaping:
     - Goal Reached (+500) & Potential-based Progress
     - Obstacle Distance Penalty & Safety Inflation Field
     - Terrain Elevation & Slope Gradient Cost
     - Wheel-Slip Traction Penalty
     - Steering Jerk / Smoothness Penalty
     - Vision Target Alignment (+2.5 * cos(bearing))
  5. Cross-Round Trajectory Evaluation & Elite Selection
  6. Slope-Compensated Feedforward PWM Synthesis for Sensorless Replay
  7. Export & Import of 'best_route.json'
=====================================================================
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any

import numpy as np

from config import DEFAULT_CONFIG, RLConfig, NavConfig, RobotConfig, TerrainConfig
from terrain_map import TerrainMap25D, SlopeEstimator, TerrainPointInfo


@dataclass
class TrajectoryPoint:
    t_sec: float
    x: float
    y: float
    theta: float
    left_pwm: int
    right_pwm: int
    v_mps: float = 0.0
    omega_radps: float = 0.0
    reward: float = 0.0
    obstacle_min_cm: float = -1.0
    slope_deg: float = 0.0
    slip_ratio: float = 0.0
    vision_bearing_rad: Optional[float] = None


@dataclass
class EpisodeSummary:
    round_number: int
    total_reward: float
    lap_time_sec: float
    path_length_m: float
    avg_speed_mps: float
    min_obstacle_dist_cm: float
    max_slope_deg: float
    avg_slip_ratio: float
    goal_reached: bool
    num_steps: int
    trajectory: List[TrajectoryPoint] = field(default_factory=list)


# Discrete Action Motion Primitives (left_pwm, right_pwm, label)
ACTION_PRIMITIVES = [
    (170, 170, "CRUISE_FORWARD"),   # Action 0: Normal cruise
    (220, 220, "FAST_FORWARD"),     # Action 1: High-speed straight sprint
    (130, 190, "GENTLE_LEFT"),      # Action 2: Smooth left curve
    (190, 130, "GENTLE_RIGHT"),     # Action 3: Smooth right curve
    (-120, 160, "SHARP_LEFT"),      # Action 4: Pivot/sharp left (avoidance)
    (160, -120, "SHARP_RIGHT"),     # Action 5: Pivot/sharp right (avoidance)
    (110, 110, "PRECISION_CREEP"),  # Action 6: Slow precision approach near dot
]


class RLTrajectoryOptimizer:
    """
    Online RL agent that learns the optimal path across exploration rounds,
    incorporating 2.5D elevation/slope analysis and obstacle costmaps,
    and synthesizes a slope-compensated sensorless replay trajectory.
    """
    def __init__(self, rl_cfg: Optional[RLConfig] = None,
                 nav_cfg: Optional[NavConfig] = None,
                 robot_cfg: Optional[RobotConfig] = None,
                 terrain_cfg: Optional[TerrainConfig] = None):
        self.cfg = rl_cfg or DEFAULT_CONFIG.rl
        self.ncfg = nav_cfg or DEFAULT_CONFIG.nav
        self.rcfg = robot_cfg or DEFAULT_CONFIG.robot
        self.tcfg = terrain_cfg or DEFAULT_CONFIG.terrain

        self.terrain_map = TerrainMap25D(terrain_cfg=self.tcfg)
        self.slope_estimator = SlopeEstimator(robot_cfg=self.rcfg, terrain_cfg=self.tcfg)

        self.q_table: Dict[Tuple, np.ndarray] = {}
        self.num_actions = len(ACTION_PRIMITIVES)
        self.current_epsilon = self.cfg.initial_epsilon

        # Trajectory History across all rounds
        self.episodes: List[EpisodeSummary] = []
        self.current_trajectory: List[TrajectoryPoint] = []
        self.episode_start_time: float = time.time()
        self.prev_distance_to_goal: Optional[float] = None
        self.prev_action_pwm: Tuple[int, int] = (0, 0)

        # Best synthesized trajectory
        self.best_episode: Optional[EpisodeSummary] = None

    def discretize_state(self, x: float, y: float, theta: float,
                         us1_cm: float, us2_cm: float,
                         slope_deg: float = 0.0,
                         vision_bearing: Optional[float] = None) -> Tuple:
        """
        Discretizes robot pose, obstacle proximity, terrain slope, and vision.
        """
        res = self.cfg.grid_resolution_m
        gx = int(round(x / res))
        gy = int(round(y / res))

        norm_theta = math.atan2(math.sin(theta), math.cos(theta))
        if norm_theta < 0:
            norm_theta += 2.0 * math.pi
        sector_size = (2.0 * math.pi) / self.cfg.heading_bins
        heading_bin = int(norm_theta / sector_size) % self.cfg.heading_bins

        # Obstacle State (0: Clear, 1: Left obstacle, 2: Right obstacle, 3: Both)
        obs_state = 0
        min_front = min(v for v in (us1_cm, us2_cm) if v > 0) if (us1_cm > 0 or us2_cm > 0) else 999.0
        if min_front < self.ncfg.obstacle_slow_cm:
            if us1_cm > 0 and us1_cm < self.ncfg.obstacle_slow_cm and (us2_cm < 0 or us1_cm < us2_cm):
                obs_state = 1
            elif us2_cm > 0 and us2_cm < self.ncfg.obstacle_slow_cm and (us1_cm < 0 or us2_cm < us1_cm):
                obs_state = 2
            else:
                obs_state = 3

        # Elevation / Slope State (0: Flat, 1: Uphill incline, 2: Downhill decline)
        elev_state = 0
        if slope_deg > 4.0:
            elev_state = 1  # Uphill
        elif slope_deg < -4.0:
            elev_state = 2  # Downhill

        # Vision Target State (0: Not visible, 1: Left, 2: Center, 3: Right)
        vis_state = 0
        if vision_bearing is not None:
            if abs(vision_bearing) < math.radians(10):
                vis_state = 2  # Dead ahead
            elif vision_bearing < 0:
                vis_state = 1  # Target to the left
            else:
                vis_state = 3  # Target to the right

        return (gx, gy, heading_bin, obs_state, elev_state, vis_state)

    def get_q_values(self, state: Tuple) -> np.ndarray:
        if state not in self.q_table:
            q = np.zeros(self.num_actions, dtype=np.float32)
            q[0] = 0.5  # Cruise forward
            q[1] = 0.8  # Fast forward
            self.q_table[state] = q
        return self.q_table[state]

    def select_action(self, state: Tuple, explore: bool = True) -> int:
        """Epsilon-greedy action selection."""
        q_vals = self.get_q_values(state)
        if explore and np.random.rand() < self.current_epsilon:
            return int(np.random.randint(self.num_actions))
        return int(np.argmax(q_vals))

    def compute_reward(self, current_x: float, current_y: float,
                       goal_x: float, goal_y: float,
                       us1_cm: float, us2_cm: float,
                       goal_reached: bool,
                       action_pwm: Tuple[int, int],
                       slope_deg: float = 0.0,
                       slip_ratio: float = 0.0,
                       vision_bearing: Optional[float] = None) -> float:
        """
        Computes composite reward with shaped potential, time penalty,
        smoothness jerk penalty, obstacle proximity penalty, elevation/slope cost,
        slip penalty, and vision alignment.
        """
        if goal_reached:
            return self.cfg.reward_goal_reached

        r = self.cfg.reward_step_penalty
        dist_to_goal = math.hypot(goal_x - current_x, goal_y - current_y)

        # 1. Distance Progress Reward (Potential-based shaping)
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - dist_to_goal
            r += progress * self.cfg.reward_progress_weight
        self.prev_distance_to_goal = dist_to_goal

        # 2. Obstacle Proximity & Costmap Penalty
        front_min = min(v for v in (us1_cm, us2_cm) if v > 0) if (us1_cm > 0 or us2_cm > 0) else 999.0
        if front_min < self.ncfg.obstacle_stop_cm:
            r += self.cfg.reward_collision
        elif front_min < self.ncfg.obstacle_slow_cm:
            proximity_factor = (self.ncfg.obstacle_slow_cm - front_min) / (self.ncfg.obstacle_slow_cm - self.ncfg.obstacle_stop_cm)
            r += self.cfg.reward_obstacle_proximity_weight * (proximity_factor ** 2)

        # 3. Elevation & Terrain Slope Penalty (penalizes steep energy-wasting slopes)
        if abs(slope_deg) > 3.0:
            r += self.cfg.reward_slope_penalty_weight * (abs(slope_deg) / 10.0)

        # 4. Wheel Slip Penalty
        if slip_ratio > 0.15:
            r -= self.tcfg.slip_penalty_weight * slip_ratio

        # 5. Steering Jerk / Smoothness Penalty
        prev_l, prev_r = self.prev_action_pwm
        curr_l, curr_r = action_pwm
        jerk = abs(curr_l - prev_l) + abs(curr_r - prev_r)
        r += self.cfg.reward_jerk_penalty_weight * (jerk / 255.0)
        self.prev_action_pwm = action_pwm

        # 6. Vision Target Alignment Reward
        if vision_bearing is not None:
            r += self.cfg.reward_vision_alignment_weight * math.cos(vision_bearing)

        return r

    def update_q_value(self, state: Tuple, action_idx: int,
                       reward: float, next_state: Tuple, done: bool):
        """Temporal Difference Q-learning update."""
        q_vals = self.get_q_values(state)
        next_q_vals = self.get_q_values(next_state)
        best_next_q = 0.0 if done else np.max(next_q_vals)

        td_target = reward + self.cfg.discount_factor * best_next_q
        td_error = td_target - q_vals[action_idx]
        q_vals[action_idx] += self.cfg.learning_rate * td_error
        self.q_table[state] = q_vals

    def record_step(self, t_sec: float, x: float, y: float, theta: float,
                    left_pwm: int, right_pwm: int, v_mps: float,
                    omega_radps: float, reward: float,
                    obstacle_min_cm: float,
                    slope_deg: float = 0.0,
                    slip_ratio: float = 0.0,
                    vision_bearing: Optional[float] = None):
        """Records telemetry and updates 2.5D terrain grid."""
        # Update spatial terrain & elevation grid
        self.terrain_map.update_cell_telemetry(x, y, slope_deg, slip_ratio)

        pt = TrajectoryPoint(
            t_sec=t_sec,
            x=x,
            y=y,
            theta=theta,
            left_pwm=left_pwm,
            right_pwm=right_pwm,
            v_mps=v_mps,
            omega_radps=omega_radps,
            reward=reward,
            obstacle_min_cm=obstacle_min_cm,
            slope_deg=slope_deg,
            slip_ratio=slip_ratio,
            vision_bearing_rad=vision_bearing,
        )
        self.current_trajectory.append(pt)

    def start_round(self, round_num: int, start_time: Optional[float] = None):
        """Prepares agent for a new exploration round."""
        self.current_trajectory = []
        self.episode_start_time = start_time if start_time is not None else time.time()
        self.prev_distance_to_goal = None
        self.prev_action_pwm = (0, 0)
        decay = self.cfg.epsilon_decay_per_round ** (round_num - 1)
        self.current_epsilon = max(self.cfg.min_epsilon, self.cfg.initial_epsilon * decay)
        print(f"[RL Agent] Starting Round {round_num} (Exploration Epsilon = {self.current_epsilon:.3f})")

    def end_round(self, round_num: int, goal_reached: bool, end_time: Optional[float] = None) -> EpisodeSummary:
        """Finalizes round and computes comprehensive terrain metrics."""
        now = end_time if end_time is not None else time.time()
        lap_time = max(0.1, now - self.episode_start_time)
        total_reward = sum(p.reward for p in self.current_trajectory)

        path_length = 0.0
        min_obs = 999.0
        max_slope = 0.0
        total_slip = 0.0

        for i in range(len(self.current_trajectory)):
            pt = self.current_trajectory[i]
            max_slope = max(max_slope, abs(pt.slope_deg))
            total_slip += pt.slip_ratio
            if pt.obstacle_min_cm > 0:
                min_obs = min(min_obs, pt.obstacle_min_cm)

            if i > 0:
                p0 = self.current_trajectory[i - 1]
                path_length += math.hypot(pt.x - p0.x, pt.y - p0.y)

        avg_speed = path_length / max(0.1, lap_time)
        avg_slip = total_slip / max(1, len(self.current_trajectory))

        summary = EpisodeSummary(
            round_number=round_num,
            total_reward=total_reward,
            lap_time_sec=lap_time,
            path_length_m=path_length,
            avg_speed_mps=avg_speed,
            min_obstacle_dist_cm=min_obs if min_obs < 900 else -1.0,
            max_slope_deg=max_slope,
            avg_slip_ratio=avg_slip,
            goal_reached=goal_reached,
            num_steps=len(self.current_trajectory),
            trajectory=list(self.current_trajectory),
        )
        self.episodes.append(summary)

        print(f"[RL Agent] Round {round_num} Complete -> Reward: {total_reward:.1f} | "
              f"Time: {lap_time:.2f}s | Dist: {path_length:.2f}m | MaxSlope: {max_slope:.1f}° | Reached: {goal_reached}")

        return summary

    def synthesize_best_route(self) -> Optional[Dict[str, Any]]:
        """
        Selects the best performing exploration round, applies trajectory
        smoothing, and generates slope-compensated feedforward PWM commands.
        """
        valid_episodes = [ep for ep in self.episodes if ep.goal_reached and len(ep.trajectory) > 5]
        if not valid_episodes:
            valid_episodes = [ep for ep in self.episodes if len(ep.trajectory) > 5]

        if not valid_episodes:
            print("[RL Agent] Warning: No valid trajectory recorded to synthesize best route.")
            return None

        # Multi-criteria scoring: Reward, time, path length, and elevation smoothness
        def score_episode(ep: EpisodeSummary) -> float:
            terrain_penalty = 0.1 * ep.max_slope_deg + 10.0 * ep.avg_slip_ratio
            score = ep.total_reward / (1.0 + 0.1 * ep.lap_time_sec + 0.05 * ep.path_length_m + terrain_penalty)
            if ep.goal_reached:
                score += 1000.0
            return score

        best_ep = max(valid_episodes, key=score_episode)
        self.best_episode = best_ep
        print(f"[RL Agent] Selected Round {best_ep.round_number} as Optimal Route Template (Score: {score_episode(best_ep):.1f}).")

        raw_pts = best_ep.trajectory
        t_start = raw_pts[0].t_sec
        smoothed_commands: List[Dict[str, Any]] = []

        window = 3
        for i in range(len(raw_pts)):
            i_start = max(0, i - window // 2)
            i_end = min(len(raw_pts), i + window // 2 + 1)
            avg_l = int(np.mean([raw_pts[k].left_pwm for k in range(i_start, i_end)]))
            avg_r = int(np.mean([raw_pts[k].right_pwm for k in range(i_start, i_end)]))
            slope = float(np.mean([raw_pts[k].slope_deg for k in range(i_start, i_end)]))

            # Apply slope feedforward compensation for hills/ramps
            comp_l = self.slope_estimator.compute_slope_compensated_pwm(avg_l, slope)
            comp_r = self.slope_estimator.compute_slope_compensated_pwm(avg_r, slope)

            rel_t = raw_pts[i].t_sec - t_start
            smoothed_commands.append({
                "time_sec": round(rel_t, 3),
                "left_pwm": comp_l,
                "right_pwm": comp_r,
                "raw_left_pwm": avg_l,
                "raw_right_pwm": avg_r,
                "slope_deg": round(slope, 1),
                "x": round(raw_pts[i].x, 3),
                "y": round(raw_pts[i].y, 3),
                "theta": round(raw_pts[i].theta, 3),
                "v_mps": round(raw_pts[i].v_mps, 3),
                "omega_radps": round(raw_pts[i].omega_radps, 3),
            })

        best_route_data = {
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "selected_round": best_ep.round_number,
                "total_reward": best_ep.total_reward,
                "lap_time_sec": round(best_ep.lap_time_sec, 2),
                "path_length_m": round(best_ep.path_length_m, 2),
                "max_slope_deg": round(best_ep.max_slope_deg, 1),
                "avg_slip_ratio": round(best_ep.avg_slip_ratio, 3),
                "num_command_steps": len(smoothed_commands),
                "sampling_period_sec": round(DEFAULT_CONFIG.nav.control_loop_hz ** -1, 3),
            },
            "commands": smoothed_commands,
        }

        filepath = self.cfg.best_route_file
        with open(filepath, "w") as f:
            json.dump(best_route_data, f, indent=2)

        print(f"[RL Agent] Optimal route synthesized & saved to '{filepath}' ({len(smoothed_commands)} slope-compensated steps).")
        return best_route_data

    @staticmethod
    def load_best_route(filepath: str = "best_route.json") -> Optional[Dict[str, Any]]:
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[RL Agent] Error loading best route from {filepath}: {e}")
            return None
