"""
Unit tests for Reinforcement Learning & Trajectory Optimization module.
"""

import math
import os
import tempfile
import unittest
from rl_agent import RLTrajectoryOptimizer, ACTION_PRIMITIVES
from config import RLConfig, NavConfig


class TestRLAgent(unittest.TestCase):
    def setUp(self):
        self.rl = RLTrajectoryOptimizer()

    def test_state_discretization(self):
        # Discretize position (0.5, 0.2), theta=0, clear obstacles, slope 5 deg (uphill), target dead ahead
        state = self.rl.discretize_state(
            x=0.5, y=0.2, theta=0.0,
            us1_cm=150.0, us2_cm=150.0,
            slope_deg=5.0,
            vision_bearing=0.0
        )
        self.assertEqual(len(state), 6)
        self.assertEqual(state[0], 5)   # 0.5m / 0.1m grid = 5
        self.assertEqual(state[1], 2)   # 0.2m / 0.1m grid = 2
        self.assertEqual(state[2], 0)   # theta=0 heading sector = 0
        self.assertEqual(state[3], 0)   # Obstacles clear = 0
        self.assertEqual(state[4], 1)   # Uphill incline = 1
        self.assertEqual(state[5], 2)   # Target center ahead = 2

    def test_q_learning_update(self):
        state = (0, 0, 0, 0, 0, 0)
        next_state = (1, 0, 0, 0, 0, 0)
        action_idx = 1  # Fast forward

        initial_q = float(self.rl.get_q_values(state)[action_idx])
        reward = 10.0

        self.rl.update_q_value(state, action_idx, reward, next_state, done=False)
        updated_q = float(self.rl.get_q_values(state)[action_idx])
        self.assertGreater(updated_q, initial_q)

    def test_reward_computation(self):
        # Progress reward
        self.rl.prev_distance_to_goal = 2.0
        # Robot moved closer: distance is now 1.8m
        r = self.rl.compute_reward(
            current_x=0.2, current_y=0.0,
            goal_x=2.0, goal_y=0.0,
            us1_cm=100.0, us2_cm=100.0,
            goal_reached=False,
            action_pwm=(170, 170),
            slope_deg=0.0,
            slip_ratio=0.0,
            vision_bearing=0.0
        )
        self.assertGreater(r, 0.0)

        # Goal reached reward
        r_goal = self.rl.compute_reward(
            current_x=2.0, current_y=0.0,
            goal_x=2.0, goal_y=0.0,
            us1_cm=100.0, us2_cm=100.0,
            goal_reached=True,
            action_pwm=(0, 0),
        )
        self.assertGreaterEqual(r_goal, 500.0)

        # Collision penalty
        r_obs = self.rl.compute_reward(
            current_x=0.5, current_y=0.0,
            goal_x=2.0, goal_y=0.0,
            us1_cm=15.0, us2_cm=18.0,  # Obstacle < 20cm
            goal_reached=False,
            action_pwm=(170, 170),
        )
        self.assertLess(r_obs, -100.0)

    def test_trajectory_recording_and_best_route_synthesis(self):
        test_file = tempfile.mktemp(suffix=".json")
        self.rl.cfg.best_route_file = test_file

        # Simulate Round 1 (10 steps)
        self.rl.start_round(1)
        for i in range(10):
            t = i * 0.05
            self.rl.record_step(
                t_sec=t, x=i*0.2, y=0.0, theta=0.0,
                left_pwm=180, right_pwm=180,
                v_mps=0.4, omega_radps=0.0,
                reward=5.0, obstacle_min_cm=100.0,
                slope_deg=2.0, slip_ratio=0.01,
            )
        self.rl.end_round(1, goal_reached=True)

        # Synthesize best route
        best_data = self.rl.synthesize_best_route()
        self.assertIsNotNone(best_data)
        self.assertTrue(os.path.exists(test_file))
        self.assertEqual(len(best_data["commands"]), 10)

        # Load back
        loaded = RLTrajectoryOptimizer.load_best_route(test_file)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["metadata"]["selected_round"], 1)

        if os.path.exists(test_file):
            os.remove(test_file)


if __name__ == "__main__":
    unittest.main()
