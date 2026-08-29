"""
Unit tests for Navigation Controller and Mission Manager.
"""

import unittest
from navigation import NavigationController, NavCommand, compute_heading_error
from config import NavConfig, RobotConfig


class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.nav = NavigationController()

    def test_goal_reached(self):
        # Already at goal
        cmd = self.nav.compute_exploration_command(
            cur_x=2.0, cur_y=0.0, cur_theta=0.0,
            target_x=2.0, target_y=0.0,
            us1_cm=100.0, us2_cm=100.0,
        )
        self.assertTrue(cmd.reached)
        self.assertEqual(cmd.left_speed, 0)
        self.assertEqual(cmd.right_speed, 0)

    def test_obstacle_avoidance_override(self):
        # Obstacle right in front (15cm) on left sensor -> should pivot right
        cmd = self.nav.compute_exploration_command(
            cur_x=0.0, cur_y=0.0, cur_theta=0.0,
            target_x=2.0, target_y=0.0,
            us1_cm=15.0, us2_cm=80.0,  # Left sensor blocked
            rl_action_idx=1  # Fast forward requested by RL
        )
        self.assertEqual(cmd.mode, "OBSTACLE_AVOID")
        # Left wheel forward, right reverse to pivot right away from left obstacle
        self.assertTrue(cmd.left_forward)
        self.assertFalse(cmd.right_forward)

    def test_sensorless_replay_step(self):
        best_route = {
            "commands": [
                {"time_sec": 0.0, "left_pwm": 150, "right_pwm": 150},
                {"time_sec": 1.0, "left_pwm": 200, "right_pwm": 200},
                {"time_sec": 2.0, "left_pwm": 0, "right_pwm": 0},
            ]
        }
        # At t=0.5s -> should interpolate to ~175 PWM
        cmd = self.nav.compute_sensorless_replay_step(0.5, best_route)
        self.assertEqual(cmd.mode, "SENSORLESS_REPLAY")
        self.assertAlmostEqual(cmd.left_speed, 175, delta=2)
        self.assertAlmostEqual(cmd.right_speed, 175, delta=2)

        # At t=2.5s (after route end) -> should signal lap reached
        cmd_end = self.nav.compute_sensorless_replay_step(2.5, best_route)
        self.assertTrue(cmd_end.reached)


if __name__ == "__main__":
    unittest.main()
