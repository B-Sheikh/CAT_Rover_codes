"""
Unit tests for Odometry and Dead-Reckoning module.
"""

import math
import unittest
from odometry import Odometry
from config import RobotConfig


class TestOdometry(unittest.TestCase):
    def setUp(self):
        self.rcfg = RobotConfig(wheel_diameter_m=0.065, wheel_base_m=0.160, ticks_per_rev=20)
        self.odom = Odometry(x=0.0, y=0.0, theta=0.0, robot_cfg=self.rcfg)

    def test_initial_pose(self):
        x, y, theta = self.odom.pose()
        self.assertEqual((x, y, theta), (0.0, 0.0, 0.0))

    def test_straight_line_motion(self):
        # 1 wheel revolution = pi * 0.065 m = 0.2042 m
        # 20 ticks per rev on both left and right wheels
        self.odom.update(0, 0, 0, 0, True, True)
        self.odom.update(20, 20, 20, 20, True, True)

        expected_dist = math.pi * self.rcfg.wheel_diameter_m
        x, y, theta = self.odom.pose()
        self.assertAlmostEqual(x, expected_dist, places=4)
        self.assertAlmostEqual(y, 0.0, places=4)
        self.assertAlmostEqual(theta, 0.0, places=4)

    def test_spin_in_place(self):
        # Spin counter-clockwise: left reverse (or 0 ticks left), right forward
        self.odom.update(0, 0, 0, 0, False, True)
        # 360 deg spin requires right arc = pi * wheel_base
        arc_len = math.pi * self.rcfg.wheel_base_m
        m_per_tick = (math.pi * self.rcfg.wheel_diameter_m) / self.rcfg.ticks_per_rev
        ticks = int(arc_len / m_per_tick)

        self.odom.update(ticks, ticks, ticks, ticks, False, True)
        x, y, theta = self.odom.pose()
        # For a pure spin about center, x and y remain ~0
        self.assertAlmostEqual(x, 0.0, delta=0.05)
        self.assertAlmostEqual(y, 0.0, delta=0.05)

    def test_wheel_slip_detection(self):
        self.odom.update(0, 0, 0, 0, True, True)
        # Huge mismatch between FL (100 ticks) and RL (20 ticks)
        self.odom.update(100, 20, 50, 50, True, True)
        self.assertGreater(self.odom.last_slip_ratio_left, 0.5)
        self.assertEqual(self.odom.last_slip_ratio_right, 0.0)

    def test_bearing_and_distance_helpers(self):
        self.odom.reset(x=0.0, y=0.0, theta=0.0)
        dist = self.odom.distance_to(3.0, 4.0)
        self.assertAlmostEqual(dist, 5.0)

        bearing = self.odom.bearing_to(0.0, 2.0)
        self.assertAlmostEqual(bearing, math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()
