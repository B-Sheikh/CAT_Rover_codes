"""
Unit tests for 2.5D Terrain, Obstacle Costmap & Elevation Analyzer.
"""

import math
import unittest
from terrain_map import TerrainMap25D, SlopeEstimator, TerrainPointInfo
from config import TerrainConfig, RobotConfig


class TestTerrain(unittest.TestCase):
    def setUp(self):
        self.tcfg = TerrainConfig(grid_resolution_m=0.05, map_size_x_m=4.0, map_size_y_m=4.0)
        self.terrain_map = TerrainMap25D(terrain_cfg=self.tcfg)
        self.slope_estimator = SlopeEstimator(terrain_cfg=self.tcfg)

    def test_grid_coordinates(self):
        gx, gy = self.terrain_map.world_to_grid(0.0, 0.0)
        wx, wy = self.terrain_map.grid_to_world(gx, gy)
        self.assertAlmostEqual(wx, 0.0, delta=0.05)
        self.assertAlmostEqual(wy, 0.0, delta=0.05)

    def test_obstacle_raycast_and_inflation(self):
        # Robot at (0, 0) facing +x. Obstacle 1.0m ahead on left sensor (+15 deg)
        self.terrain_map.update_ultrasonic_ray(
            robot_x=0.0, robot_y=0.0, robot_theta=0.0,
            us1_cm=100.0, us2_cm=-1.0
        )

        obs_x = 1.0 * math.cos(math.radians(15))
        obs_y = 1.0 * math.sin(math.radians(15))
        info = self.terrain_map.query_point(obs_x, obs_y)
        self.assertGreater(info.traversal_cost, 10.0, "Obstacle cell should have high traversal cost")

    def test_kinematic_slope_estimation(self):
        # When moving slower than expected for PWM 200 -> positive incline (uphill)
        slope_uphill = self.slope_estimator.estimate_slope(
            left_pwm=200, right_pwm=200,
            v_mps=0.15,  # Very slow
        )
        self.assertGreater(slope_uphill, 0.0, "Load deficit should indicate uphill incline")

        # Slope PWM compensation should boost motor PWM uphill
        compensated_pwm = self.slope_estimator.compute_slope_compensated_pwm(base_pwm=160, slope_deg=10.0)
        self.assertGreater(compensated_pwm, 160, "Uphill slope should boost PWM")

        # Downhill slope should decrease PWM
        comp_downhill = self.slope_estimator.compute_slope_compensated_pwm(base_pwm=160, slope_deg=-10.0)
        self.assertLess(comp_downhill, 160, "Downhill slope should reduce PWM for braking")


if __name__ == "__main__":
    unittest.main()
