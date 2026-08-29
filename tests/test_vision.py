"""
Unit tests for Computer Vision module (Green & Red Dot Detection).
"""

import math
import unittest
import cv2
import numpy as np

from vision import ColorDotDetector, ColorDetectionResult, SimulatedCameraBackend
from config import CameraConfig, VisionConfig


class TestVision(unittest.TestCase):
    def setUp(self):
        self.detector = ColorDotDetector()

    def create_test_canvas(self, w=640, h=480):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_green_dot_detection(self):
        frame = self.create_test_canvas()
        center = (450, 240)
        radius = 35
        cv2.circle(frame, center, radius, (0, 220, 0), -1)

        result = self.detector.detect_color_dot(frame, "green")
        self.assertTrue(result.found, "Green dot should be detected")
        self.assertAlmostEqual(result.center_x, 450, delta=5)
        self.assertAlmostEqual(result.center_y, 240, delta=5)
        self.assertGreater(result.circularity, 0.6)
        self.assertGreater(result.bearing_rad, 0.0)
        self.assertIsNotNone(result.distance_m)
        self.assertGreater(result.distance_m, 0.0)

    def test_red_dot_detection_wrap_around(self):
        frame = self.create_test_canvas()
        center = (180, 240)
        radius = 28
        cv2.circle(frame, center, radius, (0, 0, 230), -1)

        result = self.detector.detect_color_dot(frame, "red")
        self.assertTrue(result.found, "Red dot should be detected")
        self.assertAlmostEqual(result.center_x, 180, delta=5)
        self.assertAlmostEqual(result.center_y, 240, delta=5)
        self.assertLess(result.bearing_rad, 0.0)

    def test_detect_all_both_dots(self):
        frame = self.create_test_canvas()
        cv2.circle(frame, (150, 200), 30, (0, 220, 0), -1)  # Green
        cv2.circle(frame, (500, 200), 30, (0, 0, 230), -1)  # Red

        results = self.detector.detect_all(frame)
        self.assertTrue(results["green"].found)
        self.assertTrue(results["red"].found)
        self.assertLess(results["green"].center_x, 320)
        self.assertGreater(results["red"].center_x, 320)

        annotated = self.detector.annotate_frame(frame, results)
        self.assertEqual(annotated.shape, frame.shape)

    def test_noise_rejection(self):
        frame = self.create_test_canvas()
        cv2.circle(frame, (100, 100), 3, (0, 220, 0), -1)  # Tiny speckle

        result = self.detector.detect_color_dot(frame, "green")
        # Tiny speckle should be rejected due to min_contour_area
        self.assertFalse(result.found)

    def test_simulated_camera_backend(self):
        cfg = CameraConfig()
        sim_cam = SimulatedCameraBackend(cfg, green_pos=(0.0, 0.0), red_pos=(2.0, 0.0))
        sim_cam.update_robot_pose(x=0.5, y=0.0, theta=0.0)

        frame = sim_cam.read_frame()
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (cfg.height, cfg.width, 3))

        red_det = self.detector.detect_color_dot(frame, "red")
        self.assertTrue(red_det.found, "Simulated red dot ahead of robot should be detected")
        self.assertLess(abs(red_det.bearing_rad), math.radians(10))


if __name__ == "__main__":
    unittest.main()
