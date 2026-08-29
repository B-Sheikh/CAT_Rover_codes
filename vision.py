"""
vision.py
=====================================================================
Computer Vision pipeline for autonomous car target & landmark detection.

Features:
  1. Green Dot (Start Point) & Red Dot (Goal / Ending Point) detector:
     - Robust HSV color segmentation (with 2-range wrap-around for Red).
     - Morphological noise suppression (opening/closing).
     - Contour analysis: Area & Circularity filtering (separates dots from clutter).
     - Centroid calculation & sub-pixel moment analysis.
     - Horizontal bearing (angle in radians) & metric distance estimation.
  2. ArUco Marker detection (backward-compatible fallback).
  3. Camera Hardware Abstraction:
     - Picamera2 (Raspberry Pi Camera Module 3 / CSI ribbon cable)
     - OpenCV VideoCapture (standard USB webcams)
     - SimulatedCameraBackend (synthetic rendering for headless testing/sim)
  4. Visual annotation & debug frame rendering.
=====================================================================
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union

import cv2
import numpy as np

from config import DEFAULT_CONFIG, CameraConfig, VisionConfig


@dataclass
class ColorDetectionResult:
    found: bool
    color: str                      # "green", "red", etc.
    center_x: float = 0.0           # Pixel x coordinate (0..width)
    center_y: float = 0.0           # Pixel y coordinate (0..height)
    radius_px: float = 0.0          # Estimated radius in pixels
    area_px: float = 0.0            # Contour pixel area
    circularity: float = 0.0        # Shape circularity (1.0 = perfect circle)
    bearing_rad: float = 0.0        # Horizontal bearing (-fov/2 .. +fov/2, radians)
    distance_m: Optional[float] = None  # Estimated distance in meters
    confidence: float = 0.0         # Detection confidence (0.0 .. 1.0)


@dataclass
class ArucoDetectionResult:
    found: bool
    marker_id: int = -1
    bearing_rad: float = 0.0
    distance_m: Optional[float] = None
    corners: Optional[np.ndarray] = None


class BaseCameraBackend(ABC):
    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """Returns BGR frame as numpy ndarray, or None if unavailable."""
        pass

    @abstractmethod
    def close(self):
        """Release camera resources."""
        pass


class PiCameraBackend(BaseCameraBackend):
    def __init__(self, camera_cfg: CameraConfig):
        self.cfg = camera_cfg
        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            cam_config = self.picam2.create_preview_configuration(
                main={"size": (self.cfg.width, self.cfg.height), "format": "RGB888"}
            )
            self.picam2.configure(cam_config)
            self.picam2.start()
            # Allow camera auto-exposure and white balance to stabilize
            time.sleep(1.0)
            self._available = True
        except Exception as e:
            self._available = False
            raise RuntimeError(f"Failed to initialize Picamera2: {e}")

    def read_frame(self) -> Optional[np.ndarray]:
        if not self._available:
            return None
        # picamera2 returns RGB888, convert to BGR for standard OpenCV processing
        rgb = self.picam2.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def close(self):
        if self._available:
            try:
                self.picam2.stop()
            except Exception:
                pass


class USBCameraBackend(BaseCameraBackend):
    def __init__(self, camera_cfg: CameraConfig):
        self.cfg = camera_cfg
        self.cap = cv2.VideoCapture(self.cfg.usb_device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.framerate)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open USB camera at index {self.cfg.usb_device_id}")

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


class SimulatedCameraBackend(BaseCameraBackend):
    """
    Renders synthetic camera frames for headless testing & simulation.
    Given simulated robot pose and landmark positions, draws green & red dots.
    """
    def __init__(self, camera_cfg: CameraConfig,
                 green_pos: Tuple[float, float] = (0.0, 0.0),
                 red_pos: Tuple[float, float] = (2.0, 0.0)):
        self.cfg = camera_cfg
        self.green_pos = green_pos
        self.red_pos = red_pos
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

    def update_robot_pose(self, x: float, y: float, theta: float):
        self.robot_x = x
        self.robot_y = y
        self.robot_theta = theta

    def read_frame(self) -> np.ndarray:
        # Create dark indoor background
        frame = np.full((self.cfg.height, self.cfg.width, 3), 30, dtype=np.uint8)

        # Draw floor line
        cv2.line(frame, (0, int(self.cfg.height * 0.7)), (self.cfg.width, int(self.cfg.height * 0.7)), (60, 60, 60), 2)

        # Helper to project world (x,y) dot into camera frame
        def project_dot(target_x, target_y, color_bgr):
            dx = target_x - self.robot_x
            dy = target_y - self.robot_y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                return

            angle_to_target = math.atan2(dy, dx)
            relative_angle = math.atan2(math.sin(angle_to_target - self.robot_theta),
                                        math.cos(angle_to_target - self.robot_theta))

            half_fov = math.radians(self.cfg.horizontal_fov_deg / 2.0)
            if abs(relative_angle) <= half_fov and dist > 0.1:
                # Apparent pixel radius: r_px = (D_real * fx) / (2 * dist)
                r_px = int(max(4, (0.08 * self.cfg.focal_length_px) / (2.0 * dist)))
                # Horizontal pixel position: x_px = center_x + fx * tan(bearing)
                cx_px = int((self.cfg.width / 2.0) + self.cfg.focal_length_px * math.tan(relative_angle))
                cy_px = int(self.cfg.height * 0.55)

                if 0 <= cx_px < self.cfg.width and 0 <= cy_px < self.cfg.height:
                    cv2.circle(frame, (cx_px, cy_px), r_px, color_bgr, -1)
                    # Add highlight for realistic lighting
                    cv2.circle(frame, (cx_px - r_px // 3, cy_px - r_px // 3), max(1, r_px // 3), (255, 255, 255), -1)

        # Render Green Dot (Start) - BGR (0, 220, 0)
        project_dot(self.green_pos[0], self.green_pos[1], (0, 220, 0))
        # Render Red Dot (Goal) - BGR (0, 0, 230)
        project_dot(self.red_pos[0], self.red_pos[1], (0, 0, 230))

        return frame

    def close(self):
        pass


def create_camera_backend(backend: str = "auto",
                          cam_cfg: Optional[CameraConfig] = None) -> BaseCameraBackend:
    cfg = cam_cfg or DEFAULT_CONFIG.camera
    backend_choice = backend.lower()

    if backend_choice == "sim":
        return SimulatedCameraBackend(cfg)

    if backend_choice in ("picamera", "auto"):
        try:
            return PiCameraBackend(cfg)
        except Exception as e:
            if backend_choice == "picamera":
                raise e
            # Fallback to USB or Sim

    if backend_choice in ("usb", "auto"):
        try:
            return USBCameraBackend(cfg)
        except Exception as e:
            if backend_choice == "usb":
                raise e

    # Fallback to simulation if auto failed
    print("Warning: Hardware camera not found. Falling back to SimulatedCameraBackend.")
    return SimulatedCameraBackend(cfg)


class ColorDotDetector:
    """
    High-performance HSV Color Dot Detector for Start (Green) and Goal (Red) landmarks.
    """
    def __init__(self, vision_cfg: Optional[VisionConfig] = None,
                 cam_cfg: Optional[CameraConfig] = None):
        self.vcfg = vision_cfg or DEFAULT_CONFIG.vision
        self.ccfg = cam_cfg or DEFAULT_CONFIG.camera
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect_color_dot(self, frame_bgr: np.ndarray, color: str = "red") -> ColorDetectionResult:
        """
        Detects a circular colored dot ('green' or 'red') in the provided BGR frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return ColorDetectionResult(found=False, color=color)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]

        if color.lower() == "green":
            lower = np.array(self.vcfg.green_hsv_lower, dtype=np.uint8)
            upper = np.array(self.vcfg.green_hsv_upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
        elif color.lower() == "red":
            lower1 = np.array(self.vcfg.red_hsv_lower1, dtype=np.uint8)
            upper1 = np.array(self.vcfg.red_hsv_upper1, dtype=np.uint8)
            lower2 = np.array(self.vcfg.red_hsv_lower2, dtype=np.uint8)
            upper2 = np.array(self.vcfg.red_hsv_upper2, dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            raise ValueError(f"Unsupported color '{color}'. Use 'green' or 'red'.")

        # Morphological noise removal (opening followed by closing)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_result = ColorDetectionResult(found=False, color=color)
        max_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.vcfg.min_contour_area or area > self.vcfg.max_contour_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            # Circularity: 4 * pi * Area / (Perimeter^2)
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.vcfg.min_circularity:
                continue

            # Get centroid and minimum enclosing circle
            ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
            if radius <= 0:
                continue

            # Calculate horizontal bearing in radians relative to camera optical axis
            # Positive = to the right, Negative = to the left
            pixel_offset = cx - (w / 2.0)
            bearing_rad = float(np.arctan2(pixel_offset, self.ccfg.focal_length_px))

            # Metric distance estimation: d = (D_real * fx) / (2 * r_px)
            apparent_diameter_px = 2.0 * radius
            distance_m = (self.vcfg.dot_real_diameter_m * self.ccfg.focal_length_px) / apparent_diameter_px

            # Composite confidence score combining circularity and size
            confidence = min(1.0, (circularity * 0.7) + min(0.3, area / 5000.0))

            if confidence > max_score:
                max_score = confidence
                best_result = ColorDetectionResult(
                    found=True,
                    color=color,
                    center_x=float(cx),
                    center_y=float(cy),
                    radius_px=float(radius),
                    area_px=float(area),
                    circularity=float(circularity),
                    bearing_rad=bearing_rad,
                    distance_m=float(distance_m),
                    confidence=float(confidence),
                )

        return best_result

    def detect_all(self, frame_bgr: np.ndarray) -> Dict[str, ColorDetectionResult]:
        """Detects both green and red dots in a single frame."""
        return {
            "green": self.detect_color_dot(frame_bgr, "green"),
            "red": self.detect_color_dot(frame_bgr, "red"),
        }

    def annotate_frame(self, frame_bgr: np.ndarray,
                       detections: Dict[str, ColorDetectionResult]) -> np.ndarray:
        """
        Draws debug bounding circles, centroids, target lines, and distance/bearing telemetry.
        """
        annotated = frame_bgr.copy()
        h, w = annotated.shape[:2]
        center_screen_x = w // 2

        # Draw central crosshair
        cv2.line(annotated, (center_screen_x, 0), (center_screen_x, h), (80, 80, 80), 1)
        cv2.line(annotated, (0, h // 2), (w, h // 2), (80, 80, 80), 1)

        color_draw_bgr = {
            "green": (0, 255, 0),
            "red": (0, 0, 255),
        }

        for color_name, det in detections.items():
            if not det.found:
                continue

            draw_color = color_draw_bgr.get(color_name, (255, 255, 255))
            pt = (int(det.center_x), int(det.center_y))
            r = int(det.radius_px)

            # Circle target
            cv2.circle(annotated, pt, r, draw_color, 2)
            cv2.circle(annotated, pt, 3, (255, 255, 255), -1)

            # Line from frame center to target
            cv2.line(annotated, (center_screen_x, h), pt, draw_color, 1)

            # Info text
            deg = math.degrees(det.bearing_rad)
            dist_str = f"{det.distance_m:.2f}m" if det.distance_m else "N/A"
            label = f"{color_name.upper()} | {dist_str} | {deg:+.1f}deg | C:{det.confidence:.2f}"

            text_pos = (max(10, pt[0] - 80), max(20, pt[1] - r - 8))
            cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(annotated, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1)

        return annotated


class TargetVision:
    """
    Unified high-level Vision interface for the autonomous car.
    Combines camera capture, color dot detection, and optional ArUco fallback.
    """
    def __init__(self, camera_backend: Optional[BaseCameraBackend] = None,
                 backend_name: str = "auto",
                 vision_cfg: Optional[VisionConfig] = None,
                 cam_cfg: Optional[CameraConfig] = None):
        self.vcfg = vision_cfg or DEFAULT_CONFIG.vision
        self.ccfg = cam_cfg or DEFAULT_CONFIG.camera
        self.camera = camera_backend or create_camera_backend(backend_name, self.ccfg)
        self.dot_detector = ColorDotDetector(self.vcfg, self.ccfg)
        self._last_frame = None

        # Setup ArUco fallback detector
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.vcfg.aruco_dict_id)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except Exception:
            self.aruco_detector = None

    def capture_frame(self) -> Optional[np.ndarray]:
        self._last_frame = self.camera.read_frame()
        return self._last_frame

    def detect_target_dot(self, color: str = "red") -> ColorDetectionResult:
        """Captures frame and detects specific target dot ('green' or 'red')."""
        frame = self.capture_frame()
        if frame is None:
            return ColorDetectionResult(found=False, color=color)
        return self.dot_detector.detect_color_dot(frame, color)

    def detect_all_dots(self) -> Tuple[Dict[str, ColorDetectionResult], Optional[np.ndarray]]:
        """Captures frame, detects both dots, and returns (detections_dict, annotated_frame)."""
        frame = self.capture_frame()
        if frame is None:
            return {"green": ColorDetectionResult(found=False, color="green"),
                    "red": ColorDetectionResult(found=False, color="red")}, None

        dets = self.dot_detector.detect_all(frame)
        annotated = self.dot_detector.annotate_frame(frame, dets)
        return dets, annotated

    def detect_aruco(self, target_id: int = 0) -> ArucoDetectionResult:
        """Backward-compatible ArUco detection."""
        if self.aruco_detector is None:
            return ArucoDetectionResult(found=False)

        frame = self.capture_frame()
        if frame is None:
            return ArucoDetectionResult(found=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None:
            return ArucoDetectionResult(found=False)

        ids_flat = ids.flatten()
        if target_id not in ids_flat:
            return ArucoDetectionResult(found=False)

        idx = list(ids_flat).index(target_id)
        marker_corners = corners[idx][0]
        center_x = float(np.mean(marker_corners[:, 0]))
        pixel_offset = center_x - (self.ccfg.width / 2.0)
        bearing_rad = float(np.arctan2(pixel_offset, self.ccfg.focal_length_px))

        side_lengths = [
            np.linalg.norm(marker_corners[i] - marker_corners[(i + 1) % 4])
            for i in range(4)
        ]
        apparent_px = float(np.mean(side_lengths))
        distance_m = (self.vcfg.aruco_real_size_m * self.ccfg.focal_length_px) / apparent_px if apparent_px > 0 else None

        return ArucoDetectionResult(
            found=True,
            marker_id=target_id,
            bearing_rad=bearing_rad,
            distance_m=distance_m,
            corners=marker_corners,
        )

    def close(self):
        if self.camera:
            self.camera.close()
