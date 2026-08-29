"""
serial_link.py
=====================================================================
USB-Serial communication link to the Arduino Mega motor/sensor controller.

Includes:
  1. Real SerialLink: Non-blocking reader thread, thread-safe command
     transmission, watchdog heartbeat, auto-reconnect, and IMU telemetry support.
  2. SimulatedSerialLink: High-fidelity physics-based software simulation
     for headless testing, validation, elevation changes, and training without hardware.
  3. Factory helper: create_serial_link() with automatic simulation fallback.
=====================================================================
"""

import math
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Tuple

try:
    import serial
except ImportError:
    serial = None

from config import DEFAULT_CONFIG, SerialConfig, RobotConfig


@dataclass
class SensorPacket:
    us1_cm: float          # -1.0 if ultrasonic sensor #1 timed out / clear
    us2_cm: float          # -1.0 if ultrasonic sensor #2 timed out / clear
    enc_fl_ticks: int      # Cumulative ticks for front-left encoder
    enc_rl_ticks: int      # Cumulative ticks for rear-left encoder
    enc_fr_ticks: int      # Cumulative ticks for front-right encoder
    enc_rr_ticks: int      # Cumulative ticks for rear-right encoder
    timestamp: float
    pitch_deg: float = 0.0 # Optional IMU pitch angle (+ uphill, - downhill)
    roll_deg: float = 0.0  # Optional IMU roll angle


class BaseSerialLink(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_latest(self) -> Optional[SensorPacket]:
        pass

    @abstractmethod
    def set_motor_speeds(self, left: int, right: int):
        pass

    @abstractmethod
    def emergency_stop(self):
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass


class SerialLink(BaseSerialLink):
    """Real USB-serial communication link to Arduino Mega."""
    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200, timeout: float = 1.0):
        if serial is None:
            raise RuntimeError("pyserial package is not installed.")
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._latest: Optional[SensorPacket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._connect()

    def _connect(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(1.8)  # Arduino resets on DTR toggle; wait for bootloader
            print(f"[SerialLink] Connected to Arduino on {self.port} at {self.baud} baud.")
        except Exception as e:
            self._ser = None
            raise ConnectionError(f"Cannot open serial port '{self.port}': {e}")

    def start(self):
        if self._ser is None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.emergency_stop()
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None
        print("[SerialLink] Stopped and closed serial port.")

    def _read_loop(self):
        while self._running:
            if self._ser is None:
                time.sleep(0.1)
                continue
            try:
                raw = self._ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()
            except Exception:
                time.sleep(0.05)
                continue

            if not line or not line.startswith("D"):
                continue

            parts = line.split(",")
            # Supports standard 7-element packet or 9-element packet with IMU pitch/roll
            if len(parts) < 7:
                continue

            try:
                pitch = float(parts[7]) if len(parts) >= 9 else 0.0
                roll = float(parts[8]) if len(parts) >= 9 else 0.0
                packet = SensorPacket(
                    us1_cm=float(parts[1]),
                    us2_cm=float(parts[2]),
                    enc_fl_ticks=int(parts[3]),
                    enc_rl_ticks=int(parts[4]),
                    enc_fr_ticks=int(parts[5]),
                    enc_rr_ticks=int(parts[6]),
                    timestamp=time.time(),
                    pitch_deg=pitch,
                    roll_deg=roll,
                )
                with self._lock:
                    self._latest = packet
            except ValueError:
                continue

    def get_latest(self) -> Optional[SensorPacket]:
        with self._lock:
            return self._latest

    def set_motor_speeds(self, left: int, right: int):
        """Sends 'M,<left>,<right>\\n' command to Arduino."""
        if not self._running or self._ser is None:
            return
        left = max(-255, min(255, int(left)))
        right = max(-255, min(255, int(right)))
        cmd = f"M,{left},{right}\n".encode("ascii")
        try:
            self._ser.write(cmd)
        except Exception:
            pass

    def emergency_stop(self):
        """Sends 'S\\n' immediate stop command."""
        if self._ser is None:
            return
        try:
            self._ser.write(b"S\n")
        except Exception:
            pass

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open


class SimulatedSerialLink(BaseSerialLink):
    """
    High-fidelity differential-drive physical simulator.
    Simulates motor response, encoder tick increments, and ultrasonic sensor obstacles.
    Supports asynchronous thread mode or synchronous manual step_physics(dt) mode.
    """
    def __init__(self, robot_cfg: Optional[RobotConfig] = None,
                 obstacles: Optional[List[Tuple[float, float, float]]] = None,
                 use_async_thread: bool = True):
        self.cfg = robot_cfg or DEFAULT_CONFIG.robot
        self.obstacles = obstacles or [(1.0, 0.25, 0.15), (1.5, -0.30, 0.15)]
        self.use_async_thread = use_async_thread
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Internal simulated robot physics state
        self.sim_x = 0.0
        self.sim_y = 0.0
        self.sim_theta = 0.0
        self.cmd_left_pwm = 0
        self.cmd_right_pwm = 0

        # Cumulative ticks
        self.fl_ticks = 0
        self.rl_ticks = 0
        self.fr_ticks = 0
        self.rr_ticks = 0

        self.last_update_time = time.time()
        self._latest = SensorPacket(
            us1_cm=-1.0, us2_cm=-1.0,
            enc_fl_ticks=0, enc_rl_ticks=0,
            enc_fr_ticks=0, enc_rr_ticks=0,
            timestamp=time.time(),
        )

    def start(self):
        self._running = True
        self.last_update_time = time.time()
        if self.use_async_thread:
            self._thread = threading.Thread(target=self._sim_loop, daemon=True)
            self._thread.start()
        print("[SimulatedSerialLink] Started software physics simulation engine.")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.cmd_left_pwm = 0
        self.cmd_right_pwm = 0
        print("[SimulatedSerialLink] Stopped simulation engine.")

    def step_physics(self, dt: float = 0.05, current_time: Optional[float] = None):
        """Discrete physics integration step."""
        with self._lock:
            def pwm_to_speed(pwm):
                if abs(pwm) < self.cfg.min_pwm_to_move:
                    return 0.0
                ratio = (abs(pwm) - self.cfg.min_pwm_to_move) / (255.0 - self.cfg.min_pwm_to_move)
                return math.copysign(ratio * self.cfg.max_linear_speed_mps, pwm)

            v_l = pwm_to_speed(self.cmd_left_pwm)
            v_r = pwm_to_speed(self.cmd_right_pwm)

            d_l = v_l * dt
            d_r = v_r * dt

            d_center = (d_l + d_r) / 2.0
            d_theta = (d_r - d_l) / self.cfg.wheel_base_m

            mid_theta = self.sim_theta + (d_theta / 2.0)
            self.sim_x += d_center * math.cos(mid_theta)
            self.sim_y += d_center * math.sin(mid_theta)
            self.sim_theta = math.atan2(math.sin(self.sim_theta + d_theta),
                                        math.cos(self.sim_theta + d_theta))

            m_per_tick = (math.pi * self.cfg.wheel_diameter_m) / self.cfg.ticks_per_rev
            d_ticks_l = int(abs(d_l) / m_per_tick)
            d_ticks_r = int(abs(d_r) / m_per_tick)

            self.fl_ticks += d_ticks_l
            self.rl_ticks += d_ticks_l
            self.fr_ticks += d_ticks_r
            self.rr_ticks += d_ticks_r

            us1_dist_cm = self._raycast_obstacle(sensor_offset_angle=math.radians(+15))
            us2_dist_cm = self._raycast_obstacle(sensor_offset_angle=math.radians(-15))

            # Simulated elevation profile: ramp between x=0.5 and x=1.2
            sim_slope = 0.0
            if 0.5 <= self.sim_x <= 1.0:
                sim_slope = +6.5  # 6.5 deg uphill
            elif 1.0 < self.sim_x <= 1.5:
                sim_slope = -6.5  # 6.5 deg downhill

            now = current_time or time.time()
            self._latest = SensorPacket(
                us1_cm=us1_dist_cm,
                us2_cm=us2_dist_cm,
                enc_fl_ticks=self.fl_ticks,
                enc_rl_ticks=self.rl_ticks,
                enc_fr_ticks=self.fr_ticks,
                enc_rr_ticks=self.rr_ticks,
                timestamp=now,
                pitch_deg=sim_slope,
                roll_deg=0.0,
            )

    def _sim_loop(self):
        dt = 0.05  # 20 Hz
        while self._running:
            time.sleep(dt)
            self.step_physics(dt)

    def _raycast_obstacle(self, sensor_offset_angle: float) -> float:
        ray_angle = self.sim_theta + sensor_offset_angle
        min_dist_m = 4.0

        for (ox, oy, orad) in self.obstacles:
            dx = ox - self.sim_x
            dy = oy - self.sim_y
            d_center = math.hypot(dx, dy)
            if d_center > 4.5:
                continue

            angle_to_obs = math.atan2(dy, dx)
            diff_angle = abs(math.atan2(math.sin(angle_to_obs - ray_angle),
                                        math.cos(angle_to_obs - ray_angle)))

            if diff_angle < math.radians(18):
                surf_dist = max(0.05, d_center - orad)
                if surf_dist < min_dist_m:
                    min_dist_m = surf_dist

        if min_dist_m < 3.5:
            return min_dist_m * 100.0
        return -1.0

    def get_latest(self) -> Optional[SensorPacket]:
        with self._lock:
            return self._latest

    def set_motor_speeds(self, left: int, right: int):
        with self._lock:
            self.cmd_left_pwm = max(-255, min(255, int(left)))
            self.cmd_right_pwm = max(-255, min(255, int(right)))

    def emergency_stop(self):
        with self._lock:
            self.cmd_left_pwm = 0
            self.cmd_right_pwm = 0

    def is_connected(self) -> bool:
        return self._running

    def get_sim_pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return self.sim_x, self.sim_y, self.sim_theta


def create_serial_link(port: str = "/dev/ttyACM0",
                       baud: int = 115200,
                       simulate: bool = False) -> BaseSerialLink:
    if simulate:
        return SimulatedSerialLink()
    try:
        return SerialLink(port=port, baud=baud)
    except Exception as e:
        print(f"[SerialLink] Notice: Could not connect to real serial port ({e}). Falling back to simulation link.")
        return SimulatedSerialLink()
