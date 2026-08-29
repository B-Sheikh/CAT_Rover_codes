"""
main.py
=====================================================================
Top-Level Autonomous Car Controller & RL Trajectory Pipeline.
Runs directly on the Raspberry Pi 5.

Features:
  - 2.5D Obstacle Analysis & Elevation/Slope Gradient Optimization
  - Visual Landmark Servoing (Green Start Dot, Red Goal Dot)
  - Multi-Round Reinforcement Learning & Trajectory Parameterization
  - Sensorless Blind Replay Mode with Slope Gravity Feedforward
=====================================================================
"""

import argparse
import os
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np

from config import DEFAULT_CONFIG, AppConfig
from odometry import Odometry
from serial_link import create_serial_link, BaseSerialLink, SimulatedSerialLink
from vision import TargetVision, SimulatedCameraBackend, create_camera_backend
from rl_agent import RLTrajectoryOptimizer
from navigation import NavigationController, MultiRoundMissionManager, MissionState, NavCommand


def main():
    parser = argparse.ArgumentParser(
        description="Fully Autonomous Car with Obstacle/Elevation Analysis, RL Trajectory Optimization & Sensorless Replay."
    )
    parser.add_argument("--mode", choices=["auto", "explore", "replay"], default="auto",
                        help="Operating mode: 'auto' (3 learning rounds -> blind replay), 'explore', or 'replay'.")
    parser.add_argument("--port", default="/dev/ttyACM0",
                        help="Arduino USB serial port (e.g. /dev/ttyACM0 or /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate.")
    parser.add_argument("--simulate", action="store_true",
                        help="Run full physics & camera simulation without hardware.")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of exploration/learning rounds with camera & sensors (default: 3).")
    parser.add_argument("--replay-laps", type=int, default=5,
                        help="Number of sensorless optimal route replay laps (default: 5).")
    parser.add_argument("--goal-x", type=float, default=2.0, help="Red Dot Goal X position in meters.")
    parser.add_argument("--goal-y", type=float, default=0.0, help="Red Dot Goal Y position in meters.")
    parser.add_argument("--start-x", type=float, default=0.0, help="Green Dot Start X position in meters.")
    parser.add_argument("--start-y", type=float, default=0.0, help="Green Dot Start Y position in meters.")
    parser.add_argument("--camera-backend", default="auto", choices=["auto", "picamera", "usb", "sim"],
                        help="Camera backend to use.")
    parser.add_argument("--loop-hz", type=float, default=20.0, help="Main control loop frequency in Hz.")
    parser.add_argument("--route-file", default="best_route.json", help="Path to best route JSON file.")
    parser.add_argument("--save-debug-images", action="store_true", help="Save annotated camera frames to disk.")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  AUTONOMOUS CAR: RL TRAJECTORY OPTIMIZER & SENSORLESS CONTROLLER")
    print("  [Obstacle Costmap & 2.5D Elevation/Slope Gradient Optimization Active]")
    print(f"  Mode: {args.mode.upper()} | Simulation: {args.simulate} | Port: {args.port}")
    print(f"  Start (Green): ({args.start_x}, {args.start_y}) m | Goal (Red): ({args.goal_x}, {args.goal_y}) m")
    print("="*70 + "\n")

    # 1. Initialize Serial Communication / Hardware Interface
    serial_link = create_serial_link(port=args.port, baud=args.baud, simulate=args.simulate)
    serial_link.start()

    # 2. Initialize Odometry
    odom = Odometry(x=args.start_x, y=args.start_y, theta=0.0)

    # 3. Initialize Vision Pipeline
    vision: Optional[TargetVision] = None
    if args.mode in ("auto", "explore"):
        cam_backend_name = "sim" if args.simulate else args.camera_backend
        try:
            vision = TargetVision(backend_name=cam_backend_name)
            print(f"[Main] Vision system initialized with '{cam_backend_name}' backend.")
        except Exception as e:
            print(f"[Main] Notice: Could not initialize camera '{cam_backend_name}': {e}. Using simulated camera.")
            vision = TargetVision(backend_name="sim")

    # 4. Initialize RL Agent & Navigation Controllers
    rl_agent = RLTrajectoryOptimizer()
    rl_agent.cfg.best_route_file = args.route_file
    nav_controller = NavigationController()

    mission_manager = MultiRoundMissionManager(
        rl_agent=rl_agent,
        nav_controller=nav_controller,
        green_start_pos=(args.start_x, args.start_y),
        red_goal_pos=(args.goal_x, args.goal_y),
        exploration_rounds=args.rounds,
        replay_laps=args.replay_laps,
    )

    if args.mode == "replay":
        loaded_route = rl_agent.load_best_route(args.route_file)
        if loaded_route is None:
            print(f"[Main] Error: Cannot load '{args.route_file}'. Please run exploration rounds first.")
            serial_link.stop()
            sys.exit(1)
        mission_manager.best_route_data = loaded_route
        mission_manager.state = MissionState.SENSORLESS_REPLAY
        mission_manager.replay_start_time = time.time()
        print(f"[Main] Loaded '{args.route_file}' with {len(loaded_route.get('commands', []))} feedforward steps.")
        print(f"[Main] Running PURE SENSORLESS REPLAY for {args.replay_laps} laps...\n")
    else:
        mission_manager.start_mission()

    is_running = True
    def shutdown_handler(*_):
        nonlocal is_running
        if not is_running:
            return
        is_running = False
        print("\n[Main] Shutdown signal received. Halting robot and releasing devices...")
        serial_link.emergency_stop()
        serial_link.stop()
        if vision:
            vision.close()
        print("[Main] Clean shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    period = 1.0 / args.loop_hz
    last_left_forward = True
    last_right_forward = True
    step_count = 0

    if args.save_debug_images and not os.path.exists("debug_frames"):
        os.makedirs("debug_frames", exist_ok=True)

    try:
        while is_running and mission_manager.state != MissionState.MISSION_FINISHED:
            loop_start = time.time()
            step_count += 1

            # A. Update Odometry from Arduino Encoder Ticks
            packet = serial_link.get_latest()
            if packet is not None:
                odom.update(
                    packet.enc_fl_ticks, packet.enc_rl_ticks,
                    packet.enc_fr_ticks, packet.enc_rr_ticks,
                    last_left_forward, last_right_forward,
                    timestamp=packet.timestamp,
                )

            cur_x, cur_y, cur_theta = odom.pose()
            us1 = packet.us1_cm if packet else -1.0
            us2 = packet.us2_cm if packet else -1.0
            pitch_deg = packet.pitch_deg if packet else 0.0
            slip_ratio = max(odom.last_slip_ratio_left, odom.last_slip_ratio_right)

            # Sync simulated camera position with robot pose
            if args.simulate and isinstance(vision, TargetVision):
                if isinstance(vision.camera, SimulatedCameraBackend):
                    if isinstance(serial_link, SimulatedSerialLink):
                        sx, sy, stheta = serial_link.get_sim_pose()
                        vision.camera.update_robot_pose(sx, sy, stheta)
                    else:
                        vision.camera.update_robot_pose(cur_x, cur_y, cur_theta)

            # B. Process Vision Inputs (ONLY in Exploration Mode)
            green_bearing = None
            red_bearing = None

            if mission_manager.state in (MissionState.EXPLORE_TO_RED_GOAL,
                                         MissionState.RETURN_TO_GREEN_START) and vision is not None:
                dets, annotated_frame = vision.detect_all_dots()
                if dets["green"].found:
                    green_bearing = dets["green"].bearing_rad
                if dets["red"].found:
                    red_bearing = dets["red"].bearing_rad

                if args.save_debug_images and annotated_frame is not None and (step_count % 10 == 0):
                    cv2.imwrite(f"debug_frames/frame_{step_count:05d}.jpg", annotated_frame)
            else:
                # SENSORLESS REPLAY: Zero vision calls, zero camera processing!
                pass

            # C. Mission Step with Elevation & Terrain Slip
            cmd: NavCommand = mission_manager.step(
                cur_x=cur_x, cur_y=cur_y, cur_theta=cur_theta,
                v_mps=odom.current_v_mps, omega_radps=odom.current_omega_radps,
                us1_cm=us1, us2_cm=us2,
                slip_ratio=slip_ratio,
                imu_pitch_deg=pitch_deg,
                green_bearing=green_bearing,
                red_bearing=red_bearing,
            )

            # D. Dispatch Motor Commands
            left_signed = cmd.left_speed if cmd.left_forward else -cmd.left_speed
            right_signed = cmd.right_speed if cmd.right_forward else -cmd.right_speed
            serial_link.set_motor_speeds(left_signed, right_signed)

            last_left_forward = cmd.left_forward
            last_right_forward = cmd.right_forward

            # E. Telemetry Logging
            if step_count % 5 == 0 or cmd.reached:
                state_str = mission_manager.state.value
                red_str = f"{math.degrees(red_bearing):+.1f}deg" if red_bearing is not None else "LOST"
                us_str = f"({us1:.0f},{us2:.0f})cm" if (us1 > 0 or us2 > 0) else "(clear)"
                slope_str = f"{cmd.slope_deg:+.1f}°"

                if mission_manager.state == MissionState.SENSORLESS_REPLAY:
                    print(f"[{state_str} Lap {mission_manager.current_replay_lap}] "
                          f"Pose=({cur_x:.2f}, {cur_y:.2f}, {math.degrees(cur_theta):.0f}deg) | "
                          f"Slope={slope_str} | PWM=(L:{left_signed:+d}, R:{right_signed:+d}) | [BLIND FEEDFORWARD]")
                else:
                    print(f"[{state_str} R{mission_manager.current_round}] "
                          f"Pose=({cur_x:.2f}, {cur_y:.2f}, {math.degrees(cur_theta):.0f}deg) | "
                          f"Slope={slope_str} | RedVis={red_str} | US={us_str} | "
                          f"PWM=(L:{left_signed:+d}, R:{right_signed:+d}) | Act: {cmd.action_name}")

            # F. Frequency Timing Sleep
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        shutdown_handler()


if __name__ == "__main__":
    main()
