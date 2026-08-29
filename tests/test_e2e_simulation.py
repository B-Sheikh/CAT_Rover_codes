"""
End-to-End Autonomous Pipeline Integration Test:
  - 3 Exploration & Learning Rounds with Green & Red Dot Detection
  - Goal reaching, Trajectory Optimization & 'best_route.json' synthesis
  - Seamless switch to Sensorless Blind Replay (0 Sensor Dependency)
"""

import math
import os
import tempfile
import unittest

from config import DEFAULT_CONFIG
from odometry import Odometry
from serial_link import SimulatedSerialLink
from vision import TargetVision, SimulatedCameraBackend
from rl_agent import RLTrajectoryOptimizer
from navigation import NavigationController, MultiRoundMissionManager, MissionState


class TestEndToEndSimulation(unittest.TestCase):
    def test_complete_autonomous_mission(self):
        test_route_file = tempfile.mktemp(suffix=".json")

        # 1. Setup Synchronous Simulation Hardware (clear track)
        sim_serial = SimulatedSerialLink(obstacles=[], use_async_thread=False)
        sim_serial.start()

        # Place Red Goal at (1.2, 0.0) m and Green Start at (0.0, 0.0) m
        sim_cam = SimulatedCameraBackend(DEFAULT_CONFIG.camera, green_pos=(0.0, 0.0), red_pos=(1.2, 0.0))
        vision = TargetVision(camera_backend=sim_cam)

        odom = Odometry(x=0.0, y=0.0, theta=0.0)
        rl_agent = RLTrajectoryOptimizer()
        rl_agent.cfg.best_route_file = test_route_file
        rl_agent.cfg.max_trajectory_steps = 200
        nav_controller = NavigationController()

        mission = MultiRoundMissionManager(
            rl_agent=rl_agent,
            nav_controller=nav_controller,
            green_start_pos=(0.0, 0.0),
            red_goal_pos=(1.2, 0.0),
            exploration_rounds=3,
            replay_laps=2,
        )

        sim_time = 0.0
        mission.start_mission(start_time=sim_time)
        last_l_fwd = True
        last_r_fwd = True

        dt = 0.05
        max_steps = 2000
        step_idx = 0

        while mission.state != MissionState.MISSION_FINISHED and step_idx < max_steps:
            step_idx += 1
            sim_time += dt

            # Step physics synchronously
            sim_serial.step_physics(dt, current_time=sim_time)

            # A. Update Odometry from Simulated Arduino
            pkt = sim_serial.get_latest()
            if pkt:
                odom.update(
                    pkt.enc_fl_ticks, pkt.enc_rl_ticks,
                    pkt.enc_fr_ticks, pkt.enc_rr_ticks,
                    last_l_fwd, last_r_fwd,
                    timestamp=sim_time
                )

            cur_x, cur_y, cur_theta = odom.pose()
            us1 = pkt.us1_cm if pkt else -1.0
            us2 = pkt.us2_cm if pkt else -1.0

            # Sync simulated camera position with robot position
            sim_cam.update_robot_pose(cur_x, cur_y, cur_theta)

            # B. Process Vision Inputs (ONLY in Exploration Mode)
            green_bearing = None
            red_bearing = None

            if mission.state in (MissionState.EXPLORE_TO_RED_GOAL,
                                 MissionState.RETURN_TO_GREEN_START):
                dets = vision.dot_detector.detect_all(sim_cam.read_frame())
                if dets["green"].found:
                    green_bearing = dets["green"].bearing_rad
                if dets["red"].found:
                    red_bearing = dets["red"].bearing_rad

            # C. Mission Step with explicit simulation time
            cmd = mission.step(
                cur_x=cur_x, cur_y=cur_y, cur_theta=cur_theta,
                v_mps=odom.current_v_mps, omega_radps=odom.current_omega_radps,
                us1_cm=us1, us2_cm=us2,
                green_bearing=green_bearing,
                red_bearing=red_bearing,
                current_time=sim_time,
            )

            # D. Dispatch to simulated motors
            l_signed = cmd.left_speed if cmd.left_forward else -cmd.left_speed
            r_signed = cmd.right_speed if cmd.right_forward else -cmd.right_speed
            sim_serial.set_motor_speeds(l_signed, r_signed)

            last_l_fwd = cmd.left_forward
            last_r_fwd = cmd.right_forward

            # If round transitioned back to explore, reset odometry and sim pose for next round
            if cmd.reached and mission.state == MissionState.EXPLORE_TO_RED_GOAL:
                odom.reset(0.0, 0.0, 0.0)
                sim_serial.sim_x = 0.0
                sim_serial.sim_y = 0.0
                sim_serial.sim_theta = 0.0

        # Cleanup
        sim_serial.stop()
        vision.close()

        # Assertions
        self.assertEqual(len(rl_agent.episodes), 3, "Should have completed 3 exploration rounds")
        self.assertTrue(os.path.exists(test_route_file), "Should have generated 'best_route.json'")

        loaded = RLTrajectoryOptimizer.load_best_route(test_route_file)
        self.assertIsNotNone(loaded)
        self.assertGreater(len(loaded["commands"]), 5)
        self.assertEqual(mission.state, MissionState.MISSION_FINISHED, "Mission should finish all replay laps")

        if os.path.exists(test_route_file):
            os.remove(test_route_file)


if __name__ == "__main__":
    unittest.main()
