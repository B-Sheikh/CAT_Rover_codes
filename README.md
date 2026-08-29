# Fully Autonomous Self-Driving Car
### Real-Time Visual Landmark Detection, Multi-Round Reinforcement Learning & Sensorless Optimal Route Replay

An end-to-end autonomous 4-wheel robotic vehicle running on a **Raspberry Pi 5** (High-Level AI / Vision / RL Engine) and an **Arduino Mega 2560** (Real-Time Motor PWM / Interrupt Encoder Counting / Ultrasonic Ranging).

---

## 1. System Architecture & Core Concept

```
+-----------------------------------------------------------------------------------+
|                                 Raspberry Pi 5                                    |
|                                                                                   |
|  +--------------------+     +-----------------------+     +--------------------+  |
|  |     Camera Cues    |     |  Odometry Kinematics  |     | Ultrasonic Signals |  |
|  |  (Green Start Dot, |     | (4-Wheel Ticks & RK2  |     |  (Left & Right Obst|  |
|  |   Red Goal Dot)    |     | Midpoint Integration) |     |     Detection)     |  |
|  +---------+----------+     +-----------+-----------+     +---------+----------+  |
|            |                            |                           |             |
|            +----------------------------+---------------------------+             |
|                                         |                                         |
|                                         v                                         |
|                     +---------------------------------------+                     |
|                     |     RL Trajectory Optimizer Engine    |                     |
|                     |  - Spatial Q-Table (State Discretiz.) |                     |
|                     |  - Composite Reward Potential         |                     |
|                     |  - Multi-Round Elite Selection        |                     |
|                     +-------------------+-------------------+                     |
|                                         |                                         |
|                     +-------------------+-------------------+                     |
|                     |                                       |                     |
|             [Rounds 1 to 3]                         [Round 4 Onwards]             |
|        Exploration & Learning Mode             Sensorless Blind Replay Mode       |
|    - Epsilon-Greedy Action Selection        - Reads 'best_route.json'             |
|    - Visual Servoing to Dots                - Pure Feedforward PWM Sequence       |
|    - Reactive Ultrasonic Avoidance          - ZERO Camera Dependency              |
|    - Synthesizes 'best_route.json'          - ZERO Ultrasonic Dependency          |
|                     |                                       |                     |
|                     +-------------------+-------------------+                     |
|                                         |                                         |
|                              (USB Serial @ 115200)                                |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                Arduino MEGA 2560                                  |
|  - 4x Hardware Interrupts on Encoders (INT2, INT3, INT4, INT5)                    |
|  - Dual L298N Motor Driver Control (FL, RL, FR, RR)                               |
|  - Microsecond HC-SR04 Ultrasonic Trigger & Echo Timing                           |
|  - 500ms Watchdog Heartbeat Fail-Safe                                             |
+-----------------------------------------------------------------------------------+
```

---

## 2. Multi-Round RL & Trajectory Optimization Pipeline

### Phase 1: Exploration & Learning (Rounds 1 to 3)
1. **Starting Landmark (Green Dot)**: The camera detects the Green circle to establish start heading alignment.
2. **Ending Landmark (Red Dot)**: The camera segments the Red circle in HSV color space (with wrap-around handling: $H \in [0, 10] \cup [170, 180]$) and calculates the target bearing $\theta_{\text{bearing}}$ and estimated distance $d_{\text{target}}$.
3. **Reinforcement Learning Policy**:
   - **State Space**: $s = (x_{\text{grid}}, y_{\text{grid}}, \theta_{\text{bin}}, \text{obs\_state}, \text{vis\_state})$
   - **Reward Function**:
     $$r_t = r_{\text{goal}} + r_{\text{progress}} + r_{\text{time}} + r_{\text{smoothness}} + r_{\text{obstacle}} + r_{\text{vision}}$$
     - **Goal Bonus ($+500$)**: Awarded upon reaching within tolerance of the Red Dot.
     - **Potential-Based Progress**: $+15.0 \times (d_{t-1} - d_t)$ toward the target.
     - **Time Step Penalty**: $-0.5$ per step (drives the agent toward the fastest trajectory).
     - **Obstacle Proximity Penalty**: Quadratic penalty if ultrasonic distance $< 45\text{ cm}$.
     - **Jerk / Smoothness Penalty**: Penalizes rapid motor PWM oscillations.
     - **Vision Alignment**: $+2.5 \times \cos(\theta_{\text{bearing}})$ when target is in camera view.
   - **Exploration Rate**: $\epsilon$ decays exponentially across rounds ($\epsilon_1 = 0.35 \rightarrow \epsilon_2 = 0.21 \rightarrow \epsilon_3 = 0.12$).

### Phase 2: Trajectory Optimization & Route Parameterization
- At the end of the exploration rounds, the RL engine scores each completed trajectory:
  $$\text{Score}(\tau) = \frac{\text{Return}(\tau)}{1.0 + 0.1 \cdot T_{\text{lap}} + 0.05 \cdot L_{\text{path}}}$$
- The elite trajectory is filtered with moving-average B-Spline smoothing and serialized into `best_route.json` as a timestamped feedforward command table:
  $$\tau^*(t) = [t, \text{PWM}_{\text{left}}(t), \text{PWM}_{\text{right}}(t), x(t), y(t), \theta(t), v(t), \omega(t)]$$

### Phase 3: Sensorless Blind Replay (Rounds 4 Onwards)
- **Zero Sensor Dependence**: The robot executes the learned optimal route purely using time-synchronized feedforward motor commands.
- **High Speed & Smoothness**: Eliminates camera processing latency and sensor noise jitter, running the car along the mathematically optimal trajectory at maximum efficiency.

---

## 3. Hardware Pinout & Wiring Guide

### Arduino Mega 2560 Pin Map

| Component | Function | Arduino Mega Pin | Description |
|---|---|---|---|
| **Encoders** | Front-Left (FL) | **D2** (INT4) | Dedicated Hardware Interrupt |
| | Rear-Left (RL) | **D3** (INT5) | Dedicated Hardware Interrupt |
| | Front-Right (FR) | **D18** (INT3) | Dedicated Hardware Interrupt |
| | Rear-Right (RR) | **D19** (INT2) | Dedicated Hardware Interrupt |
| **L298N #1 (Left Motors)** | FL Motor PWM | **D5** (PWM) | ENA (Left Front Speed) |
| | FL Direction | **D22, D23** | IN1, IN2 |
| | RL Motor PWM | **D6** (PWM) | ENB (Left Rear Speed) |
| | RL Direction | **D24, D25** | IN3, IN4 |
| **L298N #2 (Right Motors)**| FR Motor PWM | **D9** (PWM) | ENA (Right Front Speed) |
| | FR Direction | **D26, D27** | IN1, IN2 |
| | RR Motor PWM | **D10** (PWM) | ENB (Right Rear Speed) |
| | RR Direction | **D28, D29** | IN3, IN4 |
| **Ultrasonic #1 (Left-Front)**| Trigger / Echo | **D30 / D31** | 5V tolerant on Mega inputs |
| **Ultrasonic #2 (Right-Front)**| Trigger / Echo | **D32 / D33** | 5V tolerant on Mega inputs |

### Raspberry Pi 5 Connections
- **Camera**: Raspberry Pi Camera Module 3 connected via CSI ribbon cable.
- **Arduino Mega**: Connected via standard USB cable to Pi 5 (enumerates as `/dev/ttyACM0` or `/dev/ttyUSB0`).

---

## 4. Software Installation

```bash
# 1. System packages
sudo apt update
sudo apt install -y python3-opencv python3-picamera2

# 2. Virtual environment setup
cd /home/fire/CAT_Auto_car
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Running the Autonomous Vehicle

### Mode 1: Full Automated Pipeline (Recommended)
Runs 3 exploration rounds with Green start and Red goal dot detection + RL trajectory optimization, then automatically switches to **Sensorless Blind Replay** for 5 laps:
```bash
python3 main.py --mode auto --rounds 3 --replay-laps 5 --goal-x 2.0 --goal-y 0.0
```

### Mode 2: Pure Sensorless Blind Replay
Replays the saved optimal trajectory (`best_route.json`) with **0 sensor calls** (no camera, no ultrasonic):
```bash
python3 main.py --mode replay --replay-laps 10 --route-file best_route.json
```

### Mode 3: Exploration & Learning Only
Runs exploration rounds to learn and save `best_route.json`:
```bash
python3 main.py --mode explore --rounds 3 --goal-x 2.0 --goal-y 0.0
```

### Mode 4: Headless Simulation Mode (No Hardware Needed)
Tests the complete pipeline in software physics and synthetic camera simulation:
```bash
python3 main.py --mode auto --simulate --rounds 3 --replay-laps 3
```

---

## 6. Running the Automated Test Suite

Validate all modules (Vision, Odometry, RL Agent, Navigation, and End-to-End Simulation):
```bash
source venv/bin/activate
python3 -m unittest discover tests
```
