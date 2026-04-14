"""
Hardware integration layer for SpdrBot.
Runs the trained PPO actor on real hardware (Jetson Orin Nano).

Observation order must exactly match spider_env_new.py:
  [0:360]   LiDAR     — distance in metres, one reading per degree (0 = no hit, 5.0 = max range / clear)
  [360:372] Joints    — 12 servo angles in radians, centred at 0
  [372:377] IMU       — roll_n, pitch_n, yaw_rate_n, cos(yaw), sin(yaw)
"""

import sys
import os
import numpy as np
import torch
import time
import board
import busio
import adafruit_mpu6050
from adafruit_servokit import ServoKit
from rplidar import RPLidar
from math import atan2, sqrt, sin, cos, pi

# ── import ActorNetwork from simulation so architecture stays in sync ─────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simulation"))
from agent import ActorNetwork

# ═══════════════════════════════════════════════════════════════
#  DEVICE  (Jetson Orin Nano — CUDA if available, else CPU)
# ═══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ═══════════════════════════════════════════════════════════════
#  HARDWARE INIT
# ═══════════════════════════════════════════════════════════════
kit   = ServoKit(channels=16)
i2c   = busio.I2C(board.SCL, board.SDA)
mpu   = adafruit_mpu6050.MPU6050(i2c)
lidar = RPLidar('/dev/ttyUSB0')   # run `ls /dev/ttyUSB*` to confirm port

# ═══════════════════════════════════════════════════════════════
#  SERVO CONFIG  (DS3218 — 270° range)
# ═══════════════════════════════════════════════════════════════
ANGLE_MIN = 0
ANGLE_MAX = 270
ANGLE_MID = 135     # neutral midpoint (maps to 0 rad in obs)

NUM_MOTORS = 12

# Leg → servo index mapping (coxa=shoulder, femur=upper, tibia=lower)
LEGS = {
    "front_left":  {"coxa": 0,  "femur": 1,  "tibia": 2},
    "front_right": {"coxa": 3,  "femur": 4,  "tibia": 5},
    "back_left":   {"coxa": 6,  "femur": 7,  "tibia": 8},
    "back_right":  {"coxa": 9,  "femur": 10, "tibia": 11},
}

current_angles = np.full(NUM_MOTORS, ANGLE_MID, dtype=np.float32)


def apply_action(action: np.ndarray):
    """
    action: 12D numpy array from PPO, values in [-1, 1] (post-tanh).
    Scaled to [0, 270] for DS3218 servos.
    Order: FL_coxa, FL_femur, FL_tibia,
           FR_coxa, FR_femur, FR_tibia,
           BL_coxa, BL_femur, BL_tibia,
           BR_coxa, BR_femur, BR_tibia
    """
    global current_angles
    angles = ((action + 1) / 2) * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
    angles = np.clip(angles, ANGLE_MIN, ANGLE_MAX)
    for i, angle in enumerate(angles):
        kit.servo[i].angle = float(angle)
    current_angles = angles


def stand():
    """All servos to neutral midpoint (action=0 → 135°)."""
    apply_action(np.zeros(NUM_MOTORS))
    time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════
#  PPO ACTOR
#  Loaded directly from simulation/agent.py — architecture stays
#  in sync with whatever was trained, no manual duplication.
# ═══════════════════════════════════════════════════════════════
OBS_DIM    = 377
ACTION_DIM = 12
WEIGHTS    = os.path.join(os.path.dirname(__file__), "..", "simulation", "tmp", "ppo", "actor_torch_ppo_2")

actor = ActorNetwork(n_actions=ACTION_DIM, input_dims=[OBS_DIM], alpha=0.0)
actor.load_checkpoint()   # loads from actor.checkpoint_file = WEIGHTS path set in ActorNetwork
actor.eval()


def predict(obs: np.ndarray) -> np.ndarray:
    """Run the actor deterministically (mean action, then tanh-squash)."""
    with torch.no_grad():
        t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
        dist = actor(t)
        # use the mean for deterministic inference — no sampling noise at deploy time
        action = torch.tanh(dist.mean)
        return action.squeeze(0).cpu().numpy()


# ═══════════════════════════════════════════════════════════════
#  IMU CALIBRATION
# ═══════════════════════════════════════════════════════════════
def calibrate_gyro(samples=100, delay=0.01):
    """
    Bot must be completely still during this phase.
    Runs ~1 second, computes mean bias for all 3 gyro axes.
    """
    print("Calibrating IMU... keep bot still.")
    bx, by, bz = 0.0, 0.0, 0.0
    for _ in range(samples):
        gx, gy, gz = mpu.gyro
        bx += gx; by += gy; bz += gz
        time.sleep(delay)
    bias = (bx / samples, by / samples, bz / samples)
    print(f"Bias → gx:{bias[0]:.4f}  gy:{bias[1]:.4f}  gz:{bias[2]:.4f}")
    return bias


# ═══════════════════════════════════════════════════════════════
#  OBSERVATION BUILDERS
#  Must match _get_observation() in spider_env_new.py exactly.
# ═══════════════════════════════════════════════════════════════

# — IMU state ——————————————————————————————————————————————————
YAW_RATE_CLIP = 5.0   # rad/s — matches spider_env_new.py

yaw    = 0.0
last_t = time.monotonic()


def get_imu_obs() -> np.ndarray:
    """
    5D IMU obs: [roll_n, pitch_n, yaw_rate_n, cos(yaw), sin(yaw)]
    Matches _get_imu() in spider_env_new.py.
      roll_n      = roll  / π
      pitch_n     = pitch / π
      yaw_rate_n  = clip(gz, ±5.0) / 5.0
      cos_yaw, sin_yaw from integrated yaw
    """
    global yaw, last_t

    ax, ay, az = mpu.acceleration
    gx, gy, gz = mpu.gyro
    gz -= gyro_bias[2]

    roll  = atan2(ay, az)
    pitch = atan2(-ax, sqrt(ay*ay + az*az))

    now    = time.monotonic()
    dt     = now - last_t
    last_t = now
    yaw   += gz * dt

    roll_n     = roll  / pi
    pitch_n    = pitch / pi
    yaw_rate_n = float(np.clip(gz, -YAW_RATE_CLIP, YAW_RATE_CLIP)) / YAW_RATE_CLIP

    return np.array([roll_n, pitch_n, yaw_rate_n, cos(yaw), sin(yaw)], dtype=np.float32)


# — LiDAR ——————————————————————————————————————————————————————
LIDAR_MAX_M = 5.0       # metres — matches MAX_LIDAR in spider_env_new.py
LIDAR_MAX_MM = 12000    # RPLidar A1M8 hardware max in mm
BINS = 360


def get_lidar_obs() -> np.ndarray:
    """
    360D array of distances in metres (one per degree).
    Missing readings default to LIDAR_MAX_M (5.0 m = clear path).
    Matches _get_lidar() in spider_env_new.py.
    """
    arr = np.full(BINS, LIDAR_MAX_M, dtype=np.float32)   # default = max range (clear)
    for scan in lidar.iter_scans():
        for quality, angle, distance_mm in scan:
            if quality > 0 and 0 < distance_mm <= LIDAR_MAX_MM:
                idx = int(angle) % BINS
                arr[idx] = min(distance_mm / 1000.0, LIDAR_MAX_M)  # mm → m, cap at 5.0
        break
    return arr


# — Joints ——————————————————————————————————————————————————————
def get_joint_obs() -> np.ndarray:
    """
    12D array of current servo angles in radians, centred at 0.
    Converts from hardware degrees [0, 270] → radians centred at ANGLE_MID.
    Matches _get_joint_positions() in spider_env_new.py.
    """
    return ((current_angles - ANGLE_MID) * (pi / 180.0)).astype(np.float32)


# — Full observation ————————————————————————————————————————————
def get_observation() -> np.ndarray:
    """
    Full 377D obs vector matching spider_env_new.py _get_observation():
      [0:360]   LiDAR  — metres, one reading per degree
      [360:372] Joints — radians, centred at 0
      [372:377] IMU    — roll_n, pitch_n, yaw_rate_n, cos(yaw), sin(yaw)
    """
    obs = np.concatenate([get_lidar_obs(), get_joint_obs(), get_imu_obs()])
    assert obs.shape == (377,), f"Obs shape mismatch: {obs.shape}"
    return obs


# ═══════════════════════════════════════════════════════════════
#  SHUTDOWN
# ═══════════════════════════════════════════════════════════════
def shutdown():
    print("Shutting down...")
    stand()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        print("Warming up IMU (30s)...")
        time.sleep(30)

        stand()
        gyro_bias = calibrate_gyro()

        print("Starting control loop at 50 Hz...")
        while True:
            obs    = get_observation()   # 377D
            action = predict(obs)        # 12D, values in [-1, 1]
            apply_action(action)
            time.sleep(0.02)             # 50 Hz

    except KeyboardInterrupt:
        shutdown()
