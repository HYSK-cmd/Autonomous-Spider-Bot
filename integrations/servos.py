# Requires ServoKit: https://github.com/JetsonHacksNano/ServoKit
# Setup: git clone https://github.com/JetsonHacksNano/ServoKit && cd ServoKit && ./installServoKit.sh

from adafruit_servokit import ServoKit

# 12 motors total, 3 per leg (shoulder, elbow, wrist), across 2 PCA9685 boards (16 channels each)
kit = ServoKit(channels=16)

NUM_LEGS = 4
MOTORS_PER_LEG = 3
NUM_MOTORS = NUM_LEGS * MOTORS_PER_LEG  # 12


def set_angle(motor_index, angle):
    """Set a single servo to a given angle (0–180°)."""
    kit.servo[motor_index].angle = angle


def set_all_angles(angles):
    """Set all 12 servos. angles must be a list/array of length 12."""
    assert len(angles) == NUM_MOTORS, f"Expected {NUM_MOTORS} angles, got {len(angles)}"
    for i, angle in enumerate(angles):
        kit.servo[i].angle = float(angle)


def center_all():
    """Move all servos to 90° (neutral position)."""
    set_all_angles([90] * NUM_MOTORS)


if __name__ == "__main__":
    center_all()
    print("All servos centered at 90°")
