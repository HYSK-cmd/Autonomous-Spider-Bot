# pre-flight hardware check for IMU, LiDAR, and Servo Motors
import sys
import time

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WAIT = "  [ ]"

checks = {
    "IMU (MPU6050)": False,
    "LiDAR (RPLidar A1M8)":  False,
    "Servos (PCA9685)": False,
}

def print_status():
    print("\033[H\033[J", end="")
    print("="*40)
    print("Spider Bot: Hardware Pre-Flight Check")
    print("="*40)
    for name, done in checks.items():
        mark = PASS if done else WAIT
        print(f"{mark}  {name}")
    print("=" * 40)


# check IMU
def check_imu():
    print_status()
    print("Checking IMU...")
    try:
        import board
        import busio
        import adafruit_mpu6050

        i2c = busio.I2C(board.SCL, board.SDA)
        mpu = adafruit_mpu6050.MPU6050(i2c)
        accel = mpu.acceleration
        gyro  = mpu.gyro
        assert len(accel) == 3 and len(gyro) == 3
        print(f"accel: {tuple(round(v,3) for v in accel)}")
        print(f" gyro:  {tuple(round(v,3) for v in gyro)}")
        checks["IMU (MPU6050)"] = True
        print_status()
        return mpu
    except Exception as e:
        print(f"{FAIL} IMU check failed: {e}")
        sys.exit(1)

# Check RPLiDAR
def check_lidar(port="/dev/ttyUSB0", baudrate=115200):
    print_status()
    print("Checking RPLIDAR A1M8...")
    try:
        import numpy as np
        from rplidar import RPLidar

        lidar = RPLidar(port, baudrate=baudrate)
        time.sleep(1.0)

        arr = np.zeros(360)
        for scan in lidar.iter_scans():
            for quality, angle, dist in scan:
                if quality > 0 and 0 < dist <= 12000:
                    arr[int(angle) % 360] = dist
            break
        hits = int((arr > 0).sum())
        print(f"{hits}/360 degree bins with valid readings")
        if hits < 10:
            raise RuntimeError(f"Too few LiDAR readings ({hits})")
        checks["LiDAR (RPLidar A1M8)"] = True
        print_status()
        return True
    except Exception as e:
        print(f"{FAIL} LiDAR check failed: {e}")
        sys.exit(1)

# Check Servo Motors
def check_servos():
    print_status()
    print("Checking Servos...")
    try:
        from adafruit_servokit import ServoKit

        kit = ServoKit(channels=16)
        for i in range(12):
            kit.servo[i].angle = 90
            time.sleep(0.05)
        print("All 12 servos set to 90 degrees")
        checks["Servos (PCA9685)"] = True
        print_status()
        return kit
    except Exception as e:
        print(f"{FAIL}  Servo check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print_status()
    check_imu()
    time.sleep(0.5)
    check_lidar()
    time.sleep(0.5)
    check_servos()
    print("\nAll hardware checks passed!\n")

