import board
import busio
import adafruit_mpu6050

i2c = busio.I2C(board.SCL, board.SDA)
mpu = adafruit_mpu6050.MPU6050(i2c)


def get_acceleration():
    return mpu.acceleration


def get_gyro():
    return mpu.gyro


if __name__ == "__main__":
    print(get_acceleration())
    print(get_gyro())
