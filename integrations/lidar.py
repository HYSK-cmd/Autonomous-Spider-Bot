# pip install rplidar-roboticia

from rplidar import RPLidar
import numpy as np

PORT_NAME = '/dev/ttyUSB0'
MAX_DIST = 12000
BINS = 360


def start_lidar(port_name=PORT_NAME):
    lidar = RPLidar(port_name)
    lidar.start_motor()
    return lidar


def get_scan(lidar):
    """
    Collect one full 360° sweep and return a 360-element 1D array where
    index = degree (0–359) and value = distance in mm (0 means no reading).
    """
    arr = np.zeros(BINS)
    for scan in lidar.iter_scans():
        for quality, angle, distance in scan:
            if quality > 0 and 0 < distance <= MAX_DIST:
                idx = int(angle) % BINS
                arr[idx] = round(distance, 1)
        break
    return arr


def stop_lidar(lidar):
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()


if __name__ == "__main__":
    lidar = start_lidar()
    distances = get_scan(lidar)
    stop_lidar(lidar)

    print("\n=== 360° distance array (mm) ===")
    print(distances)
