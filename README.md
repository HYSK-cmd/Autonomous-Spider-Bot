# AI PROJECT (AUTONOMOUS SPIDER BOT)
PPO-based navigation for a spider robot with lidar sensor 
Tested in 3D simulation via PyBullet

# **Jetson Orin Nano Setup Guide**
## Materials
Power supply: 5.5x2.5mm pigtails and power source between 9v and 20v (>25W)
Jumper caps to bridge gpio pins
## Setup
Download nvidia sdk manager deb
Click software install or use command
```bash
sudo dpkg -i <sdkmanager\_file\_name>.deb
```

### Enter recovery mode:
* While nono is powered off
* Use jumper cap to bridge 9th (RC REC) and 10th (GND) gpio pins

### Flashing Orin Nano:

* Connect to computer via its usb-c port
* Power on
#### **STEP 1:**
* prompt pops up - select the developer kit version
* Select connected device as target hardware and select 6.2.1 from jetpack sdk version
#### **STEP 2:**
* Leave all components to be default
#### * **STEP 3:**
* Enter username and password
* Select NVMe as storage device
* Hit flash
* Re-enter username and password
* Click install
* Once finished, remove the force recovery jumper

# **SETTING UP TAILSCALE (VPN)**

Download and connect to GitHub account
Install tailscale on Orin Nano
```bash
curl -fsSL https://tailscale.com/install.sh | sh
```
Boot it up
```bash
sudo tailscale up
```
Open the authentication link on computer
* Get the ip by navigating to the device name
* Gain ssh access to the jetson

# Create Virtual Environment
## How to install (in Jetson Nano)
```bash
git clone https://github.com/HYSK-cmd/Autonomous-Spider-Bot.git
conda create --name <env-name> python=3.11
conda create -p .conda-env -c conda-forge python=3.11 numpy pytorch gymnasium pybullet pyyaml -y

conda activate <env-name>
pip install -r requirements.txt
```

# Hardware
| Component | Model / Interface | Purpose |
|---|---|---|
| Controller | Jetson Orin Nano | PPO inference and sensor processing |
| IMU | MPU6050, I2C | Measures robot orientation and yaw rate |
| LiDAR | RPLidar A1M8, slamtec USB/USB-TO-TTL Adaptor | Measures surrounding obstacle distances |
| Servo controller | PCA9685, I2C | Controls 12 servo motors |
| Servo motors | 16 channels | Controls four legs with three joints per leg |

## Sensor Inputs
The PPO policy receives a 377 dimensional observation

| Input | Dimensions | Description |
|:---:|:---:|---|
| LiDAR | 360 | One distance measurement per degree |
| Joint positions | 12 | Current servo joint angles |
| IMU | 5 | Roll, pitch, yaw rate, cosine of yaw, and sine of yaw |

## Action Output
The PPO policy produces 12 continuous actions in the range `[-1, 1]`.
Each action represents a residual joint-angle correction added to the
baseline CPG gait.

## Hardware Setup (Use these links as a reference)
### Servo Motor Setup
https://youtu.be/RnGUTny1hG8?si=mq7sOZ-3lBW69IH0 
### Lidar Setup
https://www.instructables.com/Getting-Started-With-the-Low-cost-RPLIDAR-Using-Je/
### IMU Setup
https://automaticaddison.com/visualize-imu-data-using-the-mpu6050-ros-and-jetson-nano/
### PPO Algorithm
https://youtu.be/8jtAzxUwDj0?si=NiN2cJe0PG6mwH6z
### PPO Equations
go to ~/ppo_equation/