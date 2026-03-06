# AI PROJECT (AUTONOMOUS SPIDER BOT)
PPO-based navigation for a spider robot with lidar sensor
Tested in 3D simulation via PyBullet

### How to install
```bash
git clone https://github.com/HYSK-cmd/Autonomous-Spider-Bot.git
conda create --name <env-name> python=3.11
conda activate <env-name>
pip install -r requirements.txt
```

# **Orin Nano Setup Guide**
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
* **STEP 1:**
* prompt pops up - select the developer kit version
* Select connected device as target hardware and select 6.2.1 from jetpack sdk version
* **STEP 2:**
* Leave all components to be default
* **STEP 3:**
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



