# SpdrBot Simulation — PPO Agent

PyBullet simulation of the SpdrBot 4-arm / 12-servo robot trained with Proximal Policy Optimization (PPO).

## Dependencies

```bash
pip install torch numpy gymnasium pybullet
```

## File Overview

| File | Purpose |
|---|---|
| `run_env.py` | Main entry point — training loop and inference runner |
| `agent.py` | PPO agent: actor/critic networks, memory buffer, learn loop |
| `spider_env_new.py` | Gymnasium environment wrapping the PyBullet simulation |
| `SpdrBot_description_v2/` | Robot URDF, meshes, and ROS launch files |
| `models/` | Saved actor/critic weights (`actor_ppo.pth`, `critic_ppo.pth`) |

## Running

### Run with saved weights (inference)

The default mode. Loads weights from `tmp/ppo/` and runs the simulation visually.

```bash
cd simulation
python run_env.py
```

### Train from scratch

Open `run_env.py` and set the `train` flag to `True`:

```python
train = True
```

Then run:

```bash
python run_env.py
```

Training runs for `TOTAL_STEPS = 1_000_000` environment steps. Model weights are saved every 10 episodes to `tmp/ppo/`. Progress is logged to `reward_log.txt`.

## Configuration

All training hyperparameters are at the top of `run_env.py`:

```python
TOTAL_STEPS   = 1_000_000   # total env steps
SAVE_INTERVAL = 10          # save weights every N episodes
train         = False       # True = train, False = run with saved weights
```

PPO agent hyperparameters are in `agent.py` `Agent.__init__`:

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 0.99 | Discount factor |
| `alpha` | 0.0003 | Learning rate (Adam) |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `policy_clip` | 0.2 | PPO clip ratio |
| `batch_size` | 64 | Mini-batch size |
| `n_epochs` | 10 | Update epochs per rollout |
| `entropy_coef` | 0.01 | Entropy bonus weight |

## Observation & Action Space

**Observation (377-dim):**
- 360 lidar rays (distance to nearest obstacle, max 5 m)
- 12 joint angles
- Roll, pitch, yaw rate, cos(yaw), sin(yaw)

**Action (12-dim):**
- Residual joint-angle offsets added on top of CPG gait targets, squashed to `(-1, 1)` via tanh

## Simulation

The environment uses PyBullet with a GUI window by default. To run headless (no window), change the `render_mode` in `run_env.py`:

```python
env = SpiderEnv(render_mode=None)
```
