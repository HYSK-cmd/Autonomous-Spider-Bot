"""
Training loop for SpdrBot using the PPO agent (src/ppo.py).

The env runs one step at a time. When an episode ends (robot fell, reached
the goal, or hit the MAX_STEPS time limit), the agent learns from everything
it experienced, then a new episode starts automatically.
"""

import os
import sys

# Allow imports from both test/simulation/ (spider_env_new) and project root (src/)
_SIM_DIR      = os.path.dirname(os.path.abspath(__file__))   # .../test/simulation
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SIM_DIR))   # .../  (two levels up)
for _p in (_SIM_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from spider_env_new import SpiderEnv
from src.ppo import PPOAgent

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 1_000_000   # how many env steps to train for in total
SAVE_INTERVAL = 10          # save model weights every N episodes
LOG_FILE      = os.path.join(_SIM_DIR, "reward_log.txt")
train = False  # set to True to train, False to just run with saved weights

# ── Setup ─────────────────────────────────────────────────────────────────────
env    = SpiderEnv(render_mode="human")
obs, _ = env.reset()

if not train:
    # Auto-loads best/latest checkpoint inside __init__ — no extra call needed
    agent = PPOAgent(n_inputs=env.observation_space.shape[0],
                     n_actions=env.action_space.shape[0])
else:
    # Disable auto-loading so training always starts from scratch
    PPOAgent._try_load_checkpoint = lambda self: None
    agent = PPOAgent(n_inputs=env.observation_space.shape[0],
                     n_actions=env.action_space.shape[0])

# ── Training loop ─────────────────────────────────────────────────────────────
episode        = 0
episode_reward = 0.0
episode_length = 0
reward_history    = []
length_history    = []
collision_history = []

for step in range(TOTAL_STEPS):

    env_action, raw_action, log_prob, value = agent.select_action(obs)
    current_obs = obs   # obs_t — log_prob/value were computed here
    obs, reward, terminated, truncated, info = env.step(env_action)
    episode_reward += reward
    episode_length += 1
    done = terminated or truncated

    # raw_action (pre-tanh) stored in memory — consistent with how log_prob was computed
    agent.memory.store(current_obs, raw_action, value, reward, log_prob, done)

    if done:
        episode   += 1
        end_reason = info.get("end_reason", "?")
        per_step   = episode_reward / episode_length

        reward_history.append(episode_reward)
        length_history.append(episode_length)
        collision_history.append(1 if end_reason == "collision" else 0)

        mean_reward    = np.mean(reward_history[-100:])
        mean_length    = np.mean(length_history[-100:])
        collision_rate = np.mean(collision_history[-100:]) * 100.0

        if train:
            # Episode ended → terminal, so last_obs bootstrap = None (value = 0)
            actor_loss, critic_loss = agent.ppo_update(last_obs=None)
        else:
            actor_loss, critic_loss = float("nan"), float("nan")
            agent.memory.reset()   # discard rollout in eval mode

        with open(LOG_FILE, "a") as f:
            f.write(
                f"ep={episode}"
                f"  total={episode_reward:.2f}"
                f"  per_step={per_step:.3f}"
                f"  steps={episode_length}"
                f"  end={end_reason}"
                f"  mean100={mean_reward:.2f}"
                f"  mean_len={mean_length:.0f}"
                f"  coll%={collision_rate:.1f}"
                f"  a_loss={actor_loss:.4f}"
                f"  c_loss={critic_loss:.4f}"
                f"\n"
            )

        if episode % SAVE_INTERVAL == 0:
            print(
                f"ep {episode:>5} | step {step:>7} | "
                f"total {episode_reward:>8.2f} | steps {episode_length:>5} | "
                f"end={end_reason:<9} | mean100 {mean_reward:>8.2f} | "
                f"coll% {collision_rate:>5.1f} | "
                f"a_loss={actor_loss:.4f} | c_loss={critic_loss:.4f}"
            )
            if train:
                agent.save_checkpoint()

        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0

env.close()
