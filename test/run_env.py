import numpy as np
import os
import sys
import torch 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.ppo import PPOAgent
from spider_env import SpiderEnv

N_ROLLOUT_STEPS = 2048
TOTAL_TIMESTEPS = 500000
LOG_FILE = "reward_log2.txt"

def log_episode_reward(episode, ep_reward, mean_reward):
    with open(LOG_FILE, "a") as f:
        f.write(f"Episode {episode}: reward={ep_reward:.2f} mean={mean_reward:.2f}\n")

env = SpiderEnv(render_mode="human")
obs, info = env.reset()

input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

#===== Initialize PPOAgent =====#
agent = PPOAgent(n_inputs=input_dims, n_actions=n_actions)

# Force-load BEST checkpoint explicitly
best_ckpt = agent.best_one  # src/logs/best.pth
checkpoint = torch.load(best_ckpt, map_location=agent.device)
agent.actor.load_state_dict(checkpoint["actor_state"])
agent.critic.load_state_dict(checkpoint["critic_state"])

train = False
episode, episode_reward = 0, 0
reward_history = []
best_mean_reward = -np.inf

#===== Start Training =====#
for i in range(TOTAL_TIMESTEPS):
    state = obs
    raw_action, log_prob, value = agent.select_action(state)
    env_action = np.clip(raw_action, -1.0, 1.0)
    obs, reward, terminated, truncated, _ = env.step(env_action)
    episode_reward += reward
    done = terminated or truncated
    agent.memory.store(state, raw_action, value, reward, log_prob, done)

    if done:
        episode += 1

        reward_history.append(episode_reward)
        recent = reward_history[-100:]
        mean_reward = np.mean(recent)

        log_episode_reward(episode, episode_reward, mean_reward)
        if mean_reward > best_mean_reward and train and len(reward_history) >= 10:
            best_mean_reward = mean_reward
            agent.save_best_checkpoint()
        
        if train:
            agent.ppo_update()

        if episode % 10 == 0 and train:
            print(f"EPISODE: {episode}")
            agent.save_checkpoint()

        obs, info = env.reset()
        episode_reward = 0.0

env.close()