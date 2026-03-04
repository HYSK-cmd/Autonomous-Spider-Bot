import numpy as np
import time, os 

from ppo import PPOAgent
from spider_env import SpiderEnv
from rollout_buffer import RolloutBuffer

log_file = "reward_log2.txt"
reward_history = []

def log_episode_reward(episode, mean_reward):
    with open(log_file, "a") as f:
        f.write(f"Episode {episode}: {mean_reward}\n")

env = SpiderEnv(render_mode="human")

obs, info = env.reset()

input_dims = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

#===== Initialize PPOAgent =====#
agent = PPOAgent(n_inputs=input_dims, n_actions=n_actions)
#agent.load_models()
train = True
episode, episode_reward = 0, 0
best_mean_reward = -np.inf

#===== Start Training =====#
for _ in range(100000):
    state = obs
    action, log_prob, value = agent.select_action(state)
    obs, reward, terminated, truncated, _ = env.step(action)
    episode_reward += reward
    done = terminated or truncated
    agent.memory.store(state, action, value, reward, log_prob, done)

    if done:
        episode += 1

        reward_history.append(episode_reward)
        mean_reward = np.mean(reward_history)

        log_episode_reward(episode, mean_reward)
        if train:
            agent.ppo_update()

        if episode % 10 == 0 and train:
            print(f"EPISODE: {episode}")
            agent.save_checkpoint()

        obs, info = env.reset()
        episode_reward = 0

env.close()