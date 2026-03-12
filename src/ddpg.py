import copy
import random
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, batch_size: int = 64):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return {
            "obs": self.obs_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "acts": self.acts_buf[idxs],
            "rews": self.rews_buf[idxs],
            "done": self.done_buf[idxs],
        }

    def __len__(self) -> int:
        return self.size


class OUNoise:
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size, dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        x = self.state
        # OU process should use zero-mean Gaussian noise, not uniform [0, 1].
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x)).astype(np.float32)
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, init_w: float = 3e-3):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        return torch.tanh(self.out(x))


class Critic(nn.Module):
    def __init__(self, in_dim: int, init_w: float = 3e-3):
        super().__init__()
        self.hidden1 = nn.Linear(in_dim, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        return self.out(x)


class DDPGCartPoleAgent:
    """DDPG-style agent adapted for CartPole's discrete actions."""

    def __init__(
        self,
        env: gym.Env,
        memory_size: int = 100000,
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1000,
        ou_noise_theta: float = 0.15,
        ou_noise_sigma: float = 0.2,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
    ):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be Box.")

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

        # CartPole action is discrete(2); we model it as 1D continuous signal and threshold to env action.
        self.act_dim = 1 if self.is_discrete else env.action_space.shape[0]

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = int(initial_random_steps)

        self.memory = ReplayBuffer(self.obs_dim, self.act_dim, memory_size, batch_size)
        self.noise = OUNoise(self.act_dim, theta=ou_noise_theta, sigma=ou_noise_sigma)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.obs_dim + self.act_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim + self.act_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def reset_noise(self):
        self.noise.reset()

    def _to_discrete_env_action(self, actor_action: np.ndarray) -> int:
        return int(float(actor_action[0]) > 0.0)

    def _to_storage_action(self, env_action: int) -> np.ndarray:
        # Map env action {0,1} -> {-1,+1} for critic input consistency.
        return np.array([1.0 if env_action == 1 else -1.0], dtype=np.float32)

    def select_action(self, state: np.ndarray, explore: bool, step_idx: int) -> Tuple[int, np.ndarray]:
        if explore and step_idx < self.initial_random_steps:
            sampled = self.env.action_space.sample()
            if self.is_discrete:
                env_action = int(sampled)
                return env_action, self._to_storage_action(env_action)

            env_action = np.asarray(sampled, dtype=np.float32)
            return env_action, env_action

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            actor_action = self.actor(state_t).squeeze(0).cpu().numpy().astype(np.float32)

        if explore:
            actor_action = np.clip(actor_action + self.noise.sample(), -1.0, 1.0)

        if self.is_discrete:
            env_action = self._to_discrete_env_action(actor_action)
            return env_action, self._to_storage_action(env_action)

        env_action = np.clip(actor_action, self.env.action_space.low, self.env.action_space.high)
        return env_action, env_action.astype(np.float32)

    def store_transition(self, state: np.ndarray, action_for_buffer: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.memory.store(state, action_for_buffer, reward, next_state, done)

    def update_model(self) -> Tuple[float, float]:
        samples = self.memory.sample_batch()

        state = torch.as_tensor(samples["obs"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(samples["next_obs"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(samples["acts"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(samples["rews"], dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.as_tensor(samples["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        masks = 1.0 - done

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            target_q = reward + self.gamma * next_value * masks

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._target_soft_update()
        return float(actor_loss.item()), float(critic_loss.item())

    def _target_soft_update(self):
        tau = self.tau
        for t_param, l_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)
