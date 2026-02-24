from neural_network import Actor, Critic
from roll_out_buffer import RolloutBuffer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Independent

import numpy as np
import random

class PPOAgent:
    def __init__(self, state_dim:int, action_dim:int, buffer_size:int, hidden_size:int=128, gamma:float=0.99, gae_lambda:float=0.95, lr:float=3e-4):
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Intialize gym env
        self.env = None

        # Initialize Actor-Critic Network
        self.actor = Actor(in_dim=state_dim, out_dim=action_dim, hidden_size=hidden_size).to(self.device)
        self.critic = Critic(in_dim=state_dim, out_dim=1, hidden_size=hidden_size).to(self.device)

        # Initialize Rollout Buffer
        self.memory = RolloutBuffer(obs_dim=state_dim, act_dim=action_dim, batch_size=32)
        self.buffer_size = buffer_size

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # total_steps
        self.total_step = 0
        
        # state
        self.last_state = None
        self.last_done = None

        # Variables
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = 30
        self.policy_clip = 0.2
        self.value_clip = 0.2 

    def forward_pass(self, state, action=None, eps=1e-6):
        mean = self.actor(state)
        log_std = torch.clamp(self.actor.log_std, -5, 2)
        std = torch.exp(log_std)
        dist = Independent(Normal(mean, std), 1)

        if action is None:
            action = dist.rsample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        value = self.critic(state)
        return action, log_prob, value, entropy

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)  
        else:
            state = state.to(self.device).unsqueeze(0) 

        with torch.no_grad():
            action, log_prob, value, _ = self.forward_pass(state)

        return action.squeeze(0).cpu(), log_prob.squeeze(0).cpu(), value.squeeze().cpu()

    def collect_rollout(self, env, state):
        episode_rewards = []
        current_episode_reward = 0.0
        
        for _ in range(self.buffer_size):
            action, log_prob, value = self.select_action(state)
            action = action.numpy()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward, done = np.float32(reward), np.float32(done)
            self.memory.store(state, action, value.numpy(), reward, log_prob.numpy(), done)
            
            current_episode_reward += reward
            state = next_state
            
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                state, info = env.reset()
                self.last_state = None
                self.last_done = True
            else:
                self.last_state = next_state
                self.last_done = done
        
        if current_episode_reward > 0:
            episode_rewards.append(current_episode_reward)
        
        return state, episode_rewards


    def calculate_advantage_gae(self):
        values, rewards, dones = self.memory.get_raw_data_for_gae()
        values = torch.tensor(np.asarray(values, dtype=np.float32).reshape(-1), device=self.device)
        rewards = torch.tensor(np.asarray(rewards, dtype=np.float32).reshape(-1), device=self.device)
        dones = torch.tensor(np.asarray(dones, dtype=np.float32).reshape(-1), device=self.device)

        T = rewards.shape[0]
        gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)

        if not bool(self.last_done):
            with torch.no_grad():
                last_value = self.critic(torch.from_numpy(self.last_state).float().to(self.device)).view(-1)[0].detach()
        else:
            last_value = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            mask = 1 - dones[t]
            if t == T-1:
                next_value = last_value
            else:
                next_value = values[t+1]

            td_residual = rewards[t] + self.gamma * next_value * mask - values[t]
            at_gae = td_residual + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = at_gae
            gae = at_gae
            gae = at_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(self):
        states, actions, values, rewards, log_probs, dones, batches = self.memory.generate_batches()
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device).view(-1)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device).view(-1)

        advantages, returns = self.calculate_advantage_gae()
        advantages = advantages.detach()
        returns = returns.detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # for record
        epoch_actor_losses = []
        epoch_critic_losses = []

        for _ in range(self.epochs):
            for idx in batches:
                state = states[idx]
                action = actions[idx]
                old_log_prob = log_probs[idx]
                ret = returns[idx]
                advantage = advantages[idx]

                _, new_log_prob, value, entropy = self.forward_pass(state, action)

                r = torch.exp(new_log_prob - old_log_prob.detach())
                surr1 = r * advantage
                surr2 = torch.clamp(r, 1-self.policy_clip, 1+self.policy_clip) * advantage
                
                actor_loss = -torch.min(surr1, surr2).mean() # gradient ascent
                actor_loss = actor_loss - 0.0001 * entropy
                critic_loss = nn.MSELoss()(value.squeeze(-1), ret)

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
        
        self.memory.reset()
        return np.mean(epoch_actor_losses), np.mean(epoch_critic_losses)