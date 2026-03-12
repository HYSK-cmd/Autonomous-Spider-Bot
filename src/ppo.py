#===== Import files =====#
from .neural_network import Actor, Critic
from .rollout_buffer import RolloutBuffer
#===== PyTorch tools =====#
import torch
import torch.nn as nn
import torch.optim as optim
#===== Tools =====#
import numpy as np
import random
import os, sys
import yaml
#===== Initialize SAVE LOG PATH =====#
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
try:
    LOG_PATH = os.path.join(DIR_PATH, "logs")
    os.makedirs(LOG_PATH, exist_ok=True)
    SAVE_PATH = os.path.join(DIR_PATH, LOG_PATH)
except OSError as e:
    print(f"Failed creating a directory\n")

#===== PPOAgent =====#
class PPOAgent:
    def __init__(self, n_inputs, n_actions):
        # log file path
        self.log_file = os.path.join(SAVE_PATH, "latest.pth")
        self.best_one = os.path.join(SAVE_PATH, "best.pth")
        
        # initialize config file
        self.config_file = os.path.join(DIR_PATH, "config.yaml")

        # load const variables
        self.config = self.load_config(self.config_file)

        # Variables
        self.lr = self.config["learning_rate"]
        self.actor_lr = self.config.get("actor_learning_rate", self.lr) ######
        self.critic_lr = self.config.get("critic_learning_rate", self.lr) ######
        self.gamma = self.config["gamma"]
        self.gae_lambda = self.config["gae_lambda"]
        self.epochs = self.config["n_epochs"]
        self.entropy_coef = self.config["entropy_coef"]
        self.policy_clip = self.config["clip_epsilon"]
        self.value_clip = self.config["clip_epsilon"]
        self.batch_size = self.config["batch_size"]
        self.max_grad_norm = self.config["max_grad_norm"]
        self.target_kl = self.config.get("target_kl", 0.02) ######
        self.log_ratio_clip = self.config.get("log_ratio_clip", 10.0) ######

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Actor-Critic Network
        self.actor = Actor(in_dim=n_inputs, out_dim=n_actions).to(self.device)
        self.critic = Critic(in_dim=n_inputs).to(self.device)

        # Initialize Rollout Buffer
        self.memory = RolloutBuffer(obs_dim=n_inputs, act_dim=n_actions, batch_size=self.batch_size)

        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr) ######
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr) ######
 
        # total_steps
        self.total_step = 0

        self._try_load_checkpoint()

    #===== Load const variables =====#
    @staticmethod
    def load_config(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    #=====  =====#
    def save_checkpoint(self):
        checkpoint = {"actor_state" : self.actor.state_dict(), "critic_state" : self.critic.state_dict()}
        torch.save(checkpoint, self.log_file)
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def save_best_checkpoint(self):
        checkpoint = {"actor_state" : self.actor.state_dict(), "critic_state" : self.critic.state_dict()}
        torch.save(checkpoint, self.best_one)
    #=====  =====#
    def load_checkpoint(self):
        checkpoint = torch.load(self.log_file, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])

    def _try_load_checkpoint(self):
        for checkpoint_path in [self.best_one, self.log_file]:
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    self.actor.load_state_dict(checkpoint["actor_state"])
                    self.critic.load_state_dict(checkpoint["critic_state"])
                    return
                except Exception as e:
                    print(f"Failed to load checkpoint\n")
                    continue
    #=====  =====#
    def forward_pass(self, state, action=None, eps=1e-6):
        dist = self.actor(state)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        value = self.critic(state)
        return action, log_prob, value, entropy
        
    #=====  =====#
    def select_action(self, state):
        # convert to torch if state is a numpy array
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, value, _ = self.forward_pass(state)
        return (action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy(), value.squeeze(0).cpu().numpy())
    #=====  =====#
    def calculate_advantage_gae(self, last_obs=None):
        # get raw data from rollout buffer
        _, _, values, rewards, _, dones = self.memory.get_raw_data()
        values = values.view(-1)

        # get the size of rewards tensor
        T = rewards.shape[0]

        # initialize advantages and gae
        gae = torch.as_tensor(0.0, dtype=torch.float32, device=self.device)
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)

        # Bootstrap value for the state after the last stored transition
        # if the last transition was terminal, or we have no next state, bootstrap = 0
        if bool(dones[-1]) or last_obs is None:
            next_value = torch.tensor(0.0, device=self.device)
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
                next_value = self.critic(obs_t).squeeze()

        for t in reversed(range(T)):
            mask = 1 - dones[t]
            if t == T-1:
                nv = next_value
            else:
                nv = values[t+1]

            td_residual = rewards[t] + self.gamma * nv * mask - values[t]
            at_gae = td_residual + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = at_gae
            gae = at_gae

        returns = advantages + values
        return advantages, returns
    #=====  =====#
    def ppo_update(self, last_obs=None):
        states, actions, values, rewards, log_probs, dones, batches = self.memory.generate_batches()
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device).view(-1)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device).view(-1)

        advantages, returns = self.calculate_advantage_gae(last_obs)
        advantages = advantages.detach()
        returns = returns.detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # for record
        epoch_actor_losses = []
        epoch_critic_losses = []

        for _ in range(self.epochs):
            epoch_kls = [] ######
            for idx in batches:
                state = states[idx]
                action = actions[idx]
                old_log_prob = log_probs[idx]
                ret = returns[idx]
                advantage = advantages[idx]

                _, new_log_prob, value, entropy = self.forward_pass(state, action)

                # r = torch.exp(new_log_prob - old_log_prob.detach())
                log_ratio = new_log_prob - old_log_prob.detach()
                clipped_log_ratio = torch.clamp(log_ratio, -self.log_ratio_clip, self.log_ratio_clip) ######
                r = torch.exp(clipped_log_ratio) ######
                surr1 = r * advantage
                surr2 = torch.clamp(r, 1-self.policy_clip, 1+self.policy_clip) * advantage
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy # gradient ascent
                critic_loss = nn.MSELoss()(value.squeeze(-1), ret)

                approx_kl = (old_log_prob.detach() - new_log_prob).mean().item() ######
                epoch_kls.append(approx_kl) ######

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

            if epoch_kls and np.mean(epoch_kls) > self.target_kl:
                break
        
        self.memory.reset()
        return np.mean(epoch_actor_losses), np.mean(epoch_critic_losses)