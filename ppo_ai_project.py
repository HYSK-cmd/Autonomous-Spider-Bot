import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from torch.distributions import Independent, Normal
from neural_network_ai_project import ActorCritic
from rollout_buffer_ai_project import RolloutBuffer
# def load_config(path: str | Path) -> dict:
#     path = Path(path)
#     with path.open("r", encoding="utf-8") as f:
#         return yaml.safe_load(f) or {}

class PPO:
    def __init__(
        self,
        actor_critic: nn.Module,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_loss_coef: float,
        entropy_coef: float,
        batch_size: int,
        epochs: int,
    ):
        # initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # initialize neural network
        self.actor_critic = actor_critic.to(self.device)
        
        # hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.epsilon = 1e-8
        
        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # create a rollout buffer
        self.memory = RolloutBuffer(batch_size=batch_size)

        # variable to keep track of training loop
        self.total_step = 0

        # state
        self.last_state = None
        self.last_done = False

        self.epochs = epochs

    # need help
    def evaluate_actions(self, state, action) -> Tuple[torch.Tensor, torch.Tensor, float]:
        mean = self.actor_critic.actor(state)
        log_std = torch.clamp(self.actor_critic.log_std, -5, 2)
        std = torch.exp(log_std)
        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        value = self.actor_critic.critic(state)
        return log_prob, value, entropy

    def compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # convert all components to tensor array
        values, rewards, dones = self.memory.get_raw_data_for_gae()
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # get the size of rewards tensor
        T = rewards.shape[0]

        # initialize advantages and gae
        advantages = torch.zeros(T, dtype=torch.float32, device=self.device)
        gae = torch.as_tensor(0.0, dtype=torch.float32, device=self.device)
        
        # calculate the bootstrap value of the last trajectory
        self.last_done = dones[-1]
        if bool(self.last_done) or (self.last_state is None):
            next_value = torch.tensor(0.0, device=self.device)
        else:
                with torch.no_grad():
                    self.last_state = torch.as_tensor(self.last_state, dtype=torch.float32, device=self.device)
                    next_value = self.actor_critic.critic(self.last_state).to(self.device)

        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            if t == T-1:
                next_v = next_value.reshape(-1)[0]
            else:
                next_v = values[t+1]
            td_residual = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = td_residual + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        # get discounted returns
        returns = advantages + values

        return advantages, returns
    
    def update(self) -> Tuple[float, float]:
        # Step 1: retrieve training batch from rollout buffer
        states, actions, values, rewards, log_probs, dones, batch = self.memory.generate_batches()
        if len(states) == 0:
            return 0.0, 0.0

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Step 3: get advantages and discounted returns by computing gae
        advantages, returns = self.compute_gae()
        advantages = advantages.detach()

        # Step 4: advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + self.epsilon)

        # Step 5: mini batch PPO update
        actor_losses, critic_losses = [], []
        for epoch in range(self.epochs):
            for idx in batch:
                # extract mini batches
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = log_probs[idx]
                mb_adv = advantages[idx]
                mb_ret = returns[idx]
                #mb_old_values = values[idx] used for value clipping
                
                # predict next action
                new_log_prob, new_value, entropy = self.evaluate_actions(mb_states, mb_actions)
                
                # ppo update formula
                # clipped surrogate objective + value loss + entropy
                ratio = torch.exp(new_log_prob - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon)*mb_adv
                policy_loss = -torch.min(surr1, surr2).mean() # gradient ascent to maximize rewards
                value_loss = nn.MSELoss()(new_value.view(-1), mb_ret.view(-1))
                total_loss = policy_loss + value_loss - (self.entropy_coef * entropy) # enable more exploration

                # store losses
                actor_losses.append(policy_loss.item())
                critic_losses.append(value_loss.item())

                # update optimizer
                self.optimizer.zero_grad()
                total_loss.backward() # need help
                self.optimizer.step()

            self.total_step += 1

        self.total_step = 0
        self.memory.reset()
        return np.mean(actor_losses), np.mean(critic_losses)
        

    def save(self, path: str):
        # Save actor_critic and optimizer state
        checkpt = {"model":self.actor_critic.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.total_step}
        torch.save(checkpt, path)

    def load(self, path: str):
        # Load saved states
        required = ("model", "optimizer", "step")
        checkpt = torch.load(path, map_location=self.device)
        if all(x in checkpt for x in required):
            self.actor_critic.load_state_dict(checkpt["model"])
            self.optimizer.load_state_dict(checkpt["optimizer"])
            self.total_step = checkpt.get("step", 0)
            print(f"step: {self.total_step}")
        self.actor_critic.train()