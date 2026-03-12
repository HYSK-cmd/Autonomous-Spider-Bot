import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

class Actor(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_size:int=256):
        super(Actor, self).__init__()
        
        self.checkpoint_file = os.path.join(DIR_PATH, "actor_torch_ppo.pth")
        self.actor = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

        self.log_std = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, state) -> torch.Tensor:
        mean = self.actor(state)
        # std = torch.exp(self.log_std)
        log_std = torch.clamp(self.log_std, min=-2.0, max=1.0) ######
        std = torch.exp(log_std) ######
        dist = Normal(mean, std)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class Critic(nn.Module):
    def __init__(self, in_dim:int, hidden_size:int=256, std:float=0.0):
        super(Critic, self).__init__()

        self.checkpoint_file = os.path.join(DIR_PATH, "critic_torch_ppo.pth")
        self.critic = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.critic(state)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
