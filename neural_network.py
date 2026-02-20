import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_size:int, log_std:float=0.5, init_w:float=1e-3):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_size, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

        self.log_std = nn.Parameter(torch.ones(1, out_dim) * log_std)
    
    def forward(self, state) -> torch.Tensor:
        output = self.input(state)
        output = self.out(output)
        return output 

class Critic(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hidden_size:int, std:float=0.0, init_w:float=3e-3):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_size, out_dim)
        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.zero_()

    def forward(self, state):
        output = self.input(state)
        output = self.out(output)
        return output 