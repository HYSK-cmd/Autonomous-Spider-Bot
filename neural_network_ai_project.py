import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int, log_std:float, init_w: float = 3e-3):
        super(Actor, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, out_dim)
        nn.init.uniform_(self.out.weight, -init_w, init_w)
        nn.init.uniform_(self.out.bias, -init_w, init_w)
        self.log_std = nn.Parameter(torch.ones(1, out_dim) * log_std)
        
    def forward(self, state):
        output = self.input(state)
        output = self.out(output)
        return output
    
class Critic(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, out_dim:int = 1, init_w: float = 3e-3):
        super(Critic, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_dim, out_dim)
        nn.init.uniform_(self.out.weight, -init_w, init_w)
        nn.init.zeros_(self.out.bias)
        
    def forward(self, state):
        output = self.input(state)
        output = self.out(output)
        return output


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std: float = -0.5,
    ):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, hidden_dim, action_dim, log_std)
        self.critic = Critic(state_dim, hidden_dim, out_dim=1)
        self.log_std = self.actor.log_std