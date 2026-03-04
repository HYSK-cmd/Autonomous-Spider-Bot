import torch

class RolloutBuffer:
    def __init__(self, obs_dim:int, act_dim:int, batch_size:int):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size
        self.device = self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, state, action, value, reward, log_prob, done):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        log_prob = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get_raw_data(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        values = torch.stack(self.values)
        rewards = torch.stack(self.rewards)
        log_probs = torch.stack(self.log_probs)
        dones = torch.stack(self.dones)
        return (states, actions, values, rewards, log_probs, dones)

    def generate_batches(self):
        n_states = len(self.states)
        idxs = torch.randperm(n_states)
        batches = [idxs[i:i+self.batch_size] for i in range(0, n_states, self.batch_size)]
        return (*self.get_raw_data(), batches)
    
    def reset(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []