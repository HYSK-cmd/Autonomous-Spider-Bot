import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, batch_size:int):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size

    def store(self, state, action, value, reward, log_prob, done):
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
            
        self.states.append(to_numpy(state))
        self.actions.append(to_numpy(action))
        self.values.append(to_numpy(value))
        self.rewards.append(to_numpy(reward))
        self.log_probs.append(to_numpy(log_prob))
        self.dones.append(to_numpy(done))

    def get_raw_data_for_gae(self):
        return np.array(self.values), np.array(self.rewards), np.array(self.dones)

    def generate_batches(self):
        n_states = len(self.states)
        idxs = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(idxs)
        
        batches = [idxs[i:i+self.batch_size] for i in range(0, n_states, self.batch_size)]
        
        return (np.array(self.states), np.array(self.actions), np.array(self.values),\
            np.array(self.rewards), np.array(self.log_probs), np.array(self.dones), \
            batches)
    
    def reset(self):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []