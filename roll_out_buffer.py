import numpy as np

class RolloutBuffer:
    def __init__(self, obs_dim:int, act_dim:int, batch_size:int):
        self.states = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.dones = []
        self.batch_size = batch_size

    def store(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        value: np.ndarray, 
        reward: np.ndarray, 
        log_prob: np.ndarray, 
        done: np.ndarray,
        ):

        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

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