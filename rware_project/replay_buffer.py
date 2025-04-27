import random
import torch

class TrajectoryBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, obs, action, log_prob, reward, next_obs, done):
        self.buffer.append((obs, action, log_prob, reward, next_obs, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)  # Remove oldest if over capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, log_probs, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.stack(log_probs),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
