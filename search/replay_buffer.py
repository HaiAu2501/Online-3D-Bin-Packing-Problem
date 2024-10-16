# replay_buffer.py

import random
import numpy as np
from typing import Any, Tuple

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state: Any, policy: np.ndarray, reward: float):
        """Add a new experience to the buffer with priority equal to the reward."""
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, policy, reward))
        else:
            self.buffer[self.position] = (state, policy, reward)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, policies, rewards = zip(*batch)
        return states, np.array(policies), np.array(rewards)

    def __len__(self):
        return len(self.buffer)
