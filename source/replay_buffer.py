# replay_buffer.py

import numpy as np
from typing import Tuple, List, Any

class PrioritizedReplayBuffer:
    """
    Bộ nhớ Replay với ưu tiên trải nghiệm.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience: Tuple[Any, Any, Any], priority: float):
        """
        Thêm trải nghiệm vào bộ nhớ với độ ưu tiên.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[Any, Any, Any]]:
        """
        Lấy mẫu ngẫu nhiên các trải nghiệm dựa trên độ ưu tiên.
        """
        probabilities = np.array(self.priorities)
        probabilities = probabilities / probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        return len(self.buffer)
