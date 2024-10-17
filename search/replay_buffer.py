import numpy as np
from typing import Tuple, List

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize the Prioritized Replay Buffer.

        :param capacity: Maximum number of experiences the buffer can hold.
        :param alpha: How much prioritization is used (0 - no prioritization, 1 - full prioritization).
        """
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha

    def add(self, experience: Tuple, priority: float):
        """
        Add a new experience to the buffer with the given priority.

        :param experience: The experience tuple to add.
        :param priority: The priority of the experience.
        """
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max(priority, max_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List, List, List, List, List]:
        """
        Sample a batch of experiences from the buffer.

        :param batch_size: Number of experiences to sample.
        :param beta: To what degree to use importance weights (0 - no corrections, 1 - full correction).
        :return: Tuple of sampled experiences and their importance weights.
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """
        Update the priorities of sampled experiences.

        :param indices: List of indices of experiences to update.
        :param priorities: New priorities corresponding to the experiences.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
