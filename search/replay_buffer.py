# replay_buffer.py

import random
import numpy as np
from typing import Any, Tuple, List, Dict

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int):
        """
        Initialize the Prioritized Replay Buffer.

        :param capacity: Maximum number of experiences the buffer can hold.
        """
        self.capacity = capacity
        self.buffer: List[Tuple[Any, np.ndarray, float]] = []
        self.position = 0

    def add(self, state: Dict[str, np.ndarray], policy: np.ndarray, reward: float):
        """
        Add a new experience to the buffer with priority equal to the reward.

        :param state: The current state of the environment.
        :param policy: The target policy (from MCTS).
        :param reward: The reward obtained from the simulation.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, policy, reward))
        else:
            self.buffer[self.position] = (state, policy, reward)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences uniformly from the buffer.

        :param batch_size: Number of experiences to sample.
        :return: Tuple of (states, policies, rewards).
        """
        batch = random.sample(self.buffer, batch_size)
        states, policies, rewards = zip(*batch)
        return list(states), np.array(policies), np.array(rewards)

    def __len__(self):
        return len(self.buffer)
