import numpy as np
import random
from collections import deque
import torch

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 # Khung hình hiện tại

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, state, action_mask, action, reward, next_state, done, value, policy):
        experience = (state, action_mask, action, reward, next_state, done, value, policy)
        self.buffer.append(experience)
        # Gán priority ban đầu bằng max priority đã có (hoặc 1.0 nếu buffer rỗng)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def sample(self, batch_size: int):
        # Tính beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Lấy mẫu dựa trên priority
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Tính trọng số importance sampling (IS)
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() # Chuẩn hóa trọng số

        state_batch, action_mask_batch, action_batch, reward_batch, next_state_batch, done_batch, value_batch, policy_batch = zip(*samples)
        
        buffers = [state['buffer'] for state in state_batch]
        ems_lists = [state['ems_list'] for state in state_batch]

        next_buffers = [state['buffer'] for state in next_state_batch]
        next_ems_lists = [state['ems_list'] for state in next_state_batch]

        return {
            'buffer': np.array(buffers),
            'ems_list': np.array(ems_lists),
            'action_mask': np.array(action_mask_batch),
            'action': np.array(action_batch),
            'reward': np.array(reward_batch),
            'next_buffer': np.array(next_buffers),
            'next_ems_list': np.array(next_ems_lists),
            'done': np.array(done_batch),
            'value': np.array(value_batch),
            'policy': np.array(policy_batch),
            'indices': indices,
            'weights': weights
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Cập nhật priority cho các mẫu đã được lấy mẫu.

        :param indices: Indices của các mẫu.
        :param priorities: Giá trị priority mới.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5 # Cộng thêm một giá trị nhỏ để tránh priority bằng 0

    def __len__(self):
        return len(self.buffer)