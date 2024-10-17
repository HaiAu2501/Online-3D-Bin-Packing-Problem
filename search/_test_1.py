# test_mcts_integration.py

import torch
import numpy as np
import math
import sys
import os

sys.path.append(os.getcwd())

from mcts import MCTS
from models.transformer import BinPackingTransformer
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from replay_buffer import PrioritizedReplayBuffer
from env.env import BinPacking3DEnv

def test_mcts_integration():
    # Thiết lập thông số
    bin_size = (10, 10, 10)  # W, L, H
    items = [
        (2, 2, 2),
        (3, 3, 3),
        (1, 4, 2),
        (2, 1, 3),
        (3, 2, 1)
    ]
    buffer_size = 2
    num_rotations = 2
    max_ems = bin_size[0] * bin_size[1] * bin_size[2]  # 1000
    num_simulations = 10  # Giảm số lượng mô phỏng cho test
    c_param = math.sqrt(2)

    # Tạo dummy environment
    env = BinPacking3DEnv(
        bin_size=bin_size,
        items=items,
        buffer_size=buffer_size,
        num_rotations=num_rotations
    )

    # Khởi tạo các mạng neural
    transformer = BinPackingTransformer(
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_ems=max_ems
    )
    
    policy_net = PolicyNetwork(
        d_model=128,
        hidden_dim=256,
        W=bin_size[0],
        L=bin_size[1],
        num_rotations=num_rotations,
        buffer_size=buffer_size
    )
    
    value_net = ValueNetwork(
        d_model=128,
        hidden_dim=256
    )
    
    # Khởi tạo Prioritized Replay Buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=10000)
    
    # Khởi tạo MCTS
    mcts = MCTS(
        env=env,
        transformer=transformer,
        policy_network=policy_net,
        value_network=value_net,
        replay_buffer=replay_buffer,
        num_simulations=num_simulations,
        c_param=c_param
    )
    
    # Chạy MCTS search
    best_action = mcts.search()
    
    # In kết quả
    print("Best Action:", best_action)
    print("Replay Buffer Size:", len(replay_buffer))
    
    # Kiểm tra nội dung của PRB
    if len(replay_buffer) > 0:
        sample_states, sample_policies, sample_rewards = replay_buffer.sample(1)
        print("\nSample from Replay Buffer:")
        print("State:")
        print(sample_states[0])  # In state dictionary
        print("\nPolicy:")
        print(sample_policies[0])  # In policy array
        print("\nReward:")
        print(sample_rewards[0])  # In reward value

if __name__ == "__main__":
    test_mcts_integration()
