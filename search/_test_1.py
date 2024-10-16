# test_mcts_integration.py

import torch
import numpy as np
import math

from mcts import MCTS
from replay_buffer import PrioritizedReplayBuffer
from models.transformer import BinPackingTransformer
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork


def test_mcts_integration():
    # Thiết lập thông số
    batch_size = 1  # Trong MCTS, thường là một batch
    num_ems = 10
    num_items = 15
    d_model = 128
    hidden_dim = 256
    W = 10
    L = 10
    num_rotations = 4
    buffer_size = 20
    max_ems = 1000  # Ví dụ: W * L * H
    num_simulations = 10  # Giảm số lượng mô phỏng cho test
    
    # Tạo dummy environment
    env = BinPacking3DEnv(W=W, L=L, H=10, buffer_size=buffer_size)  # Điều chỉnh H theo yêu cầu

    # Tạo dummy inputs cho Transformer
    ems_input = torch.randint(0, 100, (batch_size, num_ems, 6)).float()
    buffer_input = torch.randint(0, 50, (batch_size, num_items, 3)).float()

    # Khởi tạo các mạng neural
    transformer = BinPackingTransformer(
        d_model=d_model,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        max_ems=max_ems
    )
    
    policy_net = PolicyNetwork(
        d_model=d_model,
        hidden_dim=hidden_dim,
        W=W,
        L=L,
        num_rotations=num_rotations,
        buffer_size=buffer_size
    )
    
    value_net = ValueNetwork(
        d_model=d_model,
        hidden_dim=hidden_dim
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
        c_param=math.sqrt(2)
    )
    
    # Chạy MCTS search
    best_action = mcts.search()
    
    # In kết quả
    print("Best Action:", best_action)
    print("Replay Buffer Size:", len(replay_buffer))
    
    # Kiểm tra nội dung của PRB
    if len(replay_buffer) > 0:
        sample_state, sample_policy, sample_reward = replay_buffer.sample(1)
        print("Sample from Replay Buffer:")
        print("State:", sample_state[0])
        print("Policy:", sample_policy[0])
        print("Reward:", sample_reward[0])

if __name__ == "__main__":
    test_mcts_integration()
