# main.py

import sys
import os

sys.path.append(os.getcwd())

from env.env import BinPacking3DEnv
from models.transformer import BinPackingTransformer
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from search.replay_buffer import PrioritizedReplayBuffer
from trainer import Trainer

def read_data(file_path):
    # First line is the bin size
    # Next lines are the items
    with open(file_path, 'r') as f:
        lines = f.readlines()
        bin_size = tuple(map(int, lines[0].strip().split()))
        items = [tuple(map(int, line.strip().split())) for line in lines[1:]]
    return bin_size, items

def main():
    # Khởi tạo môi trường
    bin_size, items = read_data("F:\\.GITHUB\\src\\boxes.dat")

    env = BinPacking3DEnv(bin_size=bin_size, items=items)

    # Khởi tạo mô hình
    transformer = BinPackingTransformer(
        d_model=128, 
        nhead=8, 
        num_layers=3, 
        dim_feedforward=512, 
        max_ems=env.W * env.L * env.H
    )
    
    policy_net = PolicyNetwork(
        d_model=128, 
        hidden_dim=256, 
        W=env.W, 
        L=env.L, 
        num_rotations=env.num_rotations, 
        buffer_size=env.buffer_size
    )
    
    value_net = ValueNetwork(
        d_model=128, 
        hidden_dim=256
    )

    # Khởi tạo Prioritized Replay Buffer
    prb = PrioritizedReplayBuffer(capacity=100000, alpha=0.6)

    # Khởi tạo Trainer
    trainer = Trainer(
        env=env,
        transformer=transformer,
        policy_network=policy_net,
        value_network=value_net,
        replay_buffer=prb,
        num_simulations=1000,
        batch_size=64,
        gamma=0.99,
        lr_policy=1e-4,
        lr_value=1e-3,
        beta_start=0.4,
        beta_frames=100000,
        save_path="./models/"
    )

    # Bắt đầu quá trình huấn luyện
    trainer.train(num_episodes=10000, update_every=10)

if __name__ == "__main__":
    main()
