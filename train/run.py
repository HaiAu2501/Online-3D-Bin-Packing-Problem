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
    # Environment initialization
    bin_size, items = read_data("boxes.dat")

    env = BinPacking3DEnv(bin_size=bin_size, items=items)

    # Models initialization
    transformer = BinPackingTransformer(
        d_model=128, 
        nhead=8, 
        num_layers=3, 
        dim_feedforward=256, 
        max_ems=100
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

    # Prioritized Replay Buffer initialization
    prb = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)

    # Trainer initialization
    trainer = Trainer(
        env=env,
        transformer=transformer,
        policy_network=policy_net,
        value_network=value_net,
        replay_buffer=prb,
        num_simulations=100,
        batch_size=64,
        gamma=0.99,
        lr_policy=1e-4,
        lr_value=1e-4,
        beta_start=0.4,
        beta_frames=100000,
        save_path="./models/",
        verbose=True
    )

    # Train the model
    trainer.train(num_episodes=100, update_every=1)

if __name__ == "__main__":
    main()
