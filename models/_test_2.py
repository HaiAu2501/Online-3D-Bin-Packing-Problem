# test_model.py

import torch
import torch.nn as nn
from transformer import BinPackingTransformer
from policy_net import PolicyNetwork
from value_net import ValueNetwork

def test_networks():
    # Thiết lập thông số
    batch_size = 2
    num_ems = 10
    num_items = 15
    d_model = 128
    hidden_dim = 256
    W = 10
    L = 10
    H = 10
    num_rotations = 2
    buffer_size = 2
    max_ems = W * L * H
    
    # Tạo dummy inputs
    ems_input = torch.randint(0, 100, (batch_size, num_ems, 6)).float()
    buffer_input = torch.randint(0, 50, (batch_size, num_items, 3)).float()
    
    # Tạo action_mask (ví dụ: tất cả các hành động đều hợp lệ)
    action_mask = torch.ones(batch_size, W * L * num_rotations * buffer_size).float()
    
    # Khởi tạo mô hình
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
    
    # Forward pass qua Transformer
    ems_features, item_features = transformer(ems_input, buffer_input)
    
    # Forward pass qua Policy Network
    action_probabilities = policy_net(ems_features, item_features, action_mask)
    
    # Forward pass qua Value Network
    state_value = value_net(ems_features, item_features)
    
    # In kết quả
    print("EMS Features Shape:", ems_features.shape)          # Expected: [batch_size, d_model]
    print("Item Features Shape:", item_features.shape)        # Expected: [batch_size, d_model]
    print("Action Probabilities Shape:", action_probabilities.shape)  # Expected: [batch_size, W * L * num_rotations * buffer_size]
    print("State Value Shape:", state_value.shape)            # Expected: [batch_size, 1]

if __name__ == "__main__":
    test_networks()
