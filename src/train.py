# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from networks import PolicyNetwork, ValueNetwork
from environment import BinPacking3DEnv
from mcts import MCTS
from replay_buffer import PrioritizedReplayBuffer
from typing import Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ boxes.dat
def read_boxes(file_path: str) -> List[Tuple[int, int, int]]:
    boxes = []
    with open(file_path, "r") as f:
        for line in f:
            width, depth, height = map(int, line.strip().split())
            boxes.append((width, depth, height))
    return boxes

def train(file_path: str = "boxes.dat"):
    # Thiết lập môi trường
    bin_size = (10, 10, 10)  # Ví dụ kích thước thùng
    items = read_boxes(file_path)
    env = BinPacking3DEnv(bin_size, items)
    
    # Khởi tạo mạng Policy và Value
    policy_net = PolicyNetwork(input_channels=4, num_rotations=6)  # 6 hướng xoay
    value_net = ValueNetwork(input_channels=4)
    
    # Thiết lập optimizer
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=1e-3)
    
    # Bộ nhớ replay
    replay_buffer = PrioritizedReplayBuffer(capacity=10000)
    
    # Thiết lập MCTS
    mcts = MCTS(policy_net, value_net, env, num_simulations=50)
    
    # Số lượng epochs huấn luyện
    num_episodes = 100
    
    # Danh sách lưu trữ loss và reward
    loss_history = []
    reward_history = []
    
    # Vòng lặp huấn luyện
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Tìm kiếm MCTS để chọn hành động
            policy = mcts.search(state)
            
            if not policy:
                # Không có hành động hợp lệ, kết thúc episode
                print(f"Episode {episode}: No valid actions found. Ending episode.")
                done = True
                break
            
            action = select_action(policy)
            # print(f"Episode {episode}: Selected action {action}")
            
            # Thực hiện hành động
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            # print(f"Episode {episode}: Reward {reward}, Done {done}")
            
            # Lưu trải nghiệm vào bộ nhớ với độ ưu tiên bằng phần thưởng
            replay_buffer.add((state, action, reward), priority=reward)
            # print(f"Episode {episode}: Added to replay buffer with priority {reward}")
            
            state = next_state
        
        # Huấn luyện mạng sau mỗi episode
        if len(replay_buffer) >= 32:
            batch = replay_buffer.sample(batch_size=32)
            loss = compute_loss(batch, policy_net, value_net)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            # print(f"Episode {episode}: Loss {loss.item():.4f}")
        else:
            loss_history.append(None)  # Placeholder khi chưa đủ batch size
        
        reward_history.append(total_reward)
        
        # In thông tin huấn luyện
        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{num_episodes} - Total Reward: {total_reward:.4f} - Buffer Size: {len(replay_buffer)}")
    
    # Vẽ đồ thị loss qua các episode
    plot_loss(loss_history, num_episodes)
    
    # Vẽ đồ thị reward qua các episode
    plot_reward(reward_history, num_episodes)

def select_action(policy: dict) -> int:
    """
    Chọn hành động dựa trên phân phối chính sách từ MCTS.
    """
    actions, probabilities = zip(*policy.items())
    probabilities = torch.tensor(probabilities)
    if probabilities.sum() == 0:
        probabilities = torch.ones_like(probabilities) / len(probabilities)
    else:
        probabilities = probabilities / probabilities.sum()
    action = np.random.choice(actions, p=probabilities.numpy())
    return action

def compute_loss(batch: List[Tuple[Any, Any, float]], policy_net: PolicyNetwork, value_net: ValueNetwork) -> torch.Tensor:
    """
    Tính toán hàm mất mát cho mạng Policy và Value.
    """
    states, actions, rewards = zip(*batch)
    
    # Tối ưu hóa việc chuyển đổi states thành numpy array trước khi chuyển sang tensor
    states = np.array(states)
    states = torch.tensor(states, dtype=torch.float32)
    
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    
    # Tính dự đoán của mạng Policy và Value
    policy_logits = policy_net(states)  # Output: (batch_size, num_rotations)
    value_preds = value_net(states).squeeze()  # Output: (batch_size)
    
    # Loss cho Value network
    value_loss = F.mse_loss(value_preds, rewards)
    
    # Extract rotation indices from actions
    rotation_indices = [decode_action(action)[2] for action in actions.tolist()]
    rotation_indices = torch.tensor(rotation_indices, dtype=torch.long)
    
    # Loss cho Policy network
    policy_targets = F.one_hot(rotation_indices, num_classes=policy_logits.size(-1)).float()
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = - (policy_targets * policy_log_probs).sum(dim=-1)
    policy_loss = (rewards * policy_loss).mean()
    
    # Tổng hợp loss với trọng số
    loss = value_loss + policy_loss
    return loss

def decode_action(action: int, bin_size: Tuple[int, int, int] = (10, 10, 10), num_rotations: int = 6) -> Tuple[int, int, int]:
    """
    Giải mã hành động thành vị trí x, y và hướng xoay.
    Giả sử không gian hành động được định nghĩa như trong BinPacking3DEnv.
    """
    rotation = action % num_rotations
    pos = action // num_rotations
    W, L, H = bin_size
    x = pos // L
    y = pos % L
    return x, y, rotation

def plot_loss(loss_history: List[float], num_episodes: int):
    """
    Vẽ đồ thị loss qua các episode.
    """
    episodes = range(1, num_episodes + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, loss_history, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_reward(reward_history: List[float], num_episodes: int):
    """
    Vẽ đồ thị reward qua các episode.
    """
    episodes = range(1, num_episodes + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, reward_history, label='Total Reward', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
