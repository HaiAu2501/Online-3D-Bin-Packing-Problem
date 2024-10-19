# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt

from env.env import BinPacking3DEnv
from search.replay_buffer import PrioritizedReplayBuffer
from search.node import Node
from search.mcts import MCTS
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from models.transformer import BinPackingTransformer  # Import transformer
import os

class Trainer:
    def __init__(
        self,
        env: BinPacking3DEnv,
        transformer: BinPackingTransformer,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        replay_buffer: PrioritizedReplayBuffer,
        num_simulations: int = 1000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr_policy: float = 1e-4,
        lr_value: float = 1e-3,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        save_path: str = "./models/",
        verbose: bool = False
    ):
        """
        Initialize the Trainer.

        :param env: The bin packing environment.
        :param transformer: The Transformer model for feature extraction.
        :param policy_network: The Policy Network.
        :param value_network: The Value Network.
        :param replay_buffer: The Prioritized Replay Buffer.
        :param num_simulations: Number of MCTS simulations per episode.
        :param batch_size: Batch size for training.
        :param gamma: Discount factor.
        :param lr_policy: Learning rate for Policy Network.
        :param lr_value: Learning rate for Value Network.
        :param beta_start: Initial value of beta for importance sampling.
        :param beta_frames: Number of frames over which beta will be annealed from beta_start to 1.
        :param save_path: Directory to save the trained models.
        """
        self.env = env

        # Choose device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move models to device
        self.transformer = transformer.to(self.device)
        self.policy_network = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)
        self.replay_buffer = replay_buffer
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.save_path = save_path

        # Define optimizers
        self.optimizer_policy = optim.Adam(self.policy_network.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(self.value_network.parameters(), lr=lr_value)

        self.verbose = verbose

        # Create directory to save models if it doesn't exist
        os.makedirs(self.save_path, exist_ok=True)

        self.policy_losses = []
        self.value_losses = []
        self.total_losses = []

        self.best_episode_rewards = []

    def train(self, num_episodes: int = 10000, update_every: int = 10):
        """
        Train the Policy and Value networks using MCTS and PRB.

        :param num_episodes: Number of training episodes.
        :param update_every: Number of episodes between each network update.
        """
        for episode in range(1, num_episodes + 1):
            if self.verbose:
                print(f"Episode {episode}/{num_episodes} - Starting episode.")
            self.env.reset()
            mcts = MCTS(
                env=self.env,
                transformer=self.transformer,
                policy_network=self.policy_network,
                value_network=self.value_network,
                num_simulations=self.num_simulations
            )

            # Perform MCTS search to generate data
            mcts.search()

            # Collect experiences from the MCTS tree
            experiences = self._collect_experiences(mcts.root)

            # Tính tổng reward cho episode hiện tại bằng cách chỉ cộng các reward từ các hành động thực sự
            best_reward = max(mcts.total_reward)
            print(f"Episode {episode} - Best Reward: {best_reward}")

            # Add experiences to the replay buffer
            for exp in experiences:
                state, action_index, reward, next_state, done = exp
                priority = reward  # You can adjust priority based on reward or another metric
                self.replay_buffer.add(exp, priority)

            # Update networks every 'update_every' episodes
            if episode % update_every == 0:
                self._update_networks()
                if self.verbose:
                    print(f"Episode {episode}/{num_episodes} - Networks updated.")

            # Save models periodically
            if episode % 1000 == 0:
                self._save_models(episode)
                if self.verbose:
                    print(f"Episode {episode}/{num_episodes} - Models saved.")

        # Sau khi huấn luyện xong, vẽ đồ thị loss và reward
        self.plot_losses(save_fig=True, fig_path="training_losses.png")
        self.plot_rewards(save_fig=True, fig_path="training_rewards.png")
        if self.verbose:
            print("Training completed. Loss and Reward plots saved.")

    def _collect_experiences(self, node: Node, parent_state: Optional[BinPacking3DEnv] = None) -> List[Tuple]:
        """
        Traverse the MCTS tree và thu thập các trải nghiệm, lưu trữ hành động dưới dạng chỉ số duy nhất.
        """
        experiences = []
        if node.parent is not None and parent_state is not None:
            # Clone state từ parent_state và thực hiện hành động
            state = parent_state.clone()
            _, reward, done, _, _ = state.step(node.action)
            next_state = state.clone()

            L = state.L
            num_rotations = state.num_rotations
            buffer_size = state.buffer_size
            total_rot_buffer = num_rotations * buffer_size

            x, y, rotation, item_index = node.action
            action_index = x * (L * total_rot_buffer) + y * total_rot_buffer + rotation * buffer_size + item_index

            # Sử dụng phần thưởng thực tế làm reward
            experiences.append((parent_state._get_observation(), action_index, reward, next_state._get_observation(), done))

        for child in node.children.values():
            experiences.extend(self._collect_experiences(child, node.state))

        return experiences

    def _update_networks(self):
        """
        Sample a batch from the replay buffer and perform a training step for both networks.
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        # Anneal beta from beta_start to 1 over time
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        # Sample a batch of experiences
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size, beta)

        # Convert batch of experiences to tensors
        batch = list(zip(*experiences))
        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        next_states = batch[3]
        dones = batch[4]

        # Process states and next_states for Policy and Value networks
        state_inputs = [self._prepare_input(state) for state in states]
        next_state_inputs = [self._prepare_input(state) for state in next_states]

        # Stack tensors
        buffer_states = torch.cat([s['buffer'] for s in state_inputs], dim=0).to(self.transformer.device)      # [batch_size, buffer_size, 3]
        ems_states = torch.cat([s['ems'] for s in state_inputs], dim=0).to(self.transformer.device)            # [batch_size, max_ems, 6]
        ems_masks = torch.cat([s['ems_mask'] for s in state_inputs], dim=0).to(self.transformer.device)        # [batch_size, max_ems]
        action_masks = torch.cat([s['action_mask'] for s in state_inputs], dim=0).to(self.transformer.device)  # [batch_size, action_dim]

        buffer_next_states = torch.cat([s['buffer'] for s in next_state_inputs], dim=0).to(self.transformer.device)  # [batch_size, buffer_size, 3]
        ems_next_states = torch.cat([s['ems'] for s in next_state_inputs], dim=0).to(self.transformer.device)        # [batch_size, max_ems, 6]
        ems_next_masks = torch.cat([s['ems_mask'] for s in next_state_inputs], dim=0).to(self.transformer.device)    # [batch_size, max_ems]

        actions = torch.tensor(actions, dtype=torch.long).to(self.transformer.device)        # [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.transformer.device)     # [batch_size]
        dones = torch.tensor(dones, dtype=torch.float32).to(self.transformer.device)         # [batch_size]
        weights = torch.tensor(weights, dtype=torch.float32).to(self.transformer.device)     # [batch_size]

        # -------------------- Train Policy Network --------------------
        self.transformer.train()
        self.policy_network.train()
        self.value_network.train()

        # Pass through transformer to get features
        ems_features, item_features = self.transformer(
            ems_states,
            buffer_states,
            ems_mask=ems_masks, # Or create buffer_mask if needed
        )

        # Pass through Policy Network
        policy_probs = self.policy_network(ems_features, item_features, action_masks)  # [batch_size, action_dim]

        # Gather log probabilities of the selected actions
        log_probs = torch.log(policy_probs.gather(1, actions.unsqueeze(1)) + 1e-10).squeeze(1)  # [batch_size]

        # Compute policy loss (negative log likelihood)
        policy_loss = -log_probs * weights

        # Predict state values
        state_values = self.value_network(ems_features, item_features).squeeze(1)  # [batch_size]

        # Pass through transformer for next states
        ems_next_features, item_next_features = self.transformer(
            ems_next_states,
            buffer_next_states,
            ems_mask=ems_next_masks,
        )

        # Predict next state values
        next_values = self.value_network(ems_next_features, item_next_features).squeeze(1)  # [batch_size]

        # Compute target values
        targets = rewards + self.gamma * next_values * (1 - dones)

        # Compute value loss (mean squared error)
        value_loss = (state_values - targets) ** 2 * weights

        # Tính tổng loss
        total_loss = policy_loss.mean() + value_loss.mean()  # [scalar]

        # Ghi lại các giá trị loss
        self.policy_losses.append(policy_loss.mean().item())
        self.value_losses.append(value_loss.mean().item())
        self.total_losses.append(total_loss.item())
 
        # Optimize cả Policy Network và Value Network
        self.optimizer_policy.zero_grad()
        self.optimizer_value.zero_grad()
        total_loss.backward()
        self.optimizer_policy.step()
        self.optimizer_value.step()

        # -------------------- Update Priorities in Replay Buffer --------------------
        # Use the TD error as priority
        td_errors = (state_values - targets).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-5)  # Add a small epsilon to avoid zero priority

    def _prepare_input(self, observation: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Prepare the input tensors for the networks from the observation.

        :param observation: The observation dictionary containing 'buffer' and 'ems'.
        :return: A dictionary with prepared tensors.
        """
        buffer = torch.tensor(observation['buffer'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, buffer_size, 3]
        ems = torch.tensor(observation['ems'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, max_ems, 6]
        ems_mask = torch.zeros(1, self.transformer.max_ems).bool().to(ems.device)
        
        # Giả sử các EMS giả được đánh dấu False trong mask
        ems_mask[:, :self._count_real_ems(observation['ems'])] = True  # True cho các EMS thực, False cho padding

        # Generate action_mask from state
        action_mask_np = self.env.generate_action_mask()
        action_mask = torch.tensor(action_mask_np, dtype=torch.float32).view(1, -1).to(self.transformer.device)  # [1, action_dim]

        return {
            'buffer': buffer,
            'ems': ems,
            'ems_mask': ems_mask,
            'action_mask': action_mask
        }

    def _count_real_ems(self, ems_array: np.ndarray) -> int:
        """
        Đếm số lượng EMS thực trong mảng EMS.

        :param ems_array: Mảng EMS hiện tại.
        :return: Số lượng EMS thực.
        """
        # Giả sử các EMS giả là toàn bộ 0
        return np.sum(np.any(ems_array != 0, axis=1))

    def _save_models(self, episode: int):
        """
        Save the Policy, Value networks and Transformer.

        :param episode: The current episode number (used in the filename).
        """
        policy_path = os.path.join(self.save_path, f"policy_net_episode_{episode}.pth")
        value_path = os.path.join(self.save_path, f"value_net_episode_{episode}.pth")
        transformer_path = os.path.join(self.save_path, f"transformer_episode_{episode}.pth")
        torch.save(self.policy_network.state_dict(), policy_path)
        torch.save(self.value_network.state_dict(), value_path)
        torch.save(self.transformer.state_dict(), transformer_path)

    def _load_models(self, episode: int):
        """
        Load the Policy, Value networks and Transformer from saved files.

        :param episode: The episode number of the saved models.
        """
        policy_path = os.path.join(self.save_path, f"policy_net_episode_{episode}.pth")
        value_path = os.path.join(self.save_path, f"value_net_episode_{episode}.pth")
        transformer_path = os.path.join(self.save_path, f"transformer_episode_{episode}.pth")
        self.policy_network.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.value_network.load_state_dict(torch.load(value_path, map_location=self.device))
        self.transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))

    def plot_losses(self, save_fig: bool = False, fig_path: str = "loss_plot.png"):
        """
        Vẽ đồ thị tổng loss (policy_loss, value_loss, total_loss).

        :param save_fig: Nếu True, lưu đồ thị vào file.
        :param fig_path: Đường dẫn file để lưu đồ thị nếu save_fig là True.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.policy_losses, label='Policy Loss')
        plt.plot(self.value_losses, label='Value Loss')
        plt.plot(self.total_losses, label='Total Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(fig_path)
            print(f"Loss plot saved to {fig_path}")
        else:
            plt.show()

    def plot_rewards(self, save_fig: bool = False, fig_path: str = "reward_plot.png"):
        """
        Vẽ đồ thị tổng reward qua các episode.

        :param save_fig: Nếu True, lưu đồ thị vào file.
        :param fig_path: Đường dẫn file để lưu đồ thị nếu save_fig là True.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label='Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward Over Episodes')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_fig:
            plt.savefig(fig_path)
            print(f"Reward plot saved to {fig_path}")
        else:
            plt.show()