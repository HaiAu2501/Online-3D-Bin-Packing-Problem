# search/mcts.py

from __future__ import annotations

import math
from typing import Optional, Tuple, List, TYPE_CHECKING
import torch
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

from .node import Node
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from models.transformer import BinPackingTransformer  # Import transformer

class MCTS:
    def __init__(
        self,
        env: BinPacking3DEnv,
        transformer: BinPackingTransformer,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        num_simulations: int = 1000,
        c_param: float = math.sqrt(2)
    ):
        """
        Initialize the MCTS search.

        :param env: The environment to search in.
        :param transformer: The Transformer model for feature extraction.
        :param policy_network: The Policy Network.
        :param value_network: The Value Network.
        :param num_simulations: The number of simulations to run.
        :param c_param: The exploration parameter for UCB1.
        """
        self.device = transformer.device
        self.env = env.clone()
        self.transformer = transformer.to(self.device)
        self.policy_network = policy_network.to(self.device)
        self.value_network = value_network.to(self.device)
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.root = Node(state=self.env)  # Initialize the root node with the cloned state

    def search(self):
        """
        Run the MCTS search without returning a best action.
        """
        for _ in range(self.num_simulations):
            node = self.root
            state = self.env.clone()

            # -------------------- SELECTION --------------------
            # Traverse the tree until a node is found that can be expanded
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                action = node.action
                _, _, done, _, _ = state.step(action)
                if done:
                    break

            # -------------------- EXPANSION --------------------
            # If the node is not fully expanded and not terminal, expand it by adding a child
            if not node.is_fully_expanded() and not node.is_terminal:
                child_node = node.expand()
                if child_node is not None:
                    _, _, done, truncated, _ = state.step(child_node.action)
                    child_node.is_terminal = done or truncated
                    node = child_node

            # -------------------- SIMULATION (ROLLOUT) --------------------
            # Perform a simulation from the current state
            total_reward, done = self._simulate(state)

            # -------------------- BACKPROPAGATION --------------------
            self._backpropagate(node, total_reward)

    def _simulate(self, state: BinPacking3DEnv) -> Tuple[float, bool]:
        """
        Perform a simulation (rollout) from the given state to estimate the value.

        :param state: The current state of the environment.
        :return: A tuple containing the total reward from the simulation and the done flag.
        """
        done = False
        total_reward = 0.0

        self.transformer.eval()
        self.policy_network.eval()
        self.value_network.eval()

        with torch.no_grad():  # Bắt đầu không tính gradient
            while not done:
                observation = state._get_observation()
                action_mask = state.action_mask

                # Prepare input for the transformer
                ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, num_ems, 6]
                buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, num_items, 3]

                # Create masks
                num_ems = observation['ems'].shape[0]
                ems_mask = torch.zeros(1, self.transformer.max_ems).bool().to(ems_tensor.device)
                ems_mask[:, :num_ems] = True  # True cho các phần thực, False cho padding

                # Convert action_mask from numpy to torch tensor, flatten it, and move to device
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).view(1, -1).to(ems_tensor.device)  # [1, action_dim]

                # Pass through transformer to get features
                ems_features, item_features = self.transformer(
                    ems_tensor,
                    buffer_tensor,
                    ems_mask=ems_mask,
                    buffer_mask=None  # You can create buffer_mask if needed
                )

                # Pass through Policy Network
                action_probs = self.policy_network(ems_features, item_features, action_mask_tensor)  # [1, action_dim]

                # Kiểm tra tổng xác suất
                sum_probs = action_probs.sum(dim=1, keepdim=True)  # [1, 1]
                if sum_probs.item() <= 0:
                    # Không có hành động hợp lệ, kết thúc simulation
                    _, _, done, truncated, _ = state.step(None)
                    if truncated:
                        done = True
                    continue

                # Sample an action based on probabilities
                action = torch.multinomial(action_probs, num_samples=1).item()

                # Decode action index to (x, y, rotation, item_index)
                W, L, num_rotations, buffer_size = state.W, state.L, state.num_rotations, state.buffer_size
                total_rot_buffer = num_rotations * buffer_size

                x = action // (L * total_rot_buffer)
                y = (action % (L * total_rot_buffer)) // total_rot_buffer
                rotation = (action % total_rot_buffer) // buffer_size
                item_index = action % buffer_size

                selected_action = (x, y, rotation, item_index)

                # Apply the action to the state
                _, reward, done, truncated, _ = state.step(selected_action)
                total_reward += reward

                if truncated:
                    done = True

            # Use the value network to estimate the value of the final state
            final_observation = state._get_observation()
            final_ems_tensor = torch.tensor(final_observation['ems'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, num_ems,6]
            final_buffer_tensor = torch.tensor(final_observation['buffer'], dtype=torch.float32).unsqueeze(0).to(self.transformer.device)  # [1, num_items,3]

            # Create masks for final state
            final_num_ems = final_observation['ems'].shape[0]
            final_ems_mask = torch.zeros(1, self.transformer.max_ems).bool().to(final_ems_tensor.device)
            final_ems_mask[:, :final_num_ems] = True

            # Get features from transformer
            final_ems_features, final_item_features = self.transformer(
                final_ems_tensor,
                final_buffer_tensor,
                ems_mask=final_ems_mask,
                buffer_mask=None
            )

            # Get value from Value Network
            value = self.value_network(final_ems_features, final_item_features).squeeze(1).item()
            value = torch.tanh(torch.tensor(value)).item()  # Normalize the value

            total_reward += value  # Combine simulation reward with value network's estimate

        return total_reward, done

    def _backpropagate(self, node: Node, reward: float):
        """
        Backpropagate the reward through the nodes from the given node to the root.

        :param node: The node to start backpropagation.
        :param reward: The reward to backpropagate.
        """
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
