# search/mcts.py

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict
import torch
import numpy as np

from node import Node
from env.env import BinPacking3DEnv
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from models.transformer import BinPackingTransformer
from replay_buffer import PrioritizedReplayBuffer  # Import PrioritizedReplayBuffer

class MCTS:
    def __init__(
        self,
        env: BinPacking3DEnv,
        transformer: BinPackingTransformer,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        replay_buffer: PrioritizedReplayBuffer,
        num_simulations: int = 1000,
        c_param: float = math.sqrt(2)
    ):
        """
        Initialize the MCTS search.

        :param env: The environment to search in.
        :param transformer: The Transformer network for feature extraction.
        :param policy_network: The Policy Network for guiding action selection.
        :param value_network: The Value Network for evaluating states.
        :param replay_buffer: The Prioritized Replay Buffer for storing experiences.
        :param num_simulations: The number of simulations to run.
        :param c_param: The exploration parameter for UCB1.
        """
        self.env = env.clone()  # Use the clone method to copy the environment
        self.transformer = transformer
        self.policy_network = policy_network
        self.value_network = value_network
        self.replay_buffer = replay_buffer
        self.num_simulations = num_simulations
        self.c_param = c_param
        self.root = Node(state=self.env)  # Initialize the root node with the cloned state

    def search(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Perform the MCTS search to determine the best action.

        :return: The best action determined by MCTS.
        """
        for _ in range(self.num_simulations):
            node = self.root
            state = self.env.clone()  # Clone the environment for simulation

            # -------------------- SELECTION --------------------
            # Traverse the tree until a node is found that can be expanded
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                if node is None:
                    break
                action = node.action
                _, _, done, _, _ = state.step(action)
                if done:
                    break

            # -------------------- EXPANSION --------------------
            # If the node is not fully expanded, expand it by adding a child
            if not node.is_fully_expanded() and not node.is_terminal:
                child_node = node.expand()
                if child_node is not None:
                    # Apply the action to the cloned state
                    _, _, done, truncated, _ = state.step(child_node.action)
                    child_node.is_terminal = done or truncated
                    node = child_node

                    # -------------------- LƯU TRỪNG TRẠNG THÁI VÀ POLICY CHO CHILD_NODE --------------------
                    # Trích xuất đặc trưng và policy cho child_node ngay tại thời điểm mở rộng
                    observation = child_node.state._get_observation()
                    buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32)
                    ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32)

                    with torch.no_grad():
                        ems_features, item_features = self.transformer(ems_tensor.unsqueeze(0), buffer_tensor.unsqueeze(0))
                        ems_features = ems_features  # [1, d_model]
                        item_features = item_features  # [1, d_model]

                        # Generate action_mask based on child_node's state
                        action_mask = child_node.state.action_mask
                        action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).view(1, -1)  # [1, W * L * num_rotations * buffer_size]

                        policy = self.policy_network(ems_features, item_features, action_mask_tensor)  # [1, output_dim]
                        policy = policy.squeeze(0).cpu().numpy()  # [output_dim]

                    # Lưu policy vào child_node
                    child_node.policy = policy

            # -------------------- SIMULATION (ROLLOUT) --------------------
            # Perform a simulation from the current state
            total_reward, done = self._simulate(state)

            # -------------------- BACKPROPAGATION --------------------
            self._backpropagate(node, total_reward)

            # -------------------- SAVE EXPERIENCE TO REPLAY BUFFER --------------------
            # Sử dụng policy đã lưu trong parent_node để lưu trải nghiệm vào PRB
            parent_node = node.parent
            if parent_node is not None and parent_node.policy is not None:
                parent_state = parent_node.state
                observation = parent_state._get_observation()

                # Reward is the total_reward from simulation
                reward = total_reward

                # Save (state, policy, reward) to replay buffer with priority equal to reward
                self.replay_buffer.add(observation, parent_node.policy, reward)

        # After simulations, select the action with the highest visit count
        best_action = self._get_best_action()
        return best_action

    def _simulate(self, state: BinPacking3DEnv) -> Tuple[float, bool]:
        """
        Perform a simulation (rollout) from the given state to estimate the value.

        :param state: The current state of the environment.
        :return: A tuple containing the total reward from the simulation and the done flag.
        """
        done = False
        total_reward = 0.0

        while not done:
            observation = state._get_observation()
            action_mask = state.action_mask

            # Extract buffer and EMS from observation
            buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32)
            ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32)

            # Pass through Transformer to get features
            with torch.no_grad():
                ems_features, item_features = self.transformer(ems_tensor.unsqueeze(0), buffer_tensor.unsqueeze(0))
                ems_features = ems_features  # [1, d_model]
                item_features = item_features  # [1, d_model]

                # Get policy probabilities from Policy Network
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.float32).view(1, -1)  # [1, W * L * num_rotations * buffer_size]
                policy = self.policy_network(ems_features, item_features, action_mask_tensor)  # [1, output_dim]
                policy = policy.squeeze(0).cpu().numpy()  # [output_dim]

            # Không cần nhân lại với action_mask vì Policy Network đã làm

            # Nếu không có hành động hợp lệ, terminate simulation
            if policy.sum() == 0:
                break

            # Sample an action based on the probabilities
            action_index = np.random.choice(len(policy), p=policy)

            # Decode the action index back to (x, y, rotation, item_index)
            W, L, num_rotations, buffer_size = state.W, state.L, state.num_rotations, state.buffer_size
            total_rot_buffer = num_rotations * buffer_size

            x = action_index // (L * total_rot_buffer)
            remainder = action_index % (L * total_rot_buffer)
            y = remainder // total_rot_buffer
            remainder = remainder % total_rot_buffer
            rotation = remainder // buffer_size
            item_index = remainder % buffer_size

            selected_action = (x, y, rotation, item_index)

            # Apply the action to the state
            _, reward, done, truncated, _ = state.step(selected_action)
            total_reward += reward

            if truncated:
                done = True

        # Use the Value Network to estimate the value of the final state
        final_observation = state._get_observation()
        buffer_tensor = torch.tensor(final_observation['buffer'], dtype=torch.float32)
        ems_tensor = torch.tensor(final_observation['ems'], dtype=torch.float32)

        with torch.no_grad():
            ems_features, item_features = self.transformer(ems_tensor.unsqueeze(0), buffer_tensor.unsqueeze(0))
            value = self.value_network(ems_features, item_features).squeeze(0).item()
            value = torch.tanh(torch.tensor(value)).item()  # Normalize the value to [-1, 1]

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

    def _get_best_action(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Select the best action from the root based on the highest visit count.

        :return: The best action as a tuple.
        """
        if not self.root.children:
            return None  # No actions were explored

        # Select the child with the maximum number of visits
        best_child = max(self.root.children.values(), key=lambda n: n.visits)
        return best_child.action
