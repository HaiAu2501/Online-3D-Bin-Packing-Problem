# search/mcts.py

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import numpy as np

from node import Node
from env.env import BinPacking3DEnv
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork
from models.transformer import BinPackingTransformer
from replay_buffer import PrioritizedReplayBuffer

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
            node: Node = self.root
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

            # -------------------- SIMULATION (ROLLOUT) --------------------
            # Perform a simulation from the current state
            total_reward, done = self._simulate(state)

            # -------------------- BACKPROPAGATION --------------------
            self._backpropagate(node, total_reward)

            # -------------------- SAVE EXPERIENCE TO REPLAY BUFFER --------------------
            # Extract state before action, policy, and reward
            # Here, we assume that the node corresponds to a specific state
            # and that the action taken leads to the current node
            # You might need to adjust based on your Node and environment implementation

            # Get observation from the parent node's state
            parent_state = node.parent.state if node.parent else self.env
            observation = parent_state._get_observation()

            # Extract buffer and EMS from observation
            buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32)
            ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32)

            # Pass through Transformer to get features
            ems_features, item_features = self.transformer(buffer_tensor.unsqueeze(0), ems_tensor.unsqueeze(0))
            ems_features = ems_features.detach()  # Detach to prevent gradient computation
            item_features = item_features.detach()

            # Pass features through Policy Network to get policy
            with torch.no_grad():
                policy = self.policy_network(ems_features, item_features, action_mask=torch.ones(1, self.transformer.W * self.transformer.L * self.transformer.num_rotations * self.transformer.buffer_size))
                policy = policy.squeeze(0).cpu().numpy()  # Convert to numpy array

            # Reward is the total_reward from simulation
            reward = total_reward

            # Save (state, policy, reward) to replay buffer with priority equal to reward
            self.replay_buffer.add(observation, policy, reward)

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
            action_mask = state.generate_action_mask()

            # Extract buffer and EMS from observation
            buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32)
            ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32)

            # Pass through Transformer to get features
            ems_features, item_features = self.transformer(buffer_tensor.unsqueeze(0), ems_tensor.unsqueeze(0))
            ems_features = ems_features.detach()  # Detach to prevent gradient computation
            item_features = item_features.detach()

            # Get policy probabilities from Policy Network
            policy = self.policy_network(ems_features, item_features, action_mask=torch.tensor(action_mask).unsqueeze(0))
            policy = policy.squeeze(0).cpu().numpy()  # Convert to numpy array

            # Apply action mask: set probabilities of invalid actions to 0
            policy *= action_mask.flatten()

            # Normalize the policy to ensure it sums to 1
            if policy.sum() == 0:
                # If no valid actions, terminate simulation
                break
            policy /= policy.sum()

            # Sample an action based on the probabilities
            action_index = np.random.choice(len(policy), p=policy)

            # Decode the action index back to (x, y, rotation, item_index)
            W, L, num_rotations, buffer_size = self.transformer.W, self.transformer.L, self.transformer.num_rotations, self.transformer.buffer_size
            total_rot_buffer = num_rotations * buffer_size
            total_y_buffer = L * total_rot_buffer

            x = action_index // (L * total_rot_buffer)
            y = (action_index % (L * total_rot_buffer)) // total_rot_buffer
            rotation = (action_index % total_rot_buffer) // buffer_size
            item_index = action_index % buffer_size

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
        ems_features, item_features = self.transformer(buffer_tensor.unsqueeze(0), ems_tensor.unsqueeze(0))
        ems_features = ems_features.detach()
        item_features = item_features.detach()

        with torch.no_grad():
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
