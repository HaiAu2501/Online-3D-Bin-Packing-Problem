# search/mcts.py

from __future__ import annotations

import math
from typing import Optional, Tuple, List
import torch
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

from node import Node
from models.policy_net import PolicyNetwork
from models.value_net import ValueNetwork

class MCTS:
    class NodeWrapper:
        """
        A wrapper to hold additional information for training.
        """
        def __init__(self, node: Node, parent_state: Optional[BinPacking3DEnv] = None):
            self.node = node
            self.parent_state = parent_state

    def __init__(
        self,
        env: BinPacking3DEnv,
        policy_network: PolicyNetwork,
        value_network: ValueNetwork,
        num_simulations: int = 1000,
        c_param: float = math.sqrt(2)
    ):
        """
        Initialize the MCTS search.

        :param env: The environment to search in.
        :param policy_network: The Policy Network.
        :param value_network: The Value Network.
        :param num_simulations: The number of simulations to run.
        :param c_param: The exploration parameter for UCB1.
        """
        self.env = env.clone()  # Use the clone method to copy the environment
        self.policy_network = policy_network
        self.value_network = value_network
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

        while not done:
            observation = state._get_observation()
            action_mask = state.action_mask

            # Prepare input for the policy network
            buffer_tensor = torch.tensor(observation['buffer'], dtype=torch.float32)
            ems_tensor = torch.tensor(observation['ems'], dtype=torch.float32)
            # Flatten and concatenate buffer and ems for the network input
            input_tensor = torch.cat((buffer_tensor.flatten(), ems_tensor.flatten())).unsqueeze(0)

            # Get policy logits from the policy network
            policy_logits = self.policy_network(input_tensor).squeeze(0)  # Shape: (num_actions,)

            # Apply action mask: set logits of invalid actions to a very low value
            action_mask_flat = action_mask.flatten()
            masked_logits = policy_logits + (action_mask_flat == 0).float() * -1e9  # Effectively -inf for invalid actions

            # Convert masked logits to probabilities
            action_probs = torch.softmax(masked_logits, dim=0).detach().numpy()

            # If no valid actions, terminate the simulation
            if np.sum(action_mask) == 0:
                break

            # Sample an action based on the probabilities
            action_index = np.random.choice(len(action_probs), p=action_probs / np.sum(action_probs))

            # Decode the action index back to (x, y, rotation, item_index)
            W, L, num_rotations, buffer_size = state.W, state.L, state.num_rotations, state.buffer_size
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

        # Use the value network to estimate the value of the final state
        final_observation = state._get_observation()
        buffer_tensor = torch.tensor(final_observation['buffer'], dtype=torch.float32)
        ems_tensor = torch.tensor(final_observation['ems'], dtype=torch.float32)
        input_tensor = torch.cat((buffer_tensor.flatten(), ems_tensor.flatten())).unsqueeze(0)
        value = self.value_network(input_tensor).squeeze(0).item()
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