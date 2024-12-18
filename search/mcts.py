# search/mcts.py

from __future__ import annotations

import math
from typing import Optional, Tuple, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv
    from models.policy_net import PolicyNetwork
    from models.value_net import ValueNetwork
    from replay_buffer import PrioritizedReplayBuffer

from .node import Node

class MCTS:
    def __init__(
        self,
        env: BinPacking3DEnv,
        policy_net: PolicyNetwork,
        value_net: ValueNetwork,
        replay_buffer: PrioritizedReplayBuffer,
        n_simulations: int = 1000,
        c_param: float = math.sqrt(2),
        verbose: bool = False
    ) -> None:
        """
        Initialize the MCTS search.

        :param env: The environment to search in.
        :param policy_net: The policy network to use for action selection.
        :param value_net: The value network to use for state evaluation.
        :param replay_buffer: The replay buffer to store the experience.
        :param num_simulations: The number of simulations to run.
        :param c_param: The exploration parameter for UCB1.
        """
        env.reset() # Reset the environment to the initial state
        self.env: BinPacking3DEnv = env.clone()
        self.policy_net: PolicyNetwork = policy_net
        self.value_net: ValueNetwork = value_net
        self.replay_buffer: PrioritizedReplayBuffer = replay_buffer
        self.n_simulations: int = n_simulations
        self.c_param: float = c_param
        self.verbose: bool = verbose

        # Root node of the MCTS tree is the initial state of the environment
        self.root: Node = Node(state=self.env)

    def search(self):
        """
        The main search function to run the MCTS algorithm.
        """
        for sim in range(self.n_simulations):
            if self.verbose:
                print(f"Simulation {sim + 1}/{self.n_simulations} started.")

            node: Node = self.root
            path: List[BinPacking3DEnv] = [node]  # To keep track of the path for backpropagation

            # SELECTION 
            # Traverse the tree until a node is found that can be expanded
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.c_param)
                path.append(node)
                if node.is_terminal:
                    break  # Nếu node là terminal, kết thúc selection

            # EXPANSION 
            # If the node is not fully expanded and not terminal, expand it by adding a child
            if not node.is_fully_expanded() and not node.is_terminal:
                policy_probs = self.policy_net(node.state)  # Get the policy probabilities from the policy network
                action, prior_prob = self._get_untried_action(node)  # Get an untried action from the node
                child_node = Node(state=node.state, parent=node, action=action, prior_prob=prior_prob)
                node.children[action] = child_node
                path.append(child_node)
                if child_node:
                    path.append(child_node)
                    node = child_node

            # SIMULATION (ROLLOUT) 
            # Perform a simulation from the node's state
            total_reward, done = self._simulate(node.state)

            # BACKPROPAGATION 
            self._backpropagate(path, total_reward)

            # COLLECT EXPERIENCE
            self._store_experience(path, total_reward)

    def _get_policy_probs(self, state: BinPacking3DEnv) -> Tuple[np.ndarray, float]:
        """
        Get the policy probabilities from the policy network for the given state.

        :param state: The current state of the environment.
        :return: A tuple containing the action and the prior probability of selecting the action.
        """
        pass    

    def _get_untried_action(self, node: Node) -> Tuple[int, int, int, int]:
        """
        Get an untried action from the given node.

        :param node: The current node in the MCTS tree.
        :return: A tuple representing the action.
        """
        pass

    def _simulate(self, state: BinPacking3DEnv) -> Tuple[float, bool]:
        """
        Perform a simulation (rollout) from the given state to estimate the value.

        :param state: The current state of the environment.
        :return: A tuple containing the total reward from the simulation and the done flag.
        """

    def _backpropagate(self, path: List[Node], reward: float):
        """
        Backpropagate the reward through the nodes from the given path to the root.

        :param path: The list of nodes from root to the current node.
        :param reward: The total reward from the simulation.
        """
        for node in path:
            node.visits += 1
            node.value += reward  # Accumulate the total reward from the simulation

    def _store_experience(self, path: List[Node], reward: float):
        """
        Placeholder function to store the experience into the replay buffer.

        :param path: The list of nodes from root to the current node.
        :param reward: The total reward obtained from the simulation.
        """
        # TODO: Implement the logic to store the experience into the replay buffer
        pass
