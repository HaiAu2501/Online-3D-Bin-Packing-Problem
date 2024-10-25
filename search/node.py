# search/node.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from env.env import BinPacking3DEnv

class Node:
    def __init__(
        self,
        state: BinPacking3DEnv,
        parent: Optional[Node] = None,
        action: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize a node in the MCTS tree.

        :param state: The current state of the environment.
        :param parent: Parent node that leads to this state.
        :param action: Action taken to reach this state.
        """
        self.state: BinPacking3DEnv = state.clone()  # Use the clone method instead of deepcopy
        self.parent: Node = parent
        self.action = action

        self.children: Dict[Tuple[int, int, int, int], Node] = {}  # Mapping from actions to child nodes
        self.visits: int = 0  # Number of times node has been visited
        self.value: float = 0.0  # Accumulated value of the node
        self.is_terminal: bool = False  # Flag to indicate if the node is terminal
        self.untried_actions: List[Tuple[int, int, int, int]] = self.get_valid_actions() 

        self.action_probabilities: Optional[np.ndarray] = None # 

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """
        Retrieve a list of valid actions from the current state.

        :return: A list of valid actions represented as tuples.
        """
        if self.is_terminal:
            return []

        return self.state.valid_actions

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible actions have been tried from this node.

        :return: True if fully expanded, False otherwise.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> Optional[Node]:
        """
        Select the best child node based on the UCB1 formula.

        :param c_param: Exploration parameter.
        :return: The child node with the highest UCB1 value.
        """
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                ucb1 = float('inf')  # Ensure unvisited nodes are selected first
            else:
                exploitation = child.value / child.visits
                exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
                ucb1 = exploitation + exploration
            choices_weights.append(ucb1)

        # Return the child with the maximum UCB1 value
        if choices_weights:
            return list(self.children.values())[np.argmax(choices_weights)]
        else:
            return None

    def expand(self) -> Optional[Node]:
        """
        Expand the node by adding a new child for an untried action.

        :return: The newly created child node or None if no actions to try.
        """
        if self.is_terminal:
            return None

        if not self.untried_actions:
            return None

        # Select an action from the untried actions
        action = self.untried_actions.pop()

        new_state: BinPacking3DEnv = self.state.clone()

        # Apply the action to the current state to get the next state
        _, _, done, truncated, _ = new_state.step(action)

        # Create a new child node
        child_node = Node(state=new_state, parent=self, action=action)
        child_node.is_terminal = done or truncated  # Đánh dấu terminal nếu done hoặc truncated

        self.children[action] = child_node
        return child_node

    def update(self, reward: float):
        """
        Update the node's statistics based on the received reward.

        :param reward: The reward obtained from the simulation.
        """
        self.visits += 1
        self.value += reward  # Accumulate the reward

