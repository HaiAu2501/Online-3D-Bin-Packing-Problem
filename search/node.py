# search/node.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import copy
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

        :param state: The current state of the environment (as a copy of an BinPacking3DEnv instance).
        :param parent: Parent node that leads to this state.
        :param action: Action taken to reach this state.
        """
        self.state = state
        self.parent = parent
        self.action = action

        self.children: Dict[Tuple[int, int, int, int], Node] = {}
        self.visits: int = 0
        self.value: float = 0.0
        self.untried_actions: List[Tuple[int, int, int, int]] = self.get_valid_actions()

    def get_valid_actions(self) -> List[Tuple[int, int, int, int]]:
        """
        Get a list of valid actions that can be taken from the current state.

        :return: A list of valid actions.
        """
        action_mask = self.state.generate_action_mask()
        W, L, num_rotations, buffer_size = self.state.W, self.state.L, self.state.num_rotations, self.state.buffer_size
        valid_actions = []
        for x in range(W):
            for y in range(L):
                for rot in range(num_rotations):
                    for buf_idx in range(buffer_size):
                        if action_mask[x, y, rot, buf_idx]:
                            valid_actions.append((x, y, rot, buf_idx))
        return valid_actions

    def is_fully_expanded(self) -> bool:
        """
        Check if all actions have been tried from the current state.

        :return: True if all actions have been tried, False otherwise.
        """
        return len(self.untried_actions) == 0

    def best_child(self, c_param: float = math.sqrt(2)) -> 'Node':
        """
        Select the best child node based on the UCB formula.

        :param c_param: Exploration and exploitation trade-off parameter.
        :return: Best child node based on the UCB formula.
        """
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children.values()
        ]
        return list(self.children.values())[choices_weights.index(max(choices_weights))]

    def expand(self) -> Node:
        """
        Expand the current node by selecting an untried action.

        :return: The child node after taking the selected action.
        """
        action = self.untried_actions.pop()
        # Tạo bản sao của môi trường để thực hiện hành động
        next_state = copy.deepcopy(self.state)
        _, _, done, _, _ = next_state.step(action)
        child_node = Node(state=next_state, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def update(self, reward: float):
        """
        Update the value and visit count of the node.

        :param reward: The reward received after taking the action.
        """
        self.visits += 1
        self.value += reward

    def fully_expanded_children(self) -> List['Node']:
        """
        Get a list of fully expanded child nodes.

        :return: A list of fully expanded child nodes.
        """
        return [child for child in self.children.values() if child.is_fully_expanded()]
